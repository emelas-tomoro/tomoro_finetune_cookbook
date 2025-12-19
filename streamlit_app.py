from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ar_finetune.conversations import training_data_json_to_conversation_jsonl
from ar_finetune.datasets import (
    balance_binary_labels,
    build_email_prompts_df,
    split_train_test,
    write_parquet,
    write_training_data_json,
)
from ar_finetune.metrics import compute_binary_metrics
from ar_finetune.repo_utils import find_repo_root
from ar_finetune.tinker_infer import TinkerInferParams, infer_spam_df_blocking
from ar_finetune.tinker_train import TinkerTrainParams, build_train_config, get_last_sampler_path, run_train_blocking


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("repo_root", str(find_repo_root(Path(__file__).resolve().parent)))
    ss.setdefault("run_tag", "spam_ui")
    ss.setdefault("system_prompt_path", "data/verification/system_prompt_spam.md")

    ss.setdefault("dataset_train_path", "data/finetuning/lora_train_emails.parquet")
    ss.setdefault("dataset_test_path", "data/finetuning/lora_test_emails.parquet")
    ss.setdefault("training_data_json_path", "data/finetuning/training_data_spam.json")
    ss.setdefault("conversations_jsonl_path", "data/finetuning/tinker_conversations_spam.jsonl")

    ss.setdefault("log_path", "/tmp/tinker-examples/sl_ar_phishing_spam_ui")
    ss.setdefault("model_name", "meta-llama/Llama-3.1-8B")

    ss.setdefault("last_sampler_path", "")
    ss.setdefault("df_infer", None)


def _abs(repo_root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo_root / path)


@st.cache_data(show_spinner=False)
def _load_df_from_path(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")


@st.cache_data(show_spinner=False)
def _load_df_from_upload(name: str, data: bytes) -> pd.DataFrame:
    p = Path(name)
    buf = BytesIO(data)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(buf)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(buf)
    raise ValueError(f"Unsupported upload type: {p.suffix}")


def main() -> None:
    st.set_page_config(page_title="AR fine-tune UI", layout="wide")
    _init_state()

    ss = st.session_state
    repo_root = Path(ss["repo_root"]).expanduser().resolve()
    # Streamlit does NOT auto-load .env; make it match the notebooks.
    load_dotenv(dotenv_path=repo_root / ".env", override=False)

    st.title("Fine-tune + Inference UI (Tinker SFT)")

    with st.sidebar:
        st.subheader("Project")
        ss["repo_root"] = st.text_input("Repo root", ss["repo_root"])
        ss["run_tag"] = st.text_input("Run tag", ss["run_tag"])
        ss["model_name"] = st.text_input("Model name", ss["model_name"])
        ss["system_prompt_path"] = st.text_input("System prompt path", ss["system_prompt_path"])

        has_key = bool(os.environ.get("TINKER_API_KEY"))
        st.caption(
            "Tinker auth: "
            + ("`TINKER_API_KEY` detected" if has_key else "`TINKER_API_KEY` missing")
            + " (loaded from env / repo `.env`)"
        )

    tab_dataset, tab_train, tab_infer = st.tabs(["Dataset", "Fine-tune", "Inference + Metrics"])

    with tab_dataset:
        st.subheader("Dataset")

        col_a, col_b = st.columns(2)
        with col_a:
            mode = st.radio(
                "Mode",
                ["Use existing files", "Build from an uploaded emails dataset"],
                horizontal=False,
            )

        with col_b:
            st.write("Expected columns for emails datasets:")
            st.code("subject, body, label (bool)")

        if mode == "Use existing files":
            ss["dataset_train_path"] = st.text_input("Train parquet", ss["dataset_train_path"])
            ss["dataset_test_path"] = st.text_input("Test parquet", ss["dataset_test_path"])
            ss["training_data_json_path"] = st.text_input("training_data.json", ss["training_data_json_path"])
            ss["conversations_jsonl_path"] = st.text_input("tinker_conversations.jsonl", ss["conversations_jsonl_path"])

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Preview train parquet"):
                    df = _load_df_from_path(str(_abs(repo_root, ss["dataset_train_path"])))
                    st.dataframe(df.head(50), use_container_width=True)
            with col2:
                if st.button("Preview test parquet"):
                    df = _load_df_from_path(str(_abs(repo_root, ss["dataset_test_path"])))
                    st.dataframe(df.head(50), use_container_width=True)

            st.divider()
            st.write("Convert `training_data.json` → `tinker_conversations.jsonl`")
            overwrite = st.checkbox("Overwrite outputs", value=False)
            if st.button("Convert now"):
                in_path = _abs(repo_root, ss["training_data_json_path"])
                out_path = _abs(repo_root, ss["conversations_jsonl_path"])
                training_data_json_to_conversation_jsonl(in_path=in_path, out_path=out_path, overwrite=overwrite)
                st.success(f"Wrote {out_path}")

        else:
            upload = st.file_uploader("Upload emails dataset (.parquet or .csv)", type=["parquet", "csv"])
            sample_frac = st.slider("Sample fraction", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
            balance = st.checkbox("Balance labels (downsample majority class)", value=True)
            test_size = st.slider("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
            random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

            ss["dataset_train_path"] = st.text_input(
                "Output train parquet",
                f"data/finetuning/lora_train_{ss['run_tag']}.parquet",
            )
            ss["dataset_test_path"] = st.text_input(
                "Output test parquet",
                f"data/finetuning/lora_test_{ss['run_tag']}.parquet",
            )
            ss["training_data_json_path"] = st.text_input(
                "Output training_data.json",
                f"data/finetuning/training_data_{ss['run_tag']}.json",
            )
            ss["conversations_jsonl_path"] = st.text_input(
                "Output tinker_conversations.jsonl",
                f"data/finetuning/tinker_conversations_{ss['run_tag']}.jsonl",
            )
            overwrite = st.checkbox("Overwrite outputs", value=False)

            if upload is None:
                st.info("Upload a dataset to build train/test + training files.")
            else:
                df0 = _load_df_from_upload(upload.name, upload.getvalue())
                if sample_frac < 1.0:
                    df0 = df0.sample(frac=sample_frac, random_state=int(random_state)).reset_index(drop=True)
                if balance:
                    df0 = balance_binary_labels(df0, random_state=int(random_state))
                train_df, test_df = split_train_test(df0, test_size=float(test_size), random_state=int(random_state))

                st.write("Preview (train)")
                st.dataframe(train_df.head(20), use_container_width=True)

                if st.button("Write dataset + training files"):
                    sys_prompt = _abs(repo_root, ss["system_prompt_path"]).read_text(encoding="utf-8")
                    train_out = _abs(repo_root, ss["dataset_train_path"])
                    test_out = _abs(repo_root, ss["dataset_test_path"])
                    write_parquet(df=train_df, out_path=train_out, overwrite=overwrite)
                    write_parquet(df=test_df, out_path=test_out, overwrite=overwrite)

                    prompts = build_email_prompts_df(df=train_df, system_prompt=sys_prompt)
                    td_out = _abs(repo_root, ss["training_data_json_path"])
                    write_training_data_json(prompts_df=prompts, out_path=td_out, overwrite=overwrite)

                    conv_out = _abs(repo_root, ss["conversations_jsonl_path"])
                    training_data_json_to_conversation_jsonl(
                        in_path=td_out, out_path=conv_out, overwrite=overwrite
                    )

                    st.success("Wrote train/test parquet + training_data.json + tinker_conversations.jsonl")

    with tab_train:
        st.subheader("Fine-tune (Tinker supervised)")

        ss["log_path"] = st.text_input("Log path", ss["log_path"])
        ss["conversations_jsonl_path"] = st.text_input("Conversations JSONL", ss["conversations_jsonl_path"])

        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.number_input("Batch size", min_value=1, value=16, step=1)
            max_length = st.number_input("Max length", min_value=256, value=8192, step=256)
        with col2:
            learning_rate = st.number_input("Learning rate", min_value=0.0, value=2e-4, step=1e-4, format="%.6f")
            num_epochs = st.number_input("Epochs", min_value=1, value=1, step=1)
        with col3:
            eval_every = st.number_input("Eval every", min_value=1, value=10, step=1)
            save_every = st.number_input("Save every", min_value=1, value=20, step=1)

        params = TinkerTrainParams(
            model_name=ss["model_name"],
            max_length=int(max_length),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            num_epochs=int(num_epochs),
            eval_every=int(eval_every),
            save_every=int(save_every),
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start training"):
                try:
                    with st.spinner("Training (this can take a while)..."):
                        config = build_train_config(
                            repo_root=repo_root,
                            conversations_jsonl=_abs(repo_root, ss["conversations_jsonl_path"]),
                            log_path=Path(ss["log_path"]),
                            params=params,
                        )
                        run_train_blocking(config)
                    sp = get_last_sampler_path(repo_root=repo_root, log_path=Path(ss["log_path"]))
                    ss["last_sampler_path"] = sp or ""
                except Exception as e:
                    st.exception(e)
        with col_b:
            if st.button("Refresh sampler path from checkpoints"):
                sp = get_last_sampler_path(repo_root=repo_root, log_path=Path(ss["log_path"]))
                ss["last_sampler_path"] = sp or ""

        if ss.get("last_sampler_path"):
            st.success("Last sampler path:")
            st.code(ss["last_sampler_path"])

    with tab_infer:
        st.subheader("Inference + Metrics")

        ss["dataset_test_path"] = st.text_input("Eval parquet", ss["dataset_test_path"])
        sampler_path = st.text_input("Sampler path (tinker://...)", ss.get("last_sampler_path", ""))
        max_rows = st.number_input("Max rows (0 = no limit)", min_value=0, value=0, step=10)
        concurrency = st.number_input("Concurrency", min_value=1, value=16, step=1)

        if st.button("Run inference"):
            if not sampler_path.strip():
                st.error("Sampler path is required (from training checkpoints).")
            else:
                try:
                    with st.spinner("Running inference..."):
                        df_eval = pd.read_parquet(_abs(repo_root, ss["dataset_test_path"]))
                        df_pred = infer_spam_df_blocking(
                            repo_root=repo_root,
                            sampler_path=sampler_path,
                            df_eval=df_eval,
                            system_prompt_path=_abs(repo_root, ss["system_prompt_path"]),
                            params=TinkerInferParams(model_name=ss["model_name"], concurrency=int(concurrency)),
                            max_rows=None if int(max_rows) == 0 else int(max_rows),
                        )
                        ss["df_infer"] = df_pred
                except Exception as e:
                    st.exception(e)

        df_pred = ss.get("df_infer")
        if isinstance(df_pred, pd.DataFrame) and len(df_pred) > 0:
            st.write("Predictions")
            st.dataframe(df_pred.head(50), use_container_width=True)

            # Metrics (only if label exists in eval parquet, merged by ticket_id if present)
            df_eval = pd.read_parquet(_abs(repo_root, ss["dataset_test_path"]))
            if "ticket_id" in df_eval.columns:
                df_eval2 = df_eval
            else:
                df_eval2 = df_eval.reset_index().rename(columns={"index": "ticket_id"})

            merged = df_eval2.merge(df_pred[["ticket_id", "is_spam_pred"]], on="ticket_id", how="inner")
            mask = merged["is_spam_pred"].notna()
            if "label" in merged.columns and mask.any():
                y_true = merged.loc[mask, "label"].astype(bool).tolist()
                y_pred = merged.loc[mask, "is_spam_pred"].astype(bool).tolist()
                m = compute_binary_metrics(y_true=y_true, y_pred=y_pred)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{m.accuracy:.4f}")
                c2.metric("Precision", f"{m.precision:.4f}")
                c3.metric("Recall", f"{m.recall:.4f}")
                c4.metric("F1", f"{m.f1:.4f}")

                st.write("Confusion matrix [[TN, FP], [FN, TP]]")
                st.dataframe(
                    pd.DataFrame(
                        m.confusion_matrix,
                        columns=["Pred False", "Pred True"],
                        index=["True False", "True True"],
                    )
                )
            else:
                st.info("No metrics computed (need `label` column in eval dataset, and non-null predictions).")


if __name__ == "__main__":
    main()


