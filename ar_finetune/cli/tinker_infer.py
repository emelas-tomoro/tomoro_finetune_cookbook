from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ar_finetune.metrics import compute_binary_metrics
from ar_finetune.repo_utils import find_repo_root
from ar_finetune.tinker_infer import TinkerInferParams, infer_spam_df_blocking


def main() -> None:
    repo_root = find_repo_root()

    p = argparse.ArgumentParser(description="Run Tinker inference (spam classifier) and compute metrics.")
    p.add_argument("--sampler-path", type=str, required=True, help="tinker://.../sampler_weights/...")
    p.add_argument(
        "--eval-parquet",
        type=str,
        default=str(repo_root / "data/finetuning/lora_test_emails.parquet"),
    )
    p.add_argument(
        "--system-prompt",
        type=str,
        default=str(repo_root / "data/verification/system_prompt_spam.md"),
    )
    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--max-rows", type=int, default=0, help="0 = no limit")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out-parquet", type=str, default="")
    args = p.parse_args()

    df_eval = pd.read_parquet(args.eval_parquet)
    df_pred = infer_spam_df_blocking(
        repo_root=repo_root,
        sampler_path=args.sampler_path,
        df_eval=df_eval,
        system_prompt_path=Path(args.system_prompt),
        params=TinkerInferParams(model_name=args.model_name, concurrency=args.concurrency),
        max_rows=None if args.max_rows == 0 else args.max_rows,
    )

    # Merge for metrics if possible
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
        print(f"accuracy={m.accuracy:.4f} precision={m.precision:.4f} recall={m.recall:.4f} f1={m.f1:.4f}")
        print("confusion_matrix:", m.confusion_matrix)
    else:
        print("No metrics computed (need label column + non-null predictions).")

    if args.out_parquet:
        Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
        df_pred.to_parquet(args.out_parquet, index=False)
        print("Wrote:", args.out_parquet)


if __name__ == "__main__":
    main()


