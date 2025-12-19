from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class EmailDatasetPaths:
    train_parquet: Path
    test_parquet: Path
    training_data_json: Path
    conversations_jsonl: Path


def build_email_prompts_df(
    *,
    df: pd.DataFrame,
    system_prompt: str,
) -> pd.DataFrame:
    """
    Build the exact prompt format used in the notebooks (subject/body → user_prompt).
    Expects columns: subject, body, label (bool-like).
    """
    required = {"subject", "body", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _build_user_prompt(subject: str, body: str) -> str:
        return (
            f"""
Email subject:
---------------------------------------------------
\n{subject}\n\n
---------------------------------------------------
Email body:
---------------------------------------------------
\n{body}\n\n
---------------------------------------------------
""".strip()
        )

    out = df.copy()
    out["instruction"] = system_prompt
    out["input"] = [
        _build_user_prompt(str(s), str(b)) for s, b in zip(out["subject"], out["body"], strict=True)
    ]
    out["output"] = [f"is_spam: {bool(x)}" for x in out["label"]]
    return out


def write_training_data_json(
    *,
    prompts_df: pd.DataFrame,
    out_path: Path,
    overwrite: bool = False,
) -> Path:
    """
    Write list[dict] with keys instruction/input/output.
    """
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out_path}. Pass overwrite=True to overwrite."
        )

    records = prompts_df[["instruction", "input", "output"]].to_dict(orient="records")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def balance_binary_labels(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Downsample the majority class to match the minority class size.
    """
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    true_df = df[df[label_col] == True]  # noqa: E712
    false_df = df[df[label_col] == False]  # noqa: E712
    if len(true_df) == 0 or len(false_df) == 0:
        return df.copy()

    if len(true_df) < len(false_df):
        false_down = false_df.sample(n=len(true_df), random_state=random_state)
        balanced = pd.concat([true_df, false_down], axis=0)
    else:
        true_down = true_df.sample(n=len(false_df), random_state=random_state)
        balanced = pd.concat([true_down, false_df], axis=0)

    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def write_parquet(
    *,
    df: pd.DataFrame,
    out_path: Path,
    overwrite: bool = False,
) -> Path:
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out_path}. Pass overwrite=True to overwrite."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


