from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from ar_finetune.io_utils import write_json, write_jsonl


@dataclass(frozen=True)
class EmailFields:
    subject: str = "subject"
    body: str = "body"
    label: str = "label"
    ticket_id: str = "ticket_id"


def build_email_user_prompt(*, subject: str, body: str) -> str:
    # Matches the notebook formatting used for training + inference.
    return (
        "Email subject:\n"
        "---------------------------------------------------\n"
        f"\n{subject}\n\n"
        "---------------------------------------------------\n"
        "Email body:\n"
        "---------------------------------------------------\n"
        f"\n{body}\n\n"
        "---------------------------------------------------\n"
    )


def df_to_training_examples(
    df: "pd.DataFrame",
    *,
    system_prompt: str,
    fields: EmailFields = EmailFields(),
    limit: int | None = None,
) -> list[dict[str, str]]:
    rows = df
    if limit is not None:
        rows = rows.head(limit)

    out: list[dict[str, str]] = []
    for _, r in rows.iterrows():
        subject = str(r.get(fields.subject, "") or "")
        body = str(r.get(fields.body, "") or "")
        label = r.get(fields.label, None)
        user_prompt = build_email_user_prompt(subject=subject, body=body)
        out.append(
            {
                "instruction": system_prompt,
                "input": user_prompt,
                "output": f"is_spam: {label}",
            }
        )
    return out


def training_examples_to_tinker_conversations_jsonl(
    examples: list[dict[str, str]],
) -> list[dict[str, Any]]:
    # Matches the tinker_finetune.ipynb conversion cell.
    rows: list[dict[str, Any]] = []
    for ex in examples:
        instruction = (ex.get("instruction") or "").strip()
        user_input = (ex.get("input") or "").strip()
        output = (ex.get("output") or "").strip()
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": output},
                ]
            }
        )
    return rows


def write_training_data_json(
    *,
    df: "pd.DataFrame",
    system_prompt: str,
    output_path: str | Path,
    fields: EmailFields = EmailFields(),
    limit: int | None = None,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    examples = df_to_training_examples(df, system_prompt=system_prompt, fields=fields, limit=limit)
    write_json(output_path, examples, overwrite=overwrite, indent=2)
    return examples


def write_tinker_conversations_jsonl(
    *,
    training_examples: list[dict[str, str]],
    output_path: str | Path,
    overwrite: bool = False,
) -> None:
    rows = training_examples_to_tinker_conversations_jsonl(training_examples)
    write_jsonl(output_path, rows, overwrite=overwrite)

