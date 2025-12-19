from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from ar_finetune.datasets.conversations import build_email_user_prompt
from ar_finetune.io_utils import normalize_boolish


@dataclass(frozen=True)
class GPTBenchmarkConfig:
    model: str = "gpt-5.2-2025-12-11"
    system_prompt_path: str | None = None
    system_prompt: str | None = None
    concurrency: int = 50
    max_rows: int | None = None


def parse_is_spam(text: str) -> bool | None:
    m = re.search(r"\bis_spam\s*:\s*(-1|0|1|true|false)\b", text, flags=re.IGNORECASE)
    return normalize_boolish(m.group(1)) if m else None


async def benchmark_df(df: pd.DataFrame, *, cfg: GPTBenchmarkConfig) -> pd.DataFrame:
    import pandas as pd
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from tqdm.auto import tqdm

    load_dotenv(override=False)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    if cfg.system_prompt is not None:
        system_prompt = cfg.system_prompt
    elif cfg.system_prompt_path is not None:
        system_prompt = Path(cfg.system_prompt_path).read_text(encoding="utf-8")
    else:
        raise ValueError("Provide --system-prompt or --system-prompt-path")

    client = AsyncOpenAI(api_key=api_key)

    rows = list(df.iterrows())
    if cfg.max_rows is not None:
        rows = rows[: cfg.max_rows]

    sem = asyncio.Semaphore(cfg.concurrency)

    async def process_single(row_idx: Any, row: pd.Series) -> dict[str, Any]:
        async with sem:
            subject = str(row.get("subject", "") or "")
            body = str(row.get("body", "") or "")
            ticket_id = row.get("ticket_id", row_idx)
            user_prompt = build_email_user_prompt(subject=subject, body=body).strip()

            try:
                resp = await client.responses.create(
                    model=cfg.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = (getattr(resp, "output_text", "") or "").strip()
                return {
                    "ticket_id": ticket_id,
                    "is_spam_pred": parse_is_spam(text),
                    "raw": text,
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {
                    "ticket_id": ticket_id,
                    "is_spam_pred": None,
                    "raw": None,
                    "success": False,
                    "error": repr(e),
                }

    tasks = [asyncio.create_task(process_single(i, r)) for i, r in rows]
    results: list[dict[str, Any]] = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await fut)
    return pd.DataFrame(results)


def benchmark_df_sync(df: pd.DataFrame, *, cfg: GPTBenchmarkConfig) -> pd.DataFrame:
    return asyncio.run(benchmark_df(df, cfg=cfg))

