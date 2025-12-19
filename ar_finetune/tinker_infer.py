from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ar_finetune.repo_utils import ensure_tinker_cookbook_importable


_IS_SPAM_RE = re.compile(r"\bis_spam\s*:\s*(-1|0|1|true|false)\b", flags=re.IGNORECASE)


def parse_is_spam(text: str) -> bool | None:
    m = _IS_SPAM_RE.search(text or "")
    if m is None:
        return None
    v = m.group(1).lower()
    if v == "-1":
        return None
    return v in ("1", "true")


def build_email_user_prompt(*, subject: str, body: str) -> str:
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


@dataclass(frozen=True)
class TinkerInferParams:
    model_name: str = "meta-llama/Llama-3.1-8B"
    max_tokens: int = 64
    concurrency: int = 16


def _build_tinker_completer(
    *,
    repo_root: Path,
    sampler_path: str,
    params: TinkerInferParams,
):
    ensure_tinker_cookbook_importable(repo_root)

    import tinker
    from tinker_cookbook import model_info
    from tinker_cookbook.completers import TinkerMessageCompleter
    from tinker_cookbook import renderers as cookbook_renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)

    renderer_name = model_info.get_recommended_renderer_name(params.model_name)
    tok = get_tokenizer(params.model_name)
    renderer = cookbook_renderers.get_renderer(renderer_name, tok)

    return TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=params.max_tokens,
    )


async def _infer_rows_async(
    *,
    tinker_completer,
    df: pd.DataFrame,
    system_prompt: str,
    params: TinkerInferParams,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(params.concurrency)
    rows = list(df.iterrows())
    if max_rows is not None:
        rows = rows[:max_rows]

    async def process_single(row_idx: Any, row: pd.Series) -> dict[str, Any]:
        async with sem:
            try:
                subject = str(row.get("subject", ""))
                body = str(row.get("body", ""))
                ticket_id = row.get("ticket_id", row_idx)

                convo = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_email_user_prompt(subject=subject, body=body)},
                ]

                assistant_msg = await tinker_completer(convo)
                text = str(assistant_msg.get("content", "")).strip()
                pred = parse_is_spam(text)

                return {
                    "ticket_id": ticket_id,
                    "is_spam_pred": pred,
                    "raw": text,
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {
                    "ticket_id": row.get("ticket_id", row_idx),
                    "is_spam_pred": None,
                    "raw": None,
                    "success": False,
                    "error": repr(e),
                }

    tasks = [asyncio.create_task(process_single(row_idx, row)) for row_idx, row in rows]
    return [await t for t in asyncio.as_completed(tasks)]


def infer_spam_df_blocking(
    *,
    repo_root: Path,
    sampler_path: str,
    df_eval: pd.DataFrame,
    system_prompt_path: Path,
    params: TinkerInferParams = TinkerInferParams(),
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Run Tinker inference over an eval dataframe.
    Expects subject/body columns; label is optional (used later for metrics).
    """
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    tinker_completer = _build_tinker_completer(
        repo_root=repo_root, sampler_path=sampler_path, params=params
    )

    async def _run() -> pd.DataFrame:
        rows = await _infer_rows_async(
            tinker_completer=tinker_completer,
            df=df_eval,
            system_prompt=system_prompt,
            params=params,
            max_rows=max_rows,
        )
        return pd.DataFrame(rows)

    return asyncio.run(_run())


