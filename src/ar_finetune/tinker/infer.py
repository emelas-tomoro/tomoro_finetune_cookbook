from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from ar_finetune.datasets.conversations import build_email_user_prompt
from ar_finetune.io_utils import normalize_boolish


@dataclass(frozen=True)
class InferConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    log_path: str = "/tmp/tinker-examples/sl_ar_phishing_spam"
    system_prompt_path: str | None = None
    system_prompt: str | None = None

    concurrency: int = 16
    max_rows: int | None = None
    max_tokens: int = 64
    verbose_responses: bool = False

    # Local adapter mode (Transformers + PEFT)
    use_local_adapter: bool = False
    base_model_dir: str | None = None
    adapter_dir: str | None = None


def parse_is_spam_and_notes(text: str) -> tuple[bool | None, str | None]:
    # is_spam: -1/0/1/true/false
    m = re.search(r"\bis_spam\s*:\s*(-1|0|1|true|false)\b", text, flags=re.IGNORECASE)
    is_spam = normalize_boolish(m.group(1)) if m else None
    notes_match = re.search(r"\bagent_notes\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    notes = notes_match.group(1).strip() if notes_match else None
    if notes == "":
        notes = None
    return is_spam, notes


def _render_role_colon(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"{role.title()}: {content}")
    return "\n\n".join(parts) + "\n\nAssistant:"


async def _make_local_completer(
    *,
    base_model_dir: str,
    adapter_dir: str,
    max_new_tokens: int,
) -> Any:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_dir = Path(base_model_dir).expanduser()
    ad_dir = Path(adapter_dir).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base model not found: {base_dir}")
    if not ad_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {ad_dir}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=dtype)
    base_model.to(device)

    model = PeftModel.from_pretrained(base_model, ad_dir)
    model.eval()

    async def completer(messages: list[dict[str, str]]) -> dict[str, str]:
        prompt = _render_role_colon(messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen = out[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        text = text.split("\n\nUser:", 1)[0].strip()
        return {"role": "assistant", "content": text}

    return completer


async def _make_remote_completer(*, model_name: str, log_path: str, max_tokens: int) -> Any:
    import tinker
    from tinker_cookbook import model_info
    from tinker_cookbook.checkpoint_utils import get_last_checkpoint
    from tinker_cookbook.completers import TinkerMessageCompleter
    from tinker_cookbook import renderers as cookbook_renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service_client = tinker.ServiceClient()
    ckpt_sampler = get_last_checkpoint(str(log_path), required_key="sampler_path")
    if not ckpt_sampler:
        raise RuntimeError(
            f"No sampler checkpoint found under {log_path}. "
            "Make sure training ran and wrote checkpoints.jsonl."
        )
    sampling_client = service_client.create_sampling_client(model_path=ckpt_sampler["sampler_path"])

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tok = get_tokenizer(model_name)
    renderer = cookbook_renderers.get_renderer(renderer_name, tok)

    return TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=max_tokens,
    )


async def infer_df(
    df: "pd.DataFrame",
    *,
    cfg: InferConfig,
    subject_col: str = "subject",
    body_col: str = "body",
    ticket_id_col: str = "ticket_id",
) -> "pd.DataFrame":
    import pandas as pd
    from dotenv import load_dotenv
    from tqdm.auto import tqdm

    load_dotenv(override=False)

    if cfg.system_prompt is not None:
        system_prompt = cfg.system_prompt
    elif cfg.system_prompt_path is not None:
        system_prompt = Path(cfg.system_prompt_path).read_text(encoding="utf-8")
    else:
        raise ValueError("Provide --system-prompt or --system-prompt-path")

    if cfg.use_local_adapter:
        if not cfg.base_model_dir or not cfg.adapter_dir:
            raise ValueError("Local adapter mode requires --base-model-dir and --adapter-dir")
        completer = await _make_local_completer(
            base_model_dir=cfg.base_model_dir,
            adapter_dir=cfg.adapter_dir,
            max_new_tokens=cfg.max_tokens,
        )
    else:
        completer = await _make_remote_completer(
            model_name=cfg.model_name,
            log_path=cfg.log_path,
            max_tokens=cfg.max_tokens,
        )

    rows = list(df.iterrows())
    if cfg.max_rows is not None:
        rows = rows[: cfg.max_rows]

    sem = asyncio.Semaphore(cfg.concurrency)

    async def process_single(row_idx: Any, row: pd.Series) -> dict[str, Any]:
        async with sem:
            subject = str(row.get(subject_col, "") or "")
            body = str(row.get(body_col, "") or "")
            ticket_id = row.get(ticket_id_col, row_idx)
            user_prompt = build_email_user_prompt(subject=subject, body=body)
            convo = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                assistant_msg = await completer(convo)
                text = str(assistant_msg.get("content", "") or "").strip()
                if cfg.verbose_responses:
                    print({"ticket_id": ticket_id, "assistant": text})
                is_spam, notes = parse_is_spam_and_notes(text)
                return {
                    "ticket_id": ticket_id,
                    "is_spam_pred": is_spam,
                    "agent_notes_pred": notes,
                    "raw": text,
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {
                    "ticket_id": ticket_id,
                    "is_spam_pred": None,
                    "agent_notes_pred": None,
                    "raw": None,
                    "success": False,
                    "error": repr(e),
                }

    tasks = [asyncio.create_task(process_single(i, r)) for i, r in rows]
    results: list[dict[str, Any]] = []
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await fut)
    return pd.DataFrame(results)


def infer_df_sync(df: "pd.DataFrame", *, cfg: InferConfig, **kwargs: Any) -> "pd.DataFrame":
    return asyncio.run(infer_df(df, cfg=cfg, **kwargs))

