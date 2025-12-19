from __future__ import annotations

import json
from pathlib import Path


def training_data_json_to_conversation_jsonl(
    *,
    in_path: Path,
    out_path: Path,
    overwrite: bool = False,
) -> Path:
    """
    Convert training_data.json (list of {instruction,input,output}) to Tinker Cookbook
    conversation JSONL: {"messages":[{role,content},...]}.
    """
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out_path}. Pass overwrite=True to overwrite."
        )

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {in_path}, got {type(data)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in data:
            if not isinstance(ex, dict):
                raise TypeError(f"Expected dict examples, got {type(ex)}")
            instruction = (ex.get("instruction") or "").strip()
            user_input = (ex.get("input") or "").strip()
            output = (ex.get("output") or "").strip()

            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": output},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    return out_path


