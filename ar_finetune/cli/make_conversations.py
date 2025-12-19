from __future__ import annotations

import argparse
from pathlib import Path

from ar_finetune.conversations import training_data_json_to_conversation_jsonl
from ar_finetune.repo_utils import find_repo_root


def main() -> None:
    repo_root = find_repo_root()

    p = argparse.ArgumentParser(description="Convert training_data.json → tinker_conversations.jsonl")
    p.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default=str(repo_root / "data/finetuning/training_data_spam.json"),
        help="Path to training_data*.json",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default=str(repo_root / "data/finetuning/tinker_conversations_spam.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = training_data_json_to_conversation_jsonl(
        in_path=Path(args.in_path),
        out_path=Path(args.out_path),
        overwrite=bool(args.overwrite),
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()


