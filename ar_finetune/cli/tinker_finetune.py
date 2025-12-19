from __future__ import annotations

import argparse
from pathlib import Path

from ar_finetune.repo_utils import find_repo_root
from ar_finetune.tinker_train import TinkerTrainParams, build_train_config, get_last_sampler_path, run_train_blocking


def main() -> None:
    repo_root = find_repo_root()

    p = argparse.ArgumentParser(description="Run Tinker supervised fine-tuning (SFT).")
    p.add_argument("--conversations", type=str, default=str(repo_root / "data/finetuning/tinker_conversations_spam.jsonl"))
    p.add_argument("--log-path", type=str, default="/tmp/tinker-examples/sl_ar_phishing_spam_cli")
    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=20)
    args = p.parse_args()

    params = TinkerTrainParams(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    config = build_train_config(
        repo_root=repo_root,
        conversations_jsonl=Path(args.conversations),
        log_path=Path(args.log_path),
        params=params,
    )
    run_train_blocking(config)

    sampler_path = get_last_sampler_path(repo_root=repo_root, log_path=Path(args.log_path))
    print("log_path:", args.log_path)
    print("sampler_path:", sampler_path)


if __name__ == "__main__":
    main()


