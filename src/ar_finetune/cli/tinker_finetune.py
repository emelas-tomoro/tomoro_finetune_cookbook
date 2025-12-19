from __future__ import annotations

from ar_finetune.cli._common import base_parser
from ar_finetune.tinker.sft import SFTConfig, convert_training_json_to_conversations_jsonl, run_sft_sync


def main(argv: list[str] | None = None) -> int:
    p = base_parser(
        prog="ar-ft-tinker-finetune",
        description="Run a Tinker Cookbook supervised fine-tune (SFT).",
    )
    p.add_argument("--training-json", required=True, help="Path to training_data.json (list of {instruction,input,output}).")
    p.add_argument("--conversations-jsonl", required=True, help="Path to write conversation JSONL for Tinker.")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting conversations jsonl.")
    p.add_argument("--dry-run", action="store_true", help="Convert data and print config, but do not start training.")

    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--log-path", default="/tmp/tinker-examples/sl_ar_phishing_spam")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-on-what", default="ALL_ASSISTANT_MESSAGES")
    p.add_argument("--test-size", type=int, default=128)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lr-schedule", default="linear")
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=20)

    args = p.parse_args(argv)

    convert_training_json_to_conversations_jsonl(
        training_json_path=args.training_json,
        out_jsonl_path=args.conversations_jsonl,
        overwrite=args.overwrite,
    )

    cfg = SFTConfig(
        model_name=args.model_name,
        log_path=args.log_path,
        dataset_jsonl_path=args.conversations_jsonl,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=args.train_on_what,
        test_size=args.test_size,
        shuffle_seed=args.shuffle_seed,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    if args.dry_run:
        print("Dry run; not starting training.")
        print(cfg)
        return 0

    sampler_path = run_sft_sync(cfg)
    if sampler_path:
        print("Final sampler_path:", sampler_path)
    else:
        print("Training finished, but no sampler_path found yet (check log path).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

