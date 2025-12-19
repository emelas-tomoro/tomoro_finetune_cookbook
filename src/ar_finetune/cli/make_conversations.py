from __future__ import annotations

from pathlib import Path

from ar_finetune.cli._common import base_parser
from ar_finetune.datasets.conversations import (
    EmailFields,
    df_to_training_examples,
    write_tinker_conversations_jsonl,
)
from ar_finetune.datasets.emails import BalanceSplitConfig, balance_and_split
from ar_finetune.io_utils import write_json
from ar_finetune.tabular import read_table, write_table


def main(argv: list[str] | None = None) -> int:
    p = base_parser(
        prog="ar-ft-make-conversations",
        description="Build training_data.json and (optionally) Tinker conversation JSONL.",
    )
    p.add_argument("--input", required=True, help="Input dataset (csv/parquet/jsonl). Must have subject/body/label columns.")
    p.add_argument("--system-prompt-path", required=True, help="Path to system prompt markdown/text file.")
    p.add_argument("--training-json-out", required=True, help="Output path for training_data.json")
    p.add_argument("--tinker-jsonl-out", help="Optional output path for tinker_conversations.jsonl")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting output files.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of rows (after optional split).")

    p.add_argument("--subject-col", default="subject")
    p.add_argument("--body-col", default="body")
    p.add_argument("--label-col", default="label")

    p.add_argument("--make-split", action="store_true", help="Balance and split into train/test like the notebook.")
    p.add_argument("--train-out", default=None, help="If --make-split, write train split here (parquet/csv).")
    p.add_argument("--test-out", default=None, help="If --make-split, write test split here (parquet/csv).")
    p.add_argument("--sample-frac", type=float, default=0.2, help="If --make-split, fraction sampled after balancing.")
    p.add_argument("--test-size", type=float, default=0.2, help="If --make-split, test size.")
    p.add_argument("--random-state", type=int, default=42, help="If --make-split, RNG seed.")

    args = p.parse_args(argv)

    df = read_table(args.input)

    if args.make_split:
        cfg = BalanceSplitConfig(
            sample_frac=args.sample_frac,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_df, test_df = balance_and_split(df, label_col=args.label_col, cfg=cfg)

        if args.train_out:
            write_table(train_df, args.train_out, overwrite=args.overwrite)
        if args.test_out:
            write_table(test_df, args.test_out, overwrite=args.overwrite)

        df_for_training = train_df
    else:
        df_for_training = df

    system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8")
    fields = EmailFields(subject=args.subject_col, body=args.body_col, label=args.label_col)
    examples = df_to_training_examples(df_for_training, system_prompt=system_prompt, fields=fields, limit=args.limit)

    write_json(args.training_json_out, examples, overwrite=args.overwrite, indent=2)

    if args.tinker_jsonl_out:
        write_tinker_conversations_jsonl(
            training_examples=examples,
            output_path=args.tinker_jsonl_out,
            overwrite=args.overwrite,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

