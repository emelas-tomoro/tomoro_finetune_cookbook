from __future__ import annotations

from ar_finetune.bench.gpt import GPTBenchmarkConfig, benchmark_df_sync
from ar_finetune.cli._common import base_parser
from ar_finetune.tabular import read_table, write_table


def main(argv: list[str] | None = None) -> int:
    p = base_parser(
        prog="ar-ft-gpt-benchmark",
        description="Run GPT benchmark over an email dataset (OpenAI Responses API).",
    )
    p.add_argument("--input", required=True, help="Input dataset (csv/parquet/jsonl) with subject/body columns.")
    p.add_argument("--output", required=True, help="Output results (parquet/csv/jsonl).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--system-prompt-path", required=True, help="System prompt file (markdown/text).")
    p.add_argument("--model", default="gpt-5.2-2025-12-11")
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--max-rows", type=int, default=None)

    args = p.parse_args(argv)

    df = read_table(args.input)
    cfg = GPTBenchmarkConfig(
        model=args.model,
        system_prompt_path=args.system_prompt_path,
        concurrency=args.concurrency,
        max_rows=args.max_rows,
    )
    out_df = benchmark_df_sync(df, cfg=cfg)
    write_table(out_df, args.output, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

