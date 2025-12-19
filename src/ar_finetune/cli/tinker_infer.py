from __future__ import annotations

from ar_finetune.cli._common import base_parser
from ar_finetune.tabular import read_table, write_table
from ar_finetune.tinker.infer import InferConfig, infer_df_sync


def main(argv: list[str] | None = None) -> int:
    p = base_parser(
        prog="ar-ft-tinker-infer",
        description="Run inference using either a remote Tinker sampler checkpoint or a local LoRA adapter.",
    )
    p.add_argument("--input", required=True, help="Input dataset (csv/parquet/jsonl) with subject/body columns.")
    p.add_argument("--output", required=True, help="Output results (parquet/csv/jsonl).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--system-prompt-path", required=True, help="System prompt file (markdown/text).")

    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--log-path", default="/tmp/tinker-examples/sl_ar_phishing_spam")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--verbose-responses", action="store_true")

    p.add_argument("--use-local-adapter", action="store_true", help="Use local Transformers+PEFT inference instead of remote Tinker.")
    p.add_argument("--base-model-dir", default=None, help="Local base model directory (Transformers).")
    p.add_argument("--adapter-dir", default=None, help="Local adapter directory (PEFT).")

    p.add_argument("--subject-col", default="subject")
    p.add_argument("--body-col", default="body")
    p.add_argument("--ticket-id-col", default="ticket_id")

    args = p.parse_args(argv)

    df = read_table(args.input)
    cfg = InferConfig(
        model_name=args.model_name,
        log_path=args.log_path,
        system_prompt_path=args.system_prompt_path,
        concurrency=args.concurrency,
        max_rows=args.max_rows,
        max_tokens=args.max_tokens,
        verbose_responses=args.verbose_responses,
        use_local_adapter=args.use_local_adapter,
        base_model_dir=args.base_model_dir,
        adapter_dir=args.adapter_dir,
    )

    out_df = infer_df_sync(
        df,
        cfg=cfg,
        subject_col=args.subject_col,
        body_col=args.body_col,
        ticket_id_col=args.ticket_id_col,
    )

    write_table(out_df, args.output, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

