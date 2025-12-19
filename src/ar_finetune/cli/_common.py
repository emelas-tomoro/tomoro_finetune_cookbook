from __future__ import annotations

import argparse


def base_parser(*, prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print a bit more logging (placeholder).",
    )
    return p


def placeholder_notice(*, command: str) -> str:
    return (
        f"`{command}` is wired up as a CLI entrypoint, but the actual\n"
        "business logic still lives in notebooks in `notebooks_sl/`.\n"
        "Move the relevant notebook code into `src/ar_finetune/...` and call it here."
    )

