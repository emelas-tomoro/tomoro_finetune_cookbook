from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def read_table(path: str | Path) -> "pd.DataFrame":
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(p, sep="\t" if suffix == ".tsv" else ",")
    if suffix in {".jsonl"}:
        return pd.read_json(p, lines=True)
    if suffix in {".json"}:
        return pd.read_json(p)
    raise ValueError(f"Unsupported input format: {p} (expected parquet/csv/tsv/json/jsonl)")


def write_table(df: "pd.DataFrame", path: str | Path, *, overwrite: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(p, index=False)
        return
    if suffix == ".csv":
        df.to_csv(p, index=False)
        return
    if suffix == ".jsonl":
        df.to_json(p, orient="records", lines=True)
        return
    raise ValueError(f"Unsupported output format: {p} (expected parquet/csv/jsonl)")

