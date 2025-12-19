from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class BalanceSplitConfig:
    positive_label: bool | int = True
    negative_label: bool | int = False
    downsample_to: Literal["minority"] = "minority"
    sample_frac: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


def balance_and_split(
    df: "pd.DataFrame",
    *,
    label_col: str = "label",
    cfg: BalanceSplitConfig = BalanceSplitConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    pos = df[df[label_col] == cfg.positive_label]
    neg = df[df[label_col] == cfg.negative_label]

    if cfg.downsample_to == "minority":
        n = min(len(pos), len(neg))
        pos_s = pos.sample(n=n, random_state=cfg.random_state) if len(pos) > n else pos
        neg_s = neg.sample(n=n, random_state=cfg.random_state) if len(neg) > n else neg
        balanced = pd.concat([pos_s, neg_s])
    else:
        raise ValueError(f"Unsupported downsample_to={cfg.downsample_to!r}")

    balanced = balanced.sample(frac=cfg.sample_frac, random_state=cfg.random_state).reset_index(drop=True)

    strat = balanced[label_col] if cfg.stratify else None
    train_df, test_df = train_test_split(
        balanced,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=strat,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

