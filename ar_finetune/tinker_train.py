from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from ar_finetune.repo_utils import ensure_tinker_cookbook_importable


@dataclass(frozen=True)
class TinkerTrainParams:
    model_name: str = "meta-llama/Llama-3.1-8B"
    max_length: int = 8192
    batch_size: int = 16
    learning_rate: float = 2e-4
    lr_schedule: str = "linear"
    num_epochs: int = 1
    eval_every: int = 10
    save_every: int = 20
    test_size: int = 128
    shuffle_seed: int = 0


def build_train_config(
    *,
    repo_root: Path,
    conversations_jsonl: Path,
    log_path: Path,
    params: TinkerTrainParams,
):
    """
    Mirrors `notebooks_sl/tinker_finetune.ipynb` training config creation.
    Returns a `tinker_cookbook.supervised.train.Config`.
    """
    ensure_tinker_cookbook_importable(repo_root)

    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer_name = model_info.get_recommended_renderer_name(params.model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=params.model_name,
        renderer_name=renderer_name,
        max_length=params.max_length,
        batch_size=params.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(conversations_jsonl.resolve()),
        test_size=params.test_size,
        shuffle_seed=params.shuffle_seed,
    )

    config = train.Config(
        log_path=str(log_path.expanduser().resolve()),
        model_name=params.model_name,
        dataset_builder=dataset_builder,
        learning_rate=params.learning_rate,
        lr_schedule=params.lr_schedule,
        num_epochs=params.num_epochs,
        eval_every=params.eval_every,
        save_every=params.save_every,
    )
    return config


async def run_train_async(config) -> None:
    from tinker_cookbook.supervised import train

    await train.main(config)


def run_train_blocking(config) -> None:
    """
    Run training synchronously (handy for CLIs).
    """
    asyncio.run(run_train_async(config))


def get_last_sampler_path(*, repo_root: Path, log_path: Path) -> str | None:
    ensure_tinker_cookbook_importable(repo_root)
    from tinker_cookbook.checkpoint_utils import get_last_checkpoint

    ckpt = get_last_checkpoint(str(log_path), required_key="sampler_path")
    if not ckpt:
        return None
    return ckpt.get("sampler_path")


