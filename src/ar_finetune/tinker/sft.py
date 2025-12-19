from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from ar_finetune.io_utils import read_json
from ar_finetune.datasets.conversations import (
    training_examples_to_tinker_conversations_jsonl,
)
from ar_finetune.io_utils import write_jsonl


@dataclass(frozen=True)
class SFTConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"
    log_path: str = "/tmp/tinker-examples/sl_ar_phishing_spam"
    dataset_jsonl_path: str = "data/finetuning/tinker_conversations_spam.jsonl"

    # Dataset builder
    max_length: int = 8192
    batch_size: int = 16
    train_on_what: str = "ALL_ASSISTANT_MESSAGES"
    test_size: int = 128
    shuffle_seed: int = 0

    # Training hyperparams
    learning_rate: float = 2e-4
    lr_schedule: str = "linear"
    num_epochs: int = 1
    eval_every: int = 10
    save_every: int = 20


def convert_training_json_to_conversations_jsonl(
    *,
    training_json_path: str | Path,
    out_jsonl_path: str | Path,
    overwrite: bool = False,
) -> None:
    data = read_json(training_json_path)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {training_json_path}, got {type(data)}")
    rows = training_examples_to_tinker_conversations_jsonl(data)
    write_jsonl(out_jsonl_path, rows, overwrite=overwrite)


async def run_sft(cfg: SFTConfig) -> str | None:
    """
    Run Tinker Cookbook supervised fine-tune.
    Returns last sampler_path if available.
    """
    import tinker
    from tinker_cookbook import model_info
    from tinker_cookbook.checkpoint_utils import get_last_checkpoint
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    train_on = getattr(TrainOnWhat, cfg.train_on_what, None)
    if train_on is None:
        raise ValueError(
            f"Unknown TrainOnWhat={cfg.train_on_what!r}. "
            f"Expected one of: {', '.join([x.name for x in TrainOnWhat])}"
        )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what=train_on,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(Path(cfg.dataset_jsonl_path).expanduser().resolve()),
        test_size=cfg.test_size,
        shuffle_seed=cfg.shuffle_seed,
    )

    config = train.Config(
        log_path=str(Path(cfg.log_path).expanduser()),
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
    )

    # Kick off training
    await train.main(config)

    # Return last sampler path (if any)
    ckpt_sampler = get_last_checkpoint(config.log_path, required_key="sampler_path")
    if not ckpt_sampler:
        return None
    return ckpt_sampler.get("sampler_path")


def run_sft_sync(cfg: SFTConfig) -> str | None:
    return asyncio.run(run_sft(cfg))

