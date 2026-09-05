#!/usr/bin/env python
"""Train the fusion relay."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from indra.config import IndraConfig, load_config  # noqa: E402
from indra.datasets.cache import WindowCache  # noqa: E402
from indra.datasets.index import build_index  # noqa: E402
from indra.datasets.nowcast import NowcastDataset, collate_nowcast  # noqa: E402
from indra.models.fusion import (  # noqa: E402
    MissingCheckpointError,
    build_discriminators,
    build_from_config,
)
from indra.preprocessing.normalization import (  # noqa: E402
    MissingClimatologyError,
    load_statistics,
)
from indra.training.replay_buffer import (  # noqa: E402
    ExperienceReplayBuffer,
    RetentionPolicy,
)
from indra.training.trainer import Trainer  # noqa: E402

logger = logging.getLogger("indra.train")


def apply_overrides(
    config: IndraConfig, max_steps: int | None, device: str | None
) -> IndraConfig:
    """Apply CLI overrides."""
    training = config.training
    if max_steps is not None:
        training = training.model_copy(
            update={
                "schedule": training.schedule.model_copy(
                    update={"max_steps": max_steps}
                )
            }
        )
    if device is not None:
        training = training.model_copy(
            update={"run": training.run.model_copy(update={"device": device})}
        )
    return config.model_copy(update={"training": training})


def build_policy(config: IndraConfig) -> RetentionPolicy:
    """Config block to the plain dataclass the buffer holds."""
    policy = config.training.replay.policy
    return RetentionPolicy(
        gate_frames=policy.gate_frames,
        required_streams=tuple(policy.required_streams),
        disqualifying_flags=policy.resolved_disqualifying_flags(),
        min_valid_fraction=policy.min_valid_fraction,
        min_target_valid_fraction=policy.min_target_valid_fraction,
        precip_channel=policy.precip_channel,
        min_precip_coverage=policy.min_precip_coverage,
        heavy_threshold_mm_h=policy.heavy_threshold_mm_h,
        admission_quantile=policy.admission_quantile,
        min_exceedance_area=policy.min_exceedance_area,
        score_history_size=policy.score_history_size,
        min_history=policy.min_history,
        min_separation_minutes=policy.min_separation_minutes,
    )


def build_buffer(config: IndraConfig) -> ExperienceReplayBuffer | None:
    """Load an existing replay manifest, or start a new reservoir."""
    replay = config.training.replay
    if not replay.enabled:
        logger.info("replay disabled")
        return None

    manifest = replay.resolved_manifest_path()
    if manifest.is_file():
        buffer = ExperienceReplayBuffer.load(manifest)
        logger.info("replay resumed from %s: %s", manifest, buffer.stats())
        return buffer

    logger.info("replay starting empty; manifest will be written to %s", manifest)
    return ExperienceReplayBuffer(
        capacity=replay.capacity, policy=build_policy(config), seed=replay.seed
    )


def build_loaders(
    config: IndraConfig, cache: WindowCache | None, stats: dict
) -> tuple[DataLoader, DataLoader]:
    """Enumerate, split and wrap both sides of the chronological boundary."""
    split = build_index(config.training, config.model)
    logger.info("index: %s", split.summary())

    data = config.training.data
    common = {
        "ingestion": config.ingestion,
        "preprocessing": config.preprocessing,
        "cache": cache,
        "stats": stats,
    }
    train_set = NowcastDataset(split.train, split="train", **common)
    validation_set = NowcastDataset(split.validation, split="validation", **common)

    loader_kwargs = {
        "batch_size": data.batch_size,
        "num_workers": data.num_workers,
        "collate_fn": collate_nowcast,
        "pin_memory": data.pin_memory,
    }
    if data.num_workers > 0:
        # prefetch_factor is only meaningful with worker processes, and passing
        # it at num_workers=0 raises rather than being ignored.
        loader_kwargs["prefetch_factor"] = data.prefetch_factor

    train_loader = DataLoader(
        train_set,
        shuffle=data.shuffle_train,
        drop_last=data.drop_last,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_set, shuffle=False, drop_last=False, **loader_kwargs
    )
    return train_loader, validation_loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override schedule.max_steps, for a short run",
    )
    parser.add_argument("--device", default=None, help="override run.device")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "build the model, datasets and loaders, report shapes and counts, "
            "then stop without training"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config_root)
    if config.training is None:
        logger.error(
            "no training configuration found. configs/train/training.yaml is "
            "required to train."
        )
        return 2

    config = apply_overrides(config, args.max_steps, args.device)

    # Loaded once and shared by both datasets. Reading the climatology per
    # worker would parse the same JSON num_workers times, and a missing file
    # must stop the run here rather than inside an epoch.
    #
    # Both of these carry instructions. Printed rather than raised so the
    # operator sees what to do instead of a traceback ending in it.
    try:
        stats = load_statistics(config.preprocessing.normalization)
    except MissingClimatologyError as exc:
        logger.error("%s", exc)
        return 3

    cache_config = config.training.cache
    cache = (
        None
        if cache_config.scope == "none"
        else WindowCache(
            cache_config.resolved_dir(),
            config.ingestion,
            config.preprocessing,
            scope=cache_config.scope,
        )
    )

    train_loader, validation_loader = build_loaders(config, cache, stats)

    try:
        model = build_from_config(config.model, auxiliary_head=True, load_weights=True)
    except MissingCheckpointError as exc:
        logger.error("%s", exc)
        return 3
    discriminators = build_discriminators(config.model)
    buffer = build_buffer(config)

    trainer = Trainer(
        model=model,
        discriminators=discriminators,
        config=config.training,
        train_loader=train_loader,
        validation_loader=validation_loader,
        replay_buffer=buffer,
    )

    if args.dry_run:
        logger.info(
            "dry run: %d train / %d validation windows, %s",
            len(train_loader.dataset),
            len(validation_loader.dataset),
            model.parameter_summary(),
        )
        logger.info("everything constructed; stopping before the first step")
        return 0

    try:
        state = trainer.fit()
    except KeyboardInterrupt:
        # Checkpoint what the run reached rather than discarding it.
        logger.warning("interrupted at step %d; saving", trainer.state.step)
        trainer.save_checkpoint(is_best=False)
        return 130

    if buffer is not None:
        buffer.save(config.training.replay.resolved_manifest_path())

    logger.info(
        "done at step %d; best %s = %s at step %s",
        state.step,
        config.training.checkpointing.monitor,
        state.best_metric,
        state.best_step,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
