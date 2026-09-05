#!/usr/bin/env python
"""Score a checkpoint on the held-out season and write the metrics as JSON."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from indra.config import load_config  # noqa: E402
from indra.datasets.cache import WindowCache  # noqa: E402
from indra.models.fusion import (  # noqa: E402
    MissingCheckpointError,
    build_discriminators,
    build_from_config,
)
from indra.preprocessing.normalization import (  # noqa: E402
    MissingClimatologyError,
    load_statistics,
)
from indra.training.trainer import Trainer  # noqa: E402
from train import build_loaders  # noqa: E402

logger = logging.getLogger("indra.evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to score; defaults to inference.model.checkpoint",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "metrics" / "validation.json",
        help="where to write the metrics",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    if config.training is None:
        logger.error("configs/train/training.yaml is required to evaluate")
        return 2

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
    _, validation_loader = build_loaders(config, cache, stats)

    try:
        model = build_from_config(
            config.model, auxiliary_head=False, load_weights=False
        )
        checkpoint = args.checkpoint or Path(config.inference.model.checkpoint)
        model.load_fused(checkpoint, map_location=args.device or "cpu")
    except MissingCheckpointError as exc:
        logger.error("%s", exc)
        return 3

    trainer = Trainer(
        model=model,
        discriminators=build_discriminators(config.model),
        config=config.training,
        train_loader=validation_loader,
        validation_loader=validation_loader,
        device=args.device,
    )
    results = trainer.validate()

    if not results:
        logger.error("validation produced no metrics")
        return 1

    results["checkpoint"] = str(checkpoint)
    results["monitor"] = config.training.checkpointing.monitor

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", args.out)
    for key, value in sorted(results.items()):
        logger.info("  %-20s %s", key, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
