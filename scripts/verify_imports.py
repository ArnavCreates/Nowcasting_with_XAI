#!/usr/bin/env python
"""Import every module and validate every config. No model weights required."""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Ordered roughly by dependency depth, so the first failure is the root cause
# rather than a module that failed because its import did.
MODULES = [
    "indra",
    "indra.types",
    "indra.config",
    # Stage 1
    "indra.ingestion._hdf5",
    "indra.ingestion.insat3d",
    "indra.ingestion.imdaa",
    "indra.ingestion.imd_surface",
    "indra.ingestion.static_priors",
    "indra.ingestion.hem",
    "indra.ingestion.temporal_sync",
    "indra.ingestion.qc",
    "indra.ingestion.qc.bounds",
    "indra.ingestion.qc.dropout",
    "indra.ingestion.qc.gapfill",
    "indra.ingestion.qc.parallax",
    # Stage 2
    "indra.preprocessing.normalization",
    "indra.preprocessing.reprojection",
    "indra.preprocessing.tensor_assembly",
    # Stage 3
    "indra.models.earthformer.cuboid_attention",
    "indra.models.earthformer.backbone",
    "indra.models.adapter.bridge",
    "indra.models.dgmr.generator",
    "indra.models.dgmr.discriminators",
    "indra.models.fusion",
    # Stage 4
    "indra.datasets.index",
    "indra.datasets.targets",
    "indra.datasets.cache",
    "indra.datasets.nowcast",
    "indra.training.losses",
    "indra.training.replay_buffer",
    "indra.training.trainer",
    "indra.evaluation.metrics",
    "indra.xai.baselines",
    "indra.xai.attribution",
    "indra.xai.attention",
    "indra.xai.report",
    # Stage 5
    "indra.advisory.geospatial",
    "indra.advisory.retrieval",
    "indra.advisory.schemas",
    "indra.advisory.cap",
    "indra.advisory.index_corpus",
    "indra.api.schemas",
    "indra.api.state",
    "indra.api.service",
    "indra.api.routes.dependencies",
    "indra.api.routes.forecast",
    "indra.api.routes.advisory",
    "indra.api.routes.xai",
    "indra.api.routes.health",
    # Builds the FastAPI app at import, so this also exercises the router
    # wiring and the conditional static mount.
    "indra.api.main",
]


def check_imports() -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}")
        else:
            print(f"  ok    {name}")
    return failures


def check_config() -> list[tuple[str, str]]:
    """Load and cross-validate all five YAML files."""
    failures: list[tuple[str, str]] = []
    try:
        from indra.config import load_config

        config = load_config()
        print(f"  ok    config: {config.model.fusion.name}")
        print(
            f"        input {config.model.input.sequence_length}x"
            f"{config.model.input.channels}x{config.model.input.height}x"
            f"{config.model.input.width}, "
            f"{config.model.output.lead_frames} lead frames"
        )
        print(f"        training split: {'present' if config.training else 'absent'}")
    except Exception:
        failures.append(("config", traceback.format_exc()))
        print("  FAIL  config")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-config", action="store_true")
    args = parser.parse_args(argv)

    print(f"python {sys.version.split()[0]}")
    print(f"src    {SRC}\n")

    print("imports")
    failures = check_imports()

    if not args.skip_config:
        print("\nconfiguration")
        failures += check_config()

    print()
    if failures:
        for name, tb in failures:
            print(f"{'=' * 70}\n{name}\n{'=' * 70}\n{tb}")
        print(f"{len(failures)} failure(s)")
        return 1

    total = len(MODULES) + (0 if args.skip_config else 1)
    print(f"{total} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
