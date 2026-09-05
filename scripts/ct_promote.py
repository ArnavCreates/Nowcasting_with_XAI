#!/usr/bin/env python
"""Compare a retrained checkpoint against the incumbent and promote or reject."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from indra.config import load_config  # noqa: E402
from indra.evaluation.gate import GateCriteria, evaluate_gate  # noqa: E402

logger = logging.getLogger("indra.ct")


def read_metrics(path: Path | None) -> dict | None:
    """Load a metrics file, or None when there is no incumbent yet."""
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("cannot read %s: %s", path, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=REPO_ROOT / "metrics" / "production.json",
        help="absent on the first cycle, which promotes on the event floor alone",
    )
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--min-events", type=int, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="on promotion, copy the candidate checkpoint over the deployed one",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "metrics" / "gate.json",
        help="where to write the decision",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    candidate = read_metrics(args.candidate)
    if candidate is None:
        logger.error("candidate metrics not found at %s", args.candidate)
        return 2
    incumbent = read_metrics(args.incumbent)

    config = load_config()
    defaults = GateCriteria()
    criteria = GateCriteria(
        monitor=(
            config.training.checkpointing.monitor
            if config.training
            else defaults.monitor
        ),
        min_delta=(defaults.min_delta if args.min_delta is None else args.min_delta),
        guarded=defaults.guarded,
        max_regression=defaults.max_regression,
        min_events=(
            defaults.min_events if args.min_events is None else args.min_events
        ),
    )

    decision = evaluate_gate(candidate, incumbent, criteria)

    logger.info(
        "%s: %s %s -> %s",
        "PROMOTE" if decision.promote else "REJECT",
        decision.monitor,
        decision.incumbent,
        decision.candidate,
    )
    for reason in decision.reasons:
        logger.info("  %s", reason)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision.describe(), indent=2) + "\n")

    if decision.promote and args.apply:
        source = Path(candidate.get("checkpoint", ""))
        target = Path(config.inference.model.checkpoint)
        if not target.is_absolute():
            target = REPO_ROOT / target
        if not source.is_file():
            logger.error("candidate checkpoint %s is missing; not promoting", source)
            return 2
        if source.resolve() != target.resolve():
            # Written beside the target and moved, so a partial copy never
            # replaces a working deployed model.
            staged = target.with_name(target.name + ".incoming")
            shutil.copy2(source, staged)
            staged.replace(target)
            logger.info("promoted %s -> %s", source, target)
        # The candidate's scores become the incumbent's for the next cycle.
        args.incumbent.parent.mkdir(parents=True, exist_ok=True)
        args.incumbent.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n"
        )

    return 0 if decision.promote else 1


if __name__ == "__main__":
    sys.exit(main())
