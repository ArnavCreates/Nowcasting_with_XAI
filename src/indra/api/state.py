"""Startup lifecycle: load what is present, record what is not."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..advisory.geospatial import Grid
from ..config import REPO_ROOT, IndraConfig, load_config

logger = logging.getLogger(__name__)

#: Component keys, used by endpoints to declare what they need.
CHECKPOINTS = "checkpoints"
CLIMATOLOGY = "climatology"
DISTRICTS = "districts"
NDMA = "ndma"
ADVISORY_MODEL = "advisory_model"
FRONTEND = "frontend"

#: Where a built frontend would be. Vendored as source in this repository, so
#: absent until someone runs the build.
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


class ComponentUnavailable(RuntimeError):
    """A required artefact is not loaded."""

    def __init__(self, component: str, detail: str) -> None:
        self.component = component
        self.detail = detail
        super().__init__(f"{component}: {detail}")


@dataclass
class ComponentStatus:
    """Readiness of one external artefact."""

    name: str
    available: bool
    path: str | None = None
    detail: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass
class ServiceState:
    """Everything loaded at startup, and everything that could not be."""

    config: IndraConfig
    grid: Grid
    components: dict[str, ComponentStatus] = field(default_factory=dict)

    model: Any = None
    districts: Any = None
    retriever: Any = None
    advisory_generator: Any = None
    stats: Any = None
    baseline: Any = None
    device: str = "cpu"
    frontend_dir: Path | None = None

    # ------------------------------------------------------------ readiness
    @property
    def ready(self) -> bool:
        """True when a forecast can actually be produced."""
        required = (CHECKPOINTS, CLIMATOLOGY)
        return all(
            self.components[name].available
            for name in required
            if name in self.components
        ) and all(name in self.components for name in required)

    def require(self, *components: str) -> None:
        """Assert that every named component loaded, or raise with its message."""
        for name in components:
            status = self.components.get(name)
            if status is None:
                raise ComponentUnavailable(
                    name, f"unknown component {name!r}; nothing checked it at startup"
                )
            if not status.available:
                raise ComponentUnavailable(
                    name, status.detail or f"{name} is not available"
                )

    def status_list(self) -> list[ComponentStatus]:
        return [self.components[name] for name in sorted(self.components)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _record(
    state: ServiceState,
    name: str,
    path: Path | None,
    available: bool,
    detail: str | None = None,
) -> None:
    state.components[name] = ComponentStatus(
        name=name,
        available=available,
        path=str(path) if path is not None else None,
        detail=detail,
    )
    if available:
        logger.info("component %s ready (%s)", name, path)
    else:
        logger.warning("component %s unavailable: %s", name, detail)


def _load_climatology(state: ServiceState) -> None:
    """Statistics, then the attribution baseline that derives from them."""
    from ..preprocessing.normalization import MissingClimatologyError, load_statistics
    from ..xai.baselines import build_climatological_baseline

    normalization = state.config.preprocessing.normalization
    path = normalization.resolved_stats_path()
    try:
        state.stats = load_statistics(normalization)
        state.baseline = build_climatological_baseline(
            state.config.preprocessing, state.stats
        )
        _record(state, CLIMATOLOGY, path, True)
    except MissingClimatologyError as exc:
        _record(state, CLIMATOLOGY, path, False, str(exc))
    except Exception as exc:
        _record(
            state,
            CLIMATOLOGY,
            path,
            False,
            f"climatology could not be loaded: {type(exc).__name__}: {exc}",
        )


def _load_model(state: ServiceState) -> None:
    """Build the relay and load its weights."""
    import torch

    from ..models.fusion import MissingCheckpointError, build_from_config

    spec = state.config.inference.model
    path = Path(spec.checkpoint)
    if not path.is_absolute():
        path = REPO_ROOT / path

    requested = spec.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "%s requested but unavailable; serving on %s",
            requested,
            spec.fallback_device,
        )
        requested = spec.fallback_device
    state.device = requested

    try:
        model = build_from_config(
            state.config.model, auxiliary_head=False, load_weights=False
        )
        model.load_fused(path, map_location=state.device)
        model = model.to(state.device)
        if spec.eval_mode:
            model.eval()
        state.model = model
        _record(state, CHECKPOINTS, path, True)
    except MissingCheckpointError as exc:
        _record(state, CHECKPOINTS, path, False, str(exc))
    except Exception as exc:
        _record(
            state,
            CHECKPOINTS,
            path,
            False,
            f"model could not be loaded: {type(exc).__name__}: {exc}",
        )


def _load_districts(state: ServiceState) -> None:
    """Rasterise the administrative boundaries onto the model grid, once."""
    from ..advisory.geospatial import build_district_grid

    geospatial = state.config.inference.geospatial
    path = geospatial.resolved_boundaries()
    try:
        state.districts = build_district_grid(geospatial, state.grid)
        _record(state, DISTRICTS, path, True)
    except FileNotFoundError as exc:
        _record(state, DISTRICTS, path, False, str(exc))
    except Exception as exc:
        _record(
            state,
            DISTRICTS,
            path,
            False,
            f"district boundaries could not be rasterised: "
            f"{type(exc).__name__}: {exc}",
        )


def _load_retriever(state: ServiceState) -> None:
    """Open the NDMA collection, verifying its embedding provenance."""
    from ..advisory.retrieval import (
        MissingCorpusError,
        MissingEmbeddingModelError,
        NdmaRetriever,
    )

    store = state.config.inference.advisory.vector_store
    path = Path(store.persist_directory)
    if not path.is_absolute():
        path = REPO_ROOT / path

    if not state.config.inference.advisory.enabled:
        _record(
            state, NDMA, path, False, "advisory generation is disabled in configuration"
        )
        return

    try:
        state.retriever = NdmaRetriever(store)
        _record(state, NDMA, path, True)
    except (MissingCorpusError, MissingEmbeddingModelError) as exc:
        # Two distinct absences with one consequence: the collection may be
        # present and good while the model that would query it is not on disk.
        # Both carry their own instructional message.
        _record(state, NDMA, path, False, str(exc))
    except Exception as exc:
        _record(
            state,
            NDMA,
            path,
            False,
            f"NDMA collection could not be opened: {type(exc).__name__}: {exc}",
        )


def _load_advisory_model(state: ServiceState) -> None:
    """Load the local advisory model, once, under guided decoding."""
    from ..advisory.cap import AdvisoryGenerator, MissingAdvisoryModelError

    advisory = state.config.inference.advisory
    path = advisory.local_advisory.resolved_path()

    if not advisory.enabled:
        _record(
            state,
            ADVISORY_MODEL,
            path,
            False,
            "advisory generation is disabled in configuration",
        )
        return

    try:
        state.advisory_generator = AdvisoryGenerator(advisory.local_advisory)
        _record(state, ADVISORY_MODEL, path, True)
    except MissingAdvisoryModelError as exc:
        _record(state, ADVISORY_MODEL, path, False, str(exc))
    except Exception as exc:
        _record(
            state,
            ADVISORY_MODEL,
            path,
            False,
            f"advisory model could not be loaded: {type(exc).__name__}: {exc}",
        )


def _load_frontend(state: ServiceState) -> None:
    """Locate the built frontend, if someone has built it."""
    if FRONTEND_DIST.is_dir():
        state.frontend_dir = FRONTEND_DIST
        _record(state, FRONTEND, FRONTEND_DIST, True)
        return
    state.frontend_dir = None
    _record(
        state,
        FRONTEND,
        FRONTEND_DIST,
        False,
        "Frontend build directory (frontend/dist) not found; running in "
        "headless API mode.",
    )


def load_state(config_root: Path | str | None = None) -> ServiceState:
    """Load every component, tolerating the absence of each."""
    # Not tolerated: configs/ is in the repository, so a configuration that
    # will not validate is a defect here rather than a missing mount.
    config = load_config(config_root)
    state = ServiceState(
        config=config,
        grid=Grid.from_config(config.preprocessing.target_grid),
    )

    _load_climatology(state)
    _load_model(state)
    _load_districts(state)
    _load_retriever(state)
    _load_advisory_model(state)
    _load_frontend(state)

    missing = [
        name for name, status in state.components.items() if not status.available
    ]
    if missing:
        logger.warning(
            "service starting in degraded mode; unavailable: %s",
            ", ".join(sorted(missing)),
        )
    else:
        logger.info("service ready; all components loaded")
    return state


__all__ = [
    "ADVISORY_MODEL",
    "CHECKPOINTS",
    "CLIMATOLOGY",
    "DISTRICTS",
    "FRONTEND",
    "FRONTEND_DIST",
    "NDMA",
    "ComponentStatus",
    "ComponentUnavailable",
    "ServiceState",
    "load_state",
]
