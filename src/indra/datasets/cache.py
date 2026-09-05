"""Disk-backed cache of assembled windows, keyed by configuration fingerprint."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CacheScope, IngestionConfig, PreprocessingConfig
from ..types import AssembledWindow, QCFlag, TargetWindow, masked_like

logger = logging.getLogger(__name__)

#: Bumped when the stored layout changes meaning. It participates in the
#: fingerprint, so an older cache is orphaned rather than misread.
CACHE_FORMAT_VERSION = 1

# ``CacheScope`` -- "none" | "target_only" | "both" -- is imported from
# ``config`` rather than defined here, so the configuration layer and this
# module cannot come to disagree about which scopes exist. Re-exported below
# for callers that reasonably expect to find it beside the cache.

_META_KEY = "__meta__"
_INPUT = "input"
_TARGET = "target"


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def fingerprint(ingestion: IngestionConfig, preprocessing: PreprocessingConfig) -> str:
    """Stable hash of every configured value that shapes a stored tensor."""
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "ingestion": ingestion.model_dump(mode="json"),
        "preprocessing": preprocessing.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class WindowCache:
    """Fingerprinted, atomic, disk-backed store for assembled windows."""

    def __init__(
        self,
        root: Path | str,
        ingestion: IngestionConfig,
        preprocessing: PreprocessingConfig,
        scope: CacheScope = "target_only",
    ) -> None:
        if scope not in ("none", "target_only", "both"):
            raise ValueError(
                f"unknown cache scope {scope!r}; expected one of "
                "'none', 'target_only', 'both'"
            )
        self.root = Path(root)
        self.scope = scope
        self.ingestion = ingestion
        self.preprocessing = preprocessing
        self.fingerprint = fingerprint(ingestion, preprocessing)

        grid = preprocessing.target_grid
        self.input_shape = (
            preprocessing.tensor.sequence_length,
            preprocessing.channels.count,
            grid.height,
            grid.width,
        )
        self.target_shape = (
            len(ingestion.temporal.lead_indices),
            1,
            grid.height,
            grid.width,
        )

        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0

        if scope != "none":
            logger.info(
                "window cache at %s, fingerprint %s, scope %s (%s per window)",
                self.root,
                self.fingerprint,
                scope,
                _human_bytes(self.bytes_per_window()),
            )

    # ------------------------------------------------------------- geometry
    def bytes_per_window(self) -> int:
        """Stored size of one window, for the footprint an operator must plan."""
        if self.scope == "none":
            return 0

        mask_shape = (self.input_shape[0], 1, *self.input_shape[2:])
        target_bytes = int(np.prod(self.target_shape)) * 4  # rain, float32
        target_bytes += int(np.prod(self.target_shape))  # validity, bool
        if self.scope == "target_only":
            return target_bytes

        input_bytes = int(np.prod(self.input_shape)) * 4  # tensor, float32
        input_bytes += int(np.prod(mask_shape))  # validity, bool
        return target_bytes + input_bytes

    def projected_bytes(self, windows: int) -> int:
        return self.bytes_per_window() * max(windows, 0)

    def _directory(self, kind: str, valid_time: datetime) -> Path:
        # Sharded by date. Twenty-three thousand entries in one directory is a
        # filesystem nobody wants to list, and several tools degrade sharply
        # well before that.
        moment = _as_utc(valid_time)
        return (
            self.root
            / self.fingerprint
            / kind
            / f"{moment.year:04d}"
            / f"{moment.month:02d}{moment.day:02d}"
        )

    def _path(self, kind: str, valid_time: datetime) -> Path:
        moment = _as_utc(valid_time)
        return (
            self._directory(kind, moment) / f"{moment.hour:02d}{moment.minute:02d}.npz"
        )

    # -------------------------------------------------------------- storage
    def _write(
        self, path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]
    ) -> bool:
        """Write one entry atomically. Returns whether it landed."""
        # Same directory as the destination, so os.replace stays within one
        # filesystem and is therefore atomic. The pid keeps two workers
        # writing the same window from colliding on the temporary name.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(arrays)
            payload[_META_KEY] = np.array(
                json.dumps(meta, sort_keys=True, separators=(",", ":"))
            )
            with tmp.open("wb") as handle:
                np.savez(handle, **payload)
            os.replace(tmp, path)
            self.writes += 1
            return True
        except OSError as exc:
            self.errors += 1
            logger.warning("could not write cache entry %s: %s", path, exc)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return False

    def _read(
        self, path: Path, valid_time: datetime
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
        """Read one entry, or ``None`` for a miss or anything unusable."""
        if not path.is_file():
            self.misses += 1
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if _META_KEY not in data:
                    raise ValueError("entry carries no metadata")
                meta = json.loads(str(data[_META_KEY].item()))
                arrays = {k: data[k] for k in data.files if k != _META_KEY}
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            # Never fatal, and never deleted: the next successful store
            # replaces this path atomically, so a corrupt entry self-heals.
            self.errors += 1
            logger.warning("discarding unreadable cache entry %s: %s", path, exc)
            return None

        # A directory copied between deployments, or a path reused after a
        # rename, would otherwise serve a window built under different rules.
        if meta.get("fingerprint") != self.fingerprint:
            self.errors += 1
            logger.warning(
                "cache entry %s was written under fingerprint %s, not %s; " "ignoring",
                path,
                meta.get("fingerprint"),
                self.fingerprint,
            )
            return None
        if meta.get("valid_time") != _as_utc(valid_time).isoformat():
            self.errors += 1
            logger.warning(
                "cache entry %s holds %s but %s was requested; ignoring",
                path,
                meta.get("valid_time"),
                _as_utc(valid_time).isoformat(),
            )
            return None

        self.hits += 1
        return arrays, meta

    # ---------------------------------------------------------- target I/O
    def load_target(self, valid_time: datetime) -> TargetWindow | None:
        """Rebuild a cached :class:`TargetWindow`, or ``None`` to recompute."""
        if self.scope == "none":
            return None

        result = self._read(self._path(_TARGET, valid_time), valid_time)
        if result is None:
            return None
        arrays, meta = result

        try:
            if meta.get("rejected"):
                # Rejections are stored as metadata alone. Discovering one
                # costs twelve granule reads, so it is well worth caching, and
                # its arrays are entirely masked and reconstructible from the
                # configured shape -- 6.75 MiB of NaN is not worth storing
                # once, let alone across the consecutive windows an archive gap
                # produces.
                rain = masked_like(self.target_shape)
                validity = np.zeros(self.target_shape, dtype=np.bool_)
            else:
                rain = arrays["rain_rate_mm_h"]
                validity = arrays["validity"]
                if rain.shape != self.target_shape:
                    raise ValueError(
                        f"cached target is {rain.shape}, expected "
                        f"{self.target_shape}"
                    )
                if validity.shape != rain.shape:
                    raise ValueError("cached target and validity disagree in shape")

            return TargetWindow(
                valid_time=_as_utc(valid_time),
                timestamps=tuple(
                    _as_utc(datetime.fromisoformat(t)) for t in meta["timestamps"]
                ),
                lead_indices=tuple(meta["lead_indices"]),
                interval_minutes=int(meta["interval_minutes"]),
                rain_rate_mm_h=np.ascontiguousarray(rain, dtype=np.float32),
                validity=np.ascontiguousarray(validity, dtype=np.bool_),
                units=meta["units"],
                flags=tuple(QCFlag(int(f)) for f in meta.get("flags", ())),
                observed=tuple(bool(o) for o in meta.get("observed", ())),
                grid=dict(meta.get("grid", {})),
                accepted=bool(meta["accepted"]),
                rejection_reason=meta.get("rejection_reason"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            self.errors += 1
            logger.warning(
                "cached target for %s is malformed (%s); recomputing",
                _as_utc(valid_time).isoformat(),
                exc,
            )
            return None

    def store_target(self, window: TargetWindow) -> bool:
        if self.scope == "none":
            return False

        meta: dict[str, Any] = {
            "format": CACHE_FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "kind": _TARGET,
            "valid_time": _as_utc(window.valid_time).isoformat(),
            "timestamps": [_as_utc(t).isoformat() for t in window.timestamps],
            "lead_indices": list(window.lead_indices),
            "interval_minutes": window.interval_minutes,
            "units": window.units,
            "flags": [int(f) for f in window.flags],
            "observed": [bool(o) for o in window.observed],
            "grid": dict(window.grid),
            "accepted": bool(window.accepted),
            "rejection_reason": window.rejection_reason,
            "rejected": not window.accepted,
        }
        arrays: dict[str, np.ndarray] = (
            {}
            if not window.accepted
            else {
                "rain_rate_mm_h": np.ascontiguousarray(
                    window.rain_rate_mm_h, dtype=np.float32
                ),
                "validity": np.ascontiguousarray(window.validity, dtype=np.bool_),
            }
        )
        return self._write(self._path(_TARGET, window.valid_time), arrays, meta)

    # ----------------------------------------------------------- input I/O
    def load_input(self, valid_time: datetime) -> AssembledWindow | None:
        """Rebuild a cached :class:`AssembledWindow`, or ``None`` to recompute."""
        if self.scope != "both":
            return None

        result = self._read(self._path(_INPUT, valid_time), valid_time)
        if result is None:
            return None
        arrays, meta = result

        try:
            if meta.get("rejected"):
                tensor = masked_like(self.input_shape)
                validity = np.zeros(
                    (self.input_shape[0], 1, *self.input_shape[2:]), dtype=np.bool_
                )
            else:
                tensor = arrays["tensor"]
                validity = arrays["validity"]
                if tensor.shape != self.input_shape:
                    raise ValueError(
                        f"cached input is {tensor.shape}, expected "
                        f"{self.input_shape}"
                    )

            channel_names = tuple(meta["channel_names"])
            if len(channel_names) != tensor.shape[1]:
                raise ValueError(
                    f"{len(channel_names)} channel names for "
                    f"{tensor.shape[1]} channels"
                )
            # The order is the contract between the tensor, the XAI labels and
            # the API schema. A cache entry written before a channel was
            # renamed or reordered must not be served under the new names.
            if list(channel_names) != list(self.preprocessing.channels.names):
                raise ValueError("cached channel order differs from configuration")

            return AssembledWindow(
                valid_time=_as_utc(valid_time),
                timestamps=tuple(
                    _as_utc(datetime.fromisoformat(t)) for t in meta["timestamps"]
                ),
                tensor=np.ascontiguousarray(tensor, dtype=np.float32),
                validity=np.ascontiguousarray(validity, dtype=np.bool_),
                channel_names=channel_names,
                flags={
                    stream: tuple(QCFlag(int(f)) for f in values)
                    for stream, values in meta.get("flags", {}).items()
                },
                observed={
                    key.removeprefix("observed__"): np.ascontiguousarray(
                        value, dtype=np.bool_
                    )
                    for key, value in arrays.items()
                    if key.startswith("observed__")
                },
                grid=dict(meta.get("grid", {})),
                channel_coverage=dict(meta.get("channel_coverage", {})),
                accepted=bool(meta["accepted"]),
                rejection_reason=meta.get("rejection_reason"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            self.errors += 1
            logger.warning(
                "cached input for %s is malformed (%s); recomputing",
                _as_utc(valid_time).isoformat(),
                exc,
            )
            return None

    def store_input(self, window: AssembledWindow) -> bool:
        if self.scope != "both":
            return False

        meta: dict[str, Any] = {
            "format": CACHE_FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "kind": _INPUT,
            "valid_time": _as_utc(window.valid_time).isoformat(),
            "timestamps": [_as_utc(t).isoformat() for t in window.timestamps],
            "channel_names": list(window.channel_names),
            "flags": {
                stream: [int(f) for f in values]
                for stream, values in window.flags.items()
            },
            "grid": dict(window.grid),
            "channel_coverage": dict(window.channel_coverage),
            "accepted": bool(window.accepted),
            "rejection_reason": window.rejection_reason,
            "rejected": not window.accepted,
        }

        arrays: dict[str, np.ndarray] = {}
        if window.accepted:
            arrays["tensor"] = np.ascontiguousarray(window.tensor, dtype=np.float32)
            arrays["validity"] = np.ascontiguousarray(window.validity, dtype=np.bool_)
            for stream, mask in window.observed.items():
                arrays[f"observed__{stream}"] = np.ascontiguousarray(
                    mask, dtype=np.bool_
                )
        return self._write(self._path(_INPUT, window.valid_time), arrays, meta)

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        """Counters for the training log."""
        looked_up = self.hits + self.misses
        return {
            "root": str(self.root),
            "fingerprint": self.fingerprint,
            "scope": self.scope,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "hit_rate": (self.hits / looked_up) if looked_up else None,
            "bytes_per_window": self.bytes_per_window(),
        }


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


__all__ = [
    "CACHE_FORMAT_VERSION",
    "CacheScope",
    "WindowCache",
    "fingerprint",
]
