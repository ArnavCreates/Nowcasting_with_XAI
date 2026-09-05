"""Shared domain types: quality-control flags, stream identifiers, aliases."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntFlag, StrEnum, unique
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    # Import for annotations only. ``from __future__ import annotations`` makes
    # every annotation a string, so xarray is never imported at runtime by this
    # module and the lighter parts of the package can keep importing it freely.
    import xarray as xr

# ---------------------------------------------------------------------------
# Array aliases
# ---------------------------------------------------------------------------

#: 2-D field on a native or target grid, always float32 after calibration.
FloatArray: TypeAlias = npt.NDArray[np.float32]

#: Boolean validity mask. True marks a *valid* observation, matching numpy's
#: convention for ``where``, and deliberately the inverse of
#: ``numpy.ma``'s mask so the two are never confused.
ValidMask: TypeAlias = npt.NDArray[np.bool_]

#: Curvilinear geolocation arrays, same shape as the field they describe.
GeoArray: TypeAlias = npt.NDArray[np.float64]

#: (lat_min, lat_max, lon_min, lon_max) in degrees.
BoundingBox: TypeAlias = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------


class QCFlag(IntFlag):
    """Per-frame quality-control state."""

    OK = 0

    # -- acquisition -------------------------------------------------------
    #: The expected granule was not present on disk for this timestamp.
    MISSING_FILE = 1 << 0
    #: The file exists but could not be opened or parsed: truncated download,
    #: unreadable HDF5 superblock, absent expected group.
    CORRUPT_OR_MISSING = 1 << 1
    #: Opened successfully, but a variable the configuration requires is
    #: absent from the granule.
    VARIABLE_ABSENT = 1 << 2
    #: The granule covers only part of the requested domain.
    PARTIAL_COVERAGE = 1 << 3

    # -- radiometry --------------------------------------------------------
    #: The count-to-brightness-temperature lookup table was missing or
    #: unusable, so the field remains in raw instrument counts.
    CALIBRATION_FAILED = 1 << 4
    #: Detector saturation: a large fraction of the frame sits at the top of
    #: the dynamic range.
    SATURATED = 1 << 5
    #: One or more scan lines were dropped or duplicated by the downlink.
    SCANLINE_DROPOUT = 1 << 6

    # -- geometry ----------------------------------------------------------
    #: Latitude/longitude arrays were absent or unusable.
    GEOLOCATION_MISSING = 1 << 7
    #: Parallax displacement was not removed, so elevated cloud tops remain
    #: offset from their true ground position.
    PARALLAX_UNCORRECTED = 1 << 8

    # -- physical plausibility --------------------------------------------
    #: Values fell outside the configured physical range and were masked.
    OUT_OF_PHYSICAL_RANGE = 1 << 9

    # -- reconstruction ----------------------------------------------------
    # Set by quality_control.py, never by a reader. These mark a frame that
    # was *reconstructed* rather than observed, so that anything reporting
    # skill can exclude it. Treating an interpolated frame as an observation
    # is how a nowcasting system quietly starts verifying against itself.
    GAP_FILLED_OPTICAL_FLOW = 1 << 10
    GAP_FILLED_SPLINE = 1 << 11
    #: Reconstruction was attempted and failed; the frame stays missing.
    GAP_FILL_FAILED = 1 << 12

    @property
    def is_usable(self) -> bool:
        """True when the frame carries observations a model may consume."""
        fatal = (
            QCFlag.MISSING_FILE
            | QCFlag.CORRUPT_OR_MISSING
            | QCFlag.VARIABLE_ABSENT
            | QCFlag.GAP_FILL_FAILED
        )
        return not (self & fatal)

    @property
    def is_observed(self) -> bool:
        """True only for genuinely observed data."""
        reconstructed = QCFlag.GAP_FILLED_OPTICAL_FLOW | QCFlag.GAP_FILLED_SPLINE
        return self.is_usable and not (self & reconstructed)

    def describe(self) -> list[str]:
        """Set flag names, for logging and for the ``qc_flag_names`` attribute."""
        if self is QCFlag.OK:
            return ["OK"]
        return [f.name for f in QCFlag if f.value and f & self == f and f.name]


#: Attribute key under which the integer flag value is stored on an
#: ``xarray.Dataset``. NetCDF and zarr cannot round-trip a Python enum, so the
#: raw int travels and is rehydrated with ``QCFlag(value)``.
ATTR_QC_FLAG: Final[str] = "qc_flag"
ATTR_QC_FLAG_NAMES: Final[str] = "qc_flag_names"
ATTR_SOURCE_STREAM: Final[str] = "source_stream"
ATTR_VALID_TIME: Final[str] = "valid_time"
ATTR_GRANULE_PATH: Final[str] = "granule_path"
ATTR_CALIBRATION: Final[str] = "calibration"


# ---------------------------------------------------------------------------
# Stream and field identifiers
# ---------------------------------------------------------------------------


@unique
class SourceStream(StrEnum):
    """The four heterogeneous inputs."""

    INSAT = "insat"
    IMDAA = "imdaa"
    IMD_SURFACE = "imd_surface"
    STATIC_PRIORS = "static_priors"


@unique
class Calibration(StrEnum):
    """What physical units a satellite field is currently expressed in."""

    #: Raw instrument counts; calibration has not been applied.
    COUNTS = "counts"
    #: Brightness temperature in kelvin, via the in-granule lookup table.
    BRIGHTNESS_TEMPERATURE_K = "brightness_temperature_k"
    #: An already-calibrated Level-2 geophysical product.
    L2_PRODUCT = "l2_product"


@unique
class InterpolationKind(StrEnum):
    """How a gap along the time axis was filled."""

    NONE = "none"
    OPTICAL_FLOW = "optical_flow"
    SPLINE = "spline"
    HOLD = "hold"


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

#: Missing observations are represented as NaN in float fields throughout the
#: pipeline. This is a sentinel for absence, not a value: it is never
#: substituted with zero, a climatological mean, or any other plausible number
#: before the tensor assembler does so explicitly and records a validity mask
#: alongside.
MISSING: Final[float] = float("nan")


def masked_like(shape: tuple[int, ...]) -> FloatArray:
    """An all-missing field of the given shape."""
    return np.full(shape, MISSING, dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 1 output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncedWindow:
    """One temporally aligned lookback window, still on native grids."""

    #: Nowcast time, t0.
    valid_time: datetime
    #: The T nominal slot times, oldest first, ending at ``valid_time``.
    timestamps: tuple[datetime, ...]
    #: Offsets that generated them, e.g. ``(-12, ..., 0)``.
    lookback_indices: tuple[int, ...]
    interval_minutes: int

    #: INSAT channels stacked on ``(time, y, x)``. ``None`` when the stream is
    #: disabled or nothing could be read.
    satellite: xr.Dataset | None = None
    #: IMDAA variables stacked on ``(time, level, y, x)``.
    nwp: xr.Dataset | None = None
    #: IMD surface variables, one Dataset each on ``(time, y, x)``. Separate
    #: because these products do not share a grid.
    surface: dict[str, xr.Dataset] = field(default_factory=dict)
    #: Static priors on ``(y, x)``, with no time axis: they are broadcast
    #: across the window rather than stored T times.
    static: dict[str, xr.Dataset] = field(default_factory=dict)

    #: Per-stream, per-frame quality-control flags.
    flags: dict[str, tuple[QCFlag, ...]] = field(default_factory=dict)
    #: Per-stream, per-frame observation mask. True marks a frame that was
    #: genuinely observed; a reconstructed frame is False even though it holds
    #: usable data, so nothing downstream can verify against interpolation.
    observed: dict[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    #: Gap-filling reports, keyed by stream.
    gapfill: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: False when the window failed a hard requirement -- typically a gap
    #: longer than the configured limit -- and must not be used for training
    #: or inference.
    accepted: bool = True
    rejection_reason: str | None = None

    @property
    def sequence_length(self) -> int:
        return len(self.timestamps)

    def observed_fraction(self, stream: str) -> float:
        """Share of frames in a stream that are genuine observations."""
        mask = self.observed.get(stream)
        if mask is None or mask.size == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / mask.size

    def reconstructed_frames(self, stream: str) -> list[int]:
        """Indices of frames that were gap-filled rather than observed."""
        flags = self.flags.get(stream, ())
        return [
            i for i, flag in enumerate(flags) if flag.is_usable and not flag.is_observed
        ]

    def summary(self) -> dict[str, Any]:
        """Compact description, for logging and for the API's provenance block."""
        return {
            "valid_time": self.valid_time.isoformat(),
            "sequence_length": self.sequence_length,
            "interval_minutes": self.interval_minutes,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "streams": {
                name: {
                    "observed_fraction": round(self.observed_fraction(name), 4),
                    "reconstructed_frames": self.reconstructed_frames(name),
                    "flags": [f.describe() for f in self.flags.get(name, ())],
                }
                for name in self.flags
            },
        }


@dataclass(frozen=True)
class AssembledWindow:
    """A model-ready lookback window: one tensor, and what its axes mean."""

    valid_time: datetime
    timestamps: tuple[datetime, ...]

    #: ``(T, C, H, W)`` float32, normalised, with missing cells already
    #: replaced by the configured fill value.
    tensor: npt.NDArray[np.float32]
    #: ``(T, 1, H, W)`` boolean. True marks a cell whose dynamic inputs were
    #: all present. The loss consults this; the model does not see it.
    validity: npt.NDArray[np.bool_]
    #: Channel names in tensor order. ``channel_names[i]`` describes
    #: ``tensor[:, i]``.
    channel_names: tuple[str, ...]

    #: Per-stream, per-frame quality-control flags, carried through from
    #: ingestion so the dataset layer can down-weight or drop reconstructed
    #: frames.
    flags: dict[str, tuple[QCFlag, ...]] = field(default_factory=dict)
    observed: dict[str, npt.NDArray[np.bool_]] = field(default_factory=dict)

    #: Geographic description of the H and W axes.
    grid: dict[str, float] = field(default_factory=dict)
    #: Fraction of finite cells per channel before the fill was applied.
    #: A channel at zero never arrived, and training on it teaches the model
    #: that the fill value is a measurement.
    channel_coverage: dict[str, float] = field(default_factory=dict)

    accepted: bool = True
    rejection_reason: str | None = None

    @property
    def shape(self) -> tuple[int, int, int, int]:
        s = self.tensor.shape
        return (int(s[0]), int(s[1]), int(s[2]), int(s[3]))

    @property
    def sequence_length(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.tensor.shape[1])

    def channel_index(self, name: str) -> int:
        """Tensor index of a named channel."""
        try:
            return self.channel_names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"no channel named {name!r}; available: {list(self.channel_names)}"
            ) from exc

    def channel(self, name: str) -> npt.NDArray[np.float32]:
        """The ``(T, H, W)`` slice for one named channel."""
        return self.tensor[:, self.channel_index(name)]

    def to_torch(self, device: str | None = None, add_batch: bool = True) -> Any:
        """Convert to a torch tensor."""
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(self.tensor))
        if add_batch:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device) if device else tensor

    def summary(self) -> dict[str, Any]:
        starved = [
            name for name, coverage in self.channel_coverage.items() if coverage < 0.01
        ]
        return {
            "valid_time": self.valid_time.isoformat(),
            "shape": list(self.shape),
            "channels": list(self.channel_names),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "validity_fraction": (
                round(float(np.count_nonzero(self.validity)) / self.validity.size, 4)
                if self.validity.size
                else 0.0
            ),
            "empty_channels": starved,
            "channel_coverage": {
                k: round(v, 4) for k, v in self.channel_coverage.items()
            },
        }


@dataclass(frozen=True)
class TargetWindow:
    """The ground truth a nowcast is scored against."""

    valid_time: datetime
    #: The T_out forecast times, ascending, ``valid_time`` exclusive.
    timestamps: tuple[datetime, ...]
    #: Offsets that generated them, e.g. ``(1, ..., 12)``.
    lead_indices: tuple[int, ...]
    interval_minutes: int

    #: ``(T_out, 1, H, W)`` float32 rain rate in **mm h-1**, on the target
    #: grid, with missing cells left as NaN rather than filled.
    rain_rate_mm_h: npt.NDArray[np.float32]
    #: ``(T_out, 1, H, W)`` boolean. True marks a cell carrying a genuine
    #: retrieval. The loss consults this so absent cells contribute no
    #: gradient instead of being scored as zero rain.
    validity: npt.NDArray[np.bool_]

    units: str = "mm h-1"
    #: Per-lead-frame quality-control flags, in lead order.
    flags: tuple[QCFlag, ...] = ()
    #: Per-lead-frame observation mask. See the note above on why this is
    #: recorded despite being expected to be uniformly True.
    observed: tuple[bool, ...] = ()
    #: Geographic description of the H and W axes, matching
    #: :attr:`AssembledWindow.grid`.
    grid: dict[str, float] = field(default_factory=dict)

    accepted: bool = True
    rejection_reason: str | None = None

    @property
    def shape(self) -> tuple[int, int, int, int]:
        s = self.rain_rate_mm_h.shape
        return (int(s[0]), int(s[1]), int(s[2]), int(s[3]))

    @property
    def lead_frames(self) -> int:
        return int(self.rain_rate_mm_h.shape[0])

    @property
    def valid_fraction(self) -> float:
        """Share of target cells carrying a retrieval."""
        if self.validity.size == 0:
            return 0.0
        return float(np.count_nonzero(self.validity)) / self.validity.size

    def to_torch(
        self, device: str | None = None, add_batch: bool = True
    ) -> tuple[Any, Any]:
        """Rain rate and validity as torch tensors, in that order."""
        import torch

        rain = torch.from_numpy(np.ascontiguousarray(self.rain_rate_mm_h))
        mask = torch.from_numpy(np.ascontiguousarray(self.validity))
        if add_batch:
            rain = rain.unsqueeze(0)
            mask = mask.unsqueeze(0)
        if device:
            rain, mask = rain.to(device), mask.to(device)
        return rain, mask

    def summary(self) -> dict[str, Any]:
        return {
            "valid_time": self.valid_time.isoformat(),
            "lead_frames": self.lead_frames,
            "shape": list(self.shape),
            "units": self.units,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "valid_fraction": round(self.valid_fraction, 4),
            "all_observed": all(self.observed) if self.observed else False,
            "flags": [flag.describe() for flag in self.flags],
        }


__all__ = [
    "ATTR_CALIBRATION",
    "ATTR_GRANULE_PATH",
    "ATTR_QC_FLAG",
    "ATTR_QC_FLAG_NAMES",
    "ATTR_SOURCE_STREAM",
    "ATTR_VALID_TIME",
    "MISSING",
    "AssembledWindow",
    "BoundingBox",
    "Calibration",
    "FloatArray",
    "GeoArray",
    "InterpolationKind",
    "QCFlag",
    "SourceStream",
    "SyncedWindow",
    "TargetWindow",
    "ValidMask",
    "masked_like",
]
