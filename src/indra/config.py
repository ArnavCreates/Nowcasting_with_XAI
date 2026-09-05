"""Typed configuration layer."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The one intra-package import this module makes. ``types`` holds the domain
# primitives and imports nothing from the package itself, so there is no cycle:
# QC flags and stream names are resolved from their YAML spellings here, at the
# configuration boundary, rather than as strings passed inward for some later
# module to interpret differently.
from .types import QCFlag, SourceStream

# Repository root: src/indra/config.py -> src/indra -> src -> <root>
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"

# Tolerance for float comparisons on degrees. 1e-6 deg is ~0.11 m, far below
# any meaningful geolocation error, so a disagreement larger than this is a
# genuine configuration mistake rather than floating-point noise.
_DEG_EPS = 1e-6


def _check_months(months: list[int], label: str) -> list[int]:
    """Validate a calendar-month set."""
    if not months:
        raise ValueError(
            f"{label} is empty. An empty month set matches nothing, so every "
            "date lies outside the season and the pipeline accepts no data at "
            "all -- omit the field to mean 'all months' instead."
        )
    out_of_range = [m for m in months if not 1 <= m <= 12]
    if out_of_range:
        raise ValueError(f"{label} contains non-calendar months {out_of_range}")
    if len(set(months)) != len(months):
        raise ValueError(f"{label} repeats a month: {months}")
    return sorted(months)


class StrictModel(BaseModel):
    """Base for every configuration model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Shared enumerations
# ---------------------------------------------------------------------------


class FieldKind(str, Enum):
    """How a field may be resampled and normalised."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    PRECIPITATION = "precipitation"


class Resampling(str, Enum):
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"


class NormalizationMethod(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"
    NONE = "none"


class GapFillStrategy(str, Enum):
    OPTICAL_FLOW = "optical_flow"
    SPLINE = "spline"
    NONE = "none"


# ---------------------------------------------------------------------------
# Geometry, shared by ingestion and preprocessing
# ---------------------------------------------------------------------------


class GridGeometry(StrictModel):
    """A regular lat/lon grid."""

    crs: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float

    @field_validator("lat_min", "lat_max")
    @classmethod
    def _latitude_in_range(cls, v: float) -> float:
        if not -90.0 <= v <= 90.0:
            raise ValueError(f"latitude {v} outside [-90, 90]")
        return v

    @field_validator("lon_min", "lon_max")
    @classmethod
    def _longitude_in_range(cls, v: float) -> float:
        if not -180.0 <= v <= 180.0:
            raise ValueError(f"longitude {v} outside [-180, 180]")
        return v

    @model_validator(mode="after")
    def _extents_ordered(self) -> GridGeometry:
        if self.lat_max <= self.lat_min:
            raise ValueError(
                f"lat_max ({self.lat_max}) must exceed lat_min ({self.lat_min})"
            )
        if self.lon_max <= self.lon_min:
            raise ValueError(
                f"lon_max ({self.lon_max}) must exceed lon_min ({self.lon_min})"
            )
        return self

    @property
    def lat_span(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def lon_span(self) -> float:
        return self.lon_max - self.lon_min

    def check_shape(self, height: int, width: int) -> None:
        """Assert that extents, shape and resolution agree."""
        expected_lat = height * self.resolution_deg
        expected_lon = width * self.resolution_deg
        if abs(expected_lat - self.lat_span) > _DEG_EPS:
            raise ValueError(
                f"latitude extent {self.lat_span:.10f} deg does not match "
                f"{height} rows x {self.resolution_deg:.10f} deg = {expected_lat:.10f} deg"
            )
        if abs(expected_lon - self.lon_span) > _DEG_EPS:
            raise ValueError(
                f"longitude extent {self.lon_span:.10f} deg does not match "
                f"{width} cols x {self.resolution_deg:.10f} deg = {expected_lon:.10f} deg"
            )


class DomainConfig(GridGeometry):
    name: str
    grid_height: int = Field(gt=0)
    grid_width: int = Field(gt=0)

    @model_validator(mode="after")
    def _geometry_consistent(self) -> DomainConfig:
        self.check_shape(self.grid_height, self.grid_width)
        return self


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class TemporalConfig(StrictModel):
    interval_minutes: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    lookback_indices: list[int]
    #: Forecast offsets applied to the target stream, ``[1 .. lead_frames]``.
    #: ``t0`` belongs to the history and never to the forecast, so these start
    #: at 1 and there is one fewer of them than of lookback indices.
    lead_indices: list[int]
    alignment_tolerance_minutes: int = Field(ge=0)
    timezone: str

    @model_validator(mode="after")
    def _indices_consistent(self) -> TemporalConfig:
        if len(self.lookback_indices) != self.sequence_length:
            raise ValueError(
                f"lookback_indices has {len(self.lookback_indices)} entries but "
                f"sequence_length is {self.sequence_length}"
            )
        if self.lookback_indices != sorted(self.lookback_indices):
            raise ValueError("lookback_indices must be in ascending order")
        if self.lookback_indices[-1] != 0:
            raise ValueError(
                "the final lookback index must be 0 (nowcast time t0); "
                f"got {self.lookback_indices[-1]}"
            )
        if not self.lead_indices:
            raise ValueError("lead_indices is empty; there is nothing to forecast")
        expected = list(range(1, len(self.lead_indices) + 1))
        if self.lead_indices != expected:
            raise ValueError(
                f"lead_indices must be contiguous from 1; got "
                f"{self.lead_indices}, expected {expected}. A gap would mean "
                "forecasting a frame whose predecessor is never supervised, "
                "and the ConvGRU unrolls one step at a time."
            )
        # A tolerance at or beyond half the cadence lets one observation claim
        # two adjacent slots.
        if self.alignment_tolerance_minutes * 2 >= self.interval_minutes:
            raise ValueError(
                f"alignment_tolerance_minutes ({self.alignment_tolerance_minutes}) "
                f"must be under half the {self.interval_minutes}-minute cadence, "
                "or one observation can satisfy two slots"
            )
        return self

    @property
    def span_minutes(self) -> int:
        """Observed span from the oldest frame to t0."""
        return abs(self.lookback_indices[0]) * self.interval_minutes


class InsatDataset(StrictModel):
    path: str
    description: str
    units: str
    calibration_lut: str | None = None
    valid_range: tuple[float, float]
    fill_value: float

    @model_validator(mode="after")
    def _range_ordered(self) -> InsatDataset:
        lo, hi = self.valid_range
        if hi <= lo:
            raise ValueError(f"valid_range {self.valid_range} is not ascending")
        return self


class InsatGeolocation(StrictModel):
    latitude_dataset: str
    longitude_dataset: str
    subsatellite_lon: float
    satellite_altitude_km: float = Field(gt=0)


class NativeGridShape(StrictModel):
    """Mixin for sources that declare their native array dimensions."""

    native_shape: tuple[int, int]

    @field_validator("native_shape")
    @classmethod
    def _positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"native_shape {v} must be positive in both axes")
        return v

    def implied_resolution_deg(self, domain: DomainConfig) -> tuple[float, float]:
        """Grid spacing implied by this shape over the configured domain."""
        return (
            domain.lat_span / self.native_shape[0],
            domain.lon_span / self.native_shape[1],
        )


class InsatSource(NativeGridShape):
    enabled: bool
    format: Literal["hdf5"]
    root: str
    filename_pattern: str
    datasets: dict[str, InsatDataset]
    geolocation: InsatGeolocation


class ImdaaVariable(StrictModel):
    short_name: str
    description: str
    units: str


class CfgribFilterKeys(StrictModel):
    typeOfLevel: str


class CfgribBackendKwargs(StrictModel):
    filter_by_keys: CfgribFilterKeys
    indexpath: str


class ImdaaSource(NativeGridShape):
    enabled: bool
    format: Literal["grib2", "netcdf4"]
    root: str
    filename_pattern: str
    engine: str
    backend_kwargs: CfgribBackendKwargs
    pressure_levels_hpa: list[int]
    variables: dict[str, ImdaaVariable]
    native_interval_minutes: int = Field(gt=0)
    native_crs: str

    @field_validator("pressure_levels_hpa")
    @classmethod
    def _levels_descending(cls, v: list[int]) -> list[int]:
        if v != sorted(v):
            raise ValueError(
                "pressure_levels_hpa must be ascending in hPa so that channel "
                "order is deterministic"
            )
        if len(set(v)) != len(v):
            raise ValueError("pressure_levels_hpa contains duplicates")
        return v


class ImdNativeGrid(StrictModel):
    """Native grid of an IMD product."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n_lat: int = Field(gt=0)
    n_lon: int = Field(gt=0)
    resolution_deg: float = Field(gt=0)
    dtype: str
    fill_value: float

    @property
    def expected_bytes_per_field(self) -> int:
        itemsize = {"float32": 4, "float64": 8, "int16": 2, "int32": 4}[self.dtype]
        return self.n_lat * self.n_lon * itemsize


class ImdSurfaceVariable(StrictModel):
    """One IMD surface product."""

    format: Literal["binary_grd", "netcdf4"]
    filename_pattern: str
    units: str
    native_interval_minutes: int = Field(gt=0)
    description: str | None = None
    variable: str | None = None
    u_variable: str | None = None
    v_variable: str | None = None
    derive: Literal["magnitude"] | None = None
    native_grid: ImdNativeGrid | None = None

    @model_validator(mode="after")
    def _reader_is_specified(self) -> ImdSurfaceVariable:
        if self.format == "binary_grd" and self.native_grid is None:
            raise ValueError(
                "a binary_grd product has no self-describing header, so "
                "native_grid is required to interpret it"
            )
        if self.format == "netcdf4":
            has_scalar = self.variable is not None
            has_components = self.u_variable is not None and self.v_variable is not None
            if not (has_scalar or has_components):
                raise ValueError(
                    "a netcdf4 product needs either `variable` or both "
                    "`u_variable` and `v_variable`"
                )
        if self.derive is not None and not (self.u_variable and self.v_variable):
            raise ValueError("`derive` requires both u_variable and v_variable")
        return self


class ImdSurfaceSource(StrictModel):
    enabled: bool
    root: str
    variables: dict[str, ImdSurfaceVariable]


class StaticLayer(StrictModel):
    path: str
    format: Literal["geotiff", "netcdf4"]
    description: str
    units: str
    resampling: Resampling
    fill_value: float
    #: GDAL overview pyramid level to read instead of the base array. ``None``
    #: reads base resolution. Level 0 is the first overview, typically 2x
    #: decimation; level 1 is 4x, and so on.
    overview_level: int | None = None

    @property
    def is_categorical(self) -> bool:
        return self.units == "class_index"

    @field_validator("overview_level")
    @classmethod
    def _overview_non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError(f"overview_level {v} must be >= 0, or null for base")
        return v

    @model_validator(mode="after")
    def _categorical_constraints(self) -> StaticLayer:
        if self.is_categorical and self.resampling is not Resampling.NEAREST:
            raise ValueError(
                f"a class_index raster must use nearest resampling, not "
                f"{self.resampling.value}: interpolating class indices "
                "produces values that denote nothing"
            )
        if self.is_categorical and self.overview_level is not None:
            # GDAL builds overviews with averaging unless told otherwise, so a
            # pyramid level of a class raster contains blended indices that
            # denote nothing. Reading one would silently corrupt the mask in a
            # way no range check catches, since the blended value is still a
            # number inside the class range.
            raise ValueError(
                "a class_index raster must be read at base resolution "
                "(overview_level: null). GDAL overviews are averaged, and an "
                "averaged class index is not a class"
            )
        return self


class StaticPriorsSource(StrictModel):
    enabled: bool
    root: str
    layers: dict[str, StaticLayer]


class SourcesConfig(StrictModel):
    insat: InsatSource
    imdaa: ImdaaSource
    imd_surface: ImdSurfaceSource
    static_priors: StaticPriorsSource


# ---- targets ---------------------------------------------------------------
#
# Modelled separately from sources, and that separation is the safety rail. A
# target parsed alongside the inputs is one refactor away from being stacked
# into the channel tensor, and a model that can read its own answer scores
# perfectly while forecasting nothing. Nothing that walks ``SourcesConfig`` or
# the channel groups can reach these types.


class TargetVariable(StrictModel):
    """The single field a target stream yields."""

    name: str
    path: str
    description: str
    units: Literal["mm h-1"]
    valid_range: tuple[float, float]
    fill_value: float

    @model_validator(mode="after")
    def _range_ordered(self) -> TargetVariable:
        lo, hi = self.valid_range
        if hi <= lo:
            raise ValueError(f"valid_range {self.valid_range} is not ascending")
        if lo < 0:
            raise ValueError(
                f"valid_range {self.valid_range} admits negative rain rate"
            )
        return self


class TargetStreamQC(StrictModel):
    """Quality control applicable to a target stream."""

    parallax_correction: bool
    scanline_dropout: bool
    physical_bounds: bool


class HemTargetSource(NativeGridShape):
    """INSAT-3D/3DR Hydro Estimator Method surface rain rate."""

    enabled: bool
    format: Literal["hdf5"]
    root: str
    native_interval_minutes: int = Field(gt=0)
    filename_pattern: str
    variable: TargetVariable
    geolocation: InsatGeolocation
    quality_control: TargetStreamQC

    @model_validator(mode="after")
    def _parallax_is_applied(self) -> HemTargetSource:
        if self.enabled and not self.quality_control.parallax_correction:
            raise ValueError(
                "parallax_correction is disabled for the HEM target. HEM "
                "reports rain at the pixel where the cloud top was observed, "
                "not where the rain reaches the ground, so an uncorrected "
                "target sits tens of kilometres from the truth and the model "
                "would learn to reproduce the displacement."
            )
        return self


class TargetsConfig(StrictModel):
    hem: HemTargetSource


# ---- quality control -------------------------------------------------------


class RadiometricCalibration(StrictModel):
    enabled: bool
    apply_lut: bool
    lut_interpolation: Literal["linear", "nearest"]


class ScanlineDropout(StrictModel):
    enabled: bool
    max_consecutive_fill_rows: int = Field(ge=0)
    row_constant_tolerance: float = Field(ge=0)
    saturation_fraction_threshold: float = Field(gt=0, le=1)


class ParallaxCorrection(StrictModel):
    enabled: bool
    method: Literal["geometric"]
    cloud_height_source: Literal["ctt_lapse_rate", "cth_product"]
    assumed_lapse_rate_k_per_km: float = Field(gt=0)
    reference_surface_temp_k: float = Field(gt=0)


class PhysicalBounds(BaseModel):
    """Per-variable hard limits."""

    model_config = ConfigDict(extra="allow", frozen=True)

    action: Literal["mask", "clip"]

    def bounds(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for name, value in (self.__pydantic_extra__ or {}).items():
            if not (isinstance(value, list) and len(value) == 2):
                raise ValueError(f"physical_bounds.{name} must be a [min, max] pair")
            lo, hi = float(value[0]), float(value[1])
            if hi <= lo:
                raise ValueError(
                    f"physical_bounds.{name} = [{lo}, {hi}] is not ascending"
                )
            out[name] = (lo, hi)
        return out


class OpticalFlowParams(StrictModel):
    method: Literal["farneback"]
    pyramid_scale: float = Field(gt=0, lt=1)
    levels: int = Field(gt=0)
    window_size: int = Field(gt=0)
    iterations: int = Field(gt=0)
    poly_n: int = Field(gt=0)
    poly_sigma: float = Field(gt=0)


class SplineParams(StrictModel):
    order: int = Field(ge=1, le=5)
    bidirectional: bool


class GapFilling(StrictModel):
    enabled: bool
    strategy: GapFillStrategy
    optical_flow: OpticalFlowParams
    spline: SplineParams
    max_consecutive_missing_frames: int = Field(ge=0)
    reject_sequence_if_exceeded: bool


class QualityControlConfig(StrictModel):
    radiometric_calibration: RadiometricCalibration
    scanline_dropout: ScanlineDropout
    parallax_correction: ParallaxCorrection
    physical_bounds: PhysicalBounds
    gap_filling: GapFilling


class IngestionConfig(StrictModel):
    domain: DomainConfig
    temporal: TemporalConfig
    sources: SourcesConfig
    #: Ground truth, held apart from ``sources`` so no assembly step can reach
    #: it while building the input tensor.
    targets: TargetsConfig
    quality_control: QualityControlConfig

    @model_validator(mode="after")
    def _target_cadence_matches(self) -> IngestionConfig:
        hem = self.targets.hem
        if not hem.enabled:
            return self
        if hem.native_interval_minutes != self.temporal.interval_minutes:
            raise ValueError(
                f"the HEM target arrives every {hem.native_interval_minutes} "
                f"min but the model cadence is {self.temporal.interval_minutes} "
                "min. A target lifted onto a different cadence must either "
                "invent intensity between samples or hold a value across them; "
                "the whole reason for this source is that it needs neither."
            )
        if hem.variable.name not in self.quality_control.physical_bounds.bounds():
            raise ValueError(
                f"no physical bounds declared for the target "
                f"{hem.variable.name!r}. An unbounded target lets a retrieval "
                "artefact enter the loss as record rainfall."
            )
        return self


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TargetGrid(GridGeometry):
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    cell_alignment: Literal["center", "edge"]

    @model_validator(mode="after")
    def _geometry_consistent(self) -> TargetGrid:
        self.check_shape(self.height, self.width)
        return self


class ResamplingByKind(StrictModel):
    continuous: Resampling
    categorical: Resampling
    precipitation: Resampling

    @model_validator(mode="after")
    def _categorical_is_nearest(self) -> ResamplingByKind:
        if self.categorical is not Resampling.NEAREST:
            raise ValueError(
                "categorical fields must resample with nearest neighbour; "
                f"got {self.categorical.value}"
            )
        return self


class ReprojectionConfig(StrictModel):
    resampling_by_kind: ResamplingByKind
    insat_regrid_method: Literal["curvilinear_lookup", "affine"]
    imdaa_source_crs_autodetect: bool
    fallback_source_crs: str
    nodata_policy: Literal["propagate", "fill"]


class TemporalInterpolationMethods(StrictModel):
    continuous: Literal["linear", "hold", "nearest"]
    precipitation: Literal["linear", "hold", "nearest"]
    categorical: Literal["hold", "nearest"]

    @model_validator(mode="after")
    def _precipitation_not_linear(self) -> TemporalInterpolationMethods:
        if self.precipitation == "linear":
            raise ValueError(
                "linear interpolation of an accumulated precipitation product "
                "invents drizzle between observations and destroys the "
                "intensity distribution the model must learn; use `hold`"
            )
        return self


class TemporalInterpolationConfig(StrictModel):
    method_by_kind: TemporalInterpolationMethods
    extrapolate: bool


class ChannelOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: FieldKind | None = None
    normalization: NormalizationMethod | None = None
    transform: Literal["log1p", "sqrt"] | None = None
    encoding: Literal["ordinal", "onehot"] | None = None


class ChannelGroup(StrictModel):
    indices: list[int]
    names: list[str]
    kind: FieldKind | None = None
    normalization: NormalizationMethod | None = None
    broadcast_over_time: bool = False
    per_channel_overrides: dict[str, ChannelOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _indices_match_names(self) -> ChannelGroup:
        if len(self.indices) != len(self.names):
            raise ValueError(
                f"group has {len(self.indices)} indices but {len(self.names)} names"
            )
        if self.indices != list(
            range(self.indices[0], self.indices[0] + len(self.indices))
        ):
            raise ValueError(f"group indices {self.indices} are not contiguous")
        unknown = set(self.per_channel_overrides) - set(self.names)
        if unknown:
            raise ValueError(
                f"per_channel_overrides names not in this group: {sorted(unknown)}"
            )
        return self


class ChannelsConfig(StrictModel):
    count: int = Field(gt=0)
    satellite: ChannelGroup
    nwp: ChannelGroup
    surface: ChannelGroup
    static: ChannelGroup

    @property
    def groups(self) -> dict[str, ChannelGroup]:
        return {
            "satellite": self.satellite,
            "nwp": self.nwp,
            "surface": self.surface,
            "static": self.static,
        }

    @property
    def names(self) -> list[str]:
        """Channel names in tensor order."""
        ordered: list[tuple[int, str]] = []
        for group in self.groups.values():
            ordered.extend(zip(group.indices, group.names, strict=False))
        return [name for _, name in sorted(ordered)]

    @model_validator(mode="after")
    def _partition_is_exact(self) -> ChannelsConfig:
        assigned: dict[int, str] = {}
        for group_name, group in self.groups.items():
            for idx, name in zip(group.indices, group.names, strict=False):
                if idx in assigned:
                    raise ValueError(
                        f"channel index {idx} claimed by both '{assigned[idx]}' "
                        f"and '{name}' (group {group_name})"
                    )
                assigned[idx] = name

        expected = set(range(self.count))
        if set(assigned) != expected:
            missing = sorted(expected - set(assigned))
            extra = sorted(set(assigned) - expected)
            raise ValueError(
                f"channel indices must cover 0..{self.count - 1} exactly; "
                f"missing={missing} unexpected={extra}"
            )

        names = list(assigned.values())
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate channel names: {dupes}")
        return self


class ReferencePeriod(StrictModel):
    """Span over which the climatology was computed."""

    start: date
    end: date
    months: list[int] = Field(default_factory=lambda: list(range(1, 13)))

    @model_validator(mode="after")
    def _ordered(self) -> ReferencePeriod:
        if self.end <= self.start:
            raise ValueError(
                f"reference period end {self.end} must follow start {self.start}"
            )
        return self

    @field_validator("months")
    @classmethod
    def _valid_months(cls, v: list[int]) -> list[int]:
        return _check_months(v, "reference_period.months")

    def covers(self, moment: date | datetime) -> bool:
        """True when a date falls inside both the span and the month set."""
        day = moment.date() if isinstance(moment, datetime) else moment
        return self.start <= day <= self.end and day.month in self.months


class NormalizationMethods(StrictModel):
    atmospheric: NormalizationMethod
    satellite: NormalizationMethod
    static: NormalizationMethod


class StatsSchema(StrictModel):
    required_keys: list[str]
    optional_keys: list[str]


class NormalizationConfig(StrictModel):
    """Climatological normalisation."""

    stats_path: str
    reference_period: ReferencePeriod
    method: NormalizationMethods
    stats_schema: StatsSchema
    epsilon: float = Field(gt=0)
    min_std: float = Field(gt=0)
    fallback_bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    fallback_moments: dict[str, tuple[float, float]] = Field(default_factory=dict)
    clip_to_sigma: float = Field(gt=0)

    def resolved_stats_path(self, root: Path = REPO_ROOT) -> Path:
        p = Path(self.stats_path)
        return p if p.is_absolute() else root / p


class TensorConfig(StrictModel):
    layout: str
    sequence_length: int = Field(gt=0)
    channels: int = Field(gt=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    dtype: str
    nan_fill_value: float
    emit_validity_mask: bool
    mask_layout: str

    @field_validator("layout")
    @classmethod
    def _layout_is_btchw(cls, v: str) -> str:
        if v.split() != ["B", "T", "C", "H", "W"]:
            raise ValueError(
                f"tensor layout must be 'B T C H W' to match the Earthformer "
                f"cuboid attention contract; got {v!r}"
            )
        return v

    @property
    def shape_without_batch(self) -> tuple[int, int, int, int]:
        return (self.sequence_length, self.channels, self.height, self.width)


class PreprocessingConfig(StrictModel):
    target_grid: TargetGrid
    reprojection: ReprojectionConfig
    temporal_interpolation: TemporalInterpolationConfig
    channels: ChannelsConfig
    normalization: NormalizationConfig
    tensor: TensorConfig

    @model_validator(mode="after")
    def _tensor_matches_grid_and_channels(self) -> PreprocessingConfig:
        if (self.tensor.height, self.tensor.width) != (
            self.target_grid.height,
            self.target_grid.width,
        ):
            raise ValueError(
                f"tensor is {self.tensor.height}x{self.tensor.width} but the "
                f"target grid is {self.target_grid.height}x{self.target_grid.width}"
            )
        if self.tensor.channels != self.channels.count:
            raise ValueError(
                f"tensor declares {self.tensor.channels} channels but the "
                f"channel specification defines {self.channels.count}"
            )
        return self


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ModelInput(StrictModel):
    sequence_length: int = Field(gt=0)
    channels: int = Field(gt=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)


class ModelOutput(StrictModel):
    lead_frames: int = Field(gt=0)
    lead_interval_minutes: int = Field(gt=0)
    horizon_hours: float = Field(gt=0)
    channels: int = Field(gt=0)
    units: str

    @model_validator(mode="after")
    def _horizon_matches_frames(self) -> ModelOutput:
        implied = self.lead_frames * self.lead_interval_minutes / 60.0
        if abs(implied - self.horizon_hours) > 1e-9:
            raise ValueError(
                f"{self.lead_frames} frames x {self.lead_interval_minutes} min "
                f"= {implied} h, which contradicts horizon_hours={self.horizon_hours}"
            )
        return self


class PatchEmbed(StrictModel):
    patch_size: int = Field(gt=0)
    embed_dim: int = Field(gt=0)
    norm: str


class EncoderSpec(StrictModel):
    depths: list[int]
    downsample_factors: list[int]
    dims: list[int]

    @model_validator(mode="after")
    def _stage_lists_align(self) -> EncoderSpec:
        if not (len(self.depths) == len(self.downsample_factors) == len(self.dims)):
            raise ValueError(
                "depths, downsample_factors and dims must describe the same "
                f"number of stages: got {len(self.depths)}, "
                f"{len(self.downsample_factors)}, {len(self.dims)}"
            )
        return self


class DecoderSpec(StrictModel):
    depths: list[int]
    upsample_factors: list[int]
    dims: list[int]
    cross_attention: bool
    cross_attn_heads: int = Field(gt=0)

    @model_validator(mode="after")
    def _stage_lists_align(self) -> DecoderSpec:
        if not (len(self.depths) == len(self.upsample_factors) == len(self.dims)):
            raise ValueError("decoder depths, upsample_factors and dims must align")
        return self


class CuboidBlock(StrictModel):
    size: tuple[int, int, int]
    strategy: Literal["local", "dilated"]
    shift: tuple[int, int, int]

    @model_validator(mode="after")
    def _positive_extent(self) -> CuboidBlock:
        if any(s <= 0 for s in self.size):
            raise ValueError(f"cuboid size {self.size} must be positive in every axis")
        return self


class GlobalVectors(StrictModel):
    enabled: bool
    num_vectors: int = Field(ge=0)
    update_rule: str


class PositionalEncoding(StrictModel):
    type: Literal["learned", "sinusoidal"]
    separate_temporal_spatial: bool


class CheckpointSpec(StrictModel):
    """Reference to weights supplied by the deployment."""

    load_pretrained: bool
    path: str
    strict: bool
    reinit_modules: list[str] = Field(default_factory=list)

    def resolved_path(self, root: Path = REPO_ROOT) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else root / p


class EarthformerConfig(StrictModel):
    patch_embed: PatchEmbed
    base_units: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    ffn_expansion: int = Field(gt=0)
    attn_dropout: float = Field(ge=0, lt=1)
    proj_dropout: float = Field(ge=0, lt=1)
    ffn_dropout: float = Field(ge=0, lt=1)
    activation: str
    encoder: EncoderSpec
    decoder: DecoderSpec
    cuboid_blocks: list[CuboidBlock]
    global_vectors: GlobalVectors
    positional_encoding: PositionalEncoding
    relative_position_bias: bool
    checkpoint: CheckpointSpec

    @model_validator(mode="after")
    def _heads_divide_dims(self) -> EarthformerConfig:
        for stage, dim in enumerate(self.encoder.dims):
            if dim % self.num_heads:
                raise ValueError(
                    f"encoder stage {stage} width {dim} is not divisible by "
                    f"num_heads={self.num_heads}"
                )
        return self


class AdapterConfig(StrictModel):
    in_dim: int = Field(gt=0)
    temporal_to_channel: bool
    in_sequence: int = Field(gt=0)
    output_scales: list[int]
    output_dims: list[int]
    norm: str
    groups: int = Field(gt=0)
    activation: str
    interpolation: str
    align_corners: bool
    residual_projection: bool

    @model_validator(mode="after")
    def _pyramid_aligned(self) -> AdapterConfig:
        if len(self.output_scales) != len(self.output_dims):
            raise ValueError(
                f"output_scales has {len(self.output_scales)} entries but "
                f"output_dims has {len(self.output_dims)}"
            )
        if self.output_scales != sorted(self.output_scales, reverse=True):
            raise ValueError(
                "output_scales must descend: the DGMR sampler consumes "
                "conditioning coarsest-last"
            )
        for dim in self.output_dims:
            if dim % self.groups:
                raise ValueError(
                    f"adapter output dim {dim} is not divisible by "
                    f"groups={self.groups} for group normalisation"
                )
        return self


class DGMRGenerator(StrictModel):
    conditioning_dims: list[int]
    latent_channels: int = Field(gt=0)
    # No separate sampler width: every level's channel count is fixed by
    # conditioning_dims, so a standalone hidden size would be a dead knob that
    # silently governs nothing.
    num_upsample_blocks: int = Field(gt=0)
    output_channels: int = Field(gt=0)
    forecast_steps: int = Field(gt=0)
    convgru_kernel: int = Field(gt=0)
    spectral_norm: bool


class LatentConditioningStack(StrictModel):
    noise_channels: int = Field(gt=0)
    latent_resolution: int = Field(gt=0)
    output_channels: int = Field(gt=0)
    ensemble_members: int = Field(gt=0)


class DiscriminatorSpec(StrictModel):
    enabled: bool
    base_channels: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    spectral_norm: bool
    num_sampled_frames: int | None = None
    stem_kernel: tuple[int, int, int] | None = None
    stem_stride: tuple[int, int, int] | None = None


class Discriminators(StrictModel):
    spatial: DiscriminatorSpec
    temporal: DiscriminatorSpec


class DGMRConfig(StrictModel):
    generator: DGMRGenerator
    latent_conditioning_stack: LatentConditioningStack
    discriminators: Discriminators
    checkpoint: CheckpointSpec

    @model_validator(mode="after")
    def _latent_matches_generator(self) -> DGMRConfig:
        if (
            self.latent_conditioning_stack.output_channels
            != self.generator.latent_channels
        ):
            raise ValueError(
                f"latent stack emits {self.latent_conditioning_stack.output_channels} "
                f"channels but the generator expects {self.generator.latent_channels}"
            )
        return self


class FusionSpec(StrictModel):
    name: str
    freeze_backbone: bool
    relay: list[str]
    precision: Literal["fp32", "fp16", "bf16"]
    gradient_checkpointing: bool

    @field_validator("relay")
    @classmethod
    def _relay_order(cls, v: list[str]) -> list[str]:
        expected = ["earthformer", "adapter", "dgmr"]
        if v != expected:
            raise ValueError(f"relay must be {expected}; got {v}")
        return v


class ModelConfig(StrictModel):
    input: ModelInput
    output: ModelOutput
    earthformer: EarthformerConfig
    adapter: AdapterConfig
    dgmr: DGMRConfig
    fusion: FusionSpec

    @model_validator(mode="after")
    def _relay_shapes_agree(self) -> ModelConfig:
        if self.adapter.in_dim != self.earthformer.decoder.dims[-1]:
            raise ValueError(
                f"adapter.in_dim={self.adapter.in_dim} does not match the final "
                f"Earthformer decoder width {self.earthformer.decoder.dims[-1]}"
            )
        if self.adapter.in_sequence != self.output.lead_frames:
            raise ValueError(
                f"adapter.in_sequence={self.adapter.in_sequence} does not match "
                f"output.lead_frames={self.output.lead_frames}"
            )
        if self.adapter.output_dims != self.dgmr.generator.conditioning_dims:
            raise ValueError(
                "adapter.output_dims must equal dgmr.generator.conditioning_dims; "
                f"got {self.adapter.output_dims} and "
                f"{self.dgmr.generator.conditioning_dims}"
            )
        if self.dgmr.generator.forecast_steps != self.output.lead_frames:
            raise ValueError(
                f"dgmr forecast_steps={self.dgmr.generator.forecast_steps} does "
                f"not match output.lead_frames={self.output.lead_frames}"
            )
        # The patch stem must divide the input grid, or the cuboid partition
        # silently pads and the spatial extents no longer mean what they say.
        patch = self.earthformer.patch_embed.patch_size
        if self.input.height % patch or self.input.width % patch:
            raise ValueError(
                f"input grid {self.input.height}x{self.input.width} is not "
                f"divisible by patch_size={patch}"
            )
        return self


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class RunConfig(StrictModel):
    default_valid_time: datetime
    allow_out_of_record_time: bool
    record_start: datetime
    record_end: datetime
    #: Calendar months the record actually covers. The span is four monsoon
    #: seasons, so the start and end dates alone would admit dry-season
    #: timestamps for which no observation of any source exists.
    record_months: list[int] = Field(default_factory=lambda: list(range(1, 13)))

    @field_validator("record_months")
    @classmethod
    def _valid_months(cls, v: list[int]) -> list[int]:
        return _check_months(v, "run.record_months")

    def in_record(self, moment: datetime) -> bool:
        """True when a timestamp lies inside both the span and the season."""
        return (
            self.record_start <= moment <= self.record_end
            and moment.month in self.record_months
        )

    @model_validator(mode="after")
    def _default_within_record(self) -> RunConfig:
        if self.record_end <= self.record_start:
            raise ValueError("record_end must follow record_start")
        if not self.allow_out_of_record_time:
            if not self.record_start <= self.default_valid_time <= self.record_end:
                raise ValueError(
                    f"default_valid_time {self.default_valid_time.isoformat()} lies "
                    f"outside the fine-tuning record and "
                    "allow_out_of_record_time is false"
                )
            # Checked separately from the span so the message can say which of
            # the two conditions failed. A demo date in the dry season sits
            # comfortably between record_start and record_end and still has no
            # data behind it.
            if self.default_valid_time.month not in self.record_months:
                raise ValueError(
                    f"default_valid_time "
                    f"{self.default_valid_time.isoformat()} falls in month "
                    f"{self.default_valid_time.month}, which is outside the "
                    f"record months {self.record_months}. The date lies within "
                    "record_start and record_end but in a season the record "
                    "does not cover."
                )
        return self


class InferenceModelSpec(StrictModel):
    config: str
    checkpoint: str
    device: str
    fallback_device: str
    precision: Literal["fp32", "fp16", "bf16"]
    eval_mode: bool


class EnsembleConfig(StrictModel):
    members: int = Field(gt=0)
    seed: int
    reductions: list[str]


class LeadTimesConfig(StrictModel):
    interval_minutes: int = Field(gt=0)
    frames: int = Field(gt=0)
    horizon_hours: float = Field(gt=0)
    reported_hours: list[int]

    @model_validator(mode="after")
    def _reported_within_horizon(self) -> LeadTimesConfig:
        implied = self.frames * self.interval_minutes / 60.0
        if abs(implied - self.horizon_hours) > 1e-9:
            raise ValueError(
                f"{self.frames} frames x {self.interval_minutes} min = {implied} h, "
                f"contradicting horizon_hours={self.horizon_hours}"
            )
        for hour in self.reported_hours:
            if not 0 < hour <= self.horizon_hours:
                raise ValueError(
                    f"reported hour {hour} lies outside the {self.horizon_hours} h horizon"
                )
        return self


class SeverityBand(StrictModel):
    min_probability: float = Field(ge=0, le=1)
    min_intensity_mm_h: float = Field(ge=0)


class ThresholdConfig(StrictModel):
    precipitation_mm_h: dict[str, float]
    probability_from_ensemble: bool
    min_reported_probability: float = Field(ge=0, le=1)
    severity_bands: dict[str, SeverityBand]

    @model_validator(mode="after")
    def _bands_monotonic(self) -> ThresholdConfig:
        intensities = list(self.precipitation_mm_h.values())
        if intensities != sorted(intensities):
            raise ValueError(
                "precipitation_mm_h classes must ascend in intensity; "
                f"got {self.precipitation_mm_h}"
            )
        order = ["low", "moderate", "high", "severe"]
        present = [b for b in order if b in self.severity_bands]
        probs = [self.severity_bands[b].min_probability for b in present]
        if probs != sorted(probs):
            raise ValueError("severity band probabilities must ascend with severity")
        return self


class IntegratedGradientsConfig(StrictModel):
    n_steps: int = Field(gt=0)
    baseline: Literal["climatological_mean", "zeros"]
    internal_batch_size: int = Field(gt=0)
    target: str
    #: Ensemble members carried through attribution. Fewer than the served
    #: ensemble on purpose: cost is ``n_steps x members`` relay evaluations,
    #: and at 32 steps every extra member is another 32 forward and backward
    #: passes through the whole relay.
    members: int = Field(gt=0)
    #: Relaxation width of the exceedance surrogate, in mm h-1.
    #:
    #: The configured target is an exceedance *probability*, which as a count
    #: of members over a threshold has zero gradient almost everywhere --
    #: Integrated Gradients over it returns zeros and looks like it worked.
    #: The surrogate ``sigmoid((y - tau) / T)`` restores the gradient and
    #: approaches the count as ``T`` falls.
    #:
    #: ``T`` is part of the question being asked, not an implementation
    #: detail: too large and moderate rain bleeds into a heavy-rain
    #: attribution, too small and the gradient vanishes again. It is recorded
    #: in every report for that reason.
    surrogate_temperature_mm_h: float = Field(gt=0)

    @field_validator("baseline")
    @classmethod
    def _baseline_is_meaningful(cls, v: str) -> str:
        if v == "zeros":
            raise ValueError(
                "an all-zeros baseline is meaningless in normalised space, "
                "where zero is a specific and quite ordinary atmospheric state "
                "rather than an absence of weather"
            )
        return v


class AttributionMapConfig(StrictModel):
    output_resolution: int = Field(gt=0)
    normalize: str
    colormap: str
    top_k_drivers: int = Field(gt=0)


class EvidenceFramesConfig(StrictModel):
    enabled: bool
    top_k: int = Field(gt=0)
    rank_by: str


class XAIConfig(StrictModel):
    """Explainability settings."""

    enabled: bool
    integrated_gradients: IntegratedGradientsConfig
    attribution_maps: AttributionMapConfig
    evidence_frames: EvidenceFramesConfig


class GeospatialConfig(StrictModel):
    """Administrative intersection of exceedance masks."""

    admin_boundaries: str
    admin_layer: str
    admin_crs: str
    join_predicate: str
    #: Absolute floor. Must exceed one grid cell or it suppresses nothing.
    min_affected_area_km2: float = Field(ge=0)
    #: Share of a district's own area. Carries districts smaller than the
    #: absolute floor, which it could otherwise never flag.
    min_fractional_coverage: float = Field(ge=0, le=1)
    dissolve_by: list[str]

    def resolved_boundaries(self, root: Path = REPO_ROOT) -> Path:
        p = Path(self.admin_boundaries)
        return p if p.is_absolute() else root / p

    @model_validator(mode="after")
    def _gates_are_meaningful(self) -> GeospatialConfig:
        if self.min_affected_area_km2 <= 0 and self.min_fractional_coverage <= 0:
            raise ValueError(
                "both qualification gates are zero, so every district touched "
                "by a single cell is flagged. That is a domain-wide alert on "
                "one pixel of exceedance."
            )
        if not self.dissolve_by:
            raise ValueError(
                "dissolve_by is empty; districts need a stable identity. Name "
                "alone is not one -- India has an Aurangabad in Maharashtra "
                "and another in Bihar."
            )
        return self


class LocalEmbeddingConfig(StrictModel):
    """Sentence-transformer embedding, run locally."""

    model_id: str
    dimensions: int = Field(gt=0)
    query_prefix: str
    document_prefix: str
    #: CPU on purpose. The relay holds the GPU during a forecast, and a
    #: 130 MiB embedder has no business competing with it for VRAM.
    device: str = "cpu"
    #: e5 expects cosine similarity over unit-length vectors.
    normalize: bool = True
    batch_size: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def _prefixes_differ(self) -> LocalEmbeddingConfig:
        if self.query_prefix == self.document_prefix:
            raise ValueError(
                "query_prefix and document_prefix must differ; embedding both "
                "sides identically discards the asymmetry the model was "
                "trained with and measurably degrades retrieval"
            )
        return self


class VectorStoreConfig(StrictModel):
    provider: Literal["chromadb"]
    persist_directory: str
    collection: str
    metadata_filters: list[str]
    top_k: int = Field(gt=0)
    local_embedding: LocalEmbeddingConfig


class LocalAdvisoryConfig(StrictModel):
    """The advisory model, run in-process under guided decoding."""

    model_id_or_path: str
    device: str
    fallback_device: str
    context_window: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    #: Low, not zero. Guided decoding already guarantees the shape; the
    #: temperature governs only wording, and at exactly zero a model that
    #: begins a sentence badly has no way out of it.
    temperature: float = Field(ge=0, le=2)

    def resolved_path(self, root: Path = REPO_ROOT) -> Path:
        """Local path, when ``model_id_or_path`` names a directory."""
        p = Path(self.model_id_or_path)
        return p if p.is_absolute() else root / p

    @model_validator(mode="after")
    def _tokens_fit_context(self) -> LocalAdvisoryConfig:
        if self.max_tokens >= self.context_window:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) leaves no room for the prompt "
                f"inside a {self.context_window}-token context. The retrieved "
                "NDMA excerpts and the forecast summary must fit alongside the "
                "generation."
            )
        return self


class CAPConfig(StrictModel):
    version: str
    status: str
    msg_type: str
    scope: str
    sender: str
    language: str
    categories: list[str]
    urgency_by_lead_hours: dict[str, float]
    constrain_to_retrieved: bool
    require_citations: bool

    @model_validator(mode="after")
    def _grounding_enforced(self) -> CAPConfig:
        if not self.constrain_to_retrieved:
            raise ValueError(
                "constrain_to_retrieved must be true: an advisory that is not "
                "grounded in retrieved NDMA text is model-authored safety "
                "guidance, which this system must never emit"
            )
        return self


class HazardTaxonomyConfig(StrictModel):
    """Terrain and hazard labels used to scope NDMA retrieval."""

    mountainous_elevation_m: float
    coastal_elevation_m: float
    urban_fraction: float = Field(ge=0, le=1)
    urban_lulc_classes: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bands_ordered(self) -> HazardTaxonomyConfig:
        if self.coastal_elevation_m >= self.mountainous_elevation_m:
            raise ValueError(
                f"coastal_elevation_m ({self.coastal_elevation_m}) must lie "
                f"below mountainous_elevation_m "
                f"({self.mountainous_elevation_m}); the bands would otherwise "
                "overlap and a district could be classified as both"
            )
        return self


class AdvisoryConfig(StrictModel):
    enabled: bool
    vector_store: VectorStoreConfig
    local_advisory: LocalAdvisoryConfig
    cap: CAPConfig
    taxonomy: HazardTaxonomyConfig


class CacheConfig(StrictModel):
    enabled: bool
    backend: str
    ttl_seconds: int = Field(ge=0)
    max_entries: int = Field(gt=0)


class APIConfig(StrictModel):
    host: str
    port: int = Field(gt=0, lt=65536)
    root_path: str
    cors_allow_origins: list[str]
    cors_allow_credentials: bool
    request_timeout_seconds: int = Field(gt=0)
    cache: CacheConfig


class InferenceConfig(StrictModel):
    run: RunConfig
    model: InferenceModelSpec
    ensemble: EnsembleConfig
    lead_times: LeadTimesConfig
    thresholds: ThresholdConfig
    xai: XAIConfig
    geospatial: GeospatialConfig
    advisory: AdvisoryConfig
    api: APIConfig

    @model_validator(mode="after")
    def _attribution_within_ensemble(self) -> InferenceConfig:
        """Attribution may explain fewer members than are served, never more."""
        attributed = self.xai.integrated_gradients.members
        if attributed > self.ensemble.members:
            raise ValueError(
                f"attribution carries {attributed} members but the served "
                f"ensemble has {self.ensemble.members}. An explanation cannot "
                "describe more realisations than the forecast it explains."
            )
        return self


# ---------------------------------------------------------------------------
# Training
#
# Mirrors configs/train/training.yaml. The blocks under ``losses`` and
# ``replay.policy`` deliberately match training.losses.LossWeights and
# training.replay_buffer.RetentionPolicy field for field: those two modules
# hold plain dataclasses with no configuration import, and this layer is where
# YAML becomes them.
# ---------------------------------------------------------------------------


class TrainingRunConfig(StrictModel):
    name: str
    seed: int
    deterministic: bool
    device: str
    fallback_device: str
    precision: Literal["fp32", "fp16", "bf16"]
    gradient_checkpointing: bool


class SeasonsConfig(StrictModel):
    """The monsoon seasons the record covers."""

    years: list[int]
    months: list[int]

    @field_validator("months")
    @classmethod
    def _valid_months(cls, v: list[int]) -> list[int]:
        return _check_months(v, "data.seasons.months")

    @field_validator("years")
    @classmethod
    def _valid_years(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError(
                "data.seasons.years is empty; there is nothing to train on"
            )
        if len(set(v)) != len(v):
            raise ValueError(f"data.seasons.years repeats a year: {v}")
        return sorted(v)


class SplitConfig(StrictModel):
    """The train/validation boundary."""

    strategy: Literal["chronological"]
    train_until: datetime
    validation_from: datetime
    require_window_within_season: bool

    @model_validator(mode="after")
    def _ordered(self) -> SplitConfig:
        if self.validation_from <= self.train_until:
            raise ValueError(
                f"validation_from {self.validation_from.isoformat()} must follow "
                f"train_until {self.train_until.isoformat()}; an overlapping "
                "boundary puts the same window on both sides of the split"
            )
        return self


class TrainingDataConfig(StrictModel):
    seasons: SeasonsConfig
    split: SplitConfig
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    prefetch_factor: int = Field(gt=0)
    pin_memory: bool
    shuffle_train: bool
    drop_last: bool

    @model_validator(mode="after")
    def _split_within_seasons(self) -> TrainingDataConfig:
        years = self.seasons.years
        if self.split.train_until.year not in years:
            raise ValueError(
                f"train_until falls in {self.split.train_until.year}, which is "
                f"not among the configured seasons {years}"
            )
        if self.split.validation_from.year not in years:
            raise ValueError(
                f"validation_from falls in {self.split.validation_from.year}, "
                f"which is not among the configured seasons {years}"
            )
        if self.split.train_until.year >= max(years):
            raise ValueError(
                f"train_until closes {self.split.train_until.year}, the last "
                f"configured season, leaving no year to validate on"
            )
        return self


#: Cache scopes, defined here rather than in ``datasets.cache`` so the
#: vocabulary has one owner. ``datasets.cache`` imports this alias; the
#: alternative -- the same three strings written in two modules -- is a pair
#: that eventually disagrees, with the configuration accepting a value the
#: cache does not implement.
CacheScope = Literal["none", "target_only", "both"]


class WindowCacheConfig(StrictModel):
    """Disk-backed cache of assembled windows."""

    dir: str
    scope: CacheScope

    @model_validator(mode="after")
    def _directory_required_when_active(self) -> WindowCacheConfig:
        if self.scope != "none" and not self.dir.strip():
            raise ValueError(
                f"cache.scope is {self.scope!r} but no directory is configured"
            )
        return self

    def resolved_dir(self, root: Path = REPO_ROOT) -> Path:
        p = Path(self.dir)
        return p if p.is_absolute() else root / p


class OptimizerSpec(StrictModel):
    type: Literal["adam", "adamw"]
    lr: float = Field(gt=0)
    betas: tuple[float, float]
    eps: float = Field(gt=0)

    @field_validator("betas")
    @classmethod
    def _betas_in_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        for index, beta in enumerate(v):
            if not 0.0 <= beta < 1.0:
                raise ValueError(
                    f"beta{index + 1}={beta} must lie in [0, 1); at 1.0 the "
                    "moment estimate never updates"
                )
        return v


class GeneratorTrainConfig(StrictModel):
    """Generator optimisation."""

    samples_per_step: int = Field(gt=0)
    optimizer: OptimizerSpec
    #: The Earthformer's learning rate as a fraction of ``optimizer.lr``.
    #:
    #: Bounded at or below 1: the backbone arrives pretrained and the adapter
    #: and sampler arrive random, so the pretrained half must never move faster
    #: than the half that is still finding its footing. A multiplier above 1
    #: would invert the relationship this field exists to establish.
    backbone_lr_multiplier: float = Field(gt=0, le=1)


class DiscriminatorTrainConfig(StrictModel):
    optimizer: OptimizerSpec
    steps_per_generator_step: int = Field(gt=0)


class LossWeightsConfig(StrictModel):
    """Maps onto ``training.losses.LossWeights``."""

    adversarial_spatial: float = Field(ge=0)
    adversarial_temporal: float = Field(ge=0)
    grid_cell: float = Field(ge=0)
    auxiliary: float = Field(ge=0)
    grid_cell_max_weight: float = Field(gt=0)


class ScheduleConfig(StrictModel):
    max_steps: int = Field(gt=0)
    warmup_steps: int = Field(ge=0)
    lr_schedule: Literal["constant", "cosine", "linear"]
    #: None means no clipping. Spectral normalisation already bounds the
    #: critics, and a clip on top would mask an exploding generator.
    #:
    #: Written as ``Annotated[...] | None`` rather than ``float | None`` with a
    #: ``Field(gt=0)``: the constraint belongs to the float member of the
    #: union, and hanging it on the union itself is rejected when the schema is
    #: built -- a failure that would take the whole configuration layer down on
    #: import rather than on a bad value.
    grad_clip_norm: Annotated[float, Field(gt=0)] | None = None
    #: ``None`` hands the decision to ``model.fusion.freeze_backbone`` for the
    #: whole run. An integer means the trainer owns it: the Earthformer is
    #: frozen from step 0 and released at that step. That is the only reading
    #: under which this field does something a static flag could not, and it
    #: keeps the two settings from being competing sources of one truth.
    backbone_unfreeze_step: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _warmup_within_run(self) -> ScheduleConfig:
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps={self.warmup_steps} is not shorter than "
                f"max_steps={self.max_steps}; the run would end during warmup"
            )
        if (
            self.backbone_unfreeze_step is not None
            and self.backbone_unfreeze_step >= self.max_steps
        ):
            raise ValueError(
                f"backbone_unfreeze_step={self.backbone_unfreeze_step} is not "
                f"reached before max_steps={self.max_steps}; the backbone would "
                "stay frozen for the whole run, which is what a null means"
            )
        return self


class ReplayPolicyConfig(StrictModel):
    """Maps onto ``training.replay_buffer.RetentionPolicy``."""

    gate_frames: int = Field(gt=0)
    required_streams: list[str]
    #: Flag *names*, resolved to :class:`~indra.types.QCFlag` by
    #: :meth:`resolved_disqualifying_flags`. Names rather than an integer
    #: because a bitmask in a YAML file is unreadable and unreviewable.
    disqualifying_flags: list[str]
    min_valid_fraction: float = Field(ge=0, le=1)
    min_target_valid_fraction: float = Field(ge=0, le=1)
    precip_channel: str
    min_precip_coverage: float = Field(ge=0, le=1)
    heavy_threshold_mm_h: float = Field(gt=0)
    admission_quantile: float = Field(gt=0, lt=1)
    min_exceedance_area: float = Field(ge=0, le=1)
    score_history_size: int = Field(gt=0)
    min_history: int = Field(gt=0)
    min_separation_minutes: int = Field(ge=0)

    @field_validator("required_streams")
    @classmethod
    def _known_streams(cls, v: list[str]) -> list[str]:
        known = {stream.value for stream in SourceStream}
        unknown = [name for name in v if name not in known]
        if unknown:
            raise ValueError(
                f"unknown streams {unknown} in replay.policy.required_streams; "
                f"known streams are {sorted(known)}"
            )
        if not v:
            raise ValueError(
                "replay.policy.required_streams is empty: the observation gate "
                "would check nothing, and gap-filled frames would be archived "
                "as extreme events"
            )
        return v

    @field_validator("disqualifying_flags")
    @classmethod
    def _known_flags(cls, v: list[str]) -> list[str]:
        for name in v:
            if not hasattr(QCFlag, name):
                raise ValueError(
                    f"unknown QC flag {name!r}; valid names are "
                    f"{sorted(f.name for f in QCFlag if f.name)}"
                )
            if getattr(QCFlag, name).value == 0:
                raise ValueError(
                    f"{name!r} has value 0, so including it in "
                    "disqualifying_flags adds no constraint. Listing it reads "
                    "as a gate that is silently absent."
                )
        if len(set(v)) != len(v):
            raise ValueError(f"disqualifying_flags repeats a flag: {v}")
        return v

    @model_validator(mode="after")
    def _history_reachable(self) -> ReplayPolicyConfig:
        if self.min_history > self.score_history_size:
            raise ValueError(
                f"min_history={self.min_history} exceeds "
                f"score_history_size={self.score_history_size}, so the quantile "
                "estimate can never activate and the cold-start floor would "
                "govern the whole run"
            )
        return self

    def resolved_disqualifying_flags(self) -> QCFlag:
        """The configured names as a single :class:`QCFlag` bitmask."""
        combined = QCFlag.OK
        for name in self.disqualifying_flags:
            combined |= getattr(QCFlag, name)
        return combined


class ReplayConfig(StrictModel):
    enabled: bool
    capacity: int = Field(gt=0)
    manifest: str
    seed: int
    #: Replay samples are *appended* to each fresh batch, so the effective
    #: batch is ``data.batch_size + samples_per_batch``. Defined this way
    #: rather than as a share of the batch because at batch_size 1 a share
    #: would mean every batch is replay and no new data is ever consumed.
    samples_per_batch: int = Field(ge=0)
    #: The buffer must never see a validation window, or held-out events are
    #: archived and replayed into training through a side door.
    admission_only_from_training_split: bool
    policy: ReplayPolicyConfig

    def resolved_manifest_path(self, root: Path = REPO_ROOT) -> Path:
        p = Path(self.manifest)
        return p if p.is_absolute() else root / p

    @model_validator(mode="after")
    def _replay_is_reachable(self) -> ReplayConfig:
        if self.enabled and self.samples_per_batch == 0:
            raise ValueError(
                "replay is enabled but samples_per_batch is 0, so the buffer "
                "would be filled and never drawn from. Disable replay or draw "
                "from it."
            )
        return self


class ValidationConfig(StrictModel):
    every_steps: int = Field(gt=0)
    max_batches: int = Field(gt=0)
    ensemble_members: int = Field(gt=0)
    seed: int
    #: Verification consults ``QCFlag.is_observed``, never ``is_usable``.
    #: Scoring against a gap-filled frame means scoring our own optical-flow
    #: interpolation and reporting it as skill.
    require_observed_targets: bool
    metrics: list[str]

    @field_validator("metrics")
    @classmethod
    def _metrics_present(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("validation.metrics is empty; nothing would be reported")
        if len(set(v)) != len(v):
            raise ValueError(f"validation.metrics repeats a metric: {v}")
        return v

    @model_validator(mode="after")
    def _observed_targets_warned(self) -> ValidationConfig:
        if not self.require_observed_targets:
            raise ValueError(
                "require_observed_targets is false. Validation would score the "
                "model against gap-filled frames, i.e. against this pipeline's "
                "own interpolation, and every reported metric would be "
                "inflated by an unknown amount. If this is genuinely wanted, "
                "it needs a deliberate change here, not a config flag."
            )
        return self


class CheckpointingConfig(StrictModel):
    dir: str
    every_steps: int = Field(gt=0)
    keep_last: int = Field(gt=0)
    #: Resuming an adversarial run without the Adam moments restarts both
    #: players from a standstill against opponents that are not.
    save_optimizer_state: bool
    #: Metric deciding which checkpoint ships. Cross-checked against
    #: ``validation.metrics`` and the inference threshold bands in
    #: :class:`IndraConfig`, because this one string selects the artefact and a
    #: typo in it would silently select the last checkpoint instead of the best.
    monitor: str
    mode: Literal["max", "min"]

    def resolved_dir(self, root: Path = REPO_ROOT) -> Path:
        p = Path(self.dir)
        return p if p.is_absolute() else root / p


class TrainingLoggingConfig(StrictModel):
    every_steps: int = Field(gt=0)
    #: Every ``LossBreakdown`` term separately. Collapsed critics, swamping
    #: regularisation and a stalled generator are three different failures with
    #: three different fixes, and one scalar distinguishes none of them.
    log_loss_components: bool
    replay_stats_every_steps: int = Field(gt=0)


class TrainingConfig(StrictModel):
    run: TrainingRunConfig
    data: TrainingDataConfig
    cache: WindowCacheConfig
    generator: GeneratorTrainConfig
    discriminator: DiscriminatorTrainConfig
    losses: LossWeightsConfig
    schedule: ScheduleConfig
    replay: ReplayConfig
    validation: ValidationConfig
    checkpointing: CheckpointingConfig
    logging: TrainingLoggingConfig

    @model_validator(mode="after")
    def _replay_gate_fits_window(self) -> TrainingConfig:
        # The gate covers the most recent frames of the input window, so it
        # cannot be longer than the window. Checked against the model's
        # sequence length in IndraConfig; here only the self-evident half.
        if self.replay.enabled and self.replay.policy.gate_frames < 1:
            raise ValueError("replay.policy.gate_frames must be at least 1")
        return self


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class IndraConfig(StrictModel):
    """The configuration files, validated together."""

    ingestion: IngestionConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    inference: InferenceConfig
    training: TrainingConfig | None = None

    @model_validator(mode="after")
    def _cross_file_consistency(self) -> IndraConfig:
        ing, pre, mdl, inf = (
            self.ingestion,
            self.preprocessing,
            self.model,
            self.inference,
        )

        # -- sequence length -------------------------------------------------
        lengths = {
            "ingestion.temporal.sequence_length": ing.temporal.sequence_length,
            "preprocessing.tensor.sequence_length": pre.tensor.sequence_length,
            "model.input.sequence_length": mdl.input.sequence_length,
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"sequence length disagrees across files: {lengths}")

        # -- spatial grid ----------------------------------------------------
        grids = {
            "ingestion.domain": (ing.domain.grid_height, ing.domain.grid_width),
            "preprocessing.target_grid": (
                pre.target_grid.height,
                pre.target_grid.width,
            ),
            "model.input": (mdl.input.height, mdl.input.width),
        }
        if len(set(grids.values())) != 1:
            raise ValueError(f"spatial grid disagrees across files: {grids}")

        for name, a, b in (
            ("lat_min", ing.domain.lat_min, pre.target_grid.lat_min),
            ("lat_max", ing.domain.lat_max, pre.target_grid.lat_max),
            ("lon_min", ing.domain.lon_min, pre.target_grid.lon_min),
            ("lon_max", ing.domain.lon_max, pre.target_grid.lon_max),
        ):
            if abs(a - b) > _DEG_EPS:
                raise ValueError(
                    f"domain {name} differs between ingestion ({a}) and "
                    f"preprocessing ({b})"
                )

        # -- channel count ---------------------------------------------------
        if pre.channels.count != mdl.input.channels:
            raise ValueError(
                f"preprocessing defines {pre.channels.count} channels but the "
                f"model expects {mdl.input.channels}"
            )

        # -- NWP channel names must match the declared pressure levels -------
        levels = ing.sources.imdaa.pressure_levels_hpa
        variables = list(ing.sources.imdaa.variables)
        expected_nwp = [f"{var}{lvl}" for var in variables for lvl in levels]
        if pre.channels.nwp.names != expected_nwp:
            raise ValueError(
                "NWP channel names do not match the IMDAA variables and "
                f"pressure levels declared in ingestion.\n"
                f"  expected: {expected_nwp}\n"
                f"  found:    {pre.channels.nwp.names}"
            )

        # -- static layers ---------------------------------------------------
        declared_static = set(ing.sources.static_priors.layers)
        if set(pre.channels.static.names) != declared_static:
            raise ValueError(
                f"static channels {pre.channels.static.names} do not match the "
                f"ingested static layers {sorted(declared_static)}"
            )

        # -- native grids ----------------------------------------------------
        # Each source's declared native shape must span the same domain the
        # target grid does. A shape copied from a different sector order would
        # otherwise read cleanly and reproject into the wrong place.
        for label, src in (
            ("insat", ing.sources.insat),
            ("imdaa", ing.sources.imdaa),
        ):
            d_lat, d_lon = src.implied_resolution_deg(ing.domain)
            if abs(d_lat - d_lon) > 1e-4:
                raise ValueError(
                    f"{label}.native_shape {src.native_shape} implies "
                    f"non-square cells over the domain "
                    f"({d_lat:.6f} deg lat vs {d_lon:.6f} deg lon)"
                )

        # -- lead times ------------------------------------------------------
        if inf.lead_times.frames != mdl.output.lead_frames:
            raise ValueError(
                f"inference expects {inf.lead_times.frames} lead frames but the "
                f"model produces {mdl.output.lead_frames}"
            )
        if inf.lead_times.interval_minutes != mdl.output.lead_interval_minutes:
            raise ValueError("lead interval disagrees between inference and model")

        # -- ensemble --------------------------------------------------------
        if inf.ensemble.members != mdl.dgmr.latent_conditioning_stack.ensemble_members:
            raise ValueError(
                f"inference requests {inf.ensemble.members} ensemble members but "
                f"the latent conditioning stack declares "
                f"{mdl.dgmr.latent_conditioning_stack.ensemble_members}"
            )

        # -- target stream against the model's output contract ---------------
        hem = ing.targets.hem
        if hem.enabled:
            if len(ing.temporal.lead_indices) != mdl.output.lead_frames:
                raise ValueError(
                    f"ingestion declares {len(ing.temporal.lead_indices)} lead "
                    f"offsets but the model produces "
                    f"{mdl.output.lead_frames} frames"
                )
            if hem.native_interval_minutes != mdl.output.lead_interval_minutes:
                raise ValueError(
                    f"the HEM target arrives every {hem.native_interval_minutes} "
                    f"min but the model forecasts every "
                    f"{mdl.output.lead_interval_minutes} min"
                )
            if mdl.output.units != hem.variable.units:
                raise ValueError(
                    f"the model declares its output in {mdl.output.units!r} but "
                    f"the target is {hem.variable.units!r}; the loss would "
                    "compare two different physical quantities"
                )
            # Same square-cell check the input sources get. A target on
            # non-square cells reprojects with a systematic directional
            # stretch, displacing convective cores along one axis only.
            d_lat, d_lon = hem.implied_resolution_deg(ing.domain)
            if abs(d_lat - d_lon) > 1e-4:
                raise ValueError(
                    f"targets.hem.native_shape {hem.native_shape} implies "
                    f"non-square cells over the domain ({d_lat:.6f} deg lat vs "
                    f"{d_lon:.6f} deg lon)"
                )

        # -- climatology against the served record ---------------------------
        # The statistics are computed from data inside the record, so the
        # reference period must sit within it. The reverse containment is NOT
        # required and must not be added: the record deliberately extends one
        # season past the reference period, because the held-out validation
        # year is served but never contributes to the normalisation constants.
        reference = pre.normalization.reference_period
        record_start = inf.run.record_start.date()
        record_end = inf.run.record_end.date()
        if reference.start < record_start or reference.end > record_end:
            raise ValueError(
                f"climatology reference period {reference.start}..{reference.end} "
                f"is not contained in the served record {record_start}..{record_end}; "
                "the statistics would be derived from dates the pipeline never "
                "ingests"
            )
        stray = sorted(set(reference.months) - set(inf.run.record_months))
        if stray:
            raise ValueError(
                f"climatology reference period covers months {stray}, which the "
                f"record months {inf.run.record_months} exclude. Normalising a "
                "monsoon field with statistics that include months outside the "
                "monsoon shifts every channel by an unknown amount."
            )

        # -- training --------------------------------------------------------
        trn = self.training
        if trn is not None:
            # The guard against the leak the reference period was narrowed to
            # avoid. Climatological statistics fitted on data after the
            # training cutoff would normalise the held-out season by constants
            # that had already seen it, and nothing downstream could detect
            # it: the tensors look entirely ordinary.
            cutoff = trn.data.split.train_until.date()
            if reference.end > cutoff:
                raise ValueError(
                    f"climatology reference period ends {reference.end}, after "
                    f"the training cutoff {cutoff}. Statistics fitted on data "
                    "the model never trains on leak the validation "
                    "distribution into every normalised field."
                )

            # Training and serving must agree on dtype. A checkpoint produced
            # under bf16 and loaded into an fp32 forward pass is not the same
            # function, and the difference shows up as a quiet accuracy drop
            # rather than an error.
            if trn.run.precision != mdl.fusion.precision:
                raise ValueError(
                    f"training precision {trn.run.precision!r} disagrees with "
                    f"model.fusion.precision {mdl.fusion.precision!r}"
                )

            # Seasons must be the seasons the record actually holds.
            if trn.data.seasons.months != inf.run.record_months:
                raise ValueError(
                    f"training seasons cover months {trn.data.seasons.months} "
                    f"but the record covers {inf.run.record_months}"
                )
            years = trn.data.seasons.years
            if min(years) < record_start.year or max(years) > record_end.year:
                raise ValueError(
                    f"training seasons {years} reach outside the served record "
                    f"{record_start}..{record_end}"
                )

            # The replay gate reads the most recent frames of the input window
            # and cannot be longer than the window itself.
            if trn.replay.enabled:
                if trn.replay.policy.gate_frames > mdl.input.sequence_length:
                    raise ValueError(
                        f"replay gate_frames={trn.replay.policy.gate_frames} "
                        f"exceeds the {mdl.input.sequence_length}-frame input "
                        "window"
                    )
                if trn.replay.policy.precip_channel not in pre.channels.names:
                    raise ValueError(
                        f"replay precip_channel "
                        f"{trn.replay.policy.precip_channel!r} is not a "
                        f"configured channel; available: {pre.channels.names}"
                    )
                configured_heavy = inf.thresholds.precipitation_mm_h.get("heavy")
                if (
                    configured_heavy is not None
                    and trn.replay.policy.heavy_threshold_mm_h != configured_heavy
                ):
                    raise ValueError(
                        f"replay heavy_threshold_mm_h "
                        f"{trn.replay.policy.heavy_threshold_mm_h} disagrees "
                        f"with the inference heavy threshold "
                        f"{configured_heavy}. The buffer would archive events "
                        "the service would not call heavy."
                    )

            # The monitored metric selects the shipped checkpoint, so a typo
            # in it silently ships the last checkpoint instead of the best.
            monitor = trn.checkpointing.monitor
            metrics = set(trn.validation.metrics)
            bands = set(inf.thresholds.precipitation_mm_h)
            recognised = monitor in metrics or any(
                monitor == f"{metric}_{band}" for metric in metrics for band in bands
            )
            if not recognised:
                raise ValueError(
                    f"checkpointing.monitor {monitor!r} is neither a configured "
                    f"validation metric {sorted(metrics)} nor one qualified by "
                    f"an intensity band {sorted(bands)}"
                )

        # -- XAI attribution resolution --------------------------------------
        # Attribution maps live on the patch-embedded grid, so the requested
        # resolution must be reachable by the stem's downsampling factor.
        stem = mdl.earthformer.patch_embed.patch_size
        if mdl.input.height % inf.xai.attribution_maps.output_resolution:
            raise ValueError(
                f"xai attribution output_resolution "
                f"{inf.xai.attribution_maps.output_resolution} does not divide "
                f"the input height {mdl.input.height} (patch stem {stem})"
            )

        return self

    @property
    def channel_names(self) -> list[str]:
        return self.preprocessing.channels.names


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def load_ingestion_config(path: Path | str | None = None) -> IngestionConfig:
    path = Path(path) if path else CONFIG_ROOT / "data" / "ingestion.yaml"
    return IngestionConfig.model_validate(_read_yaml(path))


def load_preprocessing_config(path: Path | str | None = None) -> PreprocessingConfig:
    path = Path(path) if path else CONFIG_ROOT / "data" / "preprocessing.yaml"
    return PreprocessingConfig.model_validate(_read_yaml(path))


def load_model_config(path: Path | str | None = None) -> ModelConfig:
    path = Path(path) if path else CONFIG_ROOT / "model" / "fusion.yaml"
    return ModelConfig.model_validate(_read_yaml(path))


def load_inference_config(path: Path | str | None = None) -> InferenceConfig:
    path = Path(path) if path else CONFIG_ROOT / "inference" / "nowcast.yaml"
    return InferenceConfig.model_validate(_read_yaml(path))


def load_training_config(path: Path | str | None = None) -> TrainingConfig:
    path = Path(path) if path else CONFIG_ROOT / "train" / "training.yaml"
    return TrainingConfig.model_validate(_read_yaml(path))


def load_config(config_root: Path | str | None = None) -> IndraConfig:
    """Load and cross-validate the complete configuration."""
    root = Path(config_root) if config_root else CONFIG_ROOT
    training_path = root / "train" / "training.yaml"
    return IndraConfig(
        ingestion=load_ingestion_config(root / "data" / "ingestion.yaml"),
        preprocessing=load_preprocessing_config(root / "data" / "preprocessing.yaml"),
        model=load_model_config(root / "model" / "fusion.yaml"),
        inference=load_inference_config(root / "inference" / "nowcast.yaml"),
        # Optional by design: a serving container ships no training
        # configuration, and its absence must not stop the API from starting.
        # When it is present it is validated against everything else.
        training=(
            load_training_config(training_path) if training_path.exists() else None
        ),
    )


__all__ = [
    "CONFIG_ROOT",
    "REPO_ROOT",
    "FieldKind",
    "GapFillStrategy",
    "NormalizationMethod",
    "Resampling",
    "IngestionConfig",
    "PreprocessingConfig",
    "ModelConfig",
    "InferenceConfig",
    "TrainingConfig",
    "IndraConfig",
    "load_config",
    "load_ingestion_config",
    "load_preprocessing_config",
    "load_model_config",
    "load_inference_config",
    "load_training_config",
]
