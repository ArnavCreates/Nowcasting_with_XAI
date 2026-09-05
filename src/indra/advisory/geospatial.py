"""Turns exceedance fields into named districts with onset and expiry times."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..config import GeospatialConfig, TargetGrid, ThresholdConfig

logger = logging.getLogger(__name__)

#: WGS84 mean radius, kilometres. The spherical approximation costs about
#: 0.3% against the ellipsoid -- far below the resolution of a 9 km grid, and
#: it cancels entirely from ``affected_fraction``.
_EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------------
# The model grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Geometry of the 384 x 384 target grid, in cell centres."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float
    height: int
    width: int

    @classmethod
    def from_config(cls, grid: TargetGrid) -> Grid:
        return cls(
            lat_min=grid.lat_min,
            lat_max=grid.lat_max,
            lon_min=grid.lon_min,
            lon_max=grid.lon_max,
            resolution_deg=grid.resolution_deg,
            height=grid.height,
            width=grid.width,
        )

    @classmethod
    def from_mapping(cls, grid: dict[str, float]) -> Grid:
        """Build from ``AssembledWindow.grid`` or ``TargetWindow.grid``."""
        return cls(
            lat_min=float(grid["lat_min"]),
            lat_max=float(grid["lat_max"]),
            lon_min=float(grid["lon_min"]),
            lon_max=float(grid["lon_max"]),
            resolution_deg=float(grid["resolution_deg"]),
            height=int(grid["height"]),
            width=int(grid["width"]),
        )

    # ------------------------------------------------------------- geometry
    @property
    def outer_bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` at the outer cell edges."""
        half = self.resolution_deg / 2.0
        return (
            self.lon_min - half,
            self.lat_min - half,
            self.lon_max + half,
            self.lat_max + half,
        )

    def contains(self, lat: float, lon: float) -> bool:
        west, south, east, north = self.outer_bounds
        return south <= lat <= north and west <= lon <= east

    def latlon_to_indices(self, lat: float, lon: float) -> tuple[int, int]:
        """Nearest cell to a geographic point, as ``(row, col)``."""
        if not self.contains(lat, lon):
            west, south, east, north = self.outer_bounds
            raise ValueError(
                f"({lat:.4f}, {lon:.4f}) lies outside the domain "
                f"[{south:.4f}, {north:.4f}] x [{west:.4f}, {east:.4f}]"
            )
        row = int(round((lat - self.lat_min) / self.resolution_deg))
        col = int(round((lon - self.lon_min) / self.resolution_deg))
        return (
            min(max(row, 0), self.height - 1),
            min(max(col, 0), self.width - 1),
        )

    def indices_to_latlon(self, row: int, col: int) -> tuple[float, float]:
        """Centre coordinates of a cell."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(
                f"({row}, {col}) is outside the {self.height} x {self.width} grid"
            )
        return (
            self.lat_min + row * self.resolution_deg,
            self.lon_min + col * self.resolution_deg,
        )

    def row_areas_km2(self) -> npt.NDArray[np.float64]:
        """Area of one cell in each row, ``(H,)``."""
        half = self.resolution_deg / 2.0
        centres = self.lat_min + np.arange(self.height) * self.resolution_deg
        south = np.radians(centres - half)
        north = np.radians(centres + half)
        d_lon = math.radians(self.resolution_deg)
        return (_EARTH_RADIUS_KM**2) * d_lon * (np.sin(north) - np.sin(south))

    def cell_areas_km2(self) -> npt.NDArray[np.float64]:
        """``(H, W)`` cell areas, broadcast from :meth:`row_areas_km2`."""
        return np.repeat(self.row_areas_km2()[:, None], self.width, axis=1)


# ---------------------------------------------------------------------------
# District membership
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistrictGrid:
    """District membership of every grid cell, in compressed form."""

    grid: Grid
    #: ``(state, district)`` pairs, in the order ``offsets`` indexes.
    keys: tuple[tuple[str, str], ...]
    cells: npt.NDArray[np.int32]
    offsets: npt.NDArray[np.int64]
    #: Total area of each district, summed over its own cells so it shares a
    #: denominator with every numerator computed against it.
    areas_km2: npt.NDArray[np.float64]
    #: Districts that rasterised to no cells at all.
    unresolved: tuple[tuple[str, str], ...] = ()

    def __len__(self) -> int:
        return len(self.keys)

    def cells_of(self, index: int) -> npt.NDArray[np.int32]:
        return self.cells[self.offsets[index] : self.offsets[index + 1]]

    def districts_at(self, row: int, col: int) -> tuple[tuple[str, str], ...]:
        """Every district covering one cell, for the API's point query."""
        flat = row * self.grid.width + col
        return tuple(
            key
            for index, key in enumerate(self.keys)
            if bool((self.cells_of(index) == flat).any())
        )

    def summary(self) -> dict[str, Any]:
        return {
            "districts": len(self.keys),
            "unresolved": len(self.unresolved),
            "mean_cells_per_district": (
                round(len(self.cells) / max(len(self.keys), 1), 2)
            ),
        }


def build_district_grid(config: GeospatialConfig, grid: Grid) -> DistrictGrid:
    """Rasterise the district boundaries onto the model grid, once."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    path = config.resolved_boundaries()
    if not path.is_file():
        raise FileNotFoundError(
            f"administrative boundaries not found at {path}. Fetch them with "
            "'bash scripts/fetch_data.sh', or mount the volume that holds "
            "them."
        )

    frame = gpd.read_file(path, layer=config.admin_layer)
    if frame.crs is None:
        logger.warning(
            "boundary layer declares no CRS; assuming %s as configured",
            config.admin_crs,
        )
        frame = frame.set_crs(config.admin_crs)
    elif frame.crs.to_string().upper() != config.admin_crs.upper():
        frame = frame.to_crs(config.admin_crs)

    missing = [column for column in config.dissolve_by if column not in frame.columns]
    if missing:
        raise KeyError(
            f"boundary layer has no {missing} column(s); dissolve_by names "
            f"{config.dissolve_by} and the available columns are "
            f"{sorted(frame.columns)}"
        )

    # Identity is the full dissolve key, never the name alone: India has an
    # Aurangabad in Maharashtra and another in Bihar, and merging them would
    # warn one state about the other's weather.
    dissolved = frame.dissolve(by=list(config.dissolve_by), as_index=True)

    west, south, east, north = grid.outer_bounds
    transform = from_origin(west, north, grid.resolution_deg, grid.resolution_deg)
    all_touched = config.join_predicate == "intersects"

    areas = grid.cell_areas_km2()
    keys: list[tuple[str, str]] = []
    cells: list[np.ndarray] = []
    offsets: list[int] = [0]
    district_areas: list[float] = []
    unresolved: list[tuple[str, str]] = []

    for key, geometry in zip(dissolved.index, dissolved.geometry, strict=False):
        label = tuple(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        if geometry is None or geometry.is_empty:
            unresolved.append(label)  # type: ignore[arg-type]
            continue

        burned = rasterize(
            [(geometry, 1)],
            out_shape=(grid.height, grid.width),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        )
        # rasterize writes north-up; the model grid ascends in latitude with
        # row index. Without this flip every district lands mirrored across
        # the domain's centre, and nothing downstream would notice.
        burned = np.flipud(burned)

        flat = np.flatnonzero(burned.reshape(-1)).astype(np.int32)
        if flat.size == 0:
            unresolved.append(label)  # type: ignore[arg-type]
            continue

        keys.append(label)  # type: ignore[arg-type]
        cells.append(flat)
        offsets.append(offsets[-1] + int(flat.size))
        district_areas.append(float(areas.reshape(-1)[flat].sum()))

    if unresolved:
        # An honest limitation rather than a bug. A district smaller than one
        # cell cannot be resolved on a 9 km grid, and reporting nothing for it
        # is preferable to attributing a neighbour's rain to it.
        logger.warning(
            "%d district(s) rasterised to no cells and cannot be reported at "
            "this resolution: %s",
            len(unresolved),
            ", ".join("/".join(k) for k in unresolved[:5]),
        )

    result = DistrictGrid(
        grid=grid,
        keys=tuple(keys),
        cells=(np.concatenate(cells) if cells else np.zeros(0, dtype=np.int32)),
        offsets=np.asarray(offsets, dtype=np.int64),
        areas_km2=np.asarray(district_areas, dtype=np.float64),
        unresolved=tuple(unresolved),
    )
    logger.info("district grid built: %s", result.summary())
    return result


# ---------------------------------------------------------------------------
# Impacts
# ---------------------------------------------------------------------------


def district_mean(
    field: npt.NDArray[np.floating], districts: DistrictGrid
) -> npt.NDArray[np.float64]:
    """Area-weighted mean of a ``(H, W)`` field within each district."""
    _check_field(field, districts)
    flat = np.asarray(field, dtype=np.float64).reshape(-1)
    areas = districts.grid.cell_areas_km2().reshape(-1)

    out = np.zeros(len(districts), dtype=np.float64)
    for index in range(len(districts)):
        cells = districts.cells_of(index)
        weights = areas[cells]
        total = float(weights.sum())
        out[index] = float((flat[cells] * weights).sum() / total) if total else np.nan
    return out


def district_class_fraction(
    field: npt.NDArray[np.floating],
    districts: DistrictGrid,
    classes: Sequence[int],
) -> npt.NDArray[np.float64]:
    """Area share of each district whose categorical value is in ``classes``."""
    _check_field(field, districts)
    if not classes:
        return np.zeros(len(districts), dtype=np.float64)

    selected = np.isin(np.asarray(field), np.asarray(list(classes))).reshape(-1)
    areas = districts.grid.cell_areas_km2().reshape(-1)

    out = np.zeros(len(districts), dtype=np.float64)
    for index in range(len(districts)):
        cells = districts.cells_of(index)
        weights = areas[cells]
        total = float(weights.sum())
        out[index] = float(weights[selected[cells]].sum() / total) if total else 0.0
    return out


def _check_field(field: npt.NDArray[np.floating], districts: DistrictGrid) -> None:
    expected = (districts.grid.height, districts.grid.width)
    if tuple(field.shape) != expected:
        raise ValueError(
            f"field is {tuple(field.shape)} but the district grid is {expected}"
        )


@dataclass(frozen=True)
class DistrictImpact:
    """What one district is forecast to experience, and when."""

    state: str
    district: str
    #: Highest exceedance probability anywhere in the district, over the whole
    #: horizon.
    peak_probability: float
    peak_lead_index: int
    #: First and last lead frames at which the district qualifies. Inclusive,
    #: so the advisory's effective window is onset through expiry.
    onset_lead_index: int
    expiry_lead_index: int
    #: Greatest affected extent at any single lead frame, not the union over
    #: the horizon: a storm crossing a district would otherwise report an
    #: affected area larger than anything present at one time.
    affected_area_km2: float
    affected_fraction: float
    district_area_km2: float
    severity: str | None = None
    peak_intensity_mm_h: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "district": self.district,
            "peak_probability": round(self.peak_probability, 4),
            "peak_lead_index": self.peak_lead_index,
            "onset_lead_index": self.onset_lead_index,
            "expiry_lead_index": self.expiry_lead_index,
            "affected_area_km2": round(self.affected_area_km2, 1),
            "affected_fraction": round(self.affected_fraction, 4),
            "district_area_km2": round(self.district_area_km2, 1),
            "severity": self.severity,
            "peak_intensity_mm_h": (
                None
                if self.peak_intensity_mm_h is None
                else round(self.peak_intensity_mm_h, 2)
            ),
        }


def _severity(
    probability: float, intensity: float | None, thresholds: ThresholdConfig
) -> str | None:
    """Highest severity band the district satisfies, or ``None``."""
    if intensity is None:
        return None
    satisfied = [
        (band.min_probability, name)
        for name, band in thresholds.severity_bands.items()
        if probability >= band.min_probability and intensity >= band.min_intensity_mm_h
    ]
    if not satisfied:
        return None
    # The most demanding band the district clears, by probability -- which
    # ThresholdConfig has already validated as ascending with severity.
    return max(satisfied)[1]


def extract_impacts(
    probability: npt.NDArray[np.float32],
    districts: DistrictGrid,
    geospatial: GeospatialConfig,
    thresholds: ThresholdConfig,
    intensity_mm_h: npt.NDArray[np.float32] | None = None,
) -> list[DistrictImpact]:
    """Districts affected by an exceedance field, ranked by peak probability."""
    if probability.ndim != 3:
        raise ValueError(
            f"probability must be (T, H, W); got {tuple(probability.shape)}"
        )
    expected = (districts.grid.height, districts.grid.width)
    if probability.shape[1:] != expected:
        raise ValueError(
            f"probability grid {probability.shape[1:]} does not match the "
            f"district grid {expected}"
        )
    if intensity_mm_h is not None and intensity_mm_h.shape != probability.shape:
        raise ValueError(
            f"intensity {tuple(intensity_mm_h.shape)} does not match "
            f"probability {tuple(probability.shape)}"
        )

    n_lead = probability.shape[0]
    flat_probability = probability.reshape(n_lead, -1)
    flat_intensity = (
        None if intensity_mm_h is None else intensity_mm_h.reshape(n_lead, -1)
    )
    cell_area = districts.grid.cell_areas_km2().reshape(-1)

    # Below this, a cell is not an exceedance at all. Suppresses the
    # single-member speckle that would otherwise paint the map in 12% contours.
    floor = thresholds.min_reported_probability

    impacts: list[DistrictImpact] = []
    for index, (state, district) in enumerate(districts.keys):
        cells = districts.cells_of(index)
        if cells.size == 0:
            continue

        total_area = float(districts.areas_km2[index])
        values = flat_probability[:, cells]  # (T, n_cells)
        affected = values >= floor
        areas = (affected * cell_area[cells]).sum(axis=1)  # (T,)
        fractions = areas / max(total_area, 1e-9)

        qualifies = (areas >= geospatial.min_affected_area_km2) | (
            fractions >= geospatial.min_fractional_coverage
        )
        if not bool(qualifies.any()):
            continue

        qualifying = np.flatnonzero(qualifies)
        peak_lead = int(np.argmax(values.max(axis=1)))
        peak_probability = float(values[peak_lead].max())
        # The greatest extent at any one lead frame. A union over the horizon
        # would report a moving storm as covering more of the district than it
        # ever does at once.
        widest = int(np.argmax(areas))

        peak_intensity = None
        if flat_intensity is not None:
            peak_intensity = float(flat_intensity[:, cells].max())

        impacts.append(
            DistrictImpact(
                state=state,
                district=district,
                peak_probability=peak_probability,
                peak_lead_index=peak_lead,
                onset_lead_index=int(qualifying[0]),
                expiry_lead_index=int(qualifying[-1]),
                affected_area_km2=float(areas[widest]),
                affected_fraction=float(fractions[widest]),
                district_area_km2=total_area,
                severity=_severity(peak_probability, peak_intensity, thresholds),
                peak_intensity_mm_h=peak_intensity,
            )
        )

    impacts.sort(key=lambda impact: impact.peak_probability, reverse=True)
    logger.info(
        "%d of %d districts qualify (gates: %.0f km2 or %.0f%% coverage)",
        len(impacts),
        len(districts.keys),
        geospatial.min_affected_area_km2,
        100 * geospatial.min_fractional_coverage,
    )
    return impacts


__all__ = [
    "DistrictGrid",
    "DistrictImpact",
    "Grid",
    "build_district_grid",
    "district_class_fraction",
    "district_mean",
    "extract_impacts",
]
