"""Geostationary parallax correction."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from ...config import ParallaxCorrection
from ...types import ATTR_QC_FLAG, ATTR_QC_FLAG_NAMES, GeoArray, QCFlag

logger = logging.getLogger(__name__)

# WGS84 ellipsoid.
_WGS84_A = 6_378_137.0  # semi-major axis, m
_WGS84_F = 1.0 / 298.257223563  # flattening
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)  # semi-minor axis, m
_E2 = 1.0 - (_WGS84_B**2) / (_WGS84_A**2)  # first eccentricity squared

#: Physical ceiling on retrieved cloud height. Tops above the tropical
#: tropopause are a retrieval artefact, and an unbounded height sends the
#: correction arbitrarily far across the domain.
_MAX_CLOUD_HEIGHT_M = 20_000.0


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def geodetic_to_ecef(
    lat_deg: GeoArray, lon_deg: GeoArray, height_m: GeoArray | float = 0.0
) -> tuple[GeoArray, GeoArray, GeoArray]:
    """Geodetic latitude/longitude/height to ECEF Cartesian metres."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    # Radius of curvature in the prime vertical.
    n = _WGS84_A / np.sqrt(1.0 - _E2 * sin_lat**2)

    x = (n + height_m) * cos_lat * np.cos(lon)
    y = (n + height_m) * cos_lat * np.sin(lon)
    z = (n * (1.0 - _E2) + height_m) * sin_lat
    return x, y, z


def ecef_to_geodetic(
    x: GeoArray, y: GeoArray, z: GeoArray, iterations: int = 5
) -> tuple[GeoArray, GeoArray, GeoArray]:
    """ECEF Cartesian to geodetic latitude/longitude/height."""
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)

    # Initial guess: geocentric latitude.
    lat = np.arctan2(z, p * (1.0 - _E2))
    n = np.full_like(lat, _WGS84_A)

    for _ in range(iterations):
        sin_lat = np.sin(lat)
        n = _WGS84_A / np.sqrt(1.0 - _E2 * sin_lat**2)
        height = p / np.maximum(np.cos(lat), 1e-12) - n
        lat = np.arctan2(z, p * (1.0 - _E2 * n / (n + height)))

    sin_lat = np.sin(lat)
    n = _WGS84_A / np.sqrt(1.0 - _E2 * sin_lat**2)
    height = p / np.maximum(np.cos(lat), 1e-12) - n
    return np.degrees(lat), np.degrees(lon), height


# ---------------------------------------------------------------------------
# Cloud height
# ---------------------------------------------------------------------------


def cloud_height_from_ctt(
    ctt_k: np.ndarray, lapse_rate_k_per_km: float, surface_temp_k: float
) -> np.ndarray:
    """Cloud-top height in metres from cloud-top temperature."""
    if lapse_rate_k_per_km <= 0:
        raise ValueError("lapse rate must be positive")

    height_km = (
        surface_temp_k - np.asarray(ctt_k, dtype=np.float64)
    ) / lapse_rate_k_per_km
    height_m = height_km * 1000.0
    return np.clip(height_m, 0.0, _MAX_CLOUD_HEIGHT_M)


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------


def correct_coordinates(
    lat_deg: GeoArray,
    lon_deg: GeoArray,
    cloud_height_m: np.ndarray,
    subsatellite_lon_deg: float,
    satellite_altitude_km: float,
) -> tuple[GeoArray, GeoArray]:
    """Shift apparent cloud-top positions to their true ground positions."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    h = np.asarray(cloud_height_m, dtype=np.float64)

    usable = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(h) & (h > 0.0)
    if not usable.any():
        return lat.copy(), lon.copy()

    # Satellite in ECEF: equatorial, at the sub-satellite longitude.
    sat_radius = _WGS84_A + satellite_altitude_km * 1000.0
    sat_lon = np.radians(subsatellite_lon_deg)
    sx = sat_radius * np.cos(sat_lon)
    sy = sat_radius * np.sin(sat_lon)
    sz = 0.0

    # Apparent surface point.
    px, py, pz = geodetic_to_ecef(lat, lon, 0.0)

    # Ray from satellite toward that point.
    dx, dy, dz = px - sx, py - sy, pz - sz

    # Find t where the ray reaches ellipsoidal height h. Solving the exact
    # ellipsoid intersection needs a scaled quadratic; scaling the z axis by
    # a/b turns the ellipsoid into a sphere of radius (a + h) so the standard
    # quadratic applies, and the result is exact rather than a spherical
    # approximation.
    scale = _WGS84_A / _WGS84_B
    ox, oy, oz = sx, sy, sz * scale
    ux, uy, uz = dx, dy, dz * scale
    radius = _WGS84_A + h

    a_coef = ux * ux + uy * uy + uz * uz
    b_coef = 2.0 * (ox * ux + oy * uy + oz * uz)
    c_coef = ox * ox + oy * oy + oz * oz - radius * radius

    disc = b_coef * b_coef - 4.0 * a_coef * c_coef
    hit = usable & (disc >= 0.0) & (a_coef > 0.0)

    lat_out = lat.copy()
    lon_out = lon.copy()
    if not hit.any():
        return lat_out, lon_out

    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[hit] = np.sqrt(disc[hit])
    # Near root: the first intersection along the line of sight, which is the
    # cloud top facing the satellite. The far root is the far side of the
    # Earth and is never what the imager saw.
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (-b_coef - sqrt_disc) / (2.0 * a_coef)

    valid = hit & np.isfinite(t) & (t > 0.0)
    if not valid.any():
        return lat_out, lon_out

    # Cloud-top position in true (unscaled) ECEF.
    cx = sx + t * dx
    cy = sy + t * dy
    cz = sz + t * dz

    # Drop to the surface: converting to geodetic and discarding the height
    # follows the ellipsoid normal, which is the correct nadir direction on an
    # ellipsoid. Rescaling the radius instead would follow the geocentric
    # direction and introduce an error growing with latitude.
    corrected_lat, corrected_lon, _ = ecef_to_geodetic(cx, cy, cz)

    lat_out[valid] = corrected_lat[valid]
    lon_out[valid] = ((corrected_lon[valid] + 180.0) % 360.0) - 180.0
    return lat_out, lon_out


def apply(
    dataset: xr.Dataset,
    config: ParallaxCorrection,
    ctt_variable: str = "insat_ctt",
) -> xr.Dataset:
    """Correct the dataset's geolocation for parallax."""
    if not config.enabled:
        return dataset

    out = dataset.copy(deep=True)
    flags = QCFlag(int(out.attrs.get(ATTR_QC_FLAG, 0)))

    if "latitude" not in out.coords or "longitude" not in out.coords:
        logger.warning(
            "cannot correct parallax without geolocation; leaving "
            "PARALLAX_UNCORRECTED set"
        )
        return out

    if config.cloud_height_source == "ctt_lapse_rate":
        if ctt_variable not in out.data_vars:
            logger.warning(
                "cloud top temperature variable %r absent; parallax not " "corrected",
                ctt_variable,
            )
            return out
        ctt = np.asarray(out[ctt_variable].values, dtype=np.float64)
        height = cloud_height_from_ctt(
            ctt, config.assumed_lapse_rate_k_per_km, config.reference_surface_temp_k
        )
    else:
        source = config.cloud_height_source
        if source not in out.data_vars:
            logger.warning(
                "cloud height product %r absent; parallax not corrected", source
            )
            return out
        height = np.asarray(out[source].values, dtype=np.float64)

    lat = np.asarray(out["latitude"].values, dtype=np.float64)
    lon = np.asarray(out["longitude"].values, dtype=np.float64)
    if height.shape != lat.shape:
        logger.error(
            "cloud height shape %s does not match geolocation %s; parallax "
            "not corrected",
            height.shape,
            lat.shape,
        )
        return out

    sub_lon = float(out.attrs.get("subsatellite_lon", 0.0))
    altitude = float(out.attrs.get("satellite_altitude_km", 0.0))
    if altitude <= 0:
        logger.error(
            "satellite altitude missing from dataset attributes; parallax "
            "not corrected"
        )
        return out

    new_lat, new_lon = correct_coordinates(lat, lon, height, sub_lon, altitude)

    moved = np.isfinite(new_lat) & np.isfinite(lat)
    if moved.any():
        # Reported in kilometres because degrees understate the displacement
        # near the top of the domain, where a degree of longitude is short.
        dlat_km = (new_lat[moved] - lat[moved]) * 111.32
        dlon_km = (
            (new_lon[moved] - lon[moved]) * 111.32 * np.cos(np.radians(lat[moved]))
        )
        shift_km = np.hypot(dlat_km, dlon_km)
        logger.info(
            "parallax: %d cells shifted, mean %.2f km, max %.2f km",
            int(np.count_nonzero(shift_km > 1e-6)),
            float(np.mean(shift_km)),
            float(np.max(shift_km)),
        )
        max_shift = float(np.max(shift_km))
    else:
        max_shift = 0.0

    out = out.assign_coords(
        latitude=(out["latitude"].dims, new_lat, dict(out["latitude"].attrs)),
        longitude=(out["longitude"].dims, new_lon, dict(out["longitude"].attrs)),
    )

    # The grid is no longer the sensor's regular sweep. Recorded so
    # reprojection treats it as scattered points rather than assuming
    # structure it no longer has.
    out.attrs["grid"] = "native_curvilinear_parallax_corrected"
    out.attrs["parallax_corrected"] = 1
    out.attrs["parallax_method"] = config.method
    out.attrs["parallax_cloud_height_source"] = config.cloud_height_source
    out.attrs["parallax_lapse_rate_k_per_km"] = config.assumed_lapse_rate_k_per_km
    out.attrs["parallax_max_shift_km"] = round(max_shift, 3)

    flags &= ~QCFlag.PARALLAX_UNCORRECTED
    out.attrs[ATTR_QC_FLAG] = int(flags)
    out.attrs[ATTR_QC_FLAG_NAMES] = ",".join(flags.describe())
    return out


__all__ = [
    "apply",
    "cloud_height_from_ctt",
    "correct_coordinates",
    "ecef_to_geodetic",
    "geodetic_to_ecef",
]
