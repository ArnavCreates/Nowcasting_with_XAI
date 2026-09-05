"""Grid geometry and cell-area arithmetic.

Needs no boundary layer: Grid is deliberately free of geopandas and rasterio
so the API's point query does not pull in a GIS stack, and that makes it
testable on its own.
"""

from __future__ import annotations

import math

import pytest

from indra.advisory.geospatial import Grid

# The configured domain: 384 x 384 at 1/12 degree over India.
GRID = Grid(
    lat_min=6.0,
    lat_max=38.0,
    lon_min=68.0,
    lon_max=100.0,
    resolution_deg=1.0 / 12.0,
    height=384,
    width=384,
)


class TestIndexing:
    def test_south_west_corner_is_row_zero(self):
        # Row 0 is the southernmost row: the model grid ascends in latitude
        # with row index, opposite to a north-up raster.
        assert GRID.latlon_to_indices(6.0, 68.0) == (0, 0)

    def test_north_east_corner_is_the_last_cell(self):
        assert GRID.latlon_to_indices(38.0, 100.0) == (383, 383)

    def test_latitude_increases_with_row(self):
        south, _ = GRID.latlon_to_indices(10.0, 77.0)
        north, _ = GRID.latlon_to_indices(30.0, 77.0)
        assert north > south

    def test_round_trip_returns_the_cell_centre(self):
        row, col = GRID.latlon_to_indices(19.0760, 72.8777)  # Mumbai
        lat, lon = GRID.indices_to_latlon(row, col)
        assert abs(lat - 19.0760) <= GRID.resolution_deg
        assert abs(lon - 72.8777) <= GRID.resolution_deg

    def test_out_of_domain_raises_rather_than_clamping(self):
        # Clamping would answer a click over the Arabian Sea with the nearest
        # coastal cell's forecast, presented as the place clicked.
        with pytest.raises(ValueError, match="outside the domain"):
            GRID.latlon_to_indices(45.0, 77.0)
        with pytest.raises(ValueError, match="outside the domain"):
            GRID.latlon_to_indices(20.0, 120.0)

    def test_half_cell_beyond_the_centre_is_still_inside(self):
        # Bounds are outer edges, so the domain extends half a cell past the
        # extreme centre coordinates.
        assert GRID.contains(6.0 - GRID.resolution_deg / 2.0, 68.0)
        assert not GRID.contains(6.0 - GRID.resolution_deg, 68.0)

    def test_indices_outside_the_grid_raise(self):
        with pytest.raises(IndexError):
            GRID.indices_to_latlon(384, 0)


class TestBounds:
    def test_outer_bounds_expand_by_half_a_cell(self):
        west, south, east, north = GRID.outer_bounds
        half = GRID.resolution_deg / 2.0
        assert west == pytest.approx(68.0 - half)
        assert south == pytest.approx(6.0 - half)
        assert east == pytest.approx(100.0 + half)
        assert north == pytest.approx(38.0 + half)


class TestCellArea:
    def test_area_shrinks_toward_the_pole(self):
        # Cells narrow by cos(lat), so the northern edge of the domain has
        # smaller cells than the southern.
        areas = GRID.row_areas_km2()
        assert areas[0] > areas[-1]

    def test_areas_match_the_spherical_formula(self):
        areas = GRID.row_areas_km2()
        radius = 6371.0088
        half = GRID.resolution_deg / 2.0
        for row in (0, 200, 383):
            centre = GRID.lat_min + row * GRID.resolution_deg
            expected = (
                radius**2
                * math.radians(GRID.resolution_deg)
                * (
                    math.sin(math.radians(centre + half))
                    - math.sin(math.radians(centre - half))
                )
            )
            assert areas[row] == pytest.approx(expected, rel=1e-12)

    def test_cell_area_spans_67_to_86_km2(self):
        # What the qualification gates are calibrated against: 150 km2 is
        # roughly two cells, and the 25 km2 floor this replaced was a third of
        # one, so it suppressed nothing.
        areas = GRID.row_areas_km2()
        assert 67.0 < areas.min() < 68.0
        assert 85.0 < areas.max() < 86.0

    @pytest.mark.parametrize(
        ("latitude", "expected_km2"),
        [(8, 85.0), (20, 80.7), (30, 74.4)],
    )
    def test_documented_areas_are_accurate(self, latitude, expected_km2):
        # These figures appear in the geospatial docstring and in
        # nowcast.yaml's threshold comments. If the area formula changes, the
        # comments become wrong silently.
        row = round((latitude - GRID.lat_min) / GRID.resolution_deg)
        assert GRID.row_areas_km2()[row] == pytest.approx(expected_km2, abs=0.1)

    def test_broadcast_matches_the_row_vector(self):
        rows = GRID.row_areas_km2()
        full = GRID.cell_areas_km2()
        assert full.shape == (GRID.height, GRID.width)
        assert full[:, 0] == pytest.approx(rows)
        assert full[100, 0] == pytest.approx(full[100, 383])

    def test_total_area_is_plausible_for_the_domain(self):
        # 32 x 32 degrees centred on the subcontinent. A spherical band that
        # size is a few million square kilometres; this catches an area
        # formula wrong by orders of magnitude.
        total = GRID.cell_areas_km2().sum()
        assert 9.0e6 < total < 1.3e7
