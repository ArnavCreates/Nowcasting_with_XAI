"""Configuration schema and the cross-file validators.

Loads the real YAML in configs/, so this is also a regression test on those
files: an edit that breaks the contract between them fails here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from indra.config import (
    CAPConfig,
    GeospatialConfig,
    HazardTaxonomyConfig,
    IntegratedGradientsConfig,
    LocalAdvisoryConfig,
    LocalEmbeddingConfig,
    ScheduleConfig,
    load_config,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


class TestRealConfigLoads:
    def test_all_five_files_load_and_cross_validate(self, config):
        assert config.ingestion and config.preprocessing
        assert config.model and config.inference
        assert config.training is not None

    def test_locked_grid(self, config):
        grid = config.preprocessing.target_grid
        assert (grid.height, grid.width) == (384, 384)
        assert grid.lat_min == 6.0 and grid.lat_max == 38.0
        assert grid.lon_min == 68.0 and grid.lon_max == 100.0

    def test_locked_sequence_and_horizon(self, config):
        assert config.model.input.sequence_length == 13
        assert config.model.input.channels == 30
        assert config.model.output.lead_frames == 12
        assert config.model.output.lead_interval_minutes == 30
        assert config.model.output.horizon_hours == 6

    def test_channel_order_is_the_documented_one(self, config):
        names = config.preprocessing.channels.names
        assert len(names) == 30
        assert names[:3] == ["insat_tir1", "insat_wv", "insat_ctt"]
        assert names[23] == "imd_precip"
        assert names[-3:] == ["dem_elevation", "lulc_mask", "soil_type"]

    def test_output_units_match_the_target(self, config):
        # The loss compares them directly, so a mismatch would compare two
        # different physical quantities.
        assert config.model.output.units == "mm h-1"
        assert config.ingestion.targets.hem.variable.units == "mm h-1"

    def test_lead_indices_match_lead_frames(self, config):
        assert len(config.ingestion.temporal.lead_indices) == (
            config.model.output.lead_frames
        )
        assert config.ingestion.temporal.lead_indices[0] == 1

    def test_climatology_stops_at_the_training_cutoff(self, config):
        # The leak guard: statistics fitted past the cutoff would carry the
        # held-out season into every normalised field.
        reference_end = config.preprocessing.normalization.reference_period.end
        cutoff = config.training.data.split.train_until.date()
        assert reference_end <= cutoff

    def test_training_and_model_precision_agree(self, config):
        assert config.training.run.precision == config.model.fusion.precision

    def test_replay_threshold_matches_the_inference_threshold(self, config):
        assert (
            config.training.replay.policy.heavy_threshold_mm_h
            == (config.inference.thresholds.precipitation_mm_h["heavy"])
        )

    def test_attribution_explains_no_more_members_than_are_served(self, config):
        assert config.inference.xai.integrated_gradients.members <= (
            config.inference.ensemble.members
        )


class TestValidators:
    def test_zeros_baseline_is_refused(self):
        # Normalised zero is an ordinary atmospheric state, and on the min-max
        # channels it is the bottom of the physical range.
        with pytest.raises(ValidationError, match="meaningless"):
            IntegratedGradientsConfig(
                n_steps=32,
                baseline="zeros",
                internal_batch_size=4,
                target="exceedance_probability",
                members=4,
                surrogate_temperature_mm_h=1.0,
            )

    def test_embedding_prefixes_must_differ(self):
        with pytest.raises(ValidationError, match="must differ"):
            LocalEmbeddingConfig(
                model_id="intfloat/e5-small-v2",
                dimensions=384,
                query_prefix="passage: ",
                document_prefix="passage: ",
            )

    def test_generation_must_fit_the_context_window(self):
        with pytest.raises(ValidationError, match="no room for the prompt"):
            LocalAdvisoryConfig(
                model_id_or_path="models/advisory/x",
                device="cpu",
                fallback_device="cpu",
                context_window=1024,
                max_tokens=2048,
                temperature=0.1,
            )

    def test_ungrounded_advisories_are_refused(self):
        # An advisory not grounded in retrieved text is model-authored safety
        # guidance.
        with pytest.raises(ValidationError, match="grounded"):
            CAPConfig(
                version="1.2",
                status="Actual",
                msg_type="Alert",
                scope="Public",
                sender="indra-nowcast",
                language="en-IN",
                categories=["Met"],
                urgency_by_lead_hours={"immediate": 2, "expected": 6},
                constrain_to_retrieved=False,
                require_citations=True,
            )

    def test_both_qualification_gates_cannot_be_zero(self):
        with pytest.raises(ValidationError, match="one pixel"):
            GeospatialConfig(
                admin_boundaries="x.gpkg",
                admin_layer="districts",
                admin_crs="EPSG:4326",
                join_predicate="intersects",
                min_affected_area_km2=0.0,
                min_fractional_coverage=0.0,
                dissolve_by=["state", "district"],
            )

    def test_district_identity_needs_a_key(self):
        # India has an Aurangabad in Maharashtra and another in Bihar.
        with pytest.raises(ValidationError, match="Aurangabad"):
            GeospatialConfig(
                admin_boundaries="x.gpkg",
                admin_layer="districts",
                admin_crs="EPSG:4326",
                join_predicate="intersects",
                min_affected_area_km2=150.0,
                min_fractional_coverage=0.05,
                dissolve_by=[],
            )

    def test_terrain_bands_must_not_overlap(self):
        with pytest.raises(ValidationError, match="overlap"):
            HazardTaxonomyConfig(
                mountainous_elevation_m=20.0,
                coastal_elevation_m=500.0,
                urban_fraction=0.3,
            )

    def test_warmup_must_end_before_the_run_does(self):
        with pytest.raises(ValidationError, match="warmup"):
            ScheduleConfig(
                max_steps=100,
                warmup_steps=1000,
                lr_schedule="constant",
            )

    def test_unknown_keys_are_rejected(self):
        # extra="forbid": a mistyped key becomes an error naming the field
        # rather than a default quietly taking effect.
        with pytest.raises(ValidationError):
            ScheduleConfig(
                max_steps=1000,
                warmup_steps=10,
                lr_schedule="constant",
                typo_field=True,
            )
