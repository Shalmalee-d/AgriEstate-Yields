"""
Tests for suitability module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

ORIGINAL_CANDIDATE_LANDS_CSV = str(Path(__file__).parent.parent / "data" / "candidate_lands.csv")

from src.suitability import (
    assess_suitability,
    assess_all_lands,
    suitability_to_dataframe,
    filter_by_usage,
    get_suitability_summary,
    SuitabilityResult,
    WEIGHTS,
    THRESHOLDS,
    compute_area_score,
    compute_sunlight_score,
    compute_water_score,
    compute_market_score,
    compute_land_type_score,
    compute_usage_score,
    determine_status,
    determine_usage_flag,
    get_limiting_factors,
)
from src.preprocessing import preprocess_candidate_lands


def _load_original_preprocessed():
    """Helper to load the original dataset for tests that expect LAND001-LAND015."""
    return preprocess_candidate_lands(ORIGINAL_CANDIDATE_LANDS_CSV)


class TestFactorScoreFunctions:
    """Tests for individual factor score computations."""

    def test_compute_area_score(self):
        """Test area score computation."""
        # Below minimum
        assert compute_area_score(100) == 0.0
        # At minimum
        assert compute_area_score(200) == 0.0
        # At maximum
        assert compute_area_score(2000) == 1.0
        # Above maximum
        assert compute_area_score(3000) == 1.0
        # Midpoint
        assert abs(compute_area_score(1100) - 0.5) < 0.01

    def test_compute_sunlight_score(self):
        """Test sunlight score (already normalized)."""
        assert compute_sunlight_score(0.0) == 0.0
        assert compute_sunlight_score(0.5) == 0.5
        assert compute_sunlight_score(1.0) == 1.0
        # Clipping
        assert compute_sunlight_score(1.5) == 1.0
        assert compute_sunlight_score(-0.5) == 0.0

    def test_compute_water_score(self):
        """Test water access score from ordinal encoding."""
        assert compute_water_score(1) == 0.0   # Low
        assert compute_water_score(2) == 0.5   # Medium
        assert compute_water_score(3) == 1.0   # High
        # Clipping
        assert compute_water_score(0) == 0.0
        assert compute_water_score(4) == 1.0

    def test_compute_market_score(self):
        """Test market proximity score."""
        assert compute_market_score(1.0) == 1.0    # 0 km
        assert abs(compute_market_score(0.5) - 0.5) < 0.01  # 1 km
        assert abs(compute_market_score(0.167) - 0.167) < 0.01  # 5 km
        # Clipping
        assert compute_market_score(1.5) == 1.0
        assert compute_market_score(-0.1) == 0.0

    def test_compute_land_type_score(self):
        """Test land type base score."""
        assert compute_land_type_score(1.0) == 1.0   # Plot
        assert compute_land_type_score(0.85) == 0.85  # Rooftop
        # Clipping
        assert compute_land_type_score(1.5) == 1.0
        assert compute_land_type_score(-0.1) == 0.0

    def test_compute_usage_score(self):
        """Test usage priority score."""
        assert compute_usage_score(3) == 1.0      # Vacant
        assert abs(compute_usage_score(2) - 0.667) < 0.01  # Underutilized
        assert abs(compute_usage_score(1) - 0.333) < 0.01  # Existing Garden
        # Clipping
        assert compute_usage_score(0) == 0.0
        assert compute_usage_score(4) == 1.0


class TestDetermineStatus:
    """Tests for status determination."""

    def test_thresholds(self):
        """Test status thresholds (0-100 scale)."""
        assert determine_status(80.0) == "Excellent"
        assert determine_status(75.0) == "Excellent"
        assert determine_status(74.0) == "Suitable"
        assert determine_status(50.0) == "Suitable"
        assert determine_status(49.0) == "Marginal"
        assert determine_status(30.0) == "Marginal"
        assert determine_status(29.0) == "Unsuitable"
        assert determine_status(0.0) == "Unsuitable"


class TestDetermineUsageFlag:
    """Tests for usage flag determination."""

    def test_flags(self):
        """Test usage flags."""
        assert determine_usage_flag("Vacant") == "available"
        assert determine_usage_flag("Underutilized") == "repurpose"
        assert determine_usage_flag("Existing Garden") == "existing_use"
        assert determine_usage_flag("Unknown") == "unknown"


class TestGetLimitingFactors:
    """Tests for limiting factor identification."""

    def test_limiting_factors(self):
        """Test limiting factor detection."""
        factors = {"area": 0.8, "sunlight": 0.3, "water": 0.6, "market": 0.4}
        limiting = get_limiting_factors(factors, threshold=0.5)
        assert "sunlight" in limiting
        assert "market" in limiting
        assert "area" not in limiting
        assert "water" not in limiting

    def test_no_limiting_factors(self):
        """Test when no factors are limiting."""
        factors = {"area": 0.8, "sunlight": 0.7, "water": 0.9}
        limiting = get_limiting_factors(factors, threshold=0.5)
        assert len(limiting) == 0


class TestAssessSuitability:
    """Tests for single land suitability assessment."""

    @pytest.fixture
    def sample_row(self):
        """Create a sample preprocessed row from original dataset."""
        df = _load_original_preprocessed()
        return df.iloc[0]

    def test_assess_returns_result(self, sample_row):
        """Test assess_suitability returns SuitabilityResult."""
        result = assess_suitability(sample_row)
        assert isinstance(result, SuitabilityResult)

    def test_result_has_all_fields(self, sample_row):
        """Test result contains all expected fields."""
        result = assess_suitability(sample_row)
        assert result.land_id == sample_row["land_id"]
        assert result.name == sample_row["name"]
        assert 0 <= result.suitability_score <= 100
        assert result.suitability_status in [
            "Excellent", "Suitable",
            "Marginal", "Unsuitable"
        ]
        assert isinstance(result.factor_scores, dict)
        assert len(result.factor_scores) == 6
        assert isinstance(result.limiting_factors, list)
        assert result.usage_flag in ["available", "repurpose", "existing_use"]
        assert result.area_m2 > 0

    def test_factor_scores_match_weights(self, sample_row):
        """Test factor scores keys match weights."""
        result = assess_suitability(sample_row)
        assert set(result.factor_scores.keys()) == set(WEIGHTS.keys())

    def test_score_calculation(self, sample_row):
        """Test suitability score matches weighted sum (0-100 scale)."""
        result = assess_suitability(sample_row)
        # factor_scores are 0-1 (rounded to 4 decimals), so weighted sum is 0-1, then * 100
        # Use wider tolerance due to rounding of individual factor scores
        expected = sum(WEIGHTS[k] * v for k, v in result.factor_scores.items()) * 100.0
        assert abs(result.suitability_score - round(expected, 4)) < 0.01


def _load_original_preprocessed():
    """Helper to load the original dataset for tests that expect LAND001-LAND015."""
    from src.preprocessing import preprocess_candidate_lands
    ORIGINAL_CANDIDATE_LANDS_CSV = str(Path(__file__).parent.parent / "data" / "candidate_lands.csv")
    return preprocess_candidate_lands(ORIGINAL_CANDIDATE_LANDS_CSV)


class TestAssessAllLands:
    """Tests for assessing all lands."""

    def test_returns_list(self):
        """Test returns list of results."""
        results = assess_all_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        assert isinstance(results, list)
        assert len(results) == 15

    def test_sorted_by_score(self):
        """Test results sorted by score descending."""
        results = assess_all_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        scores = [r.suitability_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_lands_present(self):
        """Test all 15 lands are assessed."""
        results = assess_all_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        land_ids = {r.land_id for r in results}
        expected = {f"LAND{i:03d}" for i in range(1, 16)}
        assert land_ids == expected


class TestFilterByUsage:
    """Tests for usage filtering."""

    def test_exclude_existing_garden(self):
        """Test filtering excludes existing garden by default."""
        results = assess_all_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        filtered = filter_by_usage(results, include_existing_garden=False)
        usage_flags = {r.usage_flag for r in filtered}
        assert "existing_use" not in usage_flags
        assert len(filtered) == 14  # LAND013 excluded

    def test_include_existing_garden(self):
        """Test filtering includes existing garden when requested."""
        results = assess_all_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        filtered = filter_by_usage(results, include_existing_garden=True)
        assert len(filtered) == 15

    def test_vacant_and_underutilized_included(self):
        """Test Vacant and Underutilized are always included."""
        results = assess_all_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        filtered = filter_by_usage(results, include_existing_garden=False)
        usage_flags = {r.usage_flag for r in filtered}
        assert "available" in usage_flags
        assert "repurpose" in usage_flags


class TestSuitabilityToDataFrame:
    """Tests for DataFrame conversion."""

    def test_conversion(self):
        """Test conversion to DataFrame."""
        results = assess_all_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        df = suitability_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15
        assert "land_id" in df.columns
        assert "suitability_score" in df.columns
        assert "suitability_status" in df.columns
        assert "limiting_factors" in df.columns
        # Factor columns
        factor_cols = [c for c in df.columns if c.startswith("factor_")]
        assert len(factor_cols) == 6


class TestGetSuitabilitySummary:
    """Tests for summary statistics."""

    def test_summary_structure(self):
        """Test summary has expected structure."""
        results = assess_all_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        summary = get_suitability_summary(results)

        assert "total_lands" in summary
        assert summary["total_lands"] == 15
        assert "by_status" in summary
        assert "by_usage_flag" in summary
        assert "score_stats" in summary

        stats = summary["score_stats"]
        assert "mean" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats


class TestWeightsAndThresholds:
    """Tests for weights and thresholds constants."""

    def test_weights_sum_to_one(self):
        """Test weights sum to 1.0."""
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_weights_keys_match_factors(self):
        """Test weight keys match expected factors."""
        expected = {"area", "sunlight", "water", "market", "land_type", "usage"}
        assert set(WEIGHTS.keys()) == expected

    def test_thresholds_ordered(self):
        """Test thresholds are in descending order (0-100 scale)."""
        assert THRESHOLDS["excellent"] > THRESHOLDS["suitable"]
        assert THRESHOLDS["suitable"] > THRESHOLDS["marginal"]
        assert THRESHOLDS["marginal"] > THRESHOLDS["unsuitable"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])