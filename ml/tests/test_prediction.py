"""
Tests for prediction interface module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction import (
    predict_yield_interface,
    assess_suitability_interface,
    get_all_suitability,
    get_suitability_summary_interface,
    recommend_crops_interface,
    get_all_recommendations,
    get_model_status,
    reload_model,
    PredictionRequest,
    SuitabilityRequest,
    CropRecommendationRequest,
)
from src.suitability import assess_all_lands

# Path to the original candidate lands CSV (for tests that expect the original dataset)
ORIGINAL_CANDIDATE_LANDS_CSV = str(Path(__file__).parent.parent / "data" / "candidate_lands.csv")


class TestAssessSuitabilityInterface:
    """Tests for suitability assessment interface."""

    def test_valid_request(self):
        """Test suitability assessment with valid request."""
        request = SuitabilityRequest(
            land_id="TEST001",
            name="Test Land",
            land_type="Plot",
            current_usage="Vacant",
            area_sqft=5000,
            sunlight_hours=8.0,
            water_access="High",
            market_distance_km=3.0,
        )
        response = assess_suitability_interface(request)

        assert response.land_id == "TEST001"
        assert response.name == "Test Land"
        assert 0 <= response.suitability_score <= 100
        assert response.suitability_status in [
            "Excellent", "Suitable",
            "Marginal", "Unsuitable"
        ]
        assert isinstance(response.factor_scores, dict)
        assert len(response.factor_scores) == 6
        assert isinstance(response.limiting_factors, list)
        assert response.usage_flag == "available"
        assert response.area_m2 > 0

    def test_rooftop_land_type(self):
        """Test with Rooftop land type."""
        request = SuitabilityRequest(
            land_id="TEST002",
            name="Test Rooftop",
            land_type="Rooftop",
            current_usage="Vacant",
            area_sqft=4000,
            sunlight_hours=7.5,
            water_access="Medium",
            market_distance_km=2.0,
        )
        response = assess_suitability_interface(request)
        assert response.land_id == "TEST002"
        # Rooftop should have lower land_type factor
        assert response.factor_scores["land_type"] == 0.85

    def test_underutilized_usage(self):
        """Test with Underutilized current usage."""
        request = SuitabilityRequest(
            land_id="TEST003",
            name="Test Underutilized",
            land_type="Plot",
            current_usage="Underutilized",
            area_sqft=5000,
            sunlight_hours=8.0,
            water_access="High",
            market_distance_km=3.0,
        )
        response = assess_suitability_interface(request)
        assert response.usage_flag == "repurpose"
        assert abs(response.factor_scores["usage"] - 2/3) < 0.01  # ~0.667

    def test_existing_garden_usage(self):
        """Test with Existing Garden current usage."""
        request = SuitabilityRequest(
            land_id="TEST004",
            name="Test Garden",
            land_type="Plot",
            current_usage="Existing Garden",
            area_sqft=5000,
            sunlight_hours=8.0,
            water_access="High",
            market_distance_km=3.0,
        )
        response = assess_suitability_interface(request)
        assert response.usage_flag == "existing_use"
        assert abs(response.factor_scores["usage"] - 1/3) < 0.01  # ~0.333

    def test_low_water_access(self):
        """Test with Low water access."""
        request = SuitabilityRequest(
            land_id="TEST005",
            name="Test Low Water",
            land_type="Plot",
            current_usage="Vacant",
            area_sqft=5000,
            sunlight_hours=8.0,
            water_access="Low",
            market_distance_km=3.0,
        )
        response = assess_suitability_interface(request)
        assert response.factor_scores["water"] == 0.0
        assert "water" in response.limiting_factors


class TestGetAllSuitability:
    """Tests for batch suitability assessment."""

    def test_returns_list(self):
        """Test returns list of responses."""
        results = get_all_suitability(csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)
        assert isinstance(results, list)
        # Excludes existing garden by default (LAND013)
        assert len(results) == 14

    def test_excludes_existing_garden_by_default(self):
        """Test excludes existing garden by default."""
        results = get_all_suitability(csv_path=ORIGINAL_CANDIDATE_LANDS_CSV, include_existing_garden=False)
        assert len(results) == 14
        flags = {r.usage_flag for r in results}
        assert "existing_use" not in flags

    def test_includes_existing_garden_when_requested(self):
        """Test includes existing garden when requested."""
        results = get_all_suitability(csv_path=ORIGINAL_CANDIDATE_LANDS_CSV, include_existing_garden=True)
        assert len(results) == 15

    def test_sorted_by_score(self):
        """Test results sorted by score descending."""
        results = get_all_suitability(csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)
        scores = [r.suitability_score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestGetSuitabilitySummaryInterface:
    """Tests for suitability summary."""

    def test_summary_structure(self):
        """Test summary has expected structure."""
        summary = get_suitability_summary_interface(csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)
        assert "total_lands" in summary
        assert summary["total_lands"] == 15
        assert "by_status" in summary
        assert "by_usage_flag" in summary
        assert "score_stats" in summary


class TestCropRecommendationInterface:
    """Tests for crop recommendation interface."""

    def test_recommend_for_known_land(self):
        """Test recommendations for a known land ID."""
        request = CropRecommendationRequest(land_id="LAND001", top_n=3)
        response = recommend_crops_interface(request, csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)

        assert response.land_id == "LAND001"
        assert response.land_name == "Green Tower Rooftop"
        assert len(response.recommendations) == 3
        assert all("crop" in r for r in response.recommendations)
        assert all("composite_score" in r for r in response.recommendations)
        assert "data_gaps" in response.__dict__

    def test_recommend_unknown_land(self):
        """Test recommendations for unknown land ID."""
        request = CropRecommendationRequest(land_id="UNKNOWN999", top_n=3)
        response = recommend_crops_interface(request, csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)

        assert response.land_id == "UNKNOWN999"
        assert response.suitability_status == "Not Found"
        assert len(response.recommendations) == 0
        assert "land_id not found" in response.data_gaps[0]

    def test_recommend_with_environmental_data(self):
        """Test recommendations with optional environmental data."""
        request = CropRecommendationRequest(
            land_id="LAND001",
            soil_ph=6.5,
            temperature_avg=22.0,
            rainfall_mm=500,
            top_n=2
        )
        response = recommend_crops_interface(request, csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)

        assert len(response.recommendations) == 2
        # Data gaps should not include provided data
        for rec in response.recommendations:
            assert "ph" not in rec["data_gaps"]
            assert "temperature" not in rec["data_gaps"]
            assert "rainfall" not in rec["data_gaps"]


class TestGetAllRecommendations:
    """Tests for batch crop recommendations."""

    def test_returns_list(self):
        """Test returns list of responses."""
        results = get_all_recommendations(top_n=2, csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)
        assert isinstance(results, list)
        # 14 lands (excluding existing garden by default)
        assert len(results) == 14

    def test_each_has_recommendations(self):
        """Test each land has requested number of recommendations."""
        results = get_all_recommendations(top_n=3, csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)
        for r in results:
            assert len(r.recommendations) == 3

    def test_sorted_by_suitability(self):
        """Test results follow suitability order."""
        results = get_all_recommendations(top_n=1, csv_path=ORIGINAL_CANDIDATE_LANDS_CSV)
        scores = [r.suitability_score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestGetModelStatus:
    """Tests for model status."""

    def test_returns_dict(self):
        """Test returns status dict."""
        status = get_model_status()
        assert isinstance(status, dict)
        assert "model_available" in status

    def test_model_not_available(self):
        """Test model is not available (no training data)."""
        status = get_model_status()
        # Should be False since no labeled yield dataset exists
        assert status["model_available"] is False
        assert "error" in status or "message" in status


class TestReloadModel:
    """Tests for model reload."""

    def test_reload_returns_status(self):
        """Test reload returns updated status."""
        status = reload_model()
        assert isinstance(status, dict)
        assert "model_available" in status


class TestPredictionRequestDataclasses:
    """Tests for request/response dataclasses."""

    def test_prediction_request_creation(self):
        """Test PredictionRequest can be created."""
        req = PredictionRequest(
            soil_ph=6.5,
            nitrogen=100,
            crop_type="Tomato"
        )
        assert req.soil_ph == 6.5
        assert req.nitrogen == 100
        assert req.crop_type == "Tomato"
        assert req.phosphorus is None

    def test_suitability_request_creation(self):
        """Test SuitabilityRequest creation."""
        req = SuitabilityRequest(
            land_id="TEST",
            name="Test",
            land_type="Plot",
            current_usage="Vacant",
            area_sqft=5000,
            sunlight_hours=8.0,
            water_access="High",
            market_distance_km=3.0,
        )
        assert req.land_id == "TEST"

    def test_crop_recommendation_request_creation(self):
        """Test CropRecommendationRequest creation."""
        req = CropRecommendationRequest(
            land_id="LAND001",
            soil_ph=6.5,
            top_n=5
        )
        assert req.land_id == "LAND001"
        assert req.soil_ph == 6.5
        assert req.top_n == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])