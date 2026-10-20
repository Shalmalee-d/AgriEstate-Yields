"""
Tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

ORIGINAL_CANDIDATE_LANDS_CSV = str(Path(__file__).parent.parent / "data" / "candidate_lands.csv")

from src.preprocessing import (
    load_candidate_lands,
    validate_columns,
    validate_categorical_values,
    validate_numeric_ranges,
    handle_missing_values,
    encode_categorical,
    add_derived_features,
    preprocess_candidate_lands,
    get_feature_columns,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    VALID_LAND_TYPES,
    VALID_CURRENT_USAGE,
    VALID_WATER_ACCESS,
)


def _load_original_dataset():
    """Helper to load the original dataset for tests that expect LAND001-LAND015."""
    return load_candidate_lands(ORIGINAL_CANDIDATE_LANDS_CSV)


class TestLoadCandidateLands:
    """Tests for CSV loading."""

    def test_load_default_path(self):
        """Test loading from default path (now Pune prototype)."""
        df = load_candidate_lands()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15
        # Check that all required columns are present
        assert set(REQUIRED_COLUMNS).issubset(set(df.columns))
        # Check that optional columns are also present
        assert set(OPTIONAL_COLUMNS).issubset(set(df.columns))

    def test_load_original_dataset(self):
        """Test loading the original dataset explicitly."""
        df = load_candidate_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        assert len(df) == 15
        assert set(REQUIRED_COLUMNS).issubset(set(df.columns))

    def test_load_custom_path(self):
        """Test loading from custom path."""
        df = load_candidate_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        assert len(df) == 15

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_candidate_lands("nonexistent.csv")


class TestValidateColumns:
    """Tests for column validation."""

    def test_valid_columns(self):
        """Test validation passes for correct columns."""
        df = _load_original_dataset()
        validate_columns(df)  # Should not raise

    def test_missing_columns(self):
        """Test validation fails for missing columns."""
        df = _load_original_dataset()
        df_missing = df.drop(columns=["area_sqft"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(df_missing)


class TestValidateCategoricalValues:
    """Tests for categorical value validation."""

    def test_valid_categorical(self):
        """Test validation passes for valid categories."""
        df = _load_original_dataset()
        validate_categorical_values(df)  # Should not raise

    def test_invalid_land_type(self):
        """Test validation fails for invalid land_type."""
        df = _load_original_dataset()
        df.loc[0, "land_type"] = "InvalidType"
        with pytest.raises(ValueError, match="Invalid land_type values"):
            validate_categorical_values(df)

    def test_invalid_current_usage(self):
        """Test validation fails for invalid current_usage."""
        df = _load_original_dataset()
        df.loc[0, "current_usage"] = "InvalidUsage"
        with pytest.raises(ValueError, match="Invalid current_usage values"):
            validate_categorical_values(df)

    def test_invalid_water_access(self):
        """Test validation fails for invalid water_access."""
        df = _load_original_dataset()
        df.loc[0, "water_access"] = "InvalidWater"
        with pytest.raises(ValueError, match="Invalid water_access values"):
            validate_categorical_values(df)


class TestValidateNumericRanges:
    """Tests for numeric range validation."""

    def test_valid_ranges(self):
        """Test validation passes for valid ranges."""
        df = _load_original_dataset()
        validate_numeric_ranges(df)  # Should not raise

    def test_area_out_of_range(self):
        """Test validation fails for area out of range."""
        df = _load_original_dataset()
        df.loc[0, "area_sqft"] = -100
        with pytest.raises(ValueError, match="outside range"):
            validate_numeric_ranges(df)

    def test_sunlight_out_of_range(self):
        """Test validation fails for sunlight out of range."""
        df = _load_original_dataset()
        df.loc[0, "sunlight_hours"] = 25
        with pytest.raises(ValueError, match="outside range"):
            validate_numeric_ranges(df)

    def test_market_distance_out_of_range(self):
        """Test validation fails for market distance out of range."""
        df = _load_original_dataset()
        df.loc[0, "market_distance_km"] = 150
        with pytest.raises(ValueError, match="outside range"):
            validate_numeric_ranges(df)


class TestHandleMissingValues:
    """Tests for missing value handling."""

    def test_no_missing_values(self):
        """Test passes when no missing values."""
        df = _load_original_dataset()
        result = handle_missing_values(df)
        assert len(result) == 15

    def test_missing_values_raises(self):
        """Test raises error when missing values present."""
        df = _load_original_dataset()
        df.loc[0, "area_sqft"] = np.nan
        with pytest.raises(ValueError, match="Missing values found"):
            handle_missing_values(df)


class TestEncodeCategorical:
    """Tests for categorical encoding."""

    def test_encoding_creates_expected_columns(self):
        """Test encoding adds expected columns."""
        df = _load_original_dataset()
        df = encode_categorical(df)

        assert "water_access_encoded" in df.columns
        assert "usage_priority" in df.columns
        assert "land_type_encoded" in df.columns
        assert "land_type_base" in df.columns

    def test_water_access_encoding_values(self):
        """Test water access encoding maps correctly."""
        df = _load_original_dataset()
        df = encode_categorical(df)

        mapping = df.drop_duplicates(subset=["water_access"])[["water_access", "water_access_encoded"]]
        assert mapping[mapping["water_access"] == "Low"]["water_access_encoded"].values[0] == 1
        assert mapping[mapping["water_access"] == "Medium"]["water_access_encoded"].values[0] == 2
        assert mapping[mapping["water_access"] == "High"]["water_access_encoded"].values[0] == 3

    def test_usage_priority_encoding_values(self):
        """Test usage priority encoding maps correctly."""
        df = _load_original_dataset()
        df = encode_categorical(df)

        mapping = df.drop_duplicates(subset=["current_usage"])[["current_usage", "usage_priority"]]
        assert mapping[mapping["current_usage"] == "Vacant"]["usage_priority"].values[0] == 3
        assert mapping[mapping["current_usage"] == "Underutilized"]["usage_priority"].values[0] == 2
        assert mapping[mapping["current_usage"] == "Existing Garden"]["usage_priority"].values[0] == 1

    def test_land_type_encoding(self):
        """Test land type encoding."""
        df = _load_original_dataset()
        df = encode_categorical(df)

        mapping = df.drop_duplicates(subset=["land_type"])[["land_type", "land_type_encoded", "land_type_base"]]
        plot_row = mapping[mapping["land_type"] == "Plot"].iloc[0]
        roof_row = mapping[mapping["land_type"] == "Rooftop"].iloc[0]

        assert plot_row["land_type_encoded"] == 1
        assert plot_row["land_type_base"] == 1.0
        assert roof_row["land_type_encoded"] == 0
        assert roof_row["land_type_base"] == 0.85

    def test_original_columns_preserved(self):
        """Test original categorical columns are preserved."""
        df = _load_original_dataset()
        original_cols = set(df.columns)
        df = encode_categorical(df)
        assert original_cols.issubset(set(df.columns))


class TestAddDerivedFeatures:
    """Tests for derived feature creation."""

    def test_area_m2_conversion(self):
        """Test area conversion to square meters."""
        df = _load_original_dataset()
        df = encode_categorical(df)
        df = add_derived_features(df)

        assert "area_m2" in df.columns
        expected = df["area_sqft"] * 0.092903
        np.testing.assert_array_almost_equal(df["area_m2"].values, expected.values)

    def test_sunlight_score_normalization(self):
        """Test sunlight score normalization."""
        df = _load_original_dataset()
        df = encode_categorical(df)
        df = add_derived_features(df)

        assert "sunlight_score" in df.columns
        # Max sunlight in data is 8.7, so max score should be 8.7/12 < 1
        assert df["sunlight_score"].max() <= 1.0
        assert df["sunlight_score"].min() >= 0.0
        # Check calculation
        expected = np.clip(df["sunlight_hours"] / 12.0, 0, 1)
        np.testing.assert_array_almost_equal(df["sunlight_score"].values, expected.values)

    def test_market_proximity_score(self):
        """Test market proximity score calculation."""
        df = _load_original_dataset()
        df = encode_categorical(df)
        df = add_derived_features(df)

        assert "market_proximity_score" in df.columns
        expected = 1.0 / (1.0 + df["market_distance_km"])
        np.testing.assert_array_almost_equal(df["market_proximity_score"].values, expected.values)
        # Closer = higher score
        assert df.loc[df["market_distance_km"].idxmin(), "market_proximity_score"] == df["market_proximity_score"].max()


class TestPreprocessPipeline:
    """Tests for full preprocessing pipeline."""

    def test_full_pipeline(self):
        """Test complete preprocessing pipeline with original dataset."""
        df = preprocess_candidate_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        assert len(df) == 15
        assert "area_m2" in df.columns
        assert "sunlight_score" in df.columns
        assert "market_proximity_score" in df.columns
        assert "water_access_encoded" in df.columns
        assert "usage_priority" in df.columns
        assert "land_type_encoded" in df.columns
        assert "land_type_base" in df.columns

    def test_land_id_preserved(self):
        """Test land_id is preserved through pipeline."""
        df = preprocess_candidate_lands(ORIGINAL_CANDIDATE_LANDS_CSV)
        assert "land_id" in df.columns
        assert df["land_id"].tolist() == [f"LAND{i:03d}" for i in range(1, 16)]

    def test_get_feature_columns(self):
        """Test feature column list."""
        features = get_feature_columns()
        assert isinstance(features, list)
        assert len(features) > 0
        assert "area_m2" in features
        assert "sunlight_score" in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])