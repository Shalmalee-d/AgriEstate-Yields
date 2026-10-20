"""
Tests for model module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    create_preprocessor,
    create_model,
    build_pipeline,
    validate_training_data,
    train_yield_model,
    save_model,
    load_model,
    predict_yield,
    predict_yield_with_uncertainty,
    evaluate_model,
    check_training_data_exists,
    ModelMetadata,
    EXPECTED_NUMERIC_FEATURES,
    EXPECTED_CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


class TestCreatePreprocessor:
    """Tests for preprocessor creation."""

    def test_preprocessor_created(self):
        """Test preprocessor is created with correct transformers."""
        preprocessor = create_preprocessor(
            numeric_features=["feat1", "feat2"],
            categorical_features=["cat1", "cat2"]
        )
        assert preprocessor is not None
        assert len(preprocessor.transformers) == 2

    def test_preprocessor_transform(self):
        """Test preprocessor transforms data correctly."""
        preprocessor = create_preprocessor(
            numeric_features=["num1"],
            categorical_features=["cat1"]
        )

        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["A", "B", "A"],
            "extra": [10, 20, 30]  # Should be dropped
        })

        transformed = preprocessor.fit_transform(df)
        # 1 numeric (scaled) + 2 one-hot (A, B) = 3 features
        assert transformed.shape == (3, 3)


class TestCreateModel:
    """Tests for model creation."""

    def test_default_model(self):
        """Test model created with defaults."""
        model = create_model()
        assert isinstance(model.n_estimators, int)
        assert model.random_state == 42

    def test_custom_params(self):
        """Test model with custom parameters."""
        model = create_model(n_estimators=50, max_depth=5, random_state=123)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 123


class TestBuildPipeline:
    """Tests for pipeline building."""

    def test_pipeline_created(self):
        """Test pipeline is created."""
        pipeline = build_pipeline(
            numeric_features=["num1"],
            categorical_features=["cat1"]
        )
        assert pipeline is not None
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps


class TestValidateTrainingData:
    """Tests for training data validation."""

    def test_valid_data(self):
        """Test validation passes for valid data."""
        df = pd.DataFrame({
            "num1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "cat1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "actual_yield": [100, 200, 150, 250, 180, 220, 190, 210, 170, 230],
        })
        numeric, categorical = validate_training_data(df)
        assert "num1" in numeric
        assert "cat1" in categorical
        assert TARGET_COLUMN not in numeric
        assert TARGET_COLUMN not in categorical

    def test_missing_target_raises(self):
        """Test validation fails when target missing."""
        df = pd.DataFrame({
            "num1": [1, 2, 3],
            "cat1": ["A", "B", "A"],
        })
        with pytest.raises(ValueError, match="Target column"):
            validate_training_data(df)

    def test_insufficient_samples_raises(self):
        """Test validation fails with too few samples."""
        df = pd.DataFrame({
            "num1": [1, 2],
            "actual_yield": [100, 200],
        })
        with pytest.raises(ValueError, match="Insufficient training samples"):
            validate_training_data(df)

    def test_missing_target_values_raises(self):
        """Test validation fails when target has NaN."""
        df = pd.DataFrame({
            "num1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "actual_yield": [100, 200, np.nan, 250, 180, 220, 190, 210, 170, 230],
        })
        with pytest.raises(ValueError, match="missing values"):
            validate_training_data(df)


class TestTrainYieldModel:
    """Tests for model training."""

    def test_training_creates_pipeline(self, sample_training_data):
        """Test training returns pipeline and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            sample_training_data.to_csv(csv_path, index=False)

            pipeline, metadata = train_yield_model(str(csv_path))

            assert pipeline is not None
            assert isinstance(metadata, ModelMetadata)
            assert metadata.train_samples > 0
            assert metadata.test_samples > 0
            assert metadata.mae >= 0
            assert metadata.rmse >= 0
            assert -1 <= metadata.r2 <= 1  # R² can be negative for bad models
            assert len(metadata.feature_importance) > 0

    def test_training_saves_model(self, sample_training_data):
        """Test training saves model when path provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "train.csv"
            sample_training_data.to_csv(csv_path, index=False)
            model_path = Path(tmpdir) / "model"

            pipeline, metadata = train_yield_model(str(csv_path), save_path=str(model_path))

            assert (model_path.with_suffix(".joblib")).exists()
            assert (model_path.with_suffix(".json")).exists()

    def test_invalid_csv_raises(self):
        """Test training fails for invalid CSV."""
        with pytest.raises(FileNotFoundError):
            train_yield_model("nonexistent.csv")


class TestSaveLoadModel:
    """Tests for model saving and loading."""

    def test_save_and_load(self, trained_pipeline_and_metadata):
        """Test model can be saved and loaded."""
        pipeline, metadata, tmpdir = trained_pipeline_and_metadata
        model_path = Path(tmpdir) / "test_model"

        save_model(pipeline, metadata, str(model_path))

        loaded_pipeline, loaded_metadata = load_model(str(model_path))

        assert loaded_pipeline is not None
        assert loaded_metadata is not None
        assert loaded_metadata.mae == metadata.mae
        assert loaded_metadata.rmse == metadata.rmse
        assert loaded_metadata.r2 == metadata.r2

    def test_load_nonexistent_raises(self):
        """Test loading nonexistent model raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_model("nonexistent")


class TestPredictYield:
    """Tests for yield prediction."""

    def test_predict_returns_array(self, trained_pipeline_and_metadata, sample_training_data):
        """Test prediction returns array."""
        pipeline, _, _ = trained_pipeline_and_metadata
        # Use training features (without target)
        feature_cols = [c for c in sample_training_data.columns if c != "actual_yield"]
        X = sample_training_data[feature_cols].head(5)

        predictions = predict_yield(pipeline, X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5

    def test_predict_with_uncertainty(self, trained_pipeline_and_metadata, sample_training_data):
        """Test prediction with uncertainty returns mean and std."""
        pipeline, _, _ = trained_pipeline_and_metadata
        feature_cols = [c for c in sample_training_data.columns if c != "actual_yield"]
        X = sample_training_data[feature_cols].head(5)

        mean_pred, std_pred = predict_yield_with_uncertainty(pipeline, X)

        assert isinstance(mean_pred, np.ndarray)
        assert isinstance(std_pred, np.ndarray)
        assert len(mean_pred) == 5
        assert len(std_pred) == 5
        assert all(s >= 0 for s in std_pred)


class TestEvaluateModel:
    """Tests for model evaluation."""

    def test_evaluate_returns_metrics(self, trained_pipeline_and_metadata):
        """Test evaluation returns MAE, RMSE, R²."""
        pipeline, metadata, _ = trained_pipeline_and_metadata

        # Create test data
        test_df = pd.DataFrame({
            "soil_ph": [6.5],
            "nitrogen": [100],
            "phosphorus": [50],
            "potassium": [200],
            "rainfall_mm": [500],
            "temperature_avg": [22],
            "humidity_avg": [60],
            "sunlight_hours": [8],
            "water_availability_mm": [400],
            "area_m2": [1000],
            "crop_type": ["Tomato"],
            "soil_type": ["Loam"],
            "irrigation_type": ["Drip"],
            "season": ["Spring"],
        })
        y_true = pd.Series([2500])

        metrics = evaluate_model(pipeline, test_df, y_true)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


class TestCheckTrainingDataExists:
    """Tests for checking training data existence."""

    def test_returns_none_when_no_data(self):
        """Test returns None when no training data in default location."""
        # The test data directory doesn't have yield data
        result = check_training_data_exists()
        assert result is None


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_defaults(self):
        """Test default values."""
        metadata = ModelMetadata()
        assert metadata.model_type == "RandomForestRegressor"
        assert metadata.n_estimators == 100
        assert metadata.train_samples == 0
        assert metadata.features == []
        assert metadata.feature_importance == {}

    def test_custom_values(self):
        """Test custom values."""
        metadata = ModelMetadata(
            n_estimators=200,
            train_samples=80,
            test_samples=20,
            features=["a", "b"],
            feature_importance={"a": 0.6, "b": 0.4},
        )
        assert metadata.n_estimators == 200
        assert metadata.train_samples == 80
        assert metadata.features == ["a", "b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])