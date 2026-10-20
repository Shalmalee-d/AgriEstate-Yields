"""
Shared pytest fixtures for ML module tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def sample_candidate_lands():
    """Load the candidate lands dataset once per session."""
    from preprocessing import load_candidate_lands
    return load_candidate_lands()


@pytest.fixture(scope="session")
def preprocessed_lands(sample_candidate_lands):
    """Preprocessed candidate lands."""
    from preprocessing import preprocess_candidate_lands
    return preprocess_candidate_lands()


@pytest.fixture(scope="session")
def suitability_results(preprocessed_lands):
    """Suitability assessment results for all lands."""
    from suitability import assess_all_lands
    return assess_all_lands()


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_training_data():
    """Generate sample labeled training data for yield model."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "soil_ph": np.random.uniform(5.5, 7.5, n),
        "nitrogen": np.random.uniform(50, 200, n),
        "phosphorus": np.random.uniform(20, 100, n),
        "potassium": np.random.uniform(100, 300, n),
        "rainfall_mm": np.random.uniform(300, 800, n),
        "temperature_avg": np.random.uniform(15, 30, n),
        "humidity_avg": np.random.uniform(40, 80, n),
        "sunlight_hours": np.random.uniform(6, 10, n),
        "water_availability_mm": np.random.uniform(200, 600, n),
        "area_m2": np.random.uniform(500, 5000, n),
        "crop_type": np.random.choice(["Tomato", "Lettuce", "Spinach"], n),
        "soil_type": np.random.choice(["Loam", "Clay", "Sandy"], n),
        "irrigation_type": np.random.choice(["Drip", "Sprinkler", "Flood"], n),
        "season": np.random.choice(["Spring", "Summer", "Fall"], n),
    })
    # Generate yield as function of features + noise
    df["actual_yield"] = (
        2000
        + (df["soil_ph"] - 6.5) * 100
        + df["nitrogen"] * 2
        + df["phosphorus"] * 1.5
        + df["potassium"] * 1
        + df["rainfall_mm"] * 0.5
        - (df["temperature_avg"] - 22) ** 2 * 2
        + np.random.normal(0, 100, n)
    ).clip(0, None)
    return df


@pytest.fixture
def trained_pipeline_and_metadata(sample_training_data, temp_dir):
    """Train a model and return pipeline and metadata."""
    from model import train_yield_model
    csv_path = temp_dir / "train.csv"
    sample_training_data.to_csv(csv_path, index=False)
    pipeline, metadata = train_yield_model(str(csv_path))
    return pipeline, metadata, str(temp_dir)