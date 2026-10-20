"""
Random Forest Yield Prediction Model Module.

This module implements a complete Random Forest regression pipeline for
agricultural yield prediction. It is designed for a PROPER SUPERVISED
AGRICULTURAL DATASET with actual yield labels.

CURRENT STATUS: PIPELINE READY, NOT TRAINED
============================================

The model is NOT trained because the current candidate_lands.csv dataset
does NOT contain:
- Actual yield measurements (target variable)
- Soil pH, N, P, K measurements
- Historical rainfall, temperature, humidity
- Crop type information

TRAINING DATA REQUIREMENTS
==========================

To train this model, you need a labeled dataset with columns:

FEATURES (X):
- soil_ph: float (typically 4.5-8.5)
- nitrogen: float (kg/ha or ppm)
- phosphorus: float (kg/ha or ppm)
- potassium: float (kg/ha or ppm)
- rainfall_mm: float (growing season total)
- temperature_avg: float (°C, growing season average)
- humidity_avg: float (%)
- sunlight_hours: float (daily average)
- water_availability: float (mm or categorical)
- area_m2: float
- crop_type: categorical (encoded)
- ... other relevant agronomic features

TARGET (y):
- actual_yield: float (kg/ha or kg/m2 or tons/acre)

EXPECTED DATA SOURCES:
- Agricultural research station trials
- Farmer field surveys with measured yields
- Government agricultural statistics (district-level with crop cuts)
- Remote sensing calibrated with ground truth
- Long-term experimental plots

USAGE WHEN LABELED DATA IS AVAILABLE
====================================

1. Prepare CSV with required features + actual_yield column
2. Call train_yield_model(csv_path, target_col="actual_yield")
3. Model will be saved as joblib file
4. Use predict_yield() for inference

ARCHITECTURE
============

- Algorithm: RandomForestRegressor (scikit-learn)
- Preprocessing: StandardScaler for numeric, OneHotEncoder for categorical
- Validation: Train/test split + cross-validation
- Metrics: MAE, RMSE, R²
- Persistence: joblib (model + preprocessor + metadata)
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# Expected feature columns for yield prediction (when labeled data exists)
EXPECTED_NUMERIC_FEATURES = [
    "soil_ph",
    "nitrogen",      # N kg/ha or ppm
    "phosphorus",    # P kg/ha or ppm
    "potassium",     # K kg/ha or ppm
    "rainfall_mm",
    "temperature_avg",
    "humidity_avg",
    "sunlight_hours",
    "water_availability_mm",
    "area_m2",
]

EXPECTED_CATEGORICAL_FEATURES = [
    "crop_type",
    "soil_type",
    "irrigation_type",
    "season",
]

TARGET_COLUMN = "actual_yield"  # kg/ha or kg/m2


@dataclass
class ModelMetadata:
    """Metadata about the trained model."""
    model_type: str = "RandomForestRegressor"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    train_samples: int = 0
    test_samples: int = 0
    features: List[str] = None
    target: str = TARGET_COLUMN
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    cv_mae_mean: float = 0.0
    cv_mae_std: float = 0.0
    feature_importance: Dict[str, float] = None
    trained_at: str = ""
    data_source: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.features is None:
            self.features = []
        if self.feature_importance is None:
            self.feature_importance = {}


def create_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Create preprocessing pipeline for features.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        ColumnTransformer for preprocessing.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop columns not specified
        verbose_feature_names_out=False,
    )
    return preprocessor


def create_model(n_estimators: int = 200,
                 max_depth: Optional[int] = 15,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 random_state: int = 42,
                 n_jobs: int = -1) -> RandomForestRegressor:
    """
    Create RandomForestRegressor with sensible defaults.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum depth (None = unlimited).
        min_samples_split: Minimum samples to split.
        min_samples_leaf: Minimum samples in leaf.
        random_state: Random seed.
        n_jobs: Parallel jobs (-1 = all cores).

    Returns:
        Configured RandomForestRegressor.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def build_pipeline(numeric_features: List[str],
                   categorical_features: List[str],
                   model_params: Optional[Dict] = None) -> Pipeline:
    """
    Build complete ML pipeline: preprocessing + model.

    Args:
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.
        model_params: Optional dict of RandomForest parameters.

    Returns:
        sklearn Pipeline.
    """
    if model_params is None:
        model_params = {}

    preprocessor = create_preprocessor(numeric_features, categorical_features)
    model = create_model(**model_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def validate_training_data(df: pd.DataFrame,
                           target_col: str = TARGET_COLUMN) -> tuple[List[str], List[str]]:
    """
    Validate training data and infer feature columns.

    Args:
        df: Training DataFrame.
        target_col: Target column name.

    Returns:
        Tuple of (numeric_features, categorical_features) found in data.

    Raises:
        ValueError: If target column missing, insufficient data, or missing target values.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. "
                         f"Available columns: {df.columns.tolist()}")

    if len(df) < 10:
        raise ValueError(f"Insufficient training samples: {len(df)}. Need at least 10.")

    if df[target_col].isnull().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")

    # Identify feature columns (exclude target and ID columns)
    exclude_cols = {target_col, "land_id", "name", "farm_id", "field_id", "date", "year"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after excluding target/ID columns.")

    # Separate numeric and categorical
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_features, categorical_features


def train_yield_model(csv_path: str,
                      target_col: str = TARGET_COLUMN,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      model_params: Optional[Dict] = None,
                      save_path: Optional[str] = None) -> tuple[Pipeline, ModelMetadata]:
    """
    Train Random Forest yield prediction model.

    Args:
        csv_path: Path to labeled training data CSV.
        target_col: Target column name (default: 'actual_yield').
        test_size: Fraction for test split.
        random_state: Random seed.
        model_params: Optional RandomForest parameters.
        save_path: Optional path to save model (joblib).

    Returns:
        Tuple of (trained_pipeline, metadata).

    Raises:
        FileNotFoundError: If CSV not found.
        ValueError: If data validation fails.
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Validate
    numeric_features, categorical_features = validate_training_data(df, target_col)

    # Prepare X, y
    feature_cols = numeric_features + categorical_features
    X = df[feature_cols]
    y = df[target_col]

    # Check for missing values in target
    if y.isnull().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build and train pipeline
    pipeline = build_pipeline(numeric_features, categorical_features, model_params)
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mae_mean = -cv_scores.mean()
    cv_mae_std = cv_scores.std()

    # Feature importance
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Get transformed feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    feature_importance = dict(zip(feature_names, model.feature_importances_))
    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Metadata
    from datetime import datetime
    metadata = ModelMetadata(
        n_estimators=model_params.get("n_estimators", 200) if model_params else 200,
        max_depth=model_params.get("max_depth", 15) if model_params else 15,
        min_samples_split=model_params.get("min_samples_split", 5) if model_params else 5,
        min_samples_leaf=model_params.get("min_samples_leaf", 2) if model_params else 2,
        random_state=random_state,
        train_samples=len(X_train),
        test_samples=len(X_test),
        features=feature_cols,
        target=target_col,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        r2=round(r2, 4),
        cv_mae_mean=round(cv_mae_mean, 4),
        cv_mae_std=round(cv_mae_std, 4),
        feature_importance=feature_importance,
        trained_at=datetime.now().isoformat(),
        data_source=csv_path,
        notes=f"Trained on {len(df)} samples from {csv_path}",
    )

    # Save if path provided
    if save_path:
        save_model(pipeline, metadata, save_path)

    return pipeline, metadata


def save_model(pipeline: Pipeline, metadata: ModelMetadata, save_path: str) -> None:
    """
    Save model pipeline and metadata to disk.

    Args:
        pipeline: Trained sklearn Pipeline.
        metadata: ModelMetadata.
        save_path: Path to save (will create .joblib and .json files).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    joblib.dump(pipeline, save_path.with_suffix(".joblib"))

    # Save metadata
    with open(save_path.with_suffix(".json"), "w") as f:
        json.dump(asdict(metadata), f, indent=2)


def load_model(load_path: str) -> tuple[Pipeline, ModelMetadata]:
    """
    Load model pipeline and metadata from disk.

    Args:
        load_path: Path to model (without extension, or with .joblib).

    Returns:
        Tuple of (pipeline, metadata).
    """
    load_path = Path(load_path)
    if load_path.suffix == "":
        joblib_path = load_path.with_suffix(".joblib")
        json_path = load_path.with_suffix(".json")
    elif load_path.suffix == ".joblib":
        joblib_path = load_path
        json_path = load_path.with_suffix(".json")
    else:
        raise ValueError("load_path must be a .joblib file or path without extension")

    pipeline = joblib.load(joblib_path)
    with open(json_path, "r") as f:
        metadata_dict = json.load(f)
    metadata = ModelMetadata(**metadata_dict)
    return pipeline, metadata


def predict_yield(pipeline: Pipeline,
                  features: pd.DataFrame) -> np.ndarray:
    """
    Predict yield for new data.

    Args:
        pipeline: Trained pipeline.
        features: DataFrame with feature columns (must match training features).

    Returns:
        Array of predicted yields.
    """
    return pipeline.predict(features)


def predict_yield_with_uncertainty(pipeline: Pipeline,
                                    features: pd.DataFrame,
                                    n_trees_sample: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict yield with uncertainty estimate using tree-level predictions.

    Args:
        pipeline: Trained pipeline.
        features: DataFrame with feature columns.
        n_trees_sample: Number of trees to sample for uncertainty (speed vs accuracy).

    Returns:
        Tuple of (predictions, std_dev) arrays.
    """
    # Transform features
    X_transformed = pipeline.named_steps["preprocessor"].transform(features)
    model = pipeline.named_steps["model"]

    # Get predictions from individual trees
    n_trees = min(n_trees_sample, len(model.estimators_))
    tree_preds = np.zeros((features.shape[0], n_trees))

    for i, tree in enumerate(model.estimators_[:n_trees]):
        tree_preds[:, i] = tree.predict(X_transformed)

    mean_pred = tree_preds.mean(axis=1)
    std_pred = tree_preds.std(axis=1)

    return mean_pred, std_pred


def evaluate_model(pipeline: Pipeline,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model on test data.

    Args:
        pipeline: Trained pipeline.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        Dict with MAE, RMSE, R².
    """
    y_pred = pipeline.predict(X_test)
    return {
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "r2": round(r2_score(y_test, y_pred), 4),
    }


def check_training_data_exists(data_dir: Optional[str] = None) -> Optional[str]:
    """
    Check if a labeled yield dataset exists in the data directory.

    Args:
        data_dir: Directory to search. Defaults to ml/data.

    Returns:
        Path to CSV if found, None otherwise.
    """
    if data_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "data"
    else:
        data_dir = Path(data_dir)

    # Look for common yield dataset names
    patterns = [
        "yield_data.csv",
        "training_data.csv",
        "historical_yields.csv",
        "agricultural_yields.csv",
        "*yield*.csv",
        "*training*.csv",
    ]

    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            return str(matches[0])

    return None


def train_if_data_available(data_dir: Optional[str] = None,
                            model_dir: Optional[str] = None) -> Optional[tuple[Pipeline, ModelMetadata]]:
    """
    Train model if labeled yield data is found.

    Args:
        data_dir: Directory to search for training data.
        model_dir: Directory to save model.

    Returns:
        Tuple of (pipeline, metadata) if trained, None if no data found.
    """
    csv_path = check_training_data_exists(data_dir)
    if csv_path is None:
        warnings.warn(
            "No labeled yield dataset found. Model not trained. "
            "Place a CSV with 'actual_yield' column in the data directory.",
            UserWarning
        )
        return None

    if model_dir is None:
        base_dir = Path(__file__).parent.parent.parent
        model_dir = base_dir / "models"
    else:
        model_dir = Path(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "yield_rf_model"

    return train_yield_model(csv_path, save_path=str(save_path))


if __name__ == "__main__":
    # Check for training data
    result = train_if_data_available()
    if result:
        pipeline, metadata = result
        print("Model trained successfully!")
        print(f"MAE: {metadata.mae}, RMSE: {metadata.rmse}, R²: {metadata.r2}")
        print(f"Top 10 features: {list(metadata.feature_importance.items())[:10]}")
    else:
        print("No training data found. Model not trained.")
        print("Expected: CSV with 'actual_yield' column in ml/data/")