"""
Data preprocessing module for candidate land dataset.

This module handles loading, validation, cleaning, and feature engineering
for the candidate land agricultural suitability assessment pipeline.

Dataset: candidate_lands.csv (15 records)
Features available: land_id, name, land_type, current_usage, area_sqft,
                    sunlight_hours, water_access, market_distance_km

Note: This dataset does NOT contain yield labels or environmental features
required for supervised yield prediction (soil pH, N/P/K, rainfall, etc.).
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


# Required columns in the candidate land dataset
REQUIRED_COLUMNS = [
    "land_id",
    "name",
    "land_type",
    "current_usage",
    "area_sqft",
    "sunlight_hours",
    "water_access",
    "market_distance_km",
]

# Optional columns that may have missing values
OPTIONAL_COLUMNS = ["latitude", "longitude"]

# Environmental data columns (from environmental prototype dataset)
ENVIRONMENTAL_COLUMNS = [
    "temperature_avg_c",
    "annual_rainfall_mm",
    "humidity_avg_pct",
    "soil_ph",
    "data_status",
]

# Mapping from environmental column names to internal names
ENVIRONMENTAL_COLUMN_MAP = {
    "temperature_avg_c": "temperature_avg",
    "annual_rainfall_mm": "rainfall_mm",
    "soil_ph": "soil_ph",
    "humidity_avg_pct": "humidity_avg_pct",
    "data_status": "environmental_data_status",
}

# Prototype dataset filenames
PUNE_CANDIDATE_LANDS_CSV = "pune_candidate_lands_prototype.csv"
PUNE_ENVIRONMENTAL_CSV = "pune_environmental_prototype.csv"
DEFAULT_CANDIDATE_LANDS_CSV = "candidate_lands.csv"


def get_default_candidate_lands_path() -> Path:
    """
    Resolve the default candidate lands CSV path.
    
    Prefers Pune prototype dataset if it exists, otherwise falls back to default.
    
    Returns:
        Path to the default candidate lands CSV.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    pune_path = data_dir / PUNE_CANDIDATE_LANDS_CSV
    if pune_path.exists():
        return pune_path
    
    default_path = data_dir / DEFAULT_CANDIDATE_LANDS_CSV
    return default_path


def get_default_environmental_data_path() -> Optional[Path]:
    """
    Resolve the default environmental data CSV path.
    
    Returns Pune environmental prototype path if it exists, otherwise None.
    
    Returns:
        Path to the default environmental data CSV, or None if not found.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    pune_env_path = data_dir / PUNE_ENVIRONMENTAL_CSV
    if pune_env_path.exists():
        return pune_env_path
    
    return None

# Valid categorical values
VALID_LAND_TYPES = {"Rooftop", "Plot"}
VALID_CURRENT_USAGE = {"Vacant", "Underutilized", "Existing Garden"}
VALID_WATER_ACCESS = {"High", "Medium", "Low"}

# Numeric validation ranges
NUMERIC_RANGES = {
    "area_sqft": (0, 100000),
    "sunlight_hours": (0, 24),
    "market_distance_km": (0, 100),
}

# Water access ordinal encoding
WATER_ACCESS_ENCODING = {"Low": 1, "Medium": 2, "High": 3}

# Current usage priority encoding (for suitability)
USAGE_PRIORITY = {"Vacant": 3, "Underutilized": 2, "Existing Garden": 1}

# Land type suitability base score
LAND_TYPE_BASE = {"Plot": 1.0, "Rooftop": 0.85}


def load_candidate_lands(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load candidate lands CSV from the default data directory or a custom path.

    Args:
        csv_path: Optional custom path to CSV file. If None, uses default location
                  (prefers Pune prototype dataset if it exists).

    Returns:
        DataFrame with raw candidate land data.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If required columns are missing.
    """
    if csv_path is None:
        csv_path = get_default_candidate_lands_path()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate lands CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that all required columns are present.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean categorical columns.

    Args:
        df: DataFrame to validate.

    Returns:
        DataFrame with validated categorical columns.

    Raises:
        ValueError: If invalid categorical values are found.
    """
    df = df.copy()

    # Validate land_type
    invalid_land = set(df["land_type"].unique()) - VALID_LAND_TYPES
    if invalid_land:
        raise ValueError(f"Invalid land_type values: {invalid_land}. Valid: {VALID_LAND_TYPES}")

    # Validate current_usage
    invalid_usage = set(df["current_usage"].unique()) - VALID_CURRENT_USAGE
    if invalid_usage:
        raise ValueError(f"Invalid current_usage values: {invalid_usage}. Valid: {VALID_CURRENT_USAGE}")

    # Validate water_access
    invalid_water = set(df["water_access"].unique()) - VALID_WATER_ACCESS
    if invalid_water:
        raise ValueError(f"Invalid water_access values: {invalid_water}. Valid: {VALID_WATER_ACCESS}")

    return df


def validate_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate numeric columns are within expected ranges.

    Args:
        df: DataFrame to validate.

    Returns:
        DataFrame with validated numeric columns.

    Raises:
        ValueError: If values are outside expected ranges.
    """
    df = df.copy()

    for col, (min_val, max_val) in NUMERIC_RANGES.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                raise ValueError(
                    f"Column '{col}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]"
                )

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    For this prototype dataset, we expect no missing values in required columns.
    Optional columns (latitude, longitude) may have missing values.

    Args:
        df: DataFrame to check.

    Returns:
        DataFrame (unchanged if no missing values in required columns).

    Raises:
        ValueError: If missing values are found in required columns.
    """
    missing = df.isnull().sum()
    # Only check required columns for missing values
    required_missing = missing[missing.index.isin(REQUIRED_COLUMNS) & (missing > 0)]
    if not required_missing.empty:
        missing_cols = required_missing.index.tolist()
        raise ValueError(f"Missing values found in required columns: {missing_cols}. "
                         "This prototype dataset should have no missing values in required columns. "
                         "Please check the source data.")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for modeling.

    Creates encoded columns while preserving original categorical columns.

    Args:
        df: DataFrame with validated categorical columns.

    Returns:
        DataFrame with additional encoded columns:
        - water_access_encoded (ordinal: Low=1, Medium=2, High=3)
        - usage_priority (ordinal: Vacant=3, Underutilized=2, Existing Garden=1)
        - land_type_encoded (binary: Plot=1, Rooftop=0)
        - land_type_base (suitability base: Plot=1.0, Rooftop=0.85)
    """
    df = df.copy()

    df["water_access_encoded"] = df["water_access"].map(WATER_ACCESS_ENCODING)
    df["usage_priority"] = df["current_usage"].map(USAGE_PRIORITY)
    df["land_type_encoded"] = (df["land_type"] == "Plot").astype(int)
    df["land_type_base"] = df["land_type"].map(LAND_TYPE_BASE)

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features for suitability assessment.

    Args:
        df: DataFrame with base features.

    Returns:
        DataFrame with additional derived features:
        - area_m2: area in square meters (area_sqft * 0.092903)
        - sunlight_score: normalized sunlight (0-1 scale, assuming 12h max practical)
        - market_proximity_score: inverse distance score (closer = better)
    """
    df = df.copy()

    # Convert area to square meters
    df["area_m2"] = df["area_sqft"] * 0.092903

    # Normalize sunlight hours (practical max ~12 hours for agriculture)
    df["sunlight_score"] = np.clip(df["sunlight_hours"] / 12.0, 0, 1)

    # Market proximity: inverse distance with diminishing returns
    # Score = 1 / (1 + distance_km), so 0km = 1.0, 1km = 0.5, 5km = 0.167
    df["market_proximity_score"] = 1.0 / (1.0 + df["market_distance_km"])

    return df


def preprocess_candidate_lands(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline for candidate lands.

    This is the main entry point that applies all preprocessing steps
    in the correct order.

    Pipeline:
    1. Load CSV
    2. Validate columns
    3. Handle missing values (error if any)
    4. Validate categorical values
    5. Validate numeric ranges
    6. Encode categorical features
    7. Add derived features

    Args:
        csv_path: Optional custom path to CSV file.

    Returns:
        Fully preprocessed DataFrame ready for suitability assessment.
    """
    df = load_candidate_lands(csv_path)
    df = handle_missing_values(df)
    df = validate_categorical_values(df)
    df = validate_numeric_ranges(df)
    df = encode_categorical(df)
    df = add_derived_features(df)
    return df


def get_feature_columns() -> list[str]:
    """
    Get the list of feature columns available for suitability assessment.

    Returns:
        List of feature column names.
    """
    return [
        "area_m2",
        "area_sqft",
        "sunlight_hours",
        "sunlight_score",
        "water_access_encoded",
        "market_distance_km",
        "market_proximity_score",
        "land_type_encoded",
        "land_type_base",
        "usage_priority",
    ]


def get_categorical_columns() -> list[str]:
    """
    Get the list of original categorical columns.

    Returns:
        List of categorical column names.
    """
    return ["land_type", "current_usage", "water_access"]


def load_environmental_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load environmental prototype data from CSV.

    Args:
        csv_path: Optional custom path to environmental CSV file.
                  If None, uses default (prefers Pune prototype if it exists).

    Returns:
        DataFrame with environmental data.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If required columns are missing.
    """
    if csv_path is None:
        default_path = get_default_environmental_data_path()
        if default_path is not None:
            csv_path = default_path
        else:
            # Fall back to explicit path
            base_dir = Path(__file__).parent.parent
            csv_path = base_dir / "data" / "pune_environmental_prototype.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Environmental data CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_environmental_columns(df)
    return df


def validate_environmental_columns(df: pd.DataFrame) -> None:
    """
    Validate that required environmental columns are present.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any required environmental columns are missing.
    """
    required_env_cols = ["land_id", "temperature_avg_c", "annual_rainfall_mm",
                         "humidity_avg_pct", "soil_ph", "data_status"]
    missing = set(required_env_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required environmental columns: {missing}")


def join_environmental_data(
    candidate_df: pd.DataFrame,
    environmental_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join environmental data to candidate lands on land_id.

    Maps environmental column names to internal names:
    - temperature_avg_c -> temperature_avg
    - annual_rainfall_mm -> rainfall_mm
    - soil_ph -> soil_ph
    - humidity_avg_pct -> humidity_avg_pct
    - data_status -> environmental_data_status

    Args:
        candidate_df: DataFrame with candidate land data (must have land_id)
        environmental_df: DataFrame with environmental data (must have land_id)

    Returns:
        DataFrame with candidate lands and joined environmental data.
        Environmental columns are renamed to internal names.
        Lands without matching environmental data will have NaN in environmental columns.
    """
    # Rename environmental columns to internal names
    env_renamed = environmental_df.rename(columns=ENVIRONMENTAL_COLUMN_MAP)

    # Select only the columns we want to join (land_id + mapped columns)
    join_cols = ["land_id"] + list(ENVIRONMENTAL_COLUMN_MAP.values())
    env_subset = env_renamed[join_cols]

    # Left join on land_id
    merged = candidate_df.merge(env_subset, on="land_id", how="left")

    return merged


def preprocess_candidate_lands_with_environment(
    candidate_csv_path: Optional[str] = None,
    environmental_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for candidate lands with optional environmental data join.

    Pipeline:
    1. Load candidate lands CSV
    2. Handle missing values
    3. Validate categorical values
    4. Validate numeric ranges
    5. Encode categorical features
    5. Add derived features
    6. Optionally join environmental data
    7. Map environmental columns to expected names for recommendation

    Args:
        candidate_csv_path: Optional custom path to candidate lands CSV.
        environmental_csv_path: Optional custom path to environmental data CSV.
                                If provided, environmental data will be joined.

    Returns:
        Fully preprocessed DataFrame ready for suitability assessment and crop recommendation.
    """
    df = load_candidate_lands(candidate_csv_path)
    df = handle_missing_values(df)
    df = validate_categorical_values(df)
    df = validate_numeric_ranges(df)
    df = encode_categorical(df)
    df = add_derived_features(df)

    # Join environmental data if provided
    if environmental_csv_path is not None:
        environmental_df = load_environmental_data(environmental_csv_path)
        df = join_environmental_data(df, environmental_df)

    return df


if __name__ == "__main__":
    # Quick test when run directly
    df = preprocess_candidate_lands()
    print("Preprocessed shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSample:")
    print(df[["land_id", "name", "land_type", "current_usage", "area_m2",
              "sunlight_score", "water_access_encoded", "market_proximity_score",
              "land_type_base", "usage_priority"]].to_string())