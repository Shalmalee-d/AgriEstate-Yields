"""
AgriEstate-Yields ML Module.

This module provides agricultural suitability assessment, crop recommendation,
and yield prediction capabilities for candidate land evaluation.

Architecture:
- preprocessing: Data loading, validation, feature engineering
- suitability: Rule-based agricultural suitability scoring (interpretable, not ML)
- recommendation: Crop matching framework (requires environmental data for production)
- model: RandomForestRegressor pipeline for yield prediction (needs labeled data)
- prediction: Clean interface for backend integration

Current Dataset: candidate_lands.csv (15 records, NO yield labels)
"""

from .preprocessing import (
    load_candidate_lands,
    preprocess_candidate_lands,
    validate_columns,
    validate_categorical_values,
    validate_numeric_ranges,
    handle_missing_values,
    encode_categorical,
    add_derived_features,
    get_feature_columns,
    get_categorical_columns,
    REQUIRED_COLUMNS,
    VALID_LAND_TYPES,
    VALID_CURRENT_USAGE,
    VALID_WATER_ACCESS,
    NUMERIC_RANGES,
)

from .suitability import (
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
)

from .recommendation import (
    recommend_crops_for_land,
    recommend_for_all_lands,
    recommendations_to_dataframe,
    CropRequirements,
    PLACEHOLDER_CROPS,
    score_crop_for_land,
)

from .model import (
    train_yield_model,
    load_model,
    save_model,
    predict_yield,
    predict_yield_with_uncertainty,
    evaluate_model,
    check_training_data_exists,
    train_if_data_available,
    ModelMetadata,
    EXPECTED_NUMERIC_FEATURES,
    EXPECTED_CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)

from .prediction import (
    predict_yield_interface,
    assess_suitability_interface,
    get_all_suitability,
    get_suitability_summary_interface,
    recommend_crops_interface,
    get_all_recommendations,
    get_model_status,
    reload_model,
    batch_suitability_csv,
    batch_recommendations_csv,
    PredictionRequest,
    PredictionResponse,
    SuitabilityRequest,
    SuitabilityResponse,
    CropRecommendationRequest,
    CropRecommendationResponse,
)

__version__ = "0.1.0"

__all__ = [
    # Preprocessing
    "load_candidate_lands",
    "preprocess_candidate_lands",
    "validate_columns",
    "validate_categorical_values",
    "validate_numeric_ranges",
    "handle_missing_values",
    "encode_categorical",
    "add_derived_features",
    "get_feature_columns",
    "get_categorical_columns",
    "REQUIRED_COLUMNS",
    "VALID_LAND_TYPES",
    "VALID_CURRENT_USAGE",
    "VALID_WATER_ACCESS",
    "NUMERIC_RANGES",

    # Suitability
    "assess_suitability",
    "assess_all_lands",
    "suitability_to_dataframe",
    "filter_by_usage",
    "get_suitability_summary",
    "SuitabilityResult",
    "WEIGHTS",
    "THRESHOLDS",
    "compute_area_score",
    "compute_sunlight_score",
    "compute_water_score",
    "compute_market_score",
    "compute_land_type_score",
    "compute_usage_score",

    # Recommendation
    "recommend_crops_for_land",
    "recommend_for_all_lands",
    "recommendations_to_dataframe",
    "CropRequirements",
    "PLACEHOLDER_CROPS",
    "score_crop_for_land",

    # Model
    "train_yield_model",
    "load_model",
    "save_model",
    "predict_yield",
    "predict_yield_with_uncertainty",
    "evaluate_model",
    "check_training_data_exists",
    "train_if_data_available",
    "ModelMetadata",
    "EXPECTED_NUMERIC_FEATURES",
    "EXPECTED_CATEGORICAL_FEATURES",
    "TARGET_COLUMN",

    # Prediction Interface
    "predict_yield_interface",
    "assess_suitability_interface",
    "get_all_suitability",
    "get_suitability_summary_interface",
    "recommend_crops_interface",
    "get_all_recommendations",
    "get_model_status",
    "reload_model",
    "batch_suitability_csv",
    "batch_recommendations_csv",
    "PredictionRequest",
    "PredictionResponse",
    "SuitabilityRequest",
    "SuitabilityResponse",
    "CropRecommendationRequest",
    "CropRecommendationResponse",
]