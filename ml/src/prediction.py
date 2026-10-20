"""
Prediction Interface Module.

This module provides a clean, backend-agnostic interface for yield prediction
and suitability assessment. The FastAPI backend (or any other consumer) can
import and call these functions directly.

No FastAPI dependencies in this module.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from .preprocessing import (preprocess_candidate_lands, load_candidate_lands,
                              preprocess_candidate_lands_with_environment,
                              load_environmental_data, join_environmental_data,
                              get_default_candidate_lands_path, get_default_environmental_data_path)
from .suitability import (assess_all_lands, assess_suitability, SuitabilityResult,
                          suitability_to_dataframe, filter_by_usage,
                          get_suitability_summary)
from .recommendation import (recommend_for_all_lands, recommend_crops_for_land,
                              recommendations_to_dataframe, CropRequirements,
                              PLACEHOLDER_CROPS)
from .model import (load_model, predict_yield, predict_yield_with_uncertainty,
                    check_training_data_exists, ModelMetadata, Pipeline)


# Global model cache (loaded on first use)
_cached_model: Optional[Pipeline] = None
_cached_metadata: Optional[ModelMetadata] = None
_model_load_error: Optional[str] = None


@dataclass
class PredictionRequest:
    """Request for yield prediction (when model is available)."""
    # Required features for yield model (subset - full list in model.py)
    soil_ph: Optional[float] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    rainfall_mm: Optional[float] = None
    temperature_avg: Optional[float] = None
    humidity_avg: Optional[float] = None
    sunlight_hours: Optional[float] = None
    water_availability_mm: Optional[float] = None
    area_m2: Optional[float] = None
    crop_type: Optional[str] = None
    soil_type: Optional[str] = None
    irrigation_type: Optional[str] = None
    season: Optional[str] = None


@dataclass
class PredictionResponse:
    """Response from yield prediction."""
    predicted_yield: Optional[float] = None  # kg/ha or kg/m2 (depends on training target)
    uncertainty_std: Optional[float] = None
    model_metadata: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class SuitabilityRequest:
    """Request for suitability assessment (can override CSV)."""
    land_id: str
    name: str
    land_type: str  # "Rooftop" or "Plot"
    current_usage: str  # "Vacant", "Underutilized", "Existing Garden"
    area_sqft: float
    sunlight_hours: float
    water_access: str  # "High", "Medium", "Low"
    market_distance_km: float


@dataclass
class SuitabilityResponse:
    """Response from suitability assessment."""
    land_id: str
    name: str
    suitability_score: float
    suitability_status: str
    factor_scores: Dict[str, float]
    limiting_factors: List[str]
    usage_flag: str
    area_m2: float


@dataclass
class CropRecommendationRequest:
    """Request for crop recommendations."""
    land_id: str
    # Optional environmental data (improves recommendations if available)
    soil_ph: Optional[float] = None
    temperature_avg: Optional[float] = None
    rainfall_mm: Optional[float] = None
    top_n: int = 3


@dataclass
class CropRecommendationResponse:
    """Response with crop recommendations."""
    land_id: str
    land_name: str
    suitability_score: float
    suitability_status: str
    usage_flag: str
    area_m2: float
    recommendations: List[Dict[str, Any]]
    data_gaps: List[str]
    note: str = ""  # Overall note about recommendation validity


def _load_model_if_needed() -> tuple[Optional[Pipeline], Optional[ModelMetadata]]:
    """
    Load model from disk if not already cached.

    Returns:
        Tuple of (pipeline, metadata) or (None, None) if not available.
    """
    global _cached_model, _cached_metadata, _model_load_error

    if _cached_model is not None or _model_load_error is not None:
        return _cached_model, _cached_metadata

    model_path = check_training_data_exists()
    if model_path is None:
        _model_load_error = "No trained model found. Train model first with labeled yield data."
        return None, None

    # Look for saved model in models directory
    base_dir = Path(__file__).parent.parent.parent
    model_dir = base_dir / "models"
    model_file = model_dir / "yield_rf_model.joblib"

    if not model_file.exists():
        _model_load_error = f"Model file not found at {model_file}. Run training first."
        return None, None

    try:
        _cached_model, _cached_metadata = load_model(str(model_file))
        return _cached_model, _cached_metadata
    except Exception as e:
        _model_load_error = f"Failed to load model: {e}"
        return None, None


def predict_yield_interface(request: PredictionRequest) -> PredictionResponse:
    """
    Predict yield for given environmental/agricultural features.

    This is the main interface for the FastAPI backend.

    Args:
        request: PredictionRequest with available features.

    Returns:
        PredictionResponse with predicted yield and metadata.
    """
    warnings_list = []

    # Try to load model
    pipeline, metadata = _load_model_if_needed()

    if pipeline is None:
        return PredictionResponse(
            error=_model_load_error or "Model not available",
            warnings=["Yield prediction requires a trained model with labeled historical data. "
                      "Current candidate_lands.csv does not contain yield labels."]
        )

    # Check which features are provided
    provided_features = {k: v for k, v in asdict(request).items() if v is not None}
    missing_features = [f for f in metadata.features if f not in provided_features]

    if missing_features:
        warnings_list.append(f"Missing features (will use defaults/imputation): {missing_features}")

    # Convert to DataFrame
    features_df = pd.DataFrame([provided_features])

    # Predict
    try:
        pred, std = predict_yield_with_uncertainty(pipeline, features_df)
        return PredictionResponse(
            predicted_yield=float(pred[0]),
            uncertainty_std=float(std[0]),
            model_metadata=asdict(metadata) if metadata else None,
            warnings=warnings_list,
        )
    except Exception as e:
        return PredictionResponse(
            error=f"Prediction failed: {e}",
            warnings=warnings_list,
        )


def assess_suitability_interface(request: SuitabilityRequest) -> SuitabilityResponse:
    """
    Assess agricultural suitability for a single land parcel.

    This uses the rule-based suitability engine (not ML).

    Args:
        request: SuitabilityRequest with land features.

    Returns:
        SuitabilityResponse with score and factor breakdown.
    """
    # Create a single-row DataFrame matching preprocessing expectations
    df = pd.DataFrame([{
        "land_id": request.land_id,
        "name": request.name,
        "land_type": request.land_type,
        "current_usage": request.current_usage,
        "area_sqft": request.area_sqft,
        "sunlight_hours": request.sunlight_hours,
        "water_access": request.water_access,
        "market_distance_km": request.market_distance_km,
    }])

    # Run through preprocessing
    from .preprocessing import (validate_columns, handle_missing_values,
                                 validate_categorical_values, validate_numeric_ranges,
                                 encode_categorical, add_derived_features)

    validate_columns(df)
    df = handle_missing_values(df)
    df = validate_categorical_values(df)
    df = validate_numeric_ranges(df)
    df = encode_categorical(df)
    df = add_derived_features(df)

    # Assess suitability
    result = assess_suitability(df.iloc[0])

    return SuitabilityResponse(
        land_id=result.land_id,
        name=result.name,
        suitability_score=result.suitability_score,
        suitability_status=result.suitability_status,
        factor_scores=result.factor_scores,
        limiting_factors=result.limiting_factors,
        usage_flag=result.usage_flag,
        area_m2=result.area_m2,
    )


def get_all_suitability(csv_path: Optional[str] = None,
                        include_existing_garden: bool = False) -> List[SuitabilityResponse]:
    """
    Get suitability assessment for all candidate lands.

    Args:
        csv_path: Optional custom CSV path.
        include_existing_garden: Whether to include 'Existing Garden' lands.

    Returns:
        List of SuitabilityResponse sorted by score descending.
    """
    results = assess_all_lands(csv_path)
    if not include_existing_garden:
        results = filter_by_usage(results, include_existing_garden=False)

    return [
        SuitabilityResponse(
            land_id=r.land_id,
            name=r.name,
            suitability_score=r.suitability_score,
            suitability_status=r.suitability_status,
            factor_scores=r.factor_scores,
            limiting_factors=r.limiting_factors,
            usage_flag=r.usage_flag,
            area_m2=r.area_m2,
        )
        for r in results
    ]


def get_suitability_summary_interface(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get summary statistics for all candidate lands.

    Args:
        csv_path: Optional custom CSV path.

    Returns:
        Dict with summary statistics.
    """
    results = assess_all_lands(csv_path)
    return get_suitability_summary(results)


def recommend_crops_interface(request: CropRecommendationRequest, csv_path: Optional[str] = None) -> CropRecommendationResponse:
    """
    Get crop recommendations for a specific land parcel.

    Args:
        request: CropRecommendationRequest with land_id and optional environmental data.
        csv_path: Optional custom CSV path. If None, uses auto-detected default
                  (prefers Pune prototype if available).

    Returns:
        CropRecommendationResponse with ranked crop recommendations.
        Note: If required environmental data (soil pH, temperature, rainfall) is missing,
        recommendations will have status="insufficient_data" with composite_score=None,
        estimated_yield_kg=None, estimated_revenue_usd=None.
    """
    # Use auto-detected default datasets (prefers Pune prototype if available)
    if csv_path is None:
        candidate_csv_path = str(get_default_candidate_lands_path())
    else:
        candidate_csv_path = csv_path
    
    # Find the land in the dataset
    all_results = assess_all_lands(candidate_csv_path)
    land_result = next((r for r in all_results if r.land_id == request.land_id), None)

    if land_result is None:
        return CropRecommendationResponse(
            land_id=request.land_id,
            land_name="Unknown",
            suitability_score=0,
            suitability_status="Not Found",
            usage_flag="unknown",
            area_m2=0,
            recommendations=[],
            data_gaps=["land_id not found in candidate dataset"],
            note="Land not found in candidate dataset.",
        )

    # Check if environmental data is available for this land
    # Try to load environmental data from the default path
    soil_ph = request.soil_ph
    temperature_avg = request.temperature_avg
    rainfall_mm = request.rainfall_mm

    # If not provided in request, try to load from environmental dataset
    if soil_ph is None or temperature_avg is None or rainfall_mm is None:
        try:
            env_path = get_default_environmental_data_path()
            if env_path is not None:
                env_df = load_environmental_data(str(env_path))
                env_row = env_df[env_df["land_id"] == request.land_id]
                if not env_row.empty:
                    if soil_ph is None:
                        soil_ph = float(env_row.iloc[0]["soil_ph"])
                    if temperature_avg is None:
                        temperature_avg = float(env_row.iloc[0]["temperature_avg_c"])
                    if rainfall_mm is None:
                        rainfall_mm = float(env_row.iloc[0]["annual_rainfall_mm"])
        except Exception:
            # Environmental data not available or error loading - will use request values or defaults
            pass

    # Get recommendations
    recs = recommend_crops_for_land(
        land_result,
        soil_ph=soil_ph,
        temperature_avg=temperature_avg,
        rainfall_mm=rainfall_mm,
        top_n=request.top_n,
    )

    # Collect all data gaps
    all_gaps = set()
    has_insufficient_data = False
    for r in recs:
        all_gaps.update(r["data_gaps"])
        if r.get("status") == "insufficient_data":
            has_insufficient_data = True

    note = ""
    if has_insufficient_data:
        note = ("Crop recommendations require soil pH, temperature, and rainfall data. "
                "These are not available in the current candidate-land dataset. "
                "Estimated yield and revenue are NOT provided without complete environmental data. "
                "Recommendations are marked as 'insufficient_data'.")

    return CropRecommendationResponse(
        land_id=land_result.land_id,
        land_name=land_result.name,
        suitability_score=land_result.suitability_score,
        suitability_status=land_result.suitability_status,
        usage_flag=land_result.usage_flag,
        area_m2=land_result.area_m2,
        recommendations=recs,
        data_gaps=sorted(list(all_gaps)),
        note=note,
    )


def get_all_recommendations(csv_path: Optional[str] = None,
                            top_n: int = 3,
                            environmental_csv_path: Optional[str] = None) -> List[CropRecommendationResponse]:
    """
    Get crop recommendations for all available lands.

    Args:
        csv_path: Optional custom CSV path. If None, uses auto-detected default
                  (prefers Pune prototype if it exists).
        top_n: Number of recommendations per land.
        environmental_csv_path: Optional custom path to environmental data CSV.
                                If None, uses auto-detected default (Pune prototype if it exists).

    Returns:
        List of CropRecommendationResponse.
        Note: If required environmental data (soil pH, temperature, rainfall) is missing,
        recommendations will have status="insufficient_data" with composite_score=None,
        estimated_yield_kg=None, estimated_revenue_usd=None.
    """
    # Use auto-detected defaults if not explicitly provided
    if csv_path is None:
        csv_path = str(get_default_candidate_lands_path())
    if environmental_csv_path is None:
        default_env_path = get_default_environmental_data_path()
        if default_env_path is not None:
            environmental_csv_path = str(default_env_path)

    all_recs = recommend_for_all_lands(csv_path, top_n=top_n, environmental_csv_path=environmental_csv_path)
    responses = []

    for land_id, data in all_recs.items():
        all_gaps = set()
        has_insufficient_data = False
        for r in data["recommendations"]:
            all_gaps.update(r["data_gaps"])
            if r.get("status") == "insufficient_data":
                has_insufficient_data = True

        note = ""
        if has_insufficient_data:
            note = ("Crop recommendations require soil pH, temperature, and rainfall data. "
                    "These are not available in the current candidate-land dataset. "
                    "Estimated yield and revenue are NOT provided without complete environmental data. "
                    "Recommendations are marked as 'insufficient_data'.")

        responses.append(CropRecommendationResponse(
            land_id=land_id,
            land_name=data["land_name"],
            suitability_score=data["suitability_score"],
            suitability_status=data["suitability_status"],
            usage_flag=data["usage_flag"],
            area_m2=data["area_m2"],
            recommendations=data["recommendations"],
            data_gaps=sorted(list(all_gaps)),
            note=note,
        ))

    return responses


def get_model_status() -> Dict[str, Any]:
    """
    Get status of the yield prediction model.

    Returns:
        Dict with model availability and metadata.
    """
    pipeline, metadata = _load_model_if_needed()

    if pipeline is None:
        return {
            "model_available": False,
            "error": _model_load_error,
            "message": "Place a labeled yield dataset in ml/data/ and run training.",
        }

    return {
        "model_available": True,
        "metadata": asdict(metadata) if metadata else None,
    }


def reload_model() -> Dict[str, Any]:
    """
    Force reload of the model (useful after retraining).

    Returns:
        Updated model status.
    """
    global _cached_model, _cached_metadata, _model_load_error
    _cached_model = None
    _cached_metadata = None
    _model_load_error = None
    return get_model_status()


# Convenience functions for direct DataFrame operations (batch processing)

def batch_suitability_csv(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Run suitability assessment on a CSV and optionally save results.

    Args:
        csv_path: Input CSV path.
        output_path: Optional output CSV path.

    Returns:
        DataFrame with suitability results.
    """
    results = assess_all_lands(csv_path)
    df = suitability_to_dataframe(results)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def batch_recommendations_csv(csv_path: str, output_path: Optional[str] = None,
                              top_n: int = 3) -> pd.DataFrame:
    """
    Run crop recommendations on a CSV and optionally save results.

    Args:
        csv_path: Input CSV path.
        output_path: Optional output CSV path.
        top_n: Recommendations per land.

    Returns:
        DataFrame with recommendations.
    """
    recs = recommend_for_all_lands(csv_path, top_n=top_n)
    df = recommendations_to_dataframe(recs)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    # Demo usage
    print("=== Suitability for all lands ===")
    suit = get_all_suitability()
    for s in suit:
        print(f"{s.land_id}: {s.name} | Score: {s.suitability_score:.3f} | {s.suitability_status} | {s.usage_flag}")

    print("\n=== Crop recommendations (top 2) ===")
    recs = get_all_recommendations(top_n=2)
    for r in recs:
        print(f"\n{r.land_id}: {r.land_name}")
        for rec in r.recommendations:
            print(f"  {rec['crop']}: {rec['composite_score']:.3f} | ~{rec['estimated_yield_kg']:.0f}kg | ${rec['estimated_revenue_usd']:.0f}")

    print("\n=== Model status ===")
    print(get_model_status())