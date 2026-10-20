"""
Crop Recommendation Module.

This module provides a framework for crop recommendations based on:
- Land suitability assessment
- Soil conditions (future)
- Climate data (future)
- Water availability
- Sunlight exposure
- Crop-specific requirements

CURRENT STATUS: PROTOTYPE FRAMEWORK ONLY
=========================================

This prototype does NOT contain scientifically validated crop thresholds.
The crop database and matching logic are placeholders for future development
when proper agronomic data becomes available.

FUTURE REQUIREMENTS
===================

To implement production-ready crop recommendations, the following data is needed:

1. CROP REQUIREMENTS DATABASE (per crop):
   - Optimal pH range
   - N/P/K requirements (kg/ha)
   - Water requirement (mm/season)
   - Temperature range (min/opt/max °C)
   - Sunlight requirement (hours/day)
   - Growing season length (days)
   - Soil type preference
   - Salinity tolerance
   - Disease resistance profile

2. LAND ENVIRONMENTAL DATA (per land):
   - Soil pH (measured)
   - Soil N/P/K (measured)
   - Soil texture/type
   - Historical rainfall (mm)
   - Temperature range (min/max/avg)
   - Humidity
   - Water availability (irrigation capacity)
   - Sunlight hours (measured)
   - Frost dates
   - Elevation

3. MATCHING METHODOLOGY:
   - Suitability index per crop (multi-criteria decision analysis)
   - Economic viability (market price × yield - cost)
   - Rotation compatibility
   - Risk assessment (climate, pest, market)

PROTOTYPE IMPLEMENTATION
========================

The current implementation provides:
- A CropRequirements dataclass structure
- A placeholder crop database with 5 common crops
- A basic matching function that scores crops against land features
- Clear documentation that thresholds are NOT validated

DO NOT USE FOR PRODUCTION DECISIONS.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from .suitability import SuitabilityResult, assess_all_lands
from .preprocessing import load_environmental_data, get_default_environmental_data_path


@dataclass
class CropRequirements:
    """
    Crop environmental and nutritional requirements.

    NOTE: Values below are PLACEHOLDERS for prototype structure only.
    They are NOT scientifically validated and should NOT be used for
    actual farming decisions.

    Source for future validation: FAO crop ecological requirements,
    USDA PLANTS database, local agricultural extension services.
    """
    name: str
    # Sunlight (hours/day)
    min_sunlight_hours: float
    optimal_sunlight_hours: float
    # Water (mm per growing season)
    min_water_mm: float
    optimal_water_mm: float
    # Temperature (°C)
    min_temp: float
    optimal_temp_min: float
    optimal_temp_max: float
    max_temp: float
    # Soil pH
    min_ph: float
    optimal_ph_min: float
    optimal_ph_max: float
    max_ph: float
    # Nutrients (kg/ha per season)
    n_requirement: float
    p_requirement: float
    k_requirement: float
    # Growing season (days)
    growing_season_days: int
    # Market info (placeholder)
    market_price_per_kg: float  # USD
    typical_yield_kg_per_m2: float


# PLACEHOLDER CROP DATABASE - NOT VALIDATED
# DO NOT USE FOR REAL DECISIONS
PLACEHOLDER_CROPS = [
    CropRequirements(
        name="Lettuce",
        min_sunlight_hours=4.0,
        optimal_sunlight_hours=6.0,
        min_water_mm=300,
        optimal_water_mm=450,
        min_temp=5.0,
        optimal_temp_min=15.0,
        optimal_temp_max=20.0,
        max_temp=25.0,
        min_ph=6.0,
        optimal_ph_min=6.2,
        optimal_ph_max=6.8,
        max_ph=7.0,
        n_requirement=120,
        p_requirement=50,
        k_requirement=150,
        growing_season_days=45,
        market_price_per_kg=3.50,
        typical_yield_kg_per_m2=3.0,
    ),
    CropRequirements(
        name="Tomato",
        min_sunlight_hours=6.0,
        optimal_sunlight_hours=8.0,
        min_water_mm=400,
        optimal_water_mm=600,
        min_temp=10.0,
        optimal_temp_min=20.0,
        optimal_temp_max=27.0,
        max_temp=35.0,
        min_ph=5.5,
        optimal_ph_min=6.0,
        optimal_ph_max=6.8,
        max_ph=7.5,
        n_requirement=200,
        p_requirement=80,
        k_requirement=250,
        growing_season_days=90,
        market_price_per_kg=2.80,
        typical_yield_kg_per_m2=5.0,
    ),
    CropRequirements(
        name="Spinach",
        min_sunlight_hours=3.0,
        optimal_sunlight_hours=5.0,
        min_water_mm=250,
        optimal_water_mm=400,
        min_temp=2.0,
        optimal_temp_min=10.0,
        optimal_temp_max=20.0,
        max_temp=25.0,
        min_ph=6.0,
        optimal_ph_min=6.5,
        optimal_ph_max=7.0,
        max_ph=7.5,
        n_requirement=100,
        p_requirement=40,
        k_requirement=120,
        growing_season_days=40,
        market_price_per_kg=4.00,
        typical_yield_kg_per_m2=2.5,
    ),
    CropRequirements(
        name="Kale",
        min_sunlight_hours=4.0,
        optimal_sunlight_hours=6.0,
        min_water_mm=300,
        optimal_water_mm=450,
        min_temp=5.0,
        optimal_temp_min=10.0,
        optimal_temp_max=20.0,
        max_temp=28.0,
        min_ph=6.0,
        optimal_ph_min=6.5,
        optimal_ph_max=7.0,
        max_ph=7.5,
        n_requirement=150,
        p_requirement=60,
        k_requirement=180,
        growing_season_days=60,
        market_price_per_kg=3.20,
        typical_yield_kg_per_m2=3.5,
    ),
    CropRequirements(
        name="Herbs (Basil/Mint)",
        min_sunlight_hours=5.0,
        optimal_sunlight_hours=7.0,
        min_water_mm=300,
        optimal_water_mm=450,
        min_temp=10.0,
        optimal_temp_min=18.0,
        optimal_temp_max=25.0,
        max_temp=30.0,
        min_ph=6.0,
        optimal_ph_min=6.5,
        optimal_ph_max=7.0,
        max_ph=7.5,
        n_requirement=80,
        p_requirement=30,
        k_requirement=100,
        growing_season_days=50,
        market_price_per_kg=15.00,
        typical_yield_kg_per_m2=1.5,
    ),
]


def score_crop_for_land(crop: CropRequirements, land: SuitabilityResult,
                        soil_ph: Optional[float] = None,
                        temperature_avg: Optional[float] = None,
                        rainfall_mm: Optional[float] = None) -> dict:
    """
    Score a crop's suitability for a specific land parcel.

    This is a SIMPLIFIED PLACEHOLDER scoring function.
    Production version should use multi-criteria decision analysis
    with validated agronomic models.

    IMPORTANT: This function requires soil pH, temperature, and rainfall data
    to produce scientifically valid crop recommendations. The current
    candidate-land dataset does NOT contain these measurements.

    Args:
        crop: CropRequirements for the crop.
        land: SuitabilityResult for the land.
        soil_ph: Measured soil pH (None if unavailable).
        temperature_avg: Average growing season temperature (None if unavailable).
        rainfall_mm: Seasonal rainfall (None if unavailable).

    Returns:
        Dict with score breakdown. If required environmental data is missing,
        returns status="insufficient_data" with composite_score=None,
        estimated_yield_kg=None, estimated_revenue_usd=None.
    """
    # Check for required environmental data
    required_env_data = {
        "soil_ph": soil_ph,
        "temperature": temperature_avg,
        "rainfall": rainfall_mm,
    }
    missing_env = [k for k, v in required_env_data.items() if v is None]

    # If any required environmental data is missing, return insufficient_data status
    if missing_env:
        scores = {}
        weights = {}

        # Sunlight score (available from land data)
        sunlight_hours = land.factor_scores.get("sunlight", 0) * 12  # reverse normalize
        if sunlight_hours >= crop.optimal_sunlight_hours:
            scores["sunlight"] = 1.0
        elif sunlight_hours >= crop.min_sunlight_hours:
            scores["sunlight"] = (sunlight_hours - crop.min_sunlight_hours) / \
                                 (crop.optimal_sunlight_hours - crop.min_sunlight_hours)
        else:
            scores["sunlight"] = 0.0
        weights["sunlight"] = 0.25

        # Water access score (proxy from land data)
        water_factor = land.factor_scores.get("water", 0)  # 0=Low, 0.5=Med, 1=High
        approx_water_mm = 200 + water_factor * 400  # 200-600 mm range
        if approx_water_mm >= crop.optimal_water_mm:
            scores["water"] = 1.0
        elif approx_water_mm >= crop.min_water_mm:
            scores["water"] = (approx_water_mm - crop.min_water_mm) / \
                              (crop.optimal_water_mm - crop.min_water_mm)
        else:
            scores["water"] = 0.0
        weights["water"] = 0.20

        # Land suitability score (from our assessment) - normalize from 0-100 to 0-1
        scores["land_suitability"] = land.suitability_score / 100.0
        weights["land_suitability"] = 0.20

        # Area adequacy (minimum area for viable crop)
        min_area_m2 = 50  # placeholder minimum
        scores["area"] = 1.0 if land.area_m2 >= min_area_m2 else land.area_m2 / min_area_m2
        weights["area"] = 0.0  # Not weighted in composite, just a gate

        # Note: temperature, ph, rainfall are NOT scored because data is missing
        # They are explicitly listed in data_gaps

        active_weights = {k: v for k, v in weights.items() if k != "area"}
        total_weight = sum(active_weights.values())
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

        # Compute composite with only available factors (sunlight, water, land_suitability)
        composite_score = sum(scores[k] * normalized_weights[k] for k in normalized_weights)

        return {
            "crop": crop.name,
            "status": "insufficient_data",
            "composite_score": None,
            "factor_scores": {k: round(v, 4) for k, v in scores.items()},
            "weights_used": {k: round(v, 4) for k, v in normalized_weights.items()},
            "data_gaps": missing_env,
            "estimated_yield_kg": None,
            "estimated_revenue_usd": None,
            "note": "Crop recommendation requires soil pH, temperature, and rainfall data. "
                    "These are not available in the current candidate-land dataset. "
                    "Estimated yield and revenue are NOT provided without complete environmental data.",
        }

    # All required environmental data is present - compute full scoring
    scores = {}
    weights = {}

    # Sunlight score (available from land data)
    sunlight_hours = land.factor_scores.get("sunlight", 0) * 12  # reverse normalize
    if sunlight_hours >= crop.optimal_sunlight_hours:
        scores["sunlight"] = 1.0
    elif sunlight_hours >= crop.min_sunlight_hours:
        scores["sunlight"] = (sunlight_hours - crop.min_sunlight_hours) / \
                             (crop.optimal_sunlight_hours - crop.min_sunlight_hours)
    else:
        scores["sunlight"] = 0.0
    weights["sunlight"] = 0.25

    # Water access score (proxy from land data)
    water_factor = land.factor_scores.get("water", 0)  # 0=Low, 0.5=Med, 1=High
    approx_water_mm = 200 + water_factor * 400  # 200-600 mm range
    if approx_water_mm >= crop.optimal_water_mm:
        scores["water"] = 1.0
    elif approx_water_mm >= crop.min_water_mm:
        scores["water"] = (approx_water_mm - crop.min_water_mm) / \
                          (crop.optimal_water_mm - crop.min_water_mm)
    else:
        scores["water"] = 0.0
    weights["water"] = 0.20

    # Temperature score
    if crop.optimal_temp_min <= temperature_avg <= crop.optimal_temp_max:
        scores["temperature"] = 1.0
    elif crop.min_temp <= temperature_avg <= crop.max_temp:
        if temperature_avg < crop.optimal_temp_min:
            scores["temperature"] = (temperature_avg - crop.min_temp) / \
                                    (crop.optimal_temp_min - crop.min_temp)
        else:
            scores["temperature"] = (crop.max_temp - temperature_avg) / \
                                    (crop.max_temp - crop.optimal_temp_max)
    else:
        scores["temperature"] = 0.0
    weights["temperature"] = 0.20

    # Soil pH score
    if crop.optimal_ph_min <= soil_ph <= crop.optimal_ph_max:
        scores["ph"] = 1.0
    elif crop.min_ph <= soil_ph <= crop.max_ph:
        if soil_ph < crop.optimal_ph_min:
            scores["ph"] = (soil_ph - crop.min_ph) / (crop.optimal_ph_min - crop.min_ph)
        else:
            scores["ph"] = (crop.max_ph - soil_ph) / (crop.max_ph - crop.optimal_ph_max)
    else:
        scores["ph"] = 0.0
    weights["ph"] = 0.15

    # Land suitability score (from our assessment) - normalize from 0-100 to 0-1 for composite
    scores["land_suitability"] = land.suitability_score / 100.0
    weights["land_suitability"] = 0.20

    # Area adequacy (minimum area for viable crop)
    min_area_m2 = 50  # placeholder minimum
    scores["area"] = 1.0 if land.area_m2 >= min_area_m2 else land.area_m2 / min_area_m2
    weights["area"] = 0.0  # Not weighted in composite, just a gate

    # Weighted composite score (excluding area gate)
    active_weights = {k: v for k, v in weights.items() if k != "area"}
    total_weight = sum(active_weights.values())
    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

    composite_score = sum(scores[k] * normalized_weights[k] for k in normalized_weights)

    return {
        "crop": crop.name,
        "status": "scored",
        "composite_score": round(composite_score, 4),
        "factor_scores": {k: round(v, 4) for k, v in scores.items()},
        "weights_used": {k: round(v, 4) for k, v in normalized_weights.items()},
        "data_gaps": [],
        "estimated_yield_kg": round(crop.typical_yield_kg_per_m2 * land.area_m2 * composite_score, 2),
        "estimated_revenue_usd": round(crop.typical_yield_kg_per_m2 * land.area_m2 * composite_score * crop.market_price_per_kg, 2),
    }


def recommend_crops_for_land(land,
                             crops: Optional[list[CropRequirements]] = None,
                             soil_ph: Optional[float] = None,
                             temperature_avg: Optional[float] = None,
                             rainfall_mm: Optional[float] = None,
                             top_n: int = 3) -> list[dict]:
    """
    Recommend top N crops for a land parcel.

    Args:
        land: SuitabilityResult for the land, OR land_id string (e.g., "LAND007").
              If string, looks up the SuitabilityResult from assess_all_lands()
              using the auto-detected default dataset.
        crops: List of crops to evaluate. Uses PLACEHOLDER_CROPS if None.
        soil_ph: Measured soil pH.
        temperature_avg: Average growing season temperature.
        rainfall_mm: Seasonal rainfall.
        top_n: Number of top recommendations to return.

    Returns:
        List of recommendation dicts. Scored recommendations (with all env data)
        come first, sorted by composite_score descending. Insufficient data
        recommendations come after.
    """
    # If land is a string (land_id), look up the SuitabilityResult
    if isinstance(land, str):
        all_results = assess_all_lands()
        land_result = next((r for r in all_results if r.land_id == land), None)
        if land_result is None:
            return [{
                "crop": "Unknown",
                "status": "error",
                "composite_score": None,
                "factor_scores": {},
                "weights_used": {},
                "data_gaps": [f"land_id '{land}' not found in candidate dataset"],
                "estimated_yield_kg": None,
                "estimated_revenue_usd": None,
                "note": f"Land '{land}' not found in candidate dataset"
            }]
        land = land_result
    
    if crops is None:
        crops = PLACEHOLDER_CROPS

    recommendations = []
    for crop in crops:
        result = score_crop_for_land(crop, land, soil_ph, temperature_avg, rainfall_mm)
        recommendations.append(result)

    # Sort: scored recommendations first (by composite_score desc),
    # then insufficient_data recommendations
    def sort_key(x):
        if x.get("status") == "scored":
            return (0, -x["composite_score"])
        return (1, 0)

    recommendations.sort(key=sort_key)
    return recommendations[:top_n]


def recommend_for_all_lands(csv_path: Optional[str] = None,
                            crops: Optional[list[CropRequirements]] = None,
                            top_n: int = 3,
                            environmental_csv_path: Optional[str] = None) -> dict:
    """
    Generate crop recommendations for all suitable lands.

    Args:
        csv_path: Path to candidate lands CSV. If None, uses auto-detected default
                  (prefers Pune prototype if it exists).
        crops: List of crops to evaluate.
        top_n: Number of recommendations per land.
        environmental_csv_path: Optional path to environmental data CSV.
                                If None, uses auto-detected default (Pune prototype if it exists).
                                If provided, environmental data will be loaded and used.

    Returns:
        Dict mapping land_id to recommendation results.
    """
    # Use auto-detected default if no explicit path provided
    if csv_path is None:
        csv_path = str(get_default_candidate_lands_path())
    
    suitability_results = assess_all_lands(csv_path)
    # Filter to only available/repurpose lands (exclude existing gardens by default)
    available_lands = [r for r in suitability_results if r.usage_flag in ("available", "repurpose")]

    # Load environmental data if provided or use auto-detected default
    environmental_data = {}
    env_path = environmental_csv_path
    if env_path is None:
        default_env_path = get_default_environmental_data_path()
        if default_env_path is not None:
            env_path = str(default_env_path)
    
    if env_path is not None:
        try:
            env_df = load_environmental_data(env_path)
            for _, row in env_df.iterrows():
                environmental_data[row["land_id"]] = {
                    "soil_ph": float(row["soil_ph"]),
                    "temperature_avg": float(row["temperature_avg_c"]),
                    "rainfall_mm": float(row["annual_rainfall_mm"]),
                }
        except Exception:
            pass

    all_recommendations = {}
    for land in available_lands:
        env_data = environmental_data.get(land.land_id, {})
        recs = recommend_crops_for_land(
            land,
            crops,
            soil_ph=env_data.get("soil_ph"),
            temperature_avg=env_data.get("temperature_avg"),
            rainfall_mm=env_data.get("rainfall_mm"),
            top_n=top_n,
        )
        all_recommendations[land.land_id] = {
            "land_name": land.name,
            "suitability_score": land.suitability_score,
            "suitability_status": land.suitability_status,
            "usage_flag": land.usage_flag,
            "area_m2": land.area_m2,
            "recommendations": recs,
        }

    return all_recommendations


def recommendations_to_dataframe(recommendations: dict) -> pd.DataFrame:
    """
    Flatten recommendations dict to DataFrame.

    Args:
        recommendations: Output from recommend_for_all_lands.

    Returns:
        DataFrame with one row per land-crop recommendation.
    """
    rows = []
    for land_id, data in recommendations.items():
        for rec in data["recommendations"]:
            row = {
                "land_id": land_id,
                "land_name": data["land_name"],
                "suitability_score": data["suitability_score"],
                "suitability_status": data["suitability_status"],
                "usage_flag": data["usage_flag"],
                "area_m2": data["area_m2"],
                "crop": rec["crop"],
                "crop_match_score": rec["composite_score"],
                "estimated_yield_kg": rec["estimated_yield_kg"],
                "estimated_revenue_usd": rec["estimated_revenue_usd"],
                "data_gaps": "; ".join(rec["data_gaps"]),
            }
            # Add factor scores
            for factor, score in rec["factor_scores"].items():
                row[f"crop_factor_{factor}"] = score
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test
    results = assess_all_lands()
    for land in results[:3]:
        print(f"\n--- {land.land_id}: {land.name} (Score: {land.suitability_score}) ---")
        recs = recommend_crops_for_land(land, top_n=3)
        for r in recs:
            print(f"  {r['crop']}: {r['composite_score']:.3f} | Yield: {r['estimated_yield_kg']:.1f}kg | Rev: ${r['estimated_revenue_usd']:.2f}")
            print(f"    Gaps: {r['data_gaps']}")