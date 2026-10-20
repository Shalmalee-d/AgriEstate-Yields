"""
Agricultural Suitability Assessment Module.

This module implements a transparent, rule-based suitability scoring system
for candidate lands based on available features. It is NOT a machine learning
model - it is an interpretable scoring framework with explicit weights and criteria.

METHODOLOGY
===========

The suitability score is computed as a weighted combination of normalized factors:

SUITABILITY SCORE = Σ (weight_i × factor_i) × 100

Where each factor is normalized to [0, 1] range, and the final score is scaled to [0, 100].

FACTORS AND WEIGHTS
-------------------

1. AREA SCORE (weight: 0.20)
   - Based on area_m2, normalized against practical minimum/maximum
   - Min viable: 200 m² (~2150 sqft) for meaningful cultivation
   - Max practical: 2000 m² (~21500 sqft) for this prototype scope
   - Formula: clip((area_m2 - 200) / (2000 - 200), 0, 1)

2. SUNLIGHT SCORE (weight: 0.25)
   - Directly uses sunlight_score from preprocessing (sunlight_hours / 12, clipped)
   - 12 hours = practical maximum for most crops
   - Weight reflects critical importance of sunlight for photosynthesis

3. WATER ACCESS SCORE (weight: 0.20)
   - Uses water_access_encoded (Low=1, Medium=2, High=3) normalized to [0,1]
   - Formula: (water_access_encoded - 1) / (3 - 1) = (encoded - 1) / 2
   - High=1.0, Medium=0.5, Low=0.0

4. MARKET PROXIMITY SCORE (weight: 0.15)
   - Uses market_proximity_score = 1 / (1 + distance_km)
   - Already normalized to (0, 1] with diminishing returns
   - Closer to market reduces transport cost and spoilage

5. LAND TYPE BASE SCORE (weight: 0.10)
   - Plot = 1.0 (full soil profile, no structural constraints)
   - Rooftop = 0.85 (container/raised bed limitations, weight limits, drainage)
   - Reflects inherent agricultural capacity difference

6. USAGE PRIORITY SCORE (weight: 0.10)
   - Vacant = 1.0 (immediately available, no transition needed)
   - Underutilized = 0.67 (some clearing/preparation needed)
   - Existing Garden = 0.33 (already in use, limited additional capacity)
   - Formula: usage_priority / 3.0

TOTAL WEIGHT = 1.00

SUITABILITY STATUS THRESHOLDS (0-100 scale)
-------------------------------------------
- Excellent:    score >= 75
- Suitable:     score >= 50
- Marginal:     score >= 30
- Unsuitable:   score < 30

CURRENT USAGE FILTERING RULES
-----------------------------
- Vacant: INCLUDED - highest priority for new cultivation
- Underutilized: INCLUDED - can be repurposed with some investment
- Existing Garden: FLAGGED - included in results but marked as 'existing_use'
  (LAND013 Community Garden Plot is NOT excluded but flagged)

REASONS/FACTORS
---------------
Each assessment includes a breakdown of factor scores (0-1 scale) and a list of
limiting factors (factors scoring < 0.5) for transparency.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from .preprocessing import preprocess_candidate_lands, get_feature_columns


# Explicit weights - documented and auditable
WEIGHTS = {
    "area": 0.20,
    "sunlight": 0.25,
    "water": 0.20,
    "market": 0.15,
    "land_type": 0.10,
    "usage": 0.10,
}

# Verify weights sum to 1.0
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Area normalization bounds (square meters)
AREA_MIN_M2 = 200.0      # ~2150 sqft minimum viable
AREA_MAX_M2 = 2000.0     # ~21500 sqft practical max for prototype

# Suitability thresholds (0-100 scale)
THRESHOLDS = {
    "excellent": 75.0,
    "suitable": 50.0,
    "marginal": 30.0,
    "unsuitable": 0.0,
}


@dataclass
class SuitabilityResult:
    """Container for suitability assessment results."""
    land_id: str
    name: str
    land_type: str
    current_usage: str
    area_m2: float
    suitability_score: float
    suitability_status: str
    factor_scores: dict
    limiting_factors: list[str]
    usage_flag: str  # 'available', 'repurpose', 'existing_use'


def compute_area_score(area_m2: float) -> float:
    """
    Compute normalized area score.

    Args:
        area_m2: Land area in square meters.

    Returns:
        Normalized score in [0, 1].
    """
    return float(np.clip((area_m2 - AREA_MIN_M2) / (AREA_MAX_M2 - AREA_MIN_M2), 0, 1))


def compute_sunlight_score(sunlight_score: float) -> float:
    """
    Compute sunlight factor score (already normalized in preprocessing).

    Args:
        sunlight_score: Pre-normalized sunlight score (0-1).

    Returns:
        Score in [0, 1].
    """
    return float(np.clip(sunlight_score, 0, 1))


def compute_water_score(water_access_encoded: int) -> float:
    """
    Compute water access score from ordinal encoding.

    Args:
        water_access_encoded: 1=Low, 2=Medium, 3=High.

    Returns:
        Normalized score in [0, 1].
    """
    return float(np.clip((water_access_encoded - 1) / 2, 0, 1))


def compute_market_score(market_proximity_score: float) -> float:
    """
    Compute market proximity score (already normalized in preprocessing).

    Args:
        market_proximity_score: Pre-computed 1/(1+distance) score.

    Returns:
        Score in [0, 1].
    """
    return float(np.clip(market_proximity_score, 0, 1))


def compute_land_type_score(land_type_base: float) -> float:
    """
    Compute land type base score.

    Args:
        land_type_base: 1.0 for Plot, 0.85 for Rooftop.

    Returns:
        Score in [0, 1].
    """
    return float(np.clip(land_type_base, 0, 1))


def compute_usage_score(usage_priority: int) -> float:
    """
    Compute current usage priority score.

    Args:
        usage_priority: 3=Vacant, 2=Underutilized, 1=Existing Garden.

    Returns:
        Normalized score in [0, 1].
    """
    return float(np.clip(usage_priority / 3.0, 0, 1))


def determine_status(score: float) -> str:
    """
    Determine suitability status from score.

    Args:
        score: Suitability score in [0, 100].

    Returns:
        Status string.
    """
    if score >= THRESHOLDS["excellent"]:
        return "Excellent"
    elif score >= THRESHOLDS["suitable"]:
        return "Suitable"
    elif score >= THRESHOLDS["marginal"]:
        return "Marginal"
    else:
        return "Unsuitable"


def determine_usage_flag(current_usage: str) -> str:
    """
    Determine usage flag for filtering/display.

    Args:
        current_usage: One of 'Vacant', 'Underutilized', 'Existing Garden'.

    Returns:
        Flag: 'available', 'repurpose', or 'existing_use'.
    """
    if current_usage == "Vacant":
        return "available"
    elif current_usage == "Underutilized":
        return "repurpose"
    elif current_usage == "Existing Garden":
        return "existing_use"
    else:
        return "unknown"


def get_limiting_factors(factor_scores: dict, threshold: float = 0.5) -> list[str]:
    """
    Identify factors scoring below threshold.

    Args:
        factor_scores: Dict of factor_name -> score.
        threshold: Score below which a factor is considered limiting.

    Returns:
        List of limiting factor names.
    """
    return [name for name, score in factor_scores.items() if score < threshold]


def assess_suitability(row: pd.Series) -> SuitabilityResult:
    """
    Assess suitability for a single land record.

    Args:
        row: Preprocessed DataFrame row.

    Returns:
        SuitabilityResult with score, status, and factor breakdown.
    """
    # Compute individual factor scores
    area_score = compute_area_score(row["area_m2"])
    sunlight_score = compute_sunlight_score(row["sunlight_score"])
    water_score = compute_water_score(row["water_access_encoded"])
    market_score = compute_market_score(row["market_proximity_score"])
    land_type_score = compute_land_type_score(row["land_type_base"])
    usage_score = compute_usage_score(row["usage_priority"])

    factor_scores = {
        "area": area_score,
        "sunlight": sunlight_score,
        "water": water_score,
        "market": market_score,
        "land_type": land_type_score,
        "usage": usage_score,
    }

    # Weighted composite score (0-1 scale)
    composite_score = sum(
        WEIGHTS[factor] * score for factor, score in factor_scores.items()
    )

    # Scale to 0-100 for public API
    suitability_score = composite_score * 100.0

    suitability_status = determine_status(suitability_score)
    usage_flag = determine_usage_flag(row["current_usage"])
    limiting_factors = get_limiting_factors(factor_scores)

    return SuitabilityResult(
        land_id=row["land_id"],
        name=row["name"],
        land_type=row["land_type"],
        current_usage=row["current_usage"],
        area_m2=row["area_m2"],
        suitability_score=round(suitability_score, 4),
        suitability_status=suitability_status,
        factor_scores={k: round(v, 4) for k, v in factor_scores.items()},
        limiting_factors=limiting_factors,
        usage_flag=usage_flag,
    )


def assess_all_lands(csv_path: Optional[str] = None) -> list[SuitabilityResult]:
    """
    Assess suitability for all candidate lands.

    Args:
        csv_path: Optional custom path to CSV file.

    Returns:
        List of SuitabilityResult sorted by suitability_score descending.
    """
    df = preprocess_candidate_lands(csv_path)
    results = [assess_suitability(row) for _, row in df.iterrows()]
    results.sort(key=lambda r: r.suitability_score, reverse=True)
    return results


def suitability_to_dataframe(results: list[SuitabilityResult]) -> pd.DataFrame:
    """
    Convert suitability results to DataFrame for export/display.

    Args:
        results: List of SuitabilityResult.

    Returns:
        DataFrame with suitability results.
    """
    data = []
    for r in results:
        row = {
            "land_id": r.land_id,
            "name": r.name,
            "land_type": r.land_type,
            "current_usage": r.current_usage,
            "area_m2": round(r.area_m2, 2),
            "suitability_score": r.suitability_score,
            "suitability_status": r.suitability_status,
            "usage_flag": r.usage_flag,
            "limiting_factors": "; ".join(r.limiting_factors) if r.limiting_factors else "None",
        }
        # Add individual factor scores
        for factor, score in r.factor_scores.items():
            row[f"factor_{factor}"] = score
        data.append(row)
    return pd.DataFrame(data)


def filter_by_usage(results: list[SuitabilityResult], include_existing_garden: bool = False) -> list[SuitabilityResult]:
    """
    Filter results by current usage flag.

    Args:
        results: List of SuitabilityResult.
        include_existing_garden: If False, exclude 'existing_use' lands.

    Returns:
        Filtered list of SuitabilityResult.
    """
    if include_existing_garden:
        return results
    return [r for r in results if r.usage_flag != "existing_use"]


def get_suitability_summary(results: list[SuitabilityResult]) -> dict:
    """
    Generate summary statistics for suitability results.

    Args:
        results: List of SuitabilityResult.

    Returns:
        Dict with summary statistics.
    """
    statuses = [r.suitability_status for r in results]
    usage_flags = [r.usage_flag for r in results]

    return {
        "total_lands": len(results),
        "by_status": {s: statuses.count(s) for s in set(statuses)},
        "by_usage_flag": {u: usage_flags.count(u) for u in set(usage_flags)},
        "score_stats": {
            "mean": round(np.mean([r.suitability_score for r in results]), 4),
            "median": round(np.median([r.suitability_score for r in results]), 4),
            "min": round(min(r.suitability_score for r in results), 4),
            "max": round(max(r.suitability_score for r in results), 4),
            "std": round(np.std([r.suitability_score for r in results]), 4),
        },
    }


if __name__ == "__main__":
    # Quick test
    results = assess_all_lands()
    df = suitability_to_dataframe(results)
    print(df[["land_id", "name", "suitability_score", "suitability_status", "usage_flag", "limiting_factors"]].to_string())
    print("\nSummary:", get_suitability_summary(results))