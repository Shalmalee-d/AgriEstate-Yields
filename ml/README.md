# AgriEstate-Yields ML Module

> **Agricultural Suitability Assessment & Yield Prediction Pipeline**

This module implements a two-stage ML pipeline for evaluating candidate lands for agricultural use:

```
Candidate Land Data (15 records)
         ↓
Data Preprocessing & Validation
         ↓
Agricultural Suitability Assessment (Rule-based, Interpretable)
         ↓
Suitable Candidate Lands
         ↓
Environmental/Agricultural Feature Enrichment (REQUIRED)
         ↓
Random Forest Yield Prediction (Supervised, Needs Labeled Data)
         ↓
Yield Predictions
         ↓
Crop Recommendations
```

---

## 📊 Current Dataset

**File**: `ml/data/candidate_lands.csv` (15 records)

| Column | Type | Description |
|--------|------|-------------|
| `land_id` | string | Unique identifier (LAND001-LAND015) |
| `name` | string | Land parcel name |
| `land_type` | categorical | `Rooftop` or `Plot` |
| `current_usage` | categorical | `Vacant`, `Underutilized`, `Existing Garden` |
| `area_sqft` | numeric | Area in square feet |
| `sunlight_hours` | numeric | Daily sunlight hours (6.9-8.7) |
| `water_access` | categorical | `High`, `Medium`, `Low` |
| `market_distance_km` | numeric | Distance to market (1.2-6.8 km) |

### ⚠️ Critical Limitations

**This dataset does NOT contain:**
- Soil pH, N/P/K measurements
- Rainfall, temperature, humidity
- Crop type information
- Historical or actual yield data

**Therefore:**
- ❌ **Cannot train yield prediction model** (no labels)
- ❌ **Cannot validate crop recommendations** (no soil/climate data)
- ✅ **Can assess agricultural suitability** (rule-based, interpretable)

---

## 🏗️ Module Structure

```
ml/
├── src/
│   ├── __init__.py           # Public API exports
│   ├── preprocessing.py      # Data loading, validation, feature engineering
│   ├── suitability.py        # Rule-based agricultural suitability scoring
│   ├── recommendation.py     # Crop recommendation framework (prototype)
│   ├── model.py              # RandomForestRegressor pipeline (ready for labeled data)
│   └── prediction.py         # Clean interface for FastAPI/backend integration
├── data/
│   └── candidate_lands.csv   # 15-record prototype dataset
├── models/                   # Trained models saved here (created after training)
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA notebook
├── tests/
│   ├── test_preprocessing.py
│   ├── test_suitability.py
│   ├── test_model.py
│   ├── test_prediction.py
│   └── conftest.py
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

```bash
cd ml
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- pandas, numpy, scikit-learn, joblib
- matplotlib, seaborn (for notebook)
- pytest (for tests)

---

## 🚀 Quick Start

### 1. Run Data Exploration Notebook

```bash
cd ml/notebooks
jupyter notebook 01_data_exploration.ipynb
```

### 2. Run Suitability Assessment (Python)

```python
from src import get_all_suitability, get_suitability_summary_interface

# Get all suitability scores
results = get_all_suitability()
for r in results:
    print(f"{r.land_id}: {r.name} | Score: {r.suitability_score:.3f} | {r.suitability_status}")

# Get summary statistics
summary = get_suitability_summary_interface()
print(summary)
```

### 3. Get Crop Recommendations (Prototype)

```python
from src import get_all_recommendations

# Top 3 crops per land (uses placeholder thresholds - NOT validated)
recs = get_all_recommendations(top_n=3)
for r in recs:
    print(f"\n{r.land_id}: {r.land_name} (Suitability: {r.suitability_score:.3f})")
    for rec in r.recommendations:
        print(f"  {rec['crop']}: match={rec['composite_score']:.3f}, "
              f"yield≈{rec['estimated_yield_kg']:.0f}kg, revenue≈${rec['estimated_revenue_usd']:.0f}")
```

### 4. Assess Custom Land Parcel

```python
from src import assess_suitability_interface, SuitabilityRequest

request = SuitabilityRequest(
    land_id="CUSTOM001",
    name="My Rooftop",
    land_type="Rooftop",
    current_usage="Vacant",
    area_sqft=5000,
    sunlight_hours=7.5,
    water_access="Medium",
    market_distance_km=2.5,
)

response = assess_suitability_interface(request)
print(f"Score: {response.suitability_score:.3f} ({response.suitability_status})")
print(f"Limiting factors: {response.limiting_factors}")
```

---

## 📐 Suitability Methodology (Transparent & Auditable)

The suitability score is a **weighted sum of normalized factors** — NOT a black-box ML model.

### Weights (Sum = 1.0)

| Factor | Weight | Description |
|--------|--------|-------------|
| `area` | 0.20 | Land area (200-2000 m² viable range) |
| `sunlight` | 0.25 | Daily sunlight hours (normalized to 12h max) |
| `water` | 0.20 | Water access (Low=0, Med=0.5, High=1) |
| `market` | 0.15 | Market proximity (1/(1+distance_km)) |
| `land_type` | 0.10 | Plot=1.0, Rooftop=0.85 (structural limits) |
| `usage` | 0.10 | Vacant=1.0, Underutilized=0.67, Existing=0.33 |

### Status Thresholds

| Score Range | Status |
|-------------|--------|
| ≥ 0.75 | Highly Suitable |
| 0.50 - 0.75 | Moderately Suitable |
| 0.30 - 0.50 | Marginally Suitable |
| < 0.30 | Not Suitable |

### Usage Filtering Rules

| Current Usage | Flag | Included by Default? |
|--------------|------|---------------------|
| `Vacant` | `available` | ✅ Yes |
| `Underutilized` | `repurpose` | ✅ Yes |
| `Existing Garden` | `existing_use` | ❌ No (excluded, use `include_existing_garden=True`) |

**LAND013 (Community Garden Plot)** is flagged as `existing_use` — not excluded silently.

### Limiting Factors

Any factor scoring < 0.5 is reported as a limiting factor for transparency.

---

## 🌱 Crop Recommendation Framework

**Status: PROTOTYPE — Placeholder thresholds only**

The `recommendation.py` module provides a *structure* for crop matching but uses **unvalidated placeholder values** for crop requirements.

### Required for Production

To enable real crop recommendations, you need per-crop validated data:

| Requirement | Source |
|-------------|--------|
| Optimal pH range | FAO/USDA crop databases |
| N/P/K requirements (kg/ha) | Agricultural extension |
| Water requirement (mm/season) | Crop coefficients (FAO-56) |
| Temperature range (min/opt/max °C) | Phenology models |
| Sunlight requirement (h/day) | Photosynthesis curves |
| Growing season length (days) | Variety catalogs |
| Soil type preference | NRCS/FAO soil-crop suitability |
| Market price & yield expectations | Local market surveys |

### Current Placeholder Crops

- Lettuce
- Tomato
- Spinach
- Kale
- Herbs (Basil/Mint)

**⚠️ DO NOT USE FOR REAL FARMING DECISIONS**

---

## 🌲 Random Forest Yield Model

**Status: PIPELINE READY — NOT TRAINED**

`model.py` implements a complete production-ready pipeline:

### Features Expected for Training

**Numeric:**
- `soil_ph`, `nitrogen`, `phosphorus`, `potassium`
- `rainfall_mm`, `temperature_avg`, `humidity_avg`
- `sunlight_hours`, `water_availability_mm`, `area_m2`

**Categorical:**
- `crop_type`, `soil_type`, `irrigation_type`, `season`

**Target:** `actual_yield` (kg/ha or kg/m²)

### Pipeline Components

1. **Preprocessing**: `ColumnTransformer` with `StandardScaler` + `OneHotEncoder`
2. **Model**: `RandomForestRegressor` (n_estimators=200, max_depth=15)
3. **Validation**: Train/test split + 5-fold CV
4. **Metrics**: MAE, RMSE, R²
5. **Persistence**: `joblib` (model + preprocessor + metadata)

### Training When Labeled Data Is Available

```python
from src import train_yield_model

# Prepare CSV with required features + actual_yield column
# Minimum ~50 samples recommended, 200+ for production

pipeline, metadata = train_yield_model(
    csv_path="data/yield_training_data.csv",
    target_col="actual_yield",
    test_size=0.2,
    save_path="models/yield_rf_model"
)

print(f"MAE: {metadata.mae}, RMSE: {metadata.rmse}, R²: {metadata.r2}")
print(f"Top features: {list(metadata.feature_importance.items())[:10]}")
```

### Auto-Training Helper

```python
from src import train_if_data_available

# Checks ml/data/ for yield dataset, trains if found
result = train_if_data_available()
if result:
    pipeline, metadata = result
    print("Model trained and saved!")
```

---

## 🔌 Prediction Interface (for FastAPI Backend)

`prediction.py` provides clean, backend-agnostic functions:

### Suitability Assessment

```python
from src.prediction import assess_suitability_interface, SuitabilityRequest

request = SuitabilityRequest(
    land_id="LAND001", name="Green Tower Rooftop",
    land_type="Rooftop", current_usage="Vacant",
    area_sqft=5000, sunlight_hours=8.0,
    water_access="High", market_distance_km=3.2
)

response = assess_suitability_interface(request)
# Returns SuitabilityResponse with score, status, factors, limiting_factors
```

### Yield Prediction (Requires Trained Model)

```python
from src.prediction import predict_yield_interface, PredictionRequest

request = PredictionRequest(
    soil_ph=6.5, nitrogen=120, phosphorus=60, potassium=200,
    rainfall_mm=500, temperature_avg=22, humidity_avg=65,
    sunlight_hours=8, water_availability_mm=400, area_m2=465,
    crop_type="Tomato", soil_type="Loam", irrigation_type="Drip", season="Spring"
)

response = predict_yield_interface(request)
# Returns PredictionResponse with predicted_yield, uncertainty_std, metadata
# Returns error if model not trained
```

### Crop Recommendations

```python
from src.prediction import recommend_crops_interface, CropRecommendationRequest

request = CropRecommendationRequest(
    land_id="LAND001",
    soil_ph=6.5, temperature_avg=22, rainfall_mm=500,
    top_n=3
)

response = recommend_crops_interface(request)
# Returns CropRecommendationResponse with ranked crops
```

### Model Status

```python
from src.prediction import get_model_status, reload_model

status = get_model_status()
# {'model_available': False, 'error': '...', 'message': '...'}

# After training:
reload_model()  # Refresh cache
status = get_model_status()
# {'model_available': True, 'metadata': {...}}
```

---

## 🧪 Running Tests

```bash
cd ml
pytest tests/ -v
```

**Test Coverage:**
- `test_preprocessing.py` - Loading, validation, encoding, derived features
- `test_suitability.py` - Factor scores, weighting, thresholds, filtering
- `test_model.py` - Pipeline creation, training, save/load, prediction
- `test_prediction.py` - Interface functions, request/response dataclasses

---

## 📝 Next Data Required

To unlock full pipeline capabilities, the following datasets are needed:

### 1. Labeled Yield Dataset (Minimum for Model Training)

| Requirement | Specification |
|-------------|---------------|
| **Samples** | 50+ (200+ recommended) |
| **Features** | All EXPECTED_NUMERIC_FEATURES + EXPECTED_CATEGORICAL_FEATURES |
| **Target** | `actual_yield` (kg/ha or kg/m², measured) |
| **Format** | CSV with header |
| **Source** | Research trials, farmer surveys, crop cuts, calibrated remote sensing |

**Place in**: `ml/data/yield_training_data.csv` (or similar name)

### 2. Environmental Data (For Crop Recommendations)

Per-land measurements needed:

| Variable | Measurement Method |
|----------|-------------------|
| Soil pH | Lab test (1:2.5 water) |
| Soil N/P/K | Lab test (Mehlich-3, Olsen, etc.) |
| Historical rainfall | Weather station / CHIRPS / ERA5 |
| Temperature range | Weather station / ERA5 |
| Humidity | Weather station / ERA5 |
| Water availability | Irrigation system capacity + water rights |

### 3. Validated Crop Requirements Database

Replace `PLACEHOLDER_CROPS` in `recommendation.py` with data from:
- FAO EcoCrop database
- USDA PLANTS database
- Local agricultural university extensions
- Peer-reviewed crop modeling literature (DSSAT, APSIM parameters)

---

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| `preprocessing.py` | Data loading, validation, feature engineering |
| `suitability.py` | **Core methodology** - weights, thresholds, factor computations |
| `recommendation.py` | Crop matching framework (prototype) |
| `model.py` | Random Forest pipeline (training, evaluation, persistence) |
| `prediction.py` | **Backend interface** - FastAPI-ready functions |
| `01_data_exploration.ipynb` | EDA with visualizations |
| `tests/` | Unit & integration tests |

---

## 🔄 Development Workflow

1. **Add labeled yield data** → `ml/data/yield_training.csv`
2. **Train model** → `python -m src.model` or `train_if_data_available()`
3. **Verify metrics** → Check MAE/RMSE/R² in `models/yield_rf_model.json`
4. **Backend integration** → Import functions from `src.prediction`
5. **Retrain** → Call `reload_model()` after new model saved

---

## ⚖️ License & Disclaimer

This is a prototype system for demonstration purposes.

**The yield prediction model is NOT trained** — no labeled dataset exists yet.

**Crop recommendations use PLACEHOLDER thresholds** — not scientifically validated.

**Suitability scores are rule-based heuristics** — not ML predictions.

For production use, engage qualified agronomists and validate all thresholds with local field trials.