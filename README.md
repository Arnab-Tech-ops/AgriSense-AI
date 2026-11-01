# AgriSense AI

**Machine Learning-Based Soil Quality Classification and Crop Suitability Prediction System**

AgriSense AI is a Streamlit decision-support application that combines user-supplied soil measurements, two validated crop classification models, optional SoilGrids map estimates, and live Open-Meteo weather. It distinguishes measured inputs, model predictions, rule-based screening, current observations, and forecasts.

## Features

- Manual soil entry and one-row CSV import of laboratory measurements, with unit and range checks.
- Transparent soil quality screening with nutrient/pH statuses, missing-data warnings, and general next steps.
- Scaled, distance-weighted KNN and Random Forest trained on the same stratified training split and evaluated on the same held-out rows.
- Model selected by stratified cross-validation on training data powers the default top-five crop recommendation; KNN and Random Forest can be compared explicitly.
- Accuracy, macro precision, macro recall, macro F1, class reports, confusion matrices, and Random Forest feature importance.
- Global city, town, and postal-code search with country qualifiers and configurable aliases; current conditions include feels-like temperature, wind direction, humidity, precipitation, cloud cover, and pressure, with a seven-day forecast including rain probability.
- Optional SoilGrids WCS map-cell estimates for pH and organic carbon at 0–5 cm, with product resolution and 90% prediction intervals.
- Plotly visualizations and in-session prediction history with a clear-history action.

## Run locally

Python 3.10+ and Streamlit 1.51+ are required (the app uses current layout APIs and the soil form starts empty so it never pre-fills invented measurements).

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

SoilGrids is optional. To enable its WCS reader, install the geospatial extra:

```bash
pip install -r requirements-geo.txt
```

The app trains on first run if `models/crop_models.joblib` is missing, corrupt, outdated, or built against a different dataset hash. It saves a local bundle when the models directory is writable and continues in memory if saving fails. Existing old model files are not used as a substitute for the new evaluated bundle.

Run the automated checks with:

```bash
python -m unittest discover -s tests -v
```

## Architecture

- `app.py` — Streamlit sections, state, weather caching, visualizations, and recommendation flows.
- `ml_model.py` — dataset validation, KNN/RF training and holdout metrics, prediction scores, explanations, and soil screening.
- `weather.py` — Open-Meteo geocoding and current/7-day forecast client.
- `soil_data.py` — manual/laboratory input validation, CSV parsing, and source register.
- `soil_geo.py` — optional SoilGrids WCS point sampling. It is separate from crop-model inputs.
- `data/Crop_recommendation.csv` — included public labeled crop dataset.
- `models/crop_models.joblib` — generated serialized estimators and train-only explanatory summaries.
- `.streamlit/config.toml` — explicit light theme and app colors.
- `tests/test_core.py` — model, input, screening, and mocked weather-client checks.

## Dataset and limitations

The included public Crop Recommendation dataset contains 2,200 rows, 22 crop labels, and the seven columns `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`. The commonly used original is on [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset); this workspace copy is a public [CSV mirror](https://huggingface.co/spaces/Vivswaan/crop_recommendation/resolve/main/Crop_recommendation.csv). The feature labels and scales are inherited from that dataset and do not specify a universal laboratory unit system. Its labels represent recommended crop classes in the dataset; they do not establish local yield, causality, or real field outcomes.

Manual NPK inputs use the published dataset-compatible scale. Laboratory CSV NPK must include compatible unit declaration `dataset_scale` for crop predictions. Values in `mg/kg`, `ppm`, `kg/ha`, or unknown units remain visible as user/lab inputs, but no conversion is guessed and no crop model prediction is made. Soil CSVs must contain exactly one data row with `N,P,K,ph`; optional columns include `soil_moisture` (fraction 0–1), `organic_carbon` (g/kg), climate inputs, and metadata such as `lab_name`, `sample_id`, and `report_date`. NPK unit selection is made in the UI.

## Machine-learning methodology

Both models use one stratified 80/20 train/test split (`random_state=42`) with the same held-out rows. Model selection uses stratified cross-validation within the training split only, comparing mean macro-F1 and then accuracy; the held-out test data is not used to choose the default model. KNN uses a pipeline with training-fold-only `StandardScaler` plus 5-neighbor distance-weighted Euclidean classification. Random Forest uses 300 trees and a fixed seed. The reported accuracy, macro precision, macro recall, macro F1, classification reports, and confusion matrices are computed on the same untouched holdout. These metrics describe this dataset holdout only and are not a guarantee of crop success. KNN neighbor vote shares and Random Forest class scores are ranking scores, not calibrated probabilities.

Recommendation explanations compare a submitted profile with the middle 50% feature ranges for each crop in the labeled training dataset. These are descriptive similarities, not causal feature attributions. The seven-day weather forecast is displayed as additional context and is **not passed into the crop model**. The separate forecast temperature context compares the forecast range with a crop's training examples; it is an advisory comparison only.

## Soil data and screening

Soil input provenance is kept as user entry or user-uploaded lab report. Missing values are reported; they are not filled with fabricated measurements. The rules-based screen is explicitly not an ML classifier or laboratory diagnosis. On the dataset-compatible NPK scale, N<40, P<20, and K<20 count as low screening values. pH below 5.5 or above 8.5 is an extreme flag; pH outside 6.0–7.5 is noted as suboptimal. Two low/extreme flags means Poor, one means Fair, complete data with no flags means Good, and missing data without a flag means Incomplete. Suggestions prompt local lab or extension guidance rather than prescribing fertilizer. Optional soil moisture and organic carbon are recorded but not assigned unsupported quality thresholds.

### Optional SoilGrids geospatial provider

ISRIC states that the SoilGrids REST API is temporarily paused and documents WCS as an alternative for spatial subsets. `soil_geo.py` can request the pH-in-water and organic-carbon 0–5 cm products through that WCS. Values are explicitly labeled as modelled map-cell estimates, not field/lab samples. The interface displays the product's 250 m resolution and Q0.05–Q0.95 interval when the provider responds. WCS/network errors are shown to the user; manual/lab data remain separate. SoilGrids values are not fed into crop recommendations. See [ISRIC SoilGrids data access](https://docs.isric.org/globaldata/soilgrids/) and its [WCS guide](https://docs.isric.org/globaldata/soilgrids/wcs.html).

## Weather API

The app resolves city, town, and postal-code searches through the [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api), supports country qualifiers and configurable common city aliases, then queries the selected coordinates through the [Forecast API](https://open-meteo.com/en/docs). Current data includes temperature, apparent (feels-like) temperature, condition/icon, humidity, wind speed/direction, current-interval precipitation, cloud cover, pressure, and local observation time. The seven-day forecast includes local day/date, condition, min/max temperature, maximum precipitation probability, rainfall, and maximum wind. Requests use the selected location timezone, finite timeouts, and the existing cache/manual-refresh controls. ET₀ appears only if Open-Meteo provides it. There are no API keys in this project.

## Known limitations and future improvements

- Crop results are based on a small, widely reused public dataset with limited provenance and no local validation.
- The dataset does not contain soil-health labels, so soil status remains transparent rules-based screening.
- Forecast temperature context is descriptive; no crop-specific regional agronomic model is included.
- SoilGrids WCS availability depends on the external service and optional rasterio installation. It can fail independently without breaking the app.
- Prediction history lives only in the current Streamlit session and is not written to disk.
- A future version could add region-calibrated datasets, local crop calendars, soil-test unit conversions vetted by agronomists, and a persistent consent-based farm history.
