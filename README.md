# AgriSense AI

**Machine Learning-Based Soil Quality Classification and Crop Suitability Prediction System**

AgriSense AI is a Streamlit decision-support application that combines user-supplied soil measurements, two validated crop classification models, optional SoilGrids map estimates, and current model-based Open-Meteo weather estimates with forecasts. It distinguishes measured inputs, model predictions, rule-based screening, gridded model estimates, and forecasts.

## Features

- Manual soil entry and one-row CSV import of laboratory measurements, with unit and range checks.
- Transparent soil quality screening with nutrient/pH statuses, missing-data warnings, and general next steps.
- Scaled, distance-weighted KNN and Random Forest trained on the same stratified training split and evaluated on the same held-out rows.
- Model selected by stratified cross-validation on training data powers the default top-five crop recommendation; KNN and Random Forest can be compared explicitly.
- Accuracy, macro precision, macro recall, macro F1, class reports, confusion matrices, and Random Forest feature importance.
- Global city, locality, and postal-code search with an ISO country/territory filter, country/state-qualified queries, exact postcode ranking, aliases, and selectable results. Results show postcode, administrative area, country code, coordinates, time zone, match type, and geocoding provider.
- Optional GeoNames postal/place fallback when Open-Meteo has no exact match. It is enabled with a GeoNames username; provider coverage varies by country, and partial postcode records are labeled for review.
- Optional SoilGrids WCS map-cell estimates for pH and organic carbon at 0–5 cm, with product resolution and 90% prediction intervals.
- Plotly visualizations and in-session prediction history with a clear-history action.

## Run locally

Python 3.10+ and Streamlit 1.51+ are required (the app uses current layout APIs and the soil form starts empty so it never pre-fills invented measurements). `pycountry` supplies the ISO 3166-1 alpha-2 country and territory list for the optional weather filter.

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

### Optional GeoNames fallback

Open-Meteo remains the primary geocoder and does not need credentials. To enable the optional GeoNames fallback, create a GeoNames account and set `GEONAMES_USERNAME` as an environment variable. For Streamlit Community Cloud, add this to the app's Secrets settings:

```toml
GEONAMES_USERNAME = "your-geonames-username"
```

Do not commit credentials or a local `.streamlit/secrets.toml` file. GeoNames documents a free-service limit of 10,000 credits per day and 1,000 per hour; the app only contacts it after Open-Meteo returns no usable match or no exact postcode match. GeoNames' own postal-data notes document country-specific limitations: Canada may return only the first three characters, Ireland and Malta only the first letters, Argentina uses an older four-digit dataset, and Brazil contains only major municipality codes ending in `-000`. Therefore, neither provider can guarantee every code worldwide. See the [GeoNames postal-code API policy and coverage notes](https://www.geonames.org/export/web-services.html) and [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api).

Run the automated checks with:

```bash
python -m unittest discover -s tests -v
```

## Architecture

- `app.py` — Streamlit sections, state, weather caching, visualizations, and recommendation flows.
- `ml_model.py` — dataset validation, KNN/RF training and holdout metrics, prediction scores, explanations, and soil screening.
- `weather.py` — Open-Meteo geocoding, GeoNames postal fallback, best-match current/hourly/daily weather parsing, freshness labels, and validation.
- `scripts/evaluate_forecasts.py` — separate archived forecast lead-time evaluation against ERA5 reanalysis.
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

The app resolves place names and postal codes through the [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api), with ISO country-code filtering and a searchable country/territory selector. Country/state qualifiers and common aliases are supported. Exact postcode matches rank first; leading zeros and alphanumeric codes remain text, and duplicates are removed by coordinates. If Open-Meteo has no usable match or exact postal-code match and a GeoNames username is configured, its [postal-code service](https://www.geonames.org/export/web-services.html) is queried as a secondary source. Each location result identifies its provider and match type.

Weather uses `https://api.open-meteo.com/v1/forecast` with `models=best_match`. The API automatically combines the most suitable global and regional model output; AgriSense does not average models or claim a single selected model. Requested current variables include temperature, apparent temperature, humidity, day/night, precipitation, rain, showers, weather code, cloud cover, sea-level pressure, wind speed/direction/gusts, visibility, ET₀, vapour-pressure deficit, and available modeled soil temperature/moisture layers. The app also requests 24 upcoming hourly points and a seven-day forecast (temperature and apparent-temperature highs/lows, precipitation probability/amount, wind/gust maxima, UV index, sunrise/sunset, sunshine duration, and ET₀). Unsupported or missing values appear as unavailable.

Current conditions are model-based estimates, not physical weather-station observations. The app converts Open-Meteo Unix timestamps through the selected IANA timezone so hourly times respect daylight-saving transitions. It separately labels the weather-valid time, its age, and the time the app last retrieved the API response. Fresh/delayed/stale is a **valid-timestamp-age heuristic** (fresh ≤90 minutes, delayed ≤6 hours, stale >6 hours), not proof of upstream model freshness. The automatic `best_match` response does not identify which individual model run supplied each value, and Open-Meteo's per-model update metadata cannot reliably identify the run behind its stitched response. Therefore model initialization and availability age are explicitly shown as unavailable rather than inferred. Open-Meteo notes that model updates occur at different cadences and that its redundant servers can take additional time to become consistent; see [Model Updates and Data Availability](https://open-meteo.com/en/docs/model-updates).

Forecast responses are cached for five minutes to reduce repeated requests; the user-triggered refresh bypasses the cache. If refresh fails, the last successful response remains visible with its retrieval age and a warning, never relabeled as a fresh update. The finite-timeout client performs one limited retry for transient failures only. Open-Meteo data are licensed under [CC BY 4.0](https://open-meteo.com/en/pricing); the Live Weather view links to Open-Meteo attribution. The free API is for non-commercial use and has published request limits; see [terms](https://open-meteo.com/en/terms). Do not claim global postal coverage or station-level accuracy.

### Forecast accuracy evaluation baseline

Run the separate utility against a date range covered by archived forecast runs (most Previous Runs data begins in 2024; coverage varies by model):

```powershell
python scripts/evaluate_forecasts.py --latitude 18.51957 --longitude 73.85535 --timezone Asia/Kolkata --start-date 2026-09-01 --end-date 2026-09-30
```

The script requests archived `best_match` forecasts from Open-Meteo's [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) at 1–7 day lead offsets. It aligns hourly temperature and precipitation valid times with Open-Meteo [ERA5 reanalysis](https://open-meteo.com/en/docs/historical-weather-api), then reports MAE/RMSE for hourly temperature, hourly precipitation, and complete local-day precipitation totals. Partial local days and unavailable variable values are excluded. ERA5 is a gridded reanalysis reference, not a station observation or independent ground truth; model and reanalysis grid resolutions differ. Values are a measured baseline only for the selected location/date range and do not demonstrate improved accuracy. For rainfall, ERA5 is used because the [ERA5-Land variable table](https://open-meteo.com/en/docs/historical-weather-api) does not list precipitation for that dataset.

## Known limitations and future improvements

- Crop results are based on a small, widely reused public dataset with limited provenance and no local validation.
- The dataset does not contain soil-health labels, so soil status remains transparent rules-based screening.
- Forecast temperature context is descriptive; no crop-specific regional agronomic model is included.
- SoilGrids WCS availability depends on the external service and optional rasterio installation. It can fail independently without breaking the app.
- Prediction history lives only in the current Streamlit session and is not written to disk.
- A future version could add region-calibrated datasets, local crop calendars, soil-test unit conversions vetted by agronomists, and a persistent consent-based farm history.
