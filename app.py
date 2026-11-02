from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_model import (DATA_PATH, FEATURES, classify_soil_quality,
                      crop_input_explanation, load_dataset, load_or_train_models,
                      predict_top_crops)
from soil_geo import fetch_soilgrids_estimate
from soil_data import read_lab_report, soil_data_sources, validate_inputs
from weather import (clear_weather_state, country_options, get_weather_bundle,
                     next_weather_refresh, search_locations)

st.set_page_config(page_title="AgriSense AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stSidebar"] {
  background:#F5F7F2; color:#26352D;
}
.stApp { font-family: Arial, sans-serif; }
.block-container { padding-top: 1.25rem; max-width: 1480px; }
h1,h2,h3,h4,h5,h6,[data-testid="stHeading"],[data-testid="stHeading"] * { color:#174D35; }
 p,li,span,small,[data-testid="stMarkdownContainer"] { color:#26352D; }
label,[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *,
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *,
[data-testid="stHelp"],[data-testid="stTooltipContent"],.stCaption { color:#52645A; }
input,textarea,[data-baseweb="input"],[data-baseweb="textarea"],
[data-baseweb="select"] > div,[data-baseweb="select"] input {
  background:#fff !important; color:#26352D !important;
  -webkit-text-fill-color:#26352D !important;
}
input::placeholder,textarea::placeholder { color:#52645A !important; opacity:1; }
div[data-testid="stButton"] button,div[data-testid="stFormSubmitButton"] button {
  background:#174D35; border:1px solid #174D35; color:#fff;
}
div[data-testid="stButton"] button *,div[data-testid="stFormSubmitButton"] button * { color:#fff; }
div[data-testid="stMetric"],div[data-testid="stForm"],div[data-testid="stExpander"],
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:#fff; border:1px solid #DCE5DD; border-radius:14px;
}
div[data-testid="stMetric"] { padding:1rem; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"],div[data-testid="stMetric"] label { color:#52645A; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#26352D; }
div[data-testid="stAlert"] { color:#26352D; border:1px solid #DCE5DD; }
div[data-testid="stAlert"] *,[data-testid="stNotification"] * { color:#26352D; }
.hero { background:#fff; border:1px solid #DCE5DD; border-radius:18px; padding:1.5rem 1.8rem; margin-bottom:1rem; }
.hero h1 { color:#174D35; font-size:2.15rem; margin:.2rem 0; }
.hero p { color:#26352D; margin:.35rem 0 0; }
.eyebrow { color:#52645A; letter-spacing:.13em; font-size:.72rem; font-weight:700; text-transform:uppercase; }
a,a:visited { color:#174D35; }
</style>
<div class="hero"><div class="eyebrow">Field intelligence · Soil and climate</div><h1>AgriSense AI</h1><p>Explore soil measurements, live weather estimates and crop suitability evidence.</p></div>
""", unsafe_allow_html=True)


def _format_metric(value, unit):
    return "Unavailable" if value is None or pd.isna(value) else f"{float(value):.1f} {unit}"


def _age_label(seconds):
    if seconds is None or seconds < 0:
        return "unavailable"
    seconds = int(seconds)
    if seconds < 60:
        return "less than a minute ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h {(seconds % 3600) // 60} min ago"


@st.cache_resource(show_spinner="Training and evaluating KNN and Random Forest on the labeled dataset…")
def model_bundle():
    return load_or_train_models()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_locations(query, country_code, geonames_username, refresh_key):
    return search_locations(query, country_code=country_code,
                            geonames_username=geonames_username)


WEATHER_CACHE_TTL_SECONDS = 5 * 60


@st.cache_data(ttl=WEATHER_CACHE_TTL_SECONDS, max_entries=128, show_spinner=False)
def cached_weather(latitude, longitude, timezone_name, name, admin1, country, refresh_key):
    location = {"latitude": latitude, "longitude": longitude, "timezone": timezone_name,
                "name": name, "admin1": admin1, "country": country}
    return get_weather_bundle(location)


def geonames_username():
    """Read optional GeoNames credentials from environment or Streamlit secrets."""
    configured = os.getenv("GEONAMES_USERNAME", "").strip()
    if configured:
        return configured
    try:
        return str(st.secrets.get("GEONAMES_USERNAME", "")).strip()
    except Exception:
        return ""


def clear_weather_search():
    clear_weather_state(st.session_state)


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def cached_soilgrids(latitude, longitude, refresh_key):
    return fetch_soilgrids_estimate(latitude, longitude)


def reset_soil_source_values():
    """Prevent switching manual/lab modes from silently reusing another source's values."""
    for key in ("soil_N", "soil_P", "soil_K", "soil_ph", "soil_soil_moisture",
                "moisture_missing", "soil_organic_carbon", "carbon_missing",
                "loaded_lab_report", "loaded_lab_fingerprint"):
        st.session_state.pop(key, None)


try:
    models, evaluation, model_context = model_bundle()
    dataset = load_dataset()
except Exception as exc:
    st.error(f"The model bundle could not be prepared: {exc}")
    st.info("Check that data/Crop_recommendation.csv exists and matches the documented schema, then restart the app.")
    st.stop()
if evaluation.get("persistence_warning"):
    st.warning(evaluation["persistence_warning"])
selection_method = evaluation.get("selection_method")
if not selection_method:
    selection_method = "legacy model bundle; restart Streamlit to reload the current training code"
    st.warning("A legacy model bundle was loaded. Restart the Streamlit process to rebuild the model bundle with training-only cross-validation selection.")

for key, default in (("analysis", None), ("history", []), ("weather_bundle", None),
                     ("weather_locations", []), ("weather_refresh_key", 0), ("location_refresh_key", 0),
                     ("weather_error", None), ("weather_bundle_location", None),
                     ("weather_country_filter", ""), ("weather_selected", 0)):
    st.session_state.setdefault(key, default)

overview, soil_tab, weather_tab, model_tab, history_tab = st.tabs([
    "Overview", "Soil & crop recommendations", "Live weather", "Model performance", "History & data"
])

with overview:
    st.subheader("Dashboard overview")
    a, b, c, d = st.columns(4)
    a.metric("Labeled observations", f"{evaluation['dataset_rows']:,}")
    b.metric("Crop classes", evaluation["crop_count"])
    c.metric("Selected model", evaluation["best_model"])
    d.metric("Selected model test macro F1", f"{evaluation['models'][evaluation['best_model']]['macro_f1']:.1%}")
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        soil_result = classify_soil_quality(analysis["soil"], analysis["npk_units"])
        col1, col2 = st.columns([1.2, 1], gap="large")
        with col1:
            st.markdown("#### Latest field assessment")
            if analysis.get("recommendation_available"):
                st.write(f"**Source:** {analysis['source']} · **Model:** {analysis['model_name']}")
                st.write(f"**Top crop:** {analysis['recommendations'][0]['crop'].title()}")
                st.write("**Top five:** " + ", ".join(x["crop"].title() for x in analysis["recommendations"]))
            else:
                st.info(analysis.get("prediction_reason", "Enter complete, dataset-compatible model inputs to request crop predictions."))
            st.write(f"**Soil screening:** {soil_result['quality']} · {', '.join(soil_result['notes'])}")
        with col2:
            st.markdown("#### Forecast snapshot")
            bundle = st.session_state.weather_bundle
            if bundle:
                cur = bundle["current"]
                st.metric(f"Current temperature · {bundle['place']}", _format_metric(cur["temperature_c"], "°C"), cur["condition"])
                st.caption(f"Open-Meteo model-based current conditions valid {cur['time']} · retrieved {bundle['fetched_at_utc']} UTC")
            else:
                st.info("Search a location in Live weather to load current conditions and a 7-day forecast.")
    else:
        st.info("Enter measured soil values in Soil & crop recommendations to start an assessment.")
    st.caption("Predictions describe patterns in the published dataset and do not guarantee yield or field success.")

with soil_tab:
    st.subheader("Soil analysis and crop suitability")
    st.caption("Manual and laboratory NPK values enter the crop model only when they are expressed on the dataset-compatible scale. Climate features remain manually entered seasonal estimates; current weather is not copied into the model.")
    source_mode = st.radio("Soil data source", ["Manual soil inputs", "Laboratory CSV report"], horizontal=True,
                           key="soil_source_mode", on_change=reset_soil_source_values)
    imported = None
    if source_mode == "Laboratory CSV report":
        unit_choice = st.selectbox("NPK units reported by the lab", [
            "csv_declared", "dataset_scale", "mg/kg", "ppm", "kg/ha", "unknown"
        ], format_func=lambda x: {
            "csv_declared": "Use the CSV's npk_units column",
            "dataset_scale": "Dataset-compatible scale (required for crop model)",
            "mg/kg": "mg/kg", "ppm": "ppm", "kg/ha": "kg/ha", "unknown": "Unknown / not specified",
        }[x], help="No conversion is assumed. Select Unknown if the report does not specify units. Only dataset_scale values can be used by the crop model.")
        lab_file = st.file_uploader("Upload one-row CSV soil report", type=["csv"], key="lab_csv")
        fingerprint = None
        if lab_file:
            try:
                fingerprint = hashlib.sha256(lab_file.getvalue()).hexdigest() + ":" + unit_choice
                if st.session_state.get("loaded_lab_fingerprint") != fingerprint:
                    st.session_state.loaded_lab_report = None
                    for widget_key in ("soil_N", "soil_P", "soil_K", "soil_ph", "soil_soil_moisture",
                                       "moisture_missing", "soil_organic_carbon", "carbon_missing"):
                        st.session_state.pop(widget_key, None)
                selected_units = None if unit_choice == "csv_declared" else unit_choice
                imported = read_lab_report(lab_file, npk_units=selected_units)
                if st.button("Load report values into the form", key="load_lab_report"):
                    for key, value in imported["values"].items():
                        if value is not None:
                            widget_key = {"N":"soil_N", "P":"soil_P", "K":"soil_K", "ph":"soil_ph",
                                          "soil_moisture":"soil_soil_moisture", "organic_carbon":"soil_organic_carbon"}[key]
                            st.session_state[widget_key] = float(value)
                    if imported["values"].get("soil_moisture") is not None:
                        st.session_state.moisture_missing = False
                    if imported["values"].get("organic_carbon") is not None:
                        st.session_state.carbon_missing = False
                    st.session_state.loaded_lab_report = imported
                    st.session_state.loaded_lab_fingerprint = fingerprint
                    st.success("Report values loaded. Review them below before analyzing.")
            except ValueError as exc:
                st.error(str(exc))
        if fingerprint and st.session_state.get("loaded_lab_fingerprint") == fingerprint:
            imported = st.session_state.get("loaded_lab_report")
        if imported:
            metadata = [x for x in (imported.get("lab_name"), imported.get("sample_id"), imported.get("report_date")) if x]
            st.caption("Source: user-uploaded laboratory report" + (" · " + " · ".join(metadata) if metadata else ""))
            st.caption(f"Reported NPK units: {imported['npk_units']} · model compatible: {'yes' if imported['model_compatible'] else 'no'}")

    with st.form("soil_analysis_form"):
        st.markdown("##### Soil measurements")
        n1, p1, k1 = st.columns(3)
        max_npk = 300.0 if source_mode == "Manual soil inputs" or (imported and imported["model_compatible"]) else 100000.0
        n = n1.number_input("Nitrogen · N", 0.0, max_npk, value=None, step=1.0, key="soil_N", placeholder="Enter measured N", help="Manual values use the published dataset scale; lab units are shown above.")
        p = p1.number_input("Phosphorus · P", 0.0, max_npk, value=None, step=1.0, key="soil_P", placeholder="Enter measured P")
        k = k1.number_input("Potassium · K", 0.0, max_npk, value=None, step=1.0, key="soil_K", placeholder="Enter measured K")
        ph = st.number_input("Soil pH", 0.0, 14.0, value=None, step=0.1, key="soil_ph", placeholder="Enter lab pH")
        st.markdown("##### Optional lab measurements")
        m1, m2 = st.columns(2)
        moisture = m1.number_input("Soil moisture · volumetric fraction (0–1)", 0.0, 1.0, value=None, step=0.01, key="soil_soil_moisture", placeholder="Optional")
        moisture_missing = m1.checkbox("Moisture not measured", value=True, key="moisture_missing")
        carbon = m2.number_input("Organic carbon · g/kg", 0.0, 100.0, value=None, step=0.1, key="soil_organic_carbon", placeholder="Optional")
        carbon_missing = m2.checkbox("Organic carbon not measured", value=True, key="carbon_missing")
        st.markdown("##### Seasonal crop conditions (manual estimates)")
        t1, h1, r1 = st.columns(3)
        temperature = t1.number_input("Temperature · °C", -10.0, 60.0, value=None, step=0.5, key="season_temperature", placeholder="Seasonal estimate")
        humidity = h1.number_input("Relative humidity · %", 0.0, 100.0, value=None, step=1.0, key="season_humidity", placeholder="Seasonal estimate")
        rainfall = r1.number_input("Seasonal rainfall · mm", 0.0, 2000.0, value=None, step=5.0, key="season_rainfall", placeholder="Seasonal estimate", help="The dataset rainfall feature is a seasonal-scale variable. Current precipitation and 7-day forecast are not substituted here.")
        submitted = st.form_submit_button("Analyze soil and recommend crops", type="primary", width="stretch")

    if submitted:
        values = {"N": n, "P": p, "K": k, "ph": ph, "temperature": temperature,
                  "humidity": humidity, "rainfall": rainfall,
                  "soil_moisture": None if moisture_missing else moisture,
                  "organic_carbon": None if carbon_missing else carbon}
        is_lab = source_mode == "Laboratory CSV report"
        units = (imported or {}).get("npk_units", unit_choice if is_lab else "dataset_scale")
        errors = validate_inputs(values, npk_units=units)
        model_errors = validate_inputs(values, require_model_features=True, npk_units=units)
        if is_lab and not imported:
            errors.append("Load a valid laboratory report before analyzing in laboratory mode.")
        if not any(values.get(name) is not None for name in ("N", "P", "K", "ph", "soil_moisture", "organic_carbon")):
            errors.append("Enter or import at least one soil measurement for soil screening.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            compatible = not is_lab or bool(imported and imported["model_compatible"])
            recommendation_available = compatible and not model_errors
            soil_values = dict(values)
            values_record = {
                "soil": soil_values, "input_values": values, "source": (
                    "User-entered soil measurements" if not is_lab else "User-uploaded lab report (reviewed/edited)"
                ), "npk_units": "dataset_scale" if not is_lab else (imported or {}).get("npk_units", unit_choice),
                "model_compatible": compatible, "recommendation_available": recommendation_available,
                "lab_metadata": ({key: imported.get(key) for key in ("lab_name", "sample_id", "report_date") if imported.get(key)} if is_lab and imported else {}),
            }
            if not compatible:
                values_record["prediction_reason"] = "Predictions are withheld because uploaded NPK units do not match the training dataset scale. No unit conversion was applied."
            elif model_errors:
                values_record["prediction_reason"] = "Soil screening is available, but crop recommendations need all seven model inputs: " + ", ".join(error.split(" is required")[0] for error in model_errors)
            if recommendation_available:
                chosen_name = st.session_state.get("selected_model", evaluation["best_model"])
                recommendations = predict_top_crops(models, values, model_name=chosen_name,
                                                    context=model_context, top_n=5)
                values_record.update({"model_name": chosen_name, "recommendations": recommendations})
                st.session_state.history.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "source": values_record["source"], "model": chosen_name,
                    "top_crop": recommendations[0]["crop"], "soil_quality": classify_soil_quality(soil_values, units)["quality"],
                    "recommendations": recommendations,
                })
                st.session_state.history = st.session_state.history[-25:]
            st.session_state.analysis = values_record

    record = st.session_state.analysis
    if record:
        st.divider()
        st.markdown("### Soil quality screening")
        soil_result = classify_soil_quality(record["soil"], record["npk_units"])
        q1, q2, q3 = st.columns(3)
        q1.metric("Rule-based status", soil_result["quality"])
        q2.metric("Soil data source", "Lab report" if "lab" in record["source"].lower() else "Manual")
        q3.metric("Available soil variables", f"{sum(record['soil'].get(k) is not None for k in ('N','P','K','ph','soil_moisture','organic_carbon'))}/6")
        indicators = pd.DataFrame([
            {"Measurement": "Nitrogen (N)", "Value": record["soil"].get("N"), "Status": soil_result["nutrients"]["N"]},
            {"Measurement": "Phosphorus (P)", "Value": record["soil"].get("P"), "Status": soil_result["nutrients"]["P"]},
            {"Measurement": "Potassium (K)", "Value": record["soil"].get("K"), "Status": soil_result["nutrients"]["K"]},
            {"Measurement": "Soil pH", "Value": record["soil"].get("ph"), "Status": soil_result["ph"]},
            {"Measurement": "Soil moisture", "Value": record["soil"].get("soil_moisture"), "Status": soil_result["optional"]["soil_moisture"]},
            {"Measurement": "Organic carbon", "Value": record["soil"].get("organic_carbon"), "Status": soil_result["optional"]["organic_carbon"]},
        ])
        sc1, sc2 = st.columns([1.1, 1], gap="large")
        with sc1:
            st.dataframe(indicators, hide_index=True, width="stretch")
            st.caption(f"NPK units: {record['npk_units']} · Source: {record['source']}")
            if record.get("lab_metadata"):
                st.caption("Lab report: " + " · ".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in record["lab_metadata"].items()))
        with sc2:
            npk_data = pd.DataFrame([{"Nutrient": name, "Value": record["soil"].get(name)} for name in ("N", "P", "K")]).dropna()
            if not npk_data.empty:
                npk_chart = px.bar(npk_data, x="Nutrient", y="Value", color="Nutrient",
                                   title=f"Soil NPK · {record['npk_units']}", color_discrete_sequence=px.colors.qualitative.G10)
                npk_chart.update_layout(template="plotly_white", height=280, showlegend=False,
                                        paper_bgcolor="#fff", plot_bgcolor="#fff", font_color="#26352D")
                st.plotly_chart(npk_chart, width="stretch")
            climate = record["input_values"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Soil pH", _format_metric(record["soil"].get("ph"), ""))
            c2.metric("Seasonal temperature", _format_metric(climate.get("temperature"), "°C"))
            c3.metric("Seasonal humidity", _format_metric(climate.get("humidity"), "%"))
            c4.metric("Seasonal rainfall", _format_metric(climate.get("rainfall"), "mm"))
        if soil_result["missing"]:
            st.warning("Missing soil data: " + ", ".join(soil_result["missing"]) + ". The screening is incomplete.")
        st.markdown("**Assessment details**")
        for note in soil_result["notes"]:
            st.write("• " + note)
        st.markdown("**General next steps**")
        for suggestion in soil_result["suggestions"]:
            st.write("• " + suggestion)
        st.caption("This is transparent rules-based screening, not a trained classifier, laboratory diagnosis, or fertilizer prescription.")

        with st.expander("Optional geospatial soil estimates · SoilGrids WCS"):
            st.caption("SoilGrids REST is paused; this optional WCS path retrieves modelled 250 m estimates for topsoil pH and organic carbon, with the product's 90% interval. These are not laboratory observations or crop-model inputs.")
            if not st.session_state.get("weather_locations"):
                st.info("Search a location in Live weather to use its coordinates here.")
            else:
                geo_locations = st.session_state.weather_locations
                geo_index = st.selectbox("Estimate location", list(range(len(geo_locations))),
                                         format_func=lambda i: ", ".join(x for x in (geo_locations[i]["name"], geo_locations[i]["admin1"], geo_locations[i]["country"]) if x),
                                         key="soilgrids_location")
                geo = geo_locations[geo_index]
                if st.button("Fetch SoilGrids map estimate", key="fetch_soilgrids"):
                    st.session_state.soilgrids_refresh_key = st.session_state.get("soilgrids_refresh_key", 0) + 1
                    st.session_state.soilgrids_estimate = None
                    try:
                        st.session_state.soilgrids_estimate = cached_soilgrids(geo["latitude"], geo["longitude"], st.session_state.soilgrids_refresh_key)
                    except (RuntimeError, ValueError) as exc:
                        st.error(str(exc))
                estimate = st.session_state.get("soilgrids_estimate")
                if estimate:
                    g1, g2 = st.columns(2)
                    g1.metric("Modelled pH · 0–5 cm", f"{estimate['ph']:.1f}", f"90% interval {estimate['ph_low']:.1f}–{estimate['ph_high']:.1f}")
                    g2.metric("Modelled organic carbon · 0–5 cm", f"{estimate['organic_carbon']:.1f} g/kg", f"90% interval {estimate['organic_carbon_low']:.1f}–{estimate['organic_carbon_high']:.1f}")
                    st.caption(f"{estimate['source']} · retrieved {estimate['retrieved_at_utc']} UTC · {estimate['product_resolution']} product grid · coordinates {estimate['latitude']:.4f}, {estimate['longitude']:.4f}")
                    st.warning(estimate["limitations"])

        st.markdown("### Crop suitability recommendations")
        if not record["recommendation_available"]:
            st.warning(record.get("prediction_reason", "Crop recommendations are unavailable for the submitted values."))
        else:
            selection = st.selectbox("Recommendation model", [evaluation["best_model"], "KNN", "Random Forest"],
                                     index=0, key="selected_model",
                                     help="Default model was selected by stratified cross-validation on training data. The KNN remains available for comparison.")
            recommendations = predict_top_crops(models, record["input_values"], model_name=selection,
                                                context=model_context, top_n=5)
            record["recommendations"] = recommendations
            record["model_name"] = selection
            top = recommendations[0]
            best_score = evaluation["models"][selection]["macro_f1"]
            r1, r2, r3 = st.columns(3)
            r1.metric("Best matching crop", top["crop"].title())
            r2.metric("Model score", f"{top['score']:.1%}", help=top["score_label"])
            r3.metric("Selected model test macro F1", f"{best_score:.1%}")
            st.caption(f"Score meaning: {top['score_label']}. It is a model ranking score, not a calibrated probability or yield estimate.")
            out_of_data = [f for f in FEATURES if record["input_values"][f] < dataset[f].min() or record["input_values"][f] > dataset[f].max()]
            if out_of_data:
                st.warning("The following inputs are outside the full training dataset range, so these recommendations are extrapolations: " + ", ".join(out_of_data) + ".")
            reco_frame = pd.DataFrame(recommendations)
            bar = px.bar(reco_frame.sort_values("score"), x="score", y="crop", orientation="h",
                         text=reco_frame.sort_values("score")["score"].map(lambda x: f"{x:.0%}"),
                         color="score", color_continuous_scale=["#B5D4A9", "#246447"])
            bar.update_layout(template="plotly_white", height=300, paper_bgcolor="#fff",
                              plot_bgcolor="#fff", xaxis_title=top["score_label"], yaxis_title="",
                              coloraxis_showscale=False, font_color="#26352D")
            bar.update_xaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(bar, width="stretch")
            rows = []
            for result in recommendations:
                explanation = crop_input_explanation(record["input_values"], result["crop"], model_context, dataset)
                rows.append({"Crop": result["crop"].title(), "Score": result["score"],
                             "Why it ranked": " · ".join(explanation),
                             "Potential limitation": "Check local season, soil test units, water availability, and crop-specific guidance."})
            st.dataframe(pd.DataFrame(rows).style.format({"Score": "{:.1%}"}), hide_index=True, width="stretch")
            st.caption("Why statements compare entered values with the middle range of examples for each crop in this dataset. These are descriptive similarities, not causal feature attributions.")
            weather = st.session_state.weather_bundle
            if weather:
                daily = weather["daily"]
                crop_rows = dataset[dataset["label"].astype(str).str.lower() == top["crop"].lower()]
                low, high = crop_rows["temperature"].quantile(.25), crop_rows["temperature"].quantile(.75)
                forecast_lows = [x["temperature_min"] for x in daily if x["temperature_min"] is not None and pd.notna(x["temperature_min"])]
                forecast_highs = [x["temperature_max"] for x in daily if x["temperature_max"] is not None and pd.notna(x["temperature_max"])]
                if not forecast_lows or not forecast_highs:
                    context_msg = "Seven-day forecast temperatures are unavailable, so no forecast comparison is shown."
                elif min(forecast_lows) >= low and max(forecast_highs) <= high:
                    context_msg = "The 7-day temperature range is inside the middle 50% of this crop's dataset examples."
                else:
                    context_msg = "Part of the 7-day temperature range is outside the middle 50% of this crop's dataset examples; check local agronomic guidance."
                st.info("Forecast context only (not fed into the trained model): " + context_msg)

with weather_tab:
    st.subheader("Live weather and seven-day forecast")
    st.caption("Current conditions are Open-Meteo model-based estimates, not readings from a physical weather station. Forecast data is supplementary and is not substituted for seasonal crop-model inputs.")
    wc1, wc2 = st.columns([1.1, 1], gap="large")
    with wc1:
        city = st.text_input(
            "Search cities, towns, localities, or postal codes worldwide",
            placeholder="e.g. Pune, India",
            help="Search local or international postal codes, ZIP codes, PIN codes, cities, towns, or localities.",
            key="weather_query",
        )
        countries = dict(country_options())
        country_filter = st.selectbox("Optional country / territory filter", [""] + list(countries),
                                      format_func=lambda code: "Any country" if not code else f"{countries[code]} ({code})",
                                      key="weather_country_filter",
                                      help="The list uses ISO 3166-1 alpha-2 country and territory codes.")
        search_col, clear_col = st.columns(2)
        if search_col.button("Search locations", width="stretch", key="search_weather"):
            st.session_state.weather_bundle = None
            st.session_state.weather_bundle_location = None
            st.session_state.weather_error = None
            try:
                st.session_state.weather_locations = cached_locations(
                    city.strip(), country_filter or None, geonames_username(),
                    st.session_state.location_refresh_key)
                st.session_state.weather_selected = 0
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                st.session_state.weather_locations = []
                st.session_state.weather_error = str(exc)
        clear_col.button("Clear search", width="stretch", key="clear_weather_search", on_click=clear_weather_search)
        locations = st.session_state.weather_locations
        if locations:
            options = list(range(len(locations)))

            def format_location(index):
                item = locations[index]
                parts = [item["name"]]
                if item.get("postal_code"):
                    parts.append(item["postal_code"])
                parts.extend(x for x in (item.get("admin1"), item.get("country")) if x)
                return " · ".join(parts)

            selected = st.selectbox("Choose the matching location", options,
                                    format_func=format_location,
                                    key="weather_selected")
            loc = locations[selected]
            st.caption(
                f"Selected: {format_location(selected)} · country code {loc.get('country_code') or 'unavailable'} · "
                f"{loc['latitude']:.5f}, {loc['longitude']:.5f} · time zone {loc.get('timezone') or 'auto'} · "
                f"{loc.get('provider', 'Open-Meteo')} ({loc.get('match_type', 'location match')})"
            )
            previous_location = st.session_state.get("weather_bundle_location")
            selected_coordinates = (loc["latitude"], loc["longitude"])
            if previous_location and previous_location != selected_coordinates:
                st.session_state.weather_bundle = None
            if st.button("Fetch / refresh live weather", type="primary", width="stretch", key="refresh_weather"):
                next_weather_refresh(st.session_state)
                try:
                    st.session_state.weather_bundle = cached_weather(loc["latitude"], loc["longitude"],
                                                                      loc["timezone"], loc["name"], loc["admin1"],
                                                                      loc["country"], st.session_state.weather_refresh_key)
                    st.session_state.weather_bundle_location = selected_coordinates
                    st.session_state.weather_error = None
                    st.success(f"Updated weather response received for {st.session_state.weather_bundle['place']}.")
                except (RuntimeError, ValueError) as exc:
                    st.session_state.weather_error = str(exc)
        elif city.strip() and st.button("Search locations again", width="stretch", key="refresh_location_search"):
            st.session_state.location_refresh_key += 1
            try:
                st.session_state.weather_locations = cached_locations(
                    city.strip(), country_filter or None, geonames_username(),
                    st.session_state.location_refresh_key)
            except (ValueError, RuntimeError) as exc:
                st.session_state.weather_error = str(exc)
                st.session_state.weather_locations = []
                st.error(str(exc))
    with wc2:
        weather = st.session_state.weather_bundle
        if st.session_state.weather_error and weather:
            st.warning(
                "Refresh failed. Showing the last successful weather response, not a fresh update. "
                f"It was retrieved {_age_label((datetime.now(timezone.utc) - datetime.fromisoformat(weather['retrieved_at_utc'])).total_seconds())}. "
                + st.session_state.weather_error
            )
        elif st.session_state.weather_error:
            st.error(st.session_state.weather_error)
        if weather:
            cur = weather["current"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{cur['icon']} Temperature", _format_metric(cur["temperature_c"], "°C"), cur["condition"])
            m2.metric("Feels like", _format_metric(cur["apparent_temperature_c"], "°C"))
            m3.metric("Humidity", _format_metric(cur["humidity_pct"], "%"))
            m4.metric("Wind", _format_metric(cur["wind_kmh"], "km/h"), f"Direction {_format_metric(cur['wind_direction_deg'], '°')}")
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Precipitation", _format_metric(cur["precipitation_mm"], "mm"), help=f"Accumulation over the preceding {cur['interval_seconds'] or 'API'} seconds.")
            m6.metric("Rain / showers", f"{_format_metric(cur['rain_mm'], 'mm')} / {_format_metric(cur['showers_mm'], 'mm')}")
            m7.metric("Wind gusts", _format_metric(cur["wind_gusts_kmh"], "km/h"))
            m8.metric("Visibility", _format_metric(cur["visibility_km"], "km"))
            m9, m10, m11 = st.columns(3)
            m9.metric("Cloud cover", _format_metric(cur["cloud_cover_pct"], "%"))
            m10.metric("Sea-level pressure", _format_metric(cur["pressure_hpa"], "hPa"))
            m11.metric("Daylight", "Day" if cur["is_day"] == 1 else "Night" if cur["is_day"] == 0 else "Unavailable")
            st.write(f"**{weather['place']}** · {cur['icon']} {cur['condition']}")
            st.caption(f"Weather valid at {cur['time']} ({weather['timezone']}) · Valid-time age: {_age_label(weather['valid_time_age_seconds'])} ({weather['valid_time_freshness']})")
            st.caption(f"Last API retrieval: {weather['retrieved_at_utc']} UTC · {_age_label((datetime.now(timezone.utc) - datetime.fromisoformat(weather['retrieved_at_utc'])).total_seconds())}")
            st.caption(f"Model selection: {weather['model_selection']}. Model initialization/availability age: unavailable from this blended API response.")
            st.caption(weather["source"])
            st.caption(f"Geocoding provider: {weather.get('geocoding_provider', 'Open-Meteo')} · Weather provider: {weather.get('weather_provider', 'Open-Meteo Forecast API')}")
            st.caption("Open-Meteo weather data · [Attribution and licence](https://open-meteo.com/en/terms)")
        else:
            st.info("Search and select a location to request live weather. Values appear only after a successful API response.")

    weather = st.session_state.weather_bundle
    if weather:
        st.markdown("#### Next 24 hours")
        if not weather["hourly"]:
            st.info("Hourly forecast data is currently unavailable for this response.")
        else:
            hourly_frame = pd.DataFrame(weather["hourly"])
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(x=hourly_frame["time"], y=hourly_frame["temperature_c"], name="Temperature · °C", mode="lines+markers", line={"color":"#174D35"}))
            hfig.add_trace(go.Scatter(x=hourly_frame["time"], y=hourly_frame["apparent_temperature_c"], name="Feels like · °C", mode="lines+markers", line={"color":"#6E9B7C"}))
            hfig.add_trace(go.Bar(x=hourly_frame["time"], y=hourly_frame["precipitation_probability_pct"], name="Rain probability · %", yaxis="y2", marker_color="rgba(65,125,165,.32)"))
            hfig.update_layout(template="plotly_white", paper_bgcolor="#fff", plot_bgcolor="#fff", font_color="#26352D",
                               yaxis_title="Temperature · °C", yaxis2={"title":"Rain chance · %","overlaying":"y","side":"right","range":[0,100]},
                               barmode="overlay", height=360, margin={"t":20,"b":10}, legend={"orientation":"h","y":1.12})
            st.plotly_chart(hfig, width="stretch")
            display_hourly = hourly_frame[["time_label", "temperature_c", "apparent_temperature_c", "precipitation_probability_pct", "precipitation_mm", "humidity_pct", "wind_kmh", "condition"]].rename(columns={
                "time_label":"Local time", "temperature_c":"Temperature · °C", "apparent_temperature_c":"Feels like · °C",
                "precipitation_probability_pct":"Rain chance · %", "precipitation_mm":"Precipitation · mm",
                "humidity_pct":"Humidity · %", "wind_kmh":"Wind · km/h", "condition":"Condition"})
            st.dataframe(display_hourly, hide_index=True, width="stretch")

        st.markdown("#### Seven-day forecast")
        daily_frame = pd.DataFrame(weather["daily"])
        tfig = go.Figure()
        tfig.add_trace(go.Scatter(x=daily_frame["date"], y=daily_frame["temperature_max"], name="Maximum °C", mode="lines+markers", line={"color":"#174D35"}))
        tfig.add_trace(go.Scatter(x=daily_frame["date"], y=daily_frame["temperature_min"], name="Minimum °C", mode="lines+markers", line={"color":"#6E9B7C"}, fill="tonexty", fillcolor="rgba(110,155,124,.12)"))
        tfig.update_layout(template="plotly_white", paper_bgcolor="#fff", plot_bgcolor="#fff", font_color="#26352D", yaxis_title="Temperature · °C", height=330, margin={"t":20,"b":10})
        st.plotly_chart(tfig, width="stretch")
        rain = px.bar(daily_frame, x="date", y="precipitation_mm", color="precipitation_mm", color_continuous_scale=["#DCEBDD", "#246447"], labels={"date":"Local date","precipitation_mm":"Forecast precipitation · mm"})
        rain.update_layout(template="plotly_white", paper_bgcolor="#fff", plot_bgcolor="#fff", font_color="#26352D", coloraxis_showscale=False, height=260)
        st.plotly_chart(rain, width="stretch")
        card_rows = [daily_frame.iloc[i:i + 3] for i in range(0, len(daily_frame), 3)]
        for row in card_rows:
            day_cols = st.columns(3)
            for col, (_, day) in zip(day_cols, row.iterrows()):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{day['day']} · {day['date']}**")
                        st.write(f"{day['icon']} {day['condition']} · {_format_metric(day['temperature_max'], '°C')} / {_format_metric(day['temperature_min'], '°C')}")
                        st.caption(f"Feels like max/min: {_format_metric(day['apparent_temperature_max'], '°C')} / {_format_metric(day['apparent_temperature_min'], '°C')}")
                        st.caption(f"Rain chance: {_format_metric(day['precipitation_probability_pct'], '%')} · Precipitation: {_format_metric(day['precipitation_mm'], 'mm')}")
                        st.caption(f"Wind/gusts: {_format_metric(day['wind_max_kmh'], 'km/h')} / {_format_metric(day['wind_gust_max_kmh'], 'km/h')} · UV max: {_format_metric(day['uv_index_max'], '')}")
                        st.caption(f"Sunrise/sunset: {day['sunrise'] or 'Unavailable'} / {day['sunset'] or 'Unavailable'} · Sun: {_format_metric(day['sunshine_hours'], 'h')}")
        rain_values = daily_frame["precipitation_mm"].dropna()
        total_rain = f"{rain_values.sum():.1f} mm" if not rain_values.empty else "unavailable"
        st.caption(f"Seven-day forecast precipitation total: {total_rain}.")

        st.markdown("#### Agricultural weather indicators")
        ag = cur["agriculture"]
        et0_today = daily_frame.iloc[0]["et0_mm"]
        ag1, ag2, ag3 = st.columns(3)
        ag1.metric("Today reference ET₀", _format_metric(et0_today, "mm"))
        ag2.metric("Vapour-pressure deficit", _format_metric(ag["vpd_kpa"], "kPa"))
        ag2.caption("Higher VPD indicates greater atmospheric drying demand.")
        ag3.metric("Surface soil temperature", _format_metric(ag["soil_temperature_c"]["0 cm"], "°C"))
        ag3.caption("ET₀ describes reference-grass water demand; it is not an irrigation prescription.")
        soil_rows = [{"Modeled depth": depth, "Variable": "Soil temperature", "Value": value, "Unit": "°C"}
                     for depth, value in ag["soil_temperature_c"].items()]
        soil_rows += [{"Modeled depth": depth, "Variable": "Soil moisture", "Value": value, "Unit": "m³/m³"}
                      for depth, value in ag["soil_moisture_m3_m3"].items()]
        st.dataframe(pd.DataFrame(soil_rows), hide_index=True, width="stretch")
        st.caption("Higher ET₀ indicates greater reference-grass atmospheric water demand. Soil moisture indicates modelled water content by soil layer. Neither is a field-sensor reading or a guaranteed irrigation prescription; use these as contextual guidance, not professional agronomic advice.")

with model_tab:
    st.subheader("Model comparison and evaluation")
    st.caption(f"Shared stratified 80/20 split · {evaluation['train_rows']:,} training rows · {evaluation['test_rows']:,} held-out rows · preprocessing fit on training data only.")
    compare = pd.DataFrame({name: {metric: result[metric] for metric in ("accuracy", "precision_macro", "recall_macro", "macro_f1")}
                            for name, result in evaluation["models"].items()}).T.rename(columns={
        "accuracy":"Accuracy", "precision_macro":"Macro precision", "recall_macro":"Macro recall", "macro_f1":"Macro F1"})
    st.dataframe(compare.style.format("{:.1%}").highlight_max(axis=0, color="#DCEBDD"), width="stretch")
    comp_long = compare.reset_index(names="Model").melt(id_vars="Model", var_name="Metric", value_name="Score")
    compfig = px.bar(comp_long, x="Metric", y="Score", color="Model", barmode="group", text_auto=".1%", range_y=[0, 1], color_discrete_sequence=["#174D35", "#7EAA88"])
    compfig.update_layout(template="plotly_white", paper_bgcolor="#fff", plot_bgcolor="#fff", font_color="#26352D", height=350)
    st.plotly_chart(compfig, width="stretch")
    selected_eval = st.selectbox("Confusion matrix", list(evaluation["models"]), key="confusion_choice")
    result = evaluation["models"][selected_eval]
    matrix = result["confusion_matrix"]
    labels = [x.title() for x in result["labels"]]
    cm = px.imshow(matrix, x=labels, y=labels, text_auto=True, aspect="auto", color_continuous_scale="Greens", labels={"x":"Predicted crop","y":"Actual crop","color":"Count"}, title=f"{selected_eval} · held-out confusion matrix")
    cm.update_layout(template="plotly_white", paper_bgcolor="#fff", plot_bgcolor="#fff", font_color="#26352D", height=700)
    st.plotly_chart(cm, width="stretch")
    if selected_eval == "Random Forest":
        fi = pd.DataFrame({"Feature": list(evaluation["feature_importances"]), "Importance": list(evaluation["feature_importances"].values())}).sort_values("Importance")
        st.markdown("#### Random Forest feature importance")
        st.plotly_chart(px.bar(fi, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#DCEBDD", "#174D35"]).update_layout(template="plotly_white", paper_bgcolor="#fff", plot_bgcolor="#fff", font_color="#26352D"), width="stretch")
        st.caption("Impurity-based importance describes the fitted forest; it is not a causal explanation.")
    st.caption(f"Default model: {evaluation['best_model']} (selected by {selection_method}). All displayed model metrics use the same untouched stratified holdout and are not a guarantee of agronomic success.")

with history_tab:
    st.subheader("Recent prediction history and data sources")
    if st.session_state.history:
        if st.button("Clear prediction history", key="clear_history"):
            st.session_state.history.clear()
            st.rerun()
        history_rows = [{"Created (UTC)": x["timestamp"], "Source": x["source"], "Model": x["model"],
                         "Top crop": x["top_crop"].title(), "Soil screen": x["soil_quality"]}
                        for x in reversed(st.session_state.history)]
        st.dataframe(pd.DataFrame(history_rows), hide_index=True, width="stretch")
    else:
        st.info("No predictions in this browser session yet. Empty history is normal.")
    st.markdown("#### Data source register")
    st.dataframe(pd.DataFrame(soil_data_sources()), hide_index=True, width="stretch")
    st.markdown("**Crop labels and features**")
    st.write(f"`{Path(DATA_PATH).as_posix()}` · {evaluation['dataset_rows']:,} labeled rows · {evaluation['crop_count']} crop classes · seven features: {', '.join(FEATURES)}.")
    st.write("Source: public Crop Recommendation dataset on Kaggle, with the CSV mirror linked in README. The dataset does not establish local yield, crop-specific field outcomes, or soil-health labels.")
    st.markdown("**Soil measurement provenance**")
    st.write("Soil values shown above come from user entry or a user-uploaded lab CSV. No soil measurements are synthesized. When the optional geospatial extra is installed and ISRIC WCS responds, the Soil tab can retrieve separate, modelled SoilGrids pH and organic-carbon estimates with source, retrieval time, depth, resolution, and prediction interval.")
    st.markdown("**Weather provenance**")
    st.write("Weather values are requested live from Open-Meteo. Current conditions are model-based estimates rather than physical-station observations; forecasts remain separate. Geocoding results identify their provider, and weather API failures show errors without fallback values.")

