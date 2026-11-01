"""Input/source validation for manual and laboratory soil observations."""
from io import BytesIO, StringIO

import numpy as np
import pandas as pd

from ml_model import FEATURES

RANGES = {
    "N": (0, 300), "P": (0, 300), "K": (0, 300), "ph": (0, 14),
    "temperature": (-10, 60), "humidity": (0, 100), "rainfall": (0, 2000),
    "soil_moisture": (0, 1), "organic_carbon": (0, 100),
}
LAB_FIELDS = ("N", "P", "K", "ph", "soil_moisture", "organic_carbon")
LAB_COLUMNS = {"ph_value": "ph", "pH": "ph", "organic carbon": "organic_carbon",
               "organic_carbon_g_kg": "organic_carbon", "moisture": "soil_moisture"}
CANONICAL_COLUMNS = {name.lower(): name for name in RANGES}


def validate_inputs(values, require_model_features=False, npk_units="dataset_scale"):
    """Return readable validation errors; never coerce absent or non-finite data."""
    errors = []
    required = set(FEATURES) if require_model_features else set()
    names = list(FEATURES if require_model_features else values)
    names.extend(name for name in ("soil_moisture", "organic_carbon") if name in values and name not in names)
    for name in names:
        value = values.get(name)
        if value is None or pd.isna(value):
            if name in required:
                errors.append(f"{name} is required for crop model inference.")
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric.")
            continue
        if not np.isfinite(numeric):
            errors.append(f"{name} must be finite.")
            continue
        if name in RANGES:
            low, high = RANGES[name]
            if name in {"N", "P", "K"} and npk_units not in {"dataset_scale", "published_dataset_scale"}:
                high = 100000.0 if npk_units == "kg/ha" else 5000.0
            if not low <= numeric <= high:
                errors.append(f"{name} must be between {low} and {high}.")
    return errors


def read_lab_report(uploaded_file, npk_units=None):
    """Read exactly one CSV lab row. Expected fields are N,P,K,ph and optional values.

    Include `npk_units` as `dataset_scale` to authorize model use. Other lab units
    are preserved but deliberately not converted into crop-model features.
    """
    if uploaded_file is None:
        raise ValueError("Choose a soil laboratory CSV report first.")
    content = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file
    try:
        frame = pd.read_csv(BytesIO(content) if isinstance(content, bytes) else StringIO(str(content)))
    except Exception as exc:
        raise ValueError(f"Could not read the soil report CSV: {exc}") from exc
    if len(frame) != 1:
        raise ValueError("The soil report CSV must contain exactly one sample row.")
    row = {str(k).strip(): v for k, v in frame.iloc[0].to_dict().items()}
    normalized = {LAB_COLUMNS.get(k.strip().lower(), CANONICAL_COLUMNS.get(k.strip().lower(), k.strip())): v
                  for k, v in row.items()}
    absent = {"N", "P", "K", "ph"} - set(normalized)
    if absent:
        raise ValueError("The report must contain N, P, K, and ph columns. Missing: " + ", ".join(sorted(absent)))
    allowed = set(LAB_FIELDS) | {"npk_units", "sample_id", "lab_name", "report_date", "ph_units"}
    unknown = set(normalized) - allowed
    if unknown:
        raise ValueError("Unrecognized CSV columns: " + ", ".join(sorted(unknown)))
    values = {}
    for name in LAB_FIELDS:
        raw = normalized.get(name)
        if raw is None or pd.isna(raw) or str(raw).strip() == "":
            values[name] = None
            continue
        try:
            values[name] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain a numeric value.") from exc
    declared_unit = str(normalized.get("npk_units") or "").strip().lower()
    selected_unit = str(npk_units or "").strip().lower()
    if not declared_unit and not selected_unit:
        raise ValueError("NPK units are not specified. Add an npk_units CSV column or choose Unknown/known units in the app.")
    if selected_unit and declared_unit and selected_unit != declared_unit:
        raise ValueError(f"Selected NPK units ({selected_unit}) conflict with the CSV declaration ({declared_unit}).")
    unit = selected_unit or declared_unit or "unknown"
    allowed_units = {"dataset_scale", "published_dataset_scale", "mg/kg", "ppm", "kg/ha", "unknown"}
    if unit not in allowed_units:
        raise ValueError("NPK units must be dataset_scale, mg/kg, ppm, kg/ha, or unknown.")
    errors = validate_inputs(values, npk_units=unit)
    if errors:
        raise ValueError("Invalid report values: " + " ".join(errors))
    return {
        "values": values, "npk_units": unit,
        "model_compatible": unit in {"dataset_scale", "published_dataset_scale"},
        "sample_id": _text_or_none(normalized.get("sample_id")),
        "lab_name": _text_or_none(normalized.get("lab_name")),
        "report_date": _text_or_none(normalized.get("report_date")),
        "source": "Laboratory report uploaded by user",
    }


def _text_or_none(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


class SoilDataProvider:
    """Small provider interface for future measured/remote soil data adapters."""
    source_name = "Unconfigured soil data provider"

    def fetch(self, latitude, longitude):
        raise NotImplementedError


def soil_data_sources():
    """Describe currently supported and investigated sources without inventing values."""
    return [
        {"name": "Manual input", "status": "Available", "kind": "User-provided measurement"},
        {"name": "Laboratory CSV", "status": "Available", "kind": "User-provided laboratory report"},
        {"name": "SoilGrids WCS", "status": "Optional; install requirements-geo.txt",
         "kind": "Modelled gridded estimate; 250 m product with uncertainty layers",
         "url": "https://docs.isric.org/globaldata/soilgrids/wcs.html"},
    ]
