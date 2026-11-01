"""Leakage-safe crop models and transparent soil screening rules."""
from pathlib import Path
import hashlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
DATA_PATH = Path(__file__).parent / "data" / "Crop_recommendation.csv"
MODEL_PATH = Path(__file__).parent / "models" / "crop_models.joblib"
MODEL_BUNDLE_VERSION = 3


def load_dataset(path=DATA_PATH):
    """Load the published labeled dataset; malformed observations are rejected."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. See README for its source.")
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df = df[FEATURES + [TARGET]].copy()
    for name in FEATURES:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df[TARGET] = df[TARGET].astype("string").str.strip().str.lower()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
    if len(df) < 10 or df[TARGET].nunique() < 2:
        raise ValueError("Dataset has too few valid rows or crop classes to evaluate models.")
    return df.reset_index(drop=True)


def _model_metrics(estimator, X_test, y_test, labels):
    prediction = estimator.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision_macro": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=labels).tolist(),
        "labels": list(labels),
        "report": classification_report(y_test, prediction, labels=labels, output_dict=True, zero_division=0),
    }


def train_models(path=DATA_PATH, save=True):
    """Train/evaluate KNN and RF against one stratified, shared holdout split."""
    df = load_dataset(path)
    X, y = df[FEATURES], df[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)),
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=1, class_weight="balanced_subsample",
                random_state=42, n_jobs=-1,
            )),
        ]),
    }
    # Choose a default model using training rows only. The held-out test split
    # remains untouched until the final, comparable evaluation below.
    min_class_rows = int(y_train.value_counts().min())
    if min_class_rows < 2:
        raise ValueError("Each crop class needs at least two training rows for stratified model selection.")
    n_splits = min(5, min_class_rows)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    selection = {
        name: cross_validate(estimator, X_train, y_train, scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
                             cv=cv, n_jobs=1, return_train_score=False)
        for name, estimator in models.items()
    }
    selection_scores = {
        name: {"macro_f1": float(np.mean(scores["test_macro_f1"])),
               "accuracy": float(np.mean(scores["test_accuracy"]))}
        for name, scores in selection.items()
    }
    for estimator in models.values():
        estimator.fit(X_train, y_train)
    labels = sorted(y.unique())
    metrics = {name: _model_metrics(estimator, X_test, y_test, labels)
               for name, estimator in models.items()}
    best_name = max(models, key=lambda name: (selection_scores[name]["macro_f1"], selection_scores[name]["accuracy"]))
    context = {
        "best_model": best_name,
        "train_medians": X_train.median().to_dict(),
        "train_q25": X_train.quantile(.25).to_dict(),
        "train_q75": X_train.quantile(.75).to_dict(),
        "crops": {},
    }
    train_rows = X_train.assign(label=y_train.to_numpy())
    for crop, crop_rows in train_rows.groupby("label"):
        crop_values = crop_rows[FEATURES]
        context["crops"][crop] = {
            "q25": crop_values.quantile(.25).to_dict(),
            "median": crop_values.median().to_dict(),
            "q75": crop_values.quantile(.75).to_dict(),
        }
    summary = {
        "models": metrics,
        "best_model": best_name,
        "selection_method": f"{n_splits}-fold stratified cross-validation on training split; macro-F1 then accuracy",
        "selection_cv": selection_scores,
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "dataset_rows": int(len(df)),
        "crop_count": int(y.nunique()),
        "class_counts": y.value_counts().sort_values(ascending=False),
        "feature_importances": (
            dict(zip(FEATURES, models["Random Forest"].named_steps["classifier"].feature_importances_.tolist()))
        ),
    }
    summary["dataset_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if save:
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"version": MODEL_BUNDLE_VERSION, "models": models, "best_model": best_name, "context": context,
                         "summary": summary, "features": FEATURES,
                         "dataset_sha256": summary["dataset_sha256"]}, MODEL_PATH)
        except Exception as exc:
            summary["persistence_warning"] = f"Models trained in memory but could not be saved: {exc}"
    return models, summary, context


def load_or_train_models(path=DATA_PATH):
    """Reuse a valid matching local bundle; retrain safely after load errors."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None
    if digest and MODEL_PATH.is_file():
        try:
            bundle = joblib.load(MODEL_PATH)
            summary = bundle.get("summary", {})
            context = bundle.get("context", {})
            if (bundle.get("version") == MODEL_BUNDLE_VERSION and bundle.get("features") == FEATURES and bundle.get("dataset_sha256") == digest
                    and set(bundle.get("models", {})) == {"KNN", "Random Forest"}
                    and all(key in summary for key in ("best_model", "selection_method", "selection_cv", "models", "test_rows", "train_rows"))
                    and summary.get("best_model") in bundle["models"]
                    and context.get("best_model") == summary.get("best_model")):
                return bundle["models"], summary, context
        except Exception:
            # A broken, outdated, or incompatible serialization is regenerated below.
            pass
    return train_models(path=path, save=True)


def train_model(path=DATA_PATH, save=True):
    """Backward-compatible KNN entry point for existing callers."""
    models, summary, _ = train_models(path=path, save=save)
    return models["KNN"], {
        "accuracy": summary["models"]["KNN"]["accuracy"],
        "macro_f1": summary["models"]["KNN"]["macro_f1"],
        "report": summary["models"]["KNN"]["report"],
        "test_rows": summary["test_rows"], "dataset_rows": summary["dataset_rows"],
        "crop_count": summary["crop_count"], "class_counts": summary["class_counts"],
    }


def predict_top_crops(models, values, model_name="best", context=None, top_n=5):
    """Rank crop labels with native model scores and add training-data context."""
    if model_name == "best":
        model_name = (context or {}).get("best_model")
        if model_name is None:
            raise ValueError("The selected best model is unavailable; pass the training context or a model name.")
    estimator = models[model_name] if isinstance(models, dict) else models
    row = pd.DataFrame([{name: float(values[name]) for name in FEATURES}], columns=FEATURES)
    scores = estimator.predict_proba(row)[0]
    classifier = estimator.named_steps["classifier"]
    ranked = sorted(zip(classifier.classes_, scores), key=lambda pair: pair[1], reverse=True)[:top_n]
    output = []
    for crop, score in ranked:
        close_features = []
        crop_range = context.get("crops", {}).get(str(crop)) if context else None
        outside_features = []
        if crop_range:
            deviations = []
            for feature in FEATURES:
                q25, q75 = crop_range["q25"][feature], crop_range["q75"][feature]
                scale = max(float(q75 - q25), 1e-6)
                deviations.append((abs(float(values[feature]) - crop_range["median"][feature]) / scale, feature))
                if not q25 <= float(values[feature]) <= q75:
                    outside_features.append(feature)
            close_features = [f for _, f in sorted(deviations)[:3]]
        output.append({
            "crop": str(crop), "score": float(score),
            "score_label": "KNN neighbor vote share" if isinstance(classifier, KNeighborsClassifier) else "Random Forest model score",
            "why": ("Closest crop-profile features: " + ", ".join(close_features)
                    if close_features else "Ranked by the selected model from its training patterns."),
            "outside_middle_range": outside_features,
        })
    return output


def crop_input_explanation(values, crop, context, dataset=None):
    """Explain input proximity to the named crop's training distribution."""
    crop_range = context.get("crops", {}).get(str(crop).lower())
    if crop_range is None:
        return []
    distances = []
    for feature in FEATURES:
        low, high = crop_range["q25"][feature], crop_range["q75"][feature]
        scale = max(float(high - low), 1e-6)
        distances.append((abs(float(values[feature]) - float(crop_range["median"][feature])) / scale,
                          feature, float(values[feature]), float(low), float(high)))
    distances.sort()
    explanations = [
        f"{feature}: entered {value:.1f}; within the middle half of {crop} examples ({low:.1f}–{high:.1f})"
        if low <= value <= high else
        f"{feature}: entered {value:.1f}; outside the middle half of {crop} examples ({low:.1f}–{high:.1f})"
        for _, feature, value, low, high in distances[:3]
    ]
    outside = [feature for _, feature, value, low, high in distances if not low <= value <= high]
    if outside:
        explanations.append("Outside this crop's middle example range: " + ", ".join(outside))
    return explanations


def classify_soil_quality(values, npk_units="dataset_scale"):
    """Transparent screening, not a trained classifier or lab diagnosis.

    On the published crop-dataset scale: N<40, P<20, K<20 are low flags;
    pH<5.5 or >8.5 is an extreme flag; pH outside 6.0–7.5 is suboptimal.
    Two low/extreme flags means Poor, one means Fair. Missing values are never
    treated as good: without any NPK or pH data status is Incomplete.
    """
    def present(name):
        value = values.get(name)
        return value is not None and pd.notna(value)

    flags, notes, suggestions, nutrient_status = [], [], [], {}
    compatible_npk = npk_units in {"dataset_scale", "published_dataset_scale"}
    thresholds = {"N": 40.0, "P": 20.0, "K": 20.0}
    for name, low_cutoff in thresholds.items():
        if not present(name):
            nutrient_status[name] = "Not provided"
            continue
        value = float(values[name])
        if not compatible_npk:
            nutrient_status[name] = f"Reported; {npk_units} not screened"
        elif value < low_cutoff:
            nutrient_status[name] = "Low screening value"
            flags.append(name)
            suggestions.append(f"Review {name} with a lab report; consider locally recommended organic matter or a crop-specific amendment.")
        else:
            nutrient_status[name] = "No low-value flag"
    ph_status = "Not provided"
    if present("ph"):
        ph = float(values["ph"])
        if ph < 5.5 or ph > 8.5:
            ph_status = "Extreme screening value"
            flags.append("pH")
            suggestions.append("Ask a local soil lab or extension service to interpret pH and recommend any correction.")
        elif ph < 6.0 or ph > 7.5:
            ph_status = "Outside broad 6.0–7.5 band"
            suggestions.append("Confirm pH with a lab test and use crop- and region-specific guidance.")
        else:
            ph_status = "Within broad 6.0–7.5 band"
    missing = [name for name in (*thresholds, "ph") if not present(name)]
    unusable = list(thresholds) if not compatible_npk else []
    if missing:
        notes.append("Missing measurements: " + ", ".join(missing) + ". Screening is incomplete.")
    if unusable:
        notes.append("NPK values are reported in " + npk_units + "; dataset-scale screening cutoffs were not applied.")
    if len(flags) >= 2:
        quality = "Poor"
    elif flags or ph_status == "Outside broad 6.0–7.5 band":
        quality = "Fair"
    elif missing or unusable:
        quality = "Incomplete"
    else:
        quality = "Good"
    notes.extend(f"Low screening value for {name}" for name in flags if name in thresholds)
    if "pH" in flags:
        notes.append("pH is outside the broad screening range")
    if not notes:
        notes.append("No flags under the documented screening rules")
    if not suggestions:
        suggestions.append("Keep soil-test records and follow local crop nutrient recommendations.")
    optional = {}
    for name, limits in (("soil_moisture", (0, 1)), ("organic_carbon", (0, 100))):
        value = values.get(name)
        if value is None:
            optional[name] = "Not provided"
        elif not limits[0] <= float(value) <= limits[1]:
            optional[name] = "Outside accepted input range"
        else:
            optional[name] = "User/lab value recorded; no quality threshold applied"
    return {"quality": quality, "notes": notes, "suggestions": list(dict.fromkeys(suggestions)),
            "nutrients": nutrient_status, "ph": ph_status, "missing": missing, "unusable": unusable,
            "optional": optional}
