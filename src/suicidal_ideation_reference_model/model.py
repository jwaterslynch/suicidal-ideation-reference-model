from __future__ import annotations

from importlib import resources
import math
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "k6_score",
    "male",
    "age",
    "married",
    "lgbtq",
    "veteran",
    "drug_use",
    "mental_health_tx",
    "work_hours",
]

DEFAULT_THRESHOLD = 0.17
DEFAULT_ARTIFACT = "si_xgb_full_2020_v0_1_1.joblib"

BINARY_FEATURES = [
    "male",
    "married",
    "lgbtq",
    "veteran",
    "drug_use",
    "mental_health_tx",
]
FEATURE_RANGES = {
    "k6_score": (0, 24),
    "age": (1, 4),
    "work_hours": (0, 168),
}


def load_reference_model(path: str | Path | None = None) -> dict[str, Any]:
    """Load the packaged reference-model bundle."""
    if path is not None:
        return joblib.load(path)

    artifact = resources.files(__package__).joinpath("artifacts", DEFAULT_ARTIFACT)
    with resources.as_file(artifact) as artifact_path:
        return joblib.load(artifact_path)


def validate_columns(df: pd.DataFrame, features: list[str] | None = None) -> None:
    """Raise a clear error if required model inputs are absent."""
    expected = features or FEATURE_COLUMNS
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(
            "Input data are missing required model columns: "
            + ", ".join(missing)
            + ". See docs/DATA_DICTIONARY.md for the expected schema."
        )


def validate_threshold(threshold: float | None) -> None:
    """Raise a clear error if a supplied threshold is not a probability."""
    if threshold is None:
        return
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("threshold must be a finite number between 0 and 1.")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be a finite number between 0 and 1.")


def _numeric_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return numeric model inputs, failing on nonnumeric observed values."""
    numeric = pd.DataFrame(index=df.index)
    invalid_columns: list[str] = []
    for feature in features:
        converted = pd.to_numeric(df[feature], errors="coerce")
        invalid = df[feature].notna() & converted.isna()
        if invalid.any():
            invalid_columns.append(feature)
        numeric[feature] = converted

    if invalid_columns:
        raise ValueError(
            "Input data contain nonnumeric values in required model columns: "
            + ", ".join(invalid_columns)
            + ". See docs/DATA_DICTIONARY.md for the expected schema."
        )
    return numeric


def validate_feature_values(df: pd.DataFrame, features: list[str] | None = None) -> None:
    """Raise clear errors for observed feature values outside the reference schema."""
    expected = features or FEATURE_COLUMNS
    numeric = _numeric_feature_frame(df, expected)
    violations: list[str] = []

    for feature in BINARY_FEATURES:
        if feature in numeric.columns:
            observed = numeric[feature].dropna()
            bad = observed[~observed.isin([0, 1])]
            if not bad.empty:
                violations.append(f"{feature} must be coded 0/1")

    for feature, (low, high) in FEATURE_RANGES.items():
        if feature in numeric.columns:
            observed = numeric[feature].dropna()
            bad = observed[(observed < low) | (observed > high)]
            if not bad.empty:
                violations.append(f"{feature} must be between {low} and {high}")

    if numeric[expected].empty:
        raise ValueError("Input data contain no rows to score.")

    all_missing = numeric[expected].isna().all(axis=1)
    if all_missing.any():
        row_labels = [str(label) for label in numeric.index[all_missing][:5]]
        suffix = "" if all_missing.sum() <= 5 else ", ..."
        raise ValueError(
            "Input data contain rows with all model features missing. "
            f"Refusing to score row labels: {', '.join(row_labels)}{suffix}."
        )

    if violations:
        raise ValueError(
            "Input data contain values outside the reference-model schema: "
            + "; ".join(violations)
            + ". Missing values are allowed for imputation, but observed values "
            "must match docs/DATA_DICTIONARY.md."
        )


def predict_dataframe(
    df: pd.DataFrame,
    *,
    bundle: dict[str, Any] | None = None,
    threshold: float | None = None,
    probability_column: str = "si_probability",
    flag_column: str = "si_flag",
    missing_count_column: str = "si_n_missing",
) -> pd.DataFrame:
    """Append reference-model probabilities to a dataframe.

    If `threshold` is supplied, a binary flag column is also appended. The flag
    is an operating-point convenience for validation studies, not a clinical or
    employment decision. A missing-feature count is appended for transparency;
    rows with all model inputs missing are refused rather than imputed.
    """
    validate_threshold(threshold)
    loaded = bundle or load_reference_model()
    features = list(loaded.get("features", FEATURE_COLUMNS))
    validate_columns(df, features)
    validate_feature_values(df, features)

    out = df.copy()
    output_columns = [probability_column, missing_count_column]
    if threshold is not None:
        output_columns.append(flag_column)
    collisions = [col for col in output_columns if col in out.columns]
    if collisions:
        raise ValueError(
            "Input data already contain output column(s): "
            + ", ".join(collisions)
            + ". Use different output column names or remove existing predictions."
        )

    model_input = _numeric_feature_frame(out, features)
    out[missing_count_column] = model_input[features].isna().sum(axis=1)
    proba = loaded["pipeline"].predict_proba(model_input[features])[:, 1]
    out[probability_column] = proba
    if threshold is not None:
        out[flag_column] = (out[probability_column] >= threshold).astype(int)
    return out


def score_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    threshold: float | None = None,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Score a CSV file and write predictions to another CSV file."""
    bundle = load_reference_model(model_path)
    df = pd.read_csv(input_path)
    scored = predict_dataframe(df, bundle=bundle, threshold=threshold)
    scored.to_csv(output_path, index=False)
    return scored
