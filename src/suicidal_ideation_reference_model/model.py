from __future__ import annotations

from importlib import resources
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
DEFAULT_ARTIFACT = "si_xgb_full_2020_v0_1_0.joblib"


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


def predict_dataframe(
    df: pd.DataFrame,
    *,
    bundle: dict[str, Any] | None = None,
    threshold: float | None = None,
    probability_column: str = "si_probability",
    flag_column: str = "si_flag",
) -> pd.DataFrame:
    """Append reference-model probabilities to a dataframe.

    If `threshold` is supplied, a binary flag column is also appended. The flag
    is an operating-point convenience for validation studies, not a clinical or
    employment decision.
    """
    loaded = bundle or load_reference_model()
    features = list(loaded.get("features", FEATURE_COLUMNS))
    validate_columns(df, features)

    out = df.copy()
    proba = loaded["pipeline"].predict_proba(out[features])[:, 1]
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
