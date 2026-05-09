from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from suicidal_ideation_reference_model import (
    FEATURE_COLUMNS,
    load_reference_model,
    predict_dataframe,
)
from suicidal_ideation_reference_model.model import DEFAULT_ARTIFACT


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "example_input.csv"


@pytest.fixture(scope="session")
def bundle():
    return load_reference_model()


@pytest.fixture()
def example_df() -> pd.DataFrame:
    return pd.read_csv(EXAMPLE)


def test_packaged_model_scores_example(example_df: pd.DataFrame, bundle) -> None:
    scored = predict_dataframe(example_df, bundle=bundle, threshold=0.17)

    assert "si_n_missing" in scored.columns
    assert "si_probability" in scored.columns
    assert "si_flag" in scored.columns
    assert scored["si_probability"].between(0, 1).all()
    assert len(scored) == len(example_df)


def test_predictions_are_deterministic_and_pinned(
    example_df: pd.DataFrame, bundle
) -> None:
    scored = predict_dataframe(example_df, bundle=bundle, threshold=0.17)
    expected = np.array(
        [
            0.0140510643,
            0.0714770041,
            0.3196183581,
            0.1516790507,
        ]
    )

    assert np.allclose(scored["si_probability"].to_numpy(), expected, atol=1e-9)
    assert scored["si_flag"].tolist() == [0, 0, 1, 0]


def test_missing_required_column_raises(example_df: pd.DataFrame, bundle) -> None:
    df = example_df.drop(columns=["k6_score"])

    with pytest.raises(ValueError, match="missing required model columns"):
        predict_dataframe(df, bundle=bundle)


def test_extra_columns_are_preserved(example_df: pd.DataFrame, bundle) -> None:
    df = example_df.copy()
    df["extra_context"] = "synthetic"

    scored = predict_dataframe(df, bundle=bundle)

    assert "extra_context" in scored.columns
    assert scored["extra_context"].eq("synthetic").all()


def test_shuffled_columns_match_canonical(example_df: pd.DataFrame, bundle) -> None:
    canonical = predict_dataframe(example_df, bundle=bundle)
    shuffled = predict_dataframe(
        example_df[list(reversed(FEATURE_COLUMNS))],
        bundle=bundle,
    )

    assert np.allclose(
        canonical["si_probability"].to_numpy(),
        shuffled["si_probability"].to_numpy(),
    )


def test_partial_missing_returns_missing_count(
    example_df: pd.DataFrame, bundle
) -> None:
    df = example_df.copy()
    df.loc[0, "k6_score"] = np.nan

    scored = predict_dataframe(df, bundle=bundle)

    assert scored.loc[0, "si_n_missing"] == 1
    assert np.isfinite(scored.loc[0, "si_probability"])


def test_all_missing_row_is_refused(example_df: pd.DataFrame, bundle) -> None:
    df = example_df.astype(float)
    df.loc[0, FEATURE_COLUMNS] = np.nan

    with pytest.raises(ValueError, match="all model features missing"):
        predict_dataframe(df, bundle=bundle)


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, "0.17"])
def test_invalid_thresholds_raise(example_df: pd.DataFrame, bundle, bad) -> None:
    with pytest.raises(ValueError, match="threshold"):
        predict_dataframe(example_df, bundle=bundle, threshold=bad)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("male", 5),
        ("age", 99),
        ("work_hours", -50),
        ("k6_score", 9999),
    ],
)
def test_invalid_feature_values_raise(
    example_df: pd.DataFrame, bundle, column: str, bad_value: int
) -> None:
    df = example_df.copy()
    df.loc[0, column] = bad_value

    with pytest.raises(ValueError, match="outside the reference-model schema"):
        predict_dataframe(df, bundle=bundle)


def test_nonnumeric_feature_values_raise(example_df: pd.DataFrame, bundle) -> None:
    df = example_df.copy()
    df["k6_score"] = df["k6_score"].astype(object)
    df.loc[0, "k6_score"] = "not-a-number"

    with pytest.raises(ValueError, match="nonnumeric values"):
        predict_dataframe(df, bundle=bundle)


def test_output_column_collision_raises(example_df: pd.DataFrame, bundle) -> None:
    df = example_df.copy()
    df["si_probability"] = 0.0

    with pytest.raises(ValueError, match="already contain output column"):
        predict_dataframe(df, bundle=bundle)


def test_artifact_metadata_matches_packaged_model(bundle) -> None:
    metadata_path = (
        ROOT
        / "src"
        / "suicidal_ideation_reference_model"
        / "artifacts"
        / "si_xgb_full_2020_v0_1_1.metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())

    assert DEFAULT_ARTIFACT == "si_xgb_full_2020_v0_1_1.joblib"
    assert bundle["metadata"]["model_id"] == "si_xgb_full_2020_v0_1_1"
    assert metadata["features"] == FEATURE_COLUMNS
    assert metadata["n_test"] == 3738
    assert metadata["positive_test"] == 206
    assert round(metadata["metrics"]["test_auc"], 3) == 0.872
    assert round(metadata["metrics"]["test_brier"], 4) == 0.0438
    assert "fit_environment" in metadata


def test_cli_writes_output(tmp_path: Path) -> None:
    output = tmp_path / "predictions.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suicidal_ideation_reference_model.cli",
            str(EXAMPLE),
            "--output",
            str(output),
            "--threshold",
            "0.17",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output.exists()
    scored = pd.read_csv(output)
    assert "si_probability" in scored.columns
    assert "si_n_missing" in scored.columns


def test_cli_returns_nonzero_on_missing_columns(tmp_path: Path) -> None:
    bad_input = tmp_path / "bad.csv"
    bad_input.write_text("a,b\n1,2\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suicidal_ideation_reference_model.cli",
            str(bad_input),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "missing required model columns" in result.stderr


def test_built_wheel_contains_default_artifact_if_present() -> None:
    wheels = list((ROOT / "dist").glob("*.whl"))
    if not wheels:
        pytest.skip("wheel not built")

    with zipfile.ZipFile(wheels[-1]) as archive:
        names = archive.namelist()

    assert any(name.endswith(DEFAULT_ARTIFACT) for name in names)
    assert any(name.endswith("si_xgb_full_2020_v0_1_1.metadata.json") for name in names)
