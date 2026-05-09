from pathlib import Path

import pandas as pd

from suicidal_ideation_reference_model import load_reference_model, predict_dataframe


def test_packaged_model_scores_example() -> None:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "examples" / "example_input.csv")
    bundle = load_reference_model()

    scored = predict_dataframe(df, bundle=bundle, threshold=0.17)

    assert "si_probability" in scored.columns
    assert "si_flag" in scored.columns
    assert scored["si_probability"].between(0, 1).all()
    assert len(scored) == len(df)
