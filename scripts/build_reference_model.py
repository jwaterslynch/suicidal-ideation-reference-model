#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = [
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


def import_reproduction_pipeline(source_repo: Path):
    pipeline_path = source_repo / "code" / "workplace_si_ml_pipeline.py"
    spec = importlib.util.spec_from_file_location("si_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import reproduction pipeline at {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operating_point(y_true, probabilities, threshold: float) -> dict:
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n": int(len(y_true)),
    }


def build_reference_model(source_repo: Path, output_dir: Path) -> dict:
    source_repo = source_repo.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    si_pipeline = import_reproduction_pipeline(source_repo)

    mapping_path = source_repo / "data" / "variable_mapping.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
    else:
        raw = pd.read_csv(source_repo / "data" / "NSDUH_2020.tab", sep="\t", low_memory=False)
        mapping = si_pipeline.align_variables_across_years(
            {2020: si_pipeline.validate_year_data(raw, 2020)}
        )

    clean = si_pipeline.load_and_clean_year(2020, source_repo / "data", mapping)
    if si_pipeline.EMPLOYMENT_FILTER and "employment" in clean.columns:
        clean = clean[clean["employment"] == 1].copy()

    X = clean[FEATURES]
    y = clean["suicide"]
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    valid = y.isin([0, 1])
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    base = XGBClassifier(
        max_depth=3,
        n_estimators=150,
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        random_state=42,
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=base,
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    p_train = pipeline.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, p_train)
    f1 = (2 * precision * recall) / np.where(precision + recall == 0, 1, precision + recall)
    f1_threshold = float(thresholds[np.argmax(f1)])

    p_test = pipeline.predict_proba(X_test)[:, 1]
    high_spec = min(
        (operating_point(y_test, p_test, thr) for thr in np.arange(0.05, 0.9000001, 0.01)),
        key=lambda d: abs(d["specificity"] - 0.93),
    )
    metadata = {
        "model_id": "si_xgb_full_2020_v0_1_0",
        "version": "0.1.0",
        "artifact_file": "si_xgb_full_2020_v0_1_0.joblib",
        "model_type": "Calibrated XGBoost classifier",
        "source_repository": "https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline",
        "training_data": "NSDUH 2020 public-use data, employed-adult analytic sample",
        "outcome": "Past-year suicidal ideation as coded in the paper reproduction pipeline",
        "features": FEATURES,
        "split": {"test_size": 0.30, "stratified": True, "random_state": 42},
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_train": int(y_train.sum()),
        "positive_test": int(y_test.sum()),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_test.mean()),
        "metrics": {
            "test_auc": float(roc_auc_score(y_test, p_test)),
            "test_brier": float(brier_score_loss(y_test, p_test)),
            "f1_threshold_from_train": f1_threshold,
            "f1_threshold_test_operating_point": operating_point(y_test, p_test, f1_threshold),
            "high_specificity_reference": high_spec,
        },
        "preprocessing": {
            "imputer": 'SimpleImputer(strategy="median") fitted on training split',
            "scaler": "StandardScaler fitted after median imputation on training split",
        },
        "classifier_params": {
            "max_depth": 3,
            "n_estimators": 150,
            "scale_pos_weight": float((len(y_train) - y_train.sum()) / y_train.sum()),
            "random_state": 42,
            "calibration": 'CalibratedClassifierCV(method="sigmoid", cv=5)',
        },
        "governance_boundary": (
            "Reference model for research, local validation, and governed support workflows; "
            "not a diagnostic system or automated decision tool."
        ),
    }
    bundle = {"pipeline": pipeline, "features": FEATURES, "metadata": metadata}
    joblib.dump(bundle, output_dir / metadata["artifact_file"])
    (output_dir / "si_xgb_full_2020_v0_1_0.metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the packaged reference model.")
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path("../../Research/suicidal_ideation_pipeline"),
        help="Path to the paper reproduction repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/suicidal_ideation_reference_model/artifacts"),
        help="Where to write the model artifact and metadata.",
    )
    args = parser.parse_args()
    metadata = build_reference_model(args.source_repo, args.output_dir)
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
