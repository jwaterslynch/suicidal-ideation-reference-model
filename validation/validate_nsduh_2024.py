#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
from pathlib import Path
import sys
import urllib.request

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from suicidal_ideation_reference_model import DEFAULT_THRESHOLD, predict_dataframe

DATA_URL = (
    "https://588738577887-baselight-crawlers-prod-ue1-datasets.s3.us-east-1.amazonaws.com/"
    "iceberg_catalog/samhsa-bltc58ad2a81b9da05838ae665ebc809396dbe8538fa9fd6abf5a09667bd6cab0b1/"
    "nsduh_nsduh_2024/data/data_v2026-04-12_25f5d582.parquet"
)
DICTIONARY_URL = (
    "https://588738577887-baselight-crawlers-prod-ue1-datasets.s3.us-east-1.amazonaws.com/"
    "iceberg_catalog/samhsa-bltc58ad2a81b9da05838ae665ebc809396dbe8538fa9fd6abf5a09667bd6cab0b1/"
    "nsduh_nsduh_2024/data_dictionary/data_dictionary_v2026-03-11_bb75438a.parquet"
)

RAW_DIR = Path("validation/raw")
RESULTS_DIR = Path("validation/results")
DATA_PATH = RAW_DIR / "nsduh_2024_data.parquet"
DICTIONARY_PATH = RAW_DIR / "nsduh_2024_data_dictionary.parquet"

SKIP_CODES = {85, 89, 94, 97, 98, 99}
EXTENDED_SKIP_CODES = {985, 989, 994, 997, 998, 999}

SOURCE_CANDIDATES = {
    "outcome": ["irsuicthnk", "suicthnk", "mhsuithk"],
    "k6_score": [
        "ksslr6max",
        "ksslr6yr",
        "ksslr6mon",
        "ksslr6moned",
        "k6scmax",
        "k6scyr",
        "k6scmon",
    ],
    "employment": ["wrkstatwk2", "wrkstatwk", "irwrkstat18", "irwrkstat"],
    "work_hours": ["wrkdhrswk2", "wrkdhrswk", "wrkhrsus2", "wrkhrsjob2"],
    "sex": ["irsex"],
    "age": ["catage", "age3"],
    "marital": ["irmarit"],
    "sexual_orientation": ["sexident", "irsexident", "sexident1", "sexident2"],
    "military": ["milstat", "service", "actdever"],
    "drug_use": ["illyr", "irillyr", "anyillyr", "anydrugyr"],
    "mental_health_tx": ["mhtrtpy2", "mhtrtosvpy2", "mhtrtpy", "mhtoppy2", "mhtinppy2"],
    "weight": ["analwt2_c", "analwt_c"],
    "stratum": ["vestr_c", "vestr"],
    "psu": ["verep"],
    "race": ["newrace2"],
}


def content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def download_if_needed(url: str, path: Path, *, max_bytes: int) -> None:
    if path.exists():
        return
    length = content_length(url)
    if length is not None and length > max_bytes:
        raise RuntimeError(
            f"Refusing to download {url}: {length:,} bytes exceeds "
            f"the configured limit of {max_bytes:,} bytes."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def choose_column(columns: set[str], candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a source column for {label}.")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def binary_1_yes_2_no(series: pd.Series) -> pd.Series:
    out = numeric(series)
    out = out.mask(out.isin(SKIP_CODES))
    observed = set(out.dropna().unique())
    if observed and observed.issubset({0, 1}):
        return out.where(out.isin([0, 1]))
    out = out.replace({1: 1, 2: 0})
    return out.where(out.isin([0, 1]))


def data_dictionary_columns(dictionary_path: Path) -> set[str]:
    dictionary = pq.read_table(dictionary_path, columns=["name"]).to_pandas()
    return set(dictionary["name"].str.lower())


def pivot_long_parquet(
    data_path: Path,
    variable_names: list[str],
) -> pd.DataFrame:
    expressions = [
        "questid2",
        "max(analwt2_c) as analwt2_c",
        *[
            f"max(case when variable_name = '{name}' then value end) as {name}"
            for name in variable_names
        ],
    ]
    placeholders = ",".join(["?"] * len(variable_names))
    sql = f"""
        select {", ".join(expressions)}
        from read_parquet(?)
        where variable_name in ({placeholders})
        group by questid2
    """
    return duckdb.connect().execute(sql, [str(data_path), *variable_names]).fetchdf()


def build_validation_frame(
    data_path: Path,
    dictionary_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    parquet = pq.ParquetFile(data_path)
    schema_columns = {name.lower(): name for name in parquet.schema.names}
    is_long = {"questid2", "variable_name", "value"}.issubset(schema_columns)
    all_columns = data_dictionary_columns(dictionary_path) if is_long else schema_columns
    selected_lower = {
        key: choose_column(set(all_columns), candidates, key)
        for key, candidates in SOURCE_CANDIDATES.items()
        if key != "weight" and any(candidate in all_columns for candidate in candidates)
    }
    if "analwt2_c" in schema_columns:
        selected_lower["weight"] = "analwt2_c"
    elif any(candidate in all_columns for candidate in SOURCE_CANDIDATES["weight"]):
        selected_lower["weight"] = choose_column(
            set(all_columns),
            SOURCE_CANDIDATES["weight"],
            "weight",
        )
    required = [
        "outcome",
        "k6_score",
        "employment",
        "work_hours",
        "sex",
        "age",
        "marital",
        "military",
        "drug_use",
        "mental_health_tx",
    ]
    missing = [key for key in required if key not in selected_lower]
    if missing:
        raise ValueError(f"Missing required source variables: {', '.join(missing)}")

    if is_long:
        variable_names = sorted(
            {
                name
                for key, name in selected_lower.items()
                if key != "weight" and name in all_columns
            }
        )
        raw = pivot_long_parquet(data_path, variable_names)
    else:
        physical_columns = [schema_columns[name] for name in sorted(set(selected_lower.values()))]
        raw = pq.read_table(data_path, columns=physical_columns).to_pandas()
    raw.columns = [c.lower() for c in raw.columns]

    src = selected_lower
    out = pd.DataFrame(index=raw.index)
    out["suicide"] = binary_1_yes_2_no(raw[src["outcome"]])
    out["k6_score"] = numeric(raw[src["k6_score"]]).mask(lambda s: s.isin(SKIP_CODES))
    out["employment"] = numeric(raw[src["employment"]])
    out["work_hours"] = numeric(raw[src["work_hours"]]).mask(
        lambda s: s.isin(SKIP_CODES | EXTENDED_SKIP_CODES)
    )
    out["male"] = numeric(raw[src["sex"]]).map({1: 1, 2: 0})
    out["age"] = numeric(raw[src["age"]]).mask(lambda s: s.isin(SKIP_CODES))
    out["married"] = numeric(raw[src["marital"]]).map({1: 1, 2: 0, 3: 0, 4: 0})

    if "sexual_orientation" in src:
        sexual_orientation = numeric(raw[src["sexual_orientation"]])
        out["lgbtq"] = sexual_orientation.map({1: 0, 2: 1, 3: 1, 4: 1})
        out.loc[sexual_orientation.isin(SKIP_CODES), "lgbtq"] = np.nan
    else:
        src["sexual_orientation"] = "not_available_in_2024_public_use_file"
        out["lgbtq"] = np.nan

    military = numeric(raw[src["military"]])
    out["veteran"] = np.where(military.isin([2, 3]), 1.0, 0.0)
    out.loc[military.isin({85, 94, 97, 98}), "veteran"] = np.nan

    out["drug_use"] = binary_1_yes_2_no(raw[src["drug_use"]])
    out["mental_health_tx"] = binary_1_yes_2_no(raw[src["mental_health_tx"]])

    if "weight" in src:
        out["weight"] = numeric(raw[src["weight"]])
    if "stratum" in src:
        out["stratum"] = numeric(raw[src["stratum"]])
    if "psu" in src:
        out["psu"] = numeric(raw[src["psu"]])
    if "race" in src:
        out["race"] = numeric(raw[src["race"]])

    return out, src


def operating_point(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else math.nan
    spec = tn / (tn + fp) if (tn + fp) else math.nan
    ppv = tp / (tp + fp) if (tp + fp) else math.nan
    npv = tn / (tn + fn) if (tn + fn) else math.nan
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
        "flag_rate": float(pred.mean()),
    }


def calibration_intercept_slope(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    eps = 1e-6
    logits = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=2000)
    model.fit(logits.reshape(-1, 1), y_true)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return math.nan
    return float(np.average(values[mask], weights=weights[mask]))


def discrimination_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    y = df["suicide"].to_numpy(dtype=int)
    p = df["si_probability"].to_numpy(dtype=float)
    metrics = {
        "n": int(len(df)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_probability": float(p.mean()),
    }
    metrics.update({f"threshold_{DEFAULT_THRESHOLD:.2f}": operating_point(y, p, DEFAULT_THRESHOLD)})
    metrics["calibration"] = calibration_intercept_slope(y, p)
    if "weight" in df.columns:
        metrics["weighted_prevalence"] = weighted_mean(df["suicide"], df["weight"])
        metrics["weighted_flag_rate_at_default_threshold"] = weighted_mean(
            (df["si_probability"] >= DEFAULT_THRESHOLD).astype(float), df["weight"]
        )
    return metrics


def threshold_table(df: pd.DataFrame, thresholds: list[float]) -> list[dict[str, float | int]]:
    y = df["suicide"].to_numpy(dtype=int)
    p = df["si_probability"].to_numpy(dtype=float)
    return [operating_point(y, p, threshold) for threshold in thresholds]


def subgroup_metrics(df: pd.DataFrame) -> dict[str, list[dict[str, float | int | str]]]:
    groups = {
        "male": "male",
        "age": "age",
        "lgbtq": "lgbtq",
        "veteran": "veteran",
        "race": "race",
    }
    output: dict[str, list[dict[str, float | int | str]]] = {}
    for label, column in groups.items():
        if column not in df.columns:
            continue
        rows: list[dict[str, float | int | str]] = []
        for value, group in df.dropna(subset=[column]).groupby(column):
            if len(group) < 100 or group["suicide"].sum() < 5:
                continue
            metrics = discrimination_metrics(group)
            rows.append({"value": str(int(value)), **metrics})
        output[label] = rows
    return output


def write_markdown(report: dict, path: Path) -> None:
    main = report["metrics"]
    op = main[f"threshold_{DEFAULT_THRESHOLD:.2f}"]
    lines = [
        "# NSDUH 2024 Fresh-Data Validation",
        "",
        "Frozen model: packaged suicidal-ideation reference model v0.1.1.",
        "",
        "The 2024 public-use file used here does not expose a sexual-orientation "
        "variable, so the `lgbtq` predictor is missing for all scored rows and "
        "handled by the model's packaged median imputer.",
        "",
        "## Main Results",
        "",
        f"- Public-use respondents: {report['raw_rows']:,}",
        f"- Employed respondents: {report['employed_rows']:,}",
        f"- Analytic sample: {main['n']:,} employed adults with valid outcome",
        f"- Outcome prevalence: {main['prevalence']:.3%}",
        f"- Weighted outcome prevalence: {main['weighted_prevalence']:.3%}",
        f"- AUC: {main['auc']:.3f}",
        f"- AUPRC: {main['auprc']:.3f}",
        f"- Brier score: {main['brier']:.4f}",
        f"- Mean predicted probability: {main['mean_probability']:.3%}",
        f"- Calibration intercept: {main['calibration']['intercept']:.3f}",
        f"- Calibration slope: {main['calibration']['slope']:.3f}",
        "",
        f"## Operating Point at Threshold {DEFAULT_THRESHOLD:.2f}",
        "",
        f"- Sensitivity: {op['sensitivity']:.3f}",
        f"- Specificity: {op['specificity']:.3f}",
        f"- PPV: {op['ppv']:.3f}",
        f"- NPV: {op['npv']:.3f}",
        f"- Flag rate: {op['flag_rate']:.3%}",
        f"- Weighted flag rate: {main['weighted_flag_rate_at_default_threshold']:.3%}",
        f"- Confusion matrix: TP={op['tp']}, FP={op['fp']}, FN={op['fn']}, TN={op['tn']}",
        "",
        "## Source Variable Mapping",
        "",
    ]
    for key, value in report["source_columns"].items():
        lines.append(f"- `{key}` -> `{value}`")
    lines.extend(["", "## Threshold Table", ""])
    for row in report["threshold_table"]:
        lines.append(
            f"- {row['threshold']:.2f}: sens={row['sensitivity']:.3f}, "
            f"spec={row['specificity']:.3f}, ppv={row['ppv']:.3f}, "
            f"flag={row['flag_rate']:.3%}"
        )
    path.write_text("\n".join(lines) + "\n")


def run_validation(args: argparse.Namespace) -> dict:
    max_bytes = int(args.max_download_gb * 1024**3)
    if args.download:
        download_if_needed(DATA_URL, args.data_path, max_bytes=max_bytes)
        download_if_needed(DICTIONARY_URL, args.dictionary_path, max_bytes=max_bytes)

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"{args.data_path} does not exist. Re-run with --download or pass --data-path."
        )

    validation_frame, source_columns = build_validation_frame(
        args.data_path,
        args.dictionary_path,
    )
    employed_all = validation_frame[validation_frame["employment"] == 1].copy()
    employed = employed_all[employed_all["suicide"].isin([0, 1])].copy()
    scored = predict_dataframe(
        employed,
        threshold=DEFAULT_THRESHOLD,
    )

    thresholds = [0.05, 0.10, 0.15, DEFAULT_THRESHOLD, 0.20, 0.30]
    report = {
        "data_source": DATA_URL,
        "dictionary_source": DICTIONARY_URL,
        "source_columns": source_columns,
        "raw_rows": int(len(validation_frame)),
        "employed_rows": int(len(employed_all)),
        "employed_rows_with_valid_outcome": int(len(employed)),
        "metrics": discrimination_metrics(scored),
        "threshold_table": threshold_table(scored, thresholds),
        "subgroups": subgroup_metrics(scored),
        "missingness": scored[
            [
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
        ]
        .isna()
        .mean()
        .to_dict(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "nsduh_2024_validation_report.json"
    md_path = RESULTS_DIR / "nsduh_2024_validation_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    write_markdown(report, md_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download public parquet files if absent.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--dictionary-path", type=Path, default=DICTIONARY_PATH)
    parser.add_argument(
        "--max-download-gb",
        type=float,
        default=1.0,
        help="Safety limit for any single downloaded file.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        report = run_validation(parse_args())
    except Exception as exc:
        print(f"validate_nsduh_2024: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    metrics = report["metrics"]
    print(
        "NSDUH 2024 validation complete: "
        f"n={metrics['n']:,}, prevalence={metrics['prevalence']:.3%}, "
        f"AUC={metrics['auc']:.3f}, Brier={metrics['brier']:.4f}"
    )


if __name__ == "__main__":
    main()
