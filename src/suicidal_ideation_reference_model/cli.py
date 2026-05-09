from __future__ import annotations

import argparse
from pathlib import Path

from .model import DEFAULT_THRESHOLD, score_csv

BOUNDARY_NOTE = (
    "Note: this package provides a reference model for research, validation, "
    "and governed support workflows. It is not a diagnostic system and must "
    "not be used for automated adverse decisions."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="si-risk-score",
        description="Score a CSV with the suicidal-ideation reference model.",
    )
    parser.add_argument("input_csv", type=Path, help="CSV containing model features.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV path. Defaults to predictions.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional probability threshold for an si_flag column. "
            f"The documented high-specificity reference threshold is {DEFAULT_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional path to a custom joblib model bundle.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    scored = score_csv(
        args.input_csv,
        args.output,
        threshold=args.threshold,
        model_path=args.model,
    )
    print(BOUNDARY_NOTE)
    print(f"Wrote {len(scored):,} scored rows to {args.output}")
