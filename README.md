# Suicidal Ideation Reference Model

Open-source reference model for suicidal-ideation prediction in employed adults,
released so researchers and governance teams can inspect, test, validate, and
adapt the model in controlled settings.

This repository is separate from the paper reproduction repository. The
reproduction repo reruns the full NSDUH analysis. This repo packages a reusable
model artifact, feature schema, inference code, local-validation utilities, and
governance guidance.

## What Is Included

- A fitted reference model trained on the 2020 NSDUH employed-adult analytic
  sample.
- The preprocessing steps needed to use the model correctly: NSDUH
  work-hours sentinel recoding, median imputation, and standard scaling.
- A stable feature schema.
- A Python API and command-line scorer for CSV files.
- A model card, governance guidance, and local-validation checklist.
- Synthetic example data for testing the interface.

The model predicts probability of past-year suicidal ideation as operationalized
in the source survey. It does not predict suicide death, attempt, imminent harm,
or clinical diagnosis.

## Intended Use

Appropriate uses include:

- Replication and methods research.
- External validation on new data.
- Sensitivity, calibration, fairness, and threshold experiments.
- Prototyping support-oriented governance workflows.
- Benchmarking local models against an open reference model.

Potential organizational use must be treated as a high-stakes deployment. The
model should not be used for automated employment, disciplinary, insurance,
legal, or access decisions. Any real-world use requires local validation,
consent, privacy review, clinical or occupational-health oversight, and a
non-punitive support pathway.

## Install

From a local checkout:

```bash
pip install -e .
```

## Score A CSV

```bash
si-risk-score examples/example_input.csv --output predictions.csv
```

The input CSV must contain these columns:

| Feature | Meaning |
|---|---|
| `k6_score` | Kessler psychological distress score, as coded in the source pipeline |
| `male` | 1 = male, 0 = not male/female-coded in source survey |
| `age` | NSDUH 2020 categorical age code, not raw age in years |
| `married` | 1 = married, 0 = not married |
| `lgbtq` | 1 = lesbian/gay/bisexual/other sexual-minority category in source coding |
| `veteran` | 1 = veteran/military-service indicator |
| `drug_use` | 1 = past-year illicit drug-use indicator |
| `mental_health_tx` | 1 = mental-health treatment/help indicator |
| `work_hours` | Usual weekly work hours, with NSDUH sentinel codes treated as missing |

Observed values must match the documented schema. Missing values are allowed
and are median-imputed by the fitted pipeline, but rows with all model features
missing are refused rather than scored. See `docs/DATA_DICTIONARY.md`.

The scorer appends:

- `si_n_missing`: number of missing model inputs on the row.
- `si_probability`: model-estimated probability.
- `si_flag`: optional threshold flag if `--threshold` is supplied.

Example:

```bash
si-risk-score examples/example_input.csv \
  --output predictions.csv \
  --threshold 0.17
```

## Python API

```python
import pandas as pd
from suicidal_ideation_reference_model import load_reference_model, predict_dataframe

bundle = load_reference_model()
df = pd.read_csv("examples/example_input.csv")
scores = predict_dataframe(df, bundle=bundle)
print(scores[["si_n_missing", "si_probability"]])
```

Use `predict_dataframe` or `score_csv` for scoring. Do not call
`bundle["pipeline"].predict_proba(...)` directly with NumPy arrays, because
plain arrays do not preserve or enforce feature names.

## Model Artifact

Default artifact:

- `si_xgb_full_2020_v0_1_1.joblib`

The artifact is a dictionary containing:

- `pipeline`: a fitted scikit-learn pipeline with imputer, scaler, and
  calibrated XGBoost classifier.
- `features`: ordered feature list.
- `metadata`: training sample, validation metrics, thresholds, and source
  provenance.

## Validation Snapshot

The default reference model follows the paper's 2020 full-model specification:
calibrated XGBoost, 9 predictors, 70/30 stratified train-test split, seed 42,
and employed-adult filter. Version 0.1.1 additionally recodes extended NSDUH
work-hours sentinel values (`985`, `989`, `994`, `997`, `998`, `999`) to
missing before fitting.

Packaged holdout validation:

| Metric | Value |
|---|---:|
| Test N | 3,738 |
| Positive cases | 206 |
| Outcome prevalence | 5.51% |
| AUC | 0.872 |
| Brier score | 0.0438 |
| High-specificity reference threshold | 0.17 |
| Sensitivity at threshold 0.17 | 0.529 |
| Specificity at threshold 0.17 | 0.928 |

These metrics are evidence about the source context, not a guarantee of
performance in other populations or organizations.

## Rebuild The Artifact

If the paper reproduction repository is available locally, rebuild the packaged
artifact with:

```bash
python scripts/build_reference_model.py \
  --source-repo /path/to/Workplace-SI-ML-Pipeline
```

## Governance Boundary

This repository makes the model inspectable and testable. It does not make the
model deployment-ready. See:

- `docs/GOVERNANCE.md`
- `docs/LOCAL_VALIDATION.md`
- `docs/MODEL_CARD.md`

## Research Citation

If you use the model in research, cite the software release and the associated
paper or working paper.

```bibtex
@software{waterslynch_suicidal_ideation_reference_model_2026,
  title = {Suicidal Ideation Reference Model},
  author = {Waters-Lynch, Julian},
  year = {2026},
  url = {https://github.com/jwaterslynch/suicidal-ideation-reference-model},
  version = {0.1.1}
}
```

## Related Repository

Paper reproduction pipeline:

- https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline

## License

MIT. See `LICENSE`.
