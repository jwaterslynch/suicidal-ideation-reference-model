# Model Card

## Model Details

- Name: Suicidal Ideation Reference Model
- Package version: 0.1.2
- Default artifact: `si_xgb_full_2020_v0_1_1.joblib`
- Artifact version: 0.1.1
- Model type: calibrated XGBoost classifier
- Training data: 2020 NSDUH public-use data, employed-adult analytic sample
- Target: past-year suicidal ideation as coded in the source survey pipeline
- Feature count: 9
- License: MIT

## Intended Use

The model is intended for research, external validation, benchmarking,
governance prototyping, and support-oriented experimentation. It is not a
diagnostic system and should not be used for automated adverse decisions.

## Factors

The model uses:

- Psychological distress score.
- Sex-coded male indicator.
- Categorical age.
- Marital-status indicator.
- Sexual-minority indicator.
- Veteran/military-service indicator.
- Past-year illicit drug-use indicator.
- Mental-health treatment/help indicator.
- Work-hours measure.

## Training Procedure

The packaged reference artifact follows the paper's 2020 full-model setup:

- Employed-adult filter.
- Extended NSDUH work-hours sentinel values (`985`, `989`, `994`, `997`,
  `998`, `999`) recoded to missing before fitting.
- 70/30 stratified train-test split.
- Random seed 42.
- Median imputation.
- Standard scaling.
- XGBoost classifier with 150 estimators and max depth 3.
- Sigmoid calibration with 5-fold cross-validation on the training split.

## Validation Metrics

### Packaged 2020 Holdout

| Metric | Value |
|---|---:|
| Total analytic N | 12,458 |
| Training N | 8,720 |
| Test N | 3,738 |
| Test positive cases | 206 |
| Test prevalence | 5.51% |
| Test AUC | 0.872 |
| Test Brier score | 0.0438 |

High-specificity reference operating point:

| Quantity | Value |
|---|---:|
| Threshold | 0.17 |
| Sensitivity | 0.529 |
| Specificity | 0.928 |
| PPV | 0.301 |
| NPV | 0.971 |
| TP / FP / FN / TN | 109 / 253 / 97 / 3,279 |

The training-fold F1-selected threshold was 0.249. On the heldout test split it
produced sensitivity 0.316 and specificity 0.967. This threshold was selected
on the training predictions and should be treated as descriptive, not as a
portable operating point.

### Fresh NSDUH 2024 Validation

Version 0.1.2 includes a reproducible fresh-data validation on the 2024 NSDUH
public-use file. The workflow is in `validation/validate_nsduh_2024.py`; the
aggregate report is in `validation/results/nsduh_2024_validation_report.md`.
An independent rerun reproduced the Markdown report byte-for-byte and matched
JSON metrics to ordinary floating-point tolerance.

| Metric | Value |
|---|---:|
| Public-use respondents | 58,633 |
| Employed respondents | 20,781 |
| Analytic N with valid outcome | 20,588 |
| Positive cases | 1,292 |
| Outcome prevalence | 6.28% |
| Weighted outcome prevalence | 5.00% |
| AUC | 0.830 |
| AUPRC | 0.304 |
| Brier score | 0.0513 |
| Mean predicted probability | 7.95% |
| Calibration intercept | -0.342 |
| Calibration slope | 0.972 |

Operating point at the packaged reference threshold:

| Quantity | Value |
|---|---:|
| Threshold | 0.17 |
| Sensitivity | 0.721 |
| Specificity | 0.828 |
| PPV | 0.219 |
| NPV | 0.978 |
| Flag rate | 20.65% |
| Weighted flag rate | 15.96% |
| TP / FP / FN / TN | 931 / 3,320 / 361 / 15,976 |

The 2024 public-use file used here does not expose a sexual-orientation
variable, so the `lgbtq` predictor is missing for every scored row and handled
by the packaged median imputer. Treat this as a partial-feature temporal
validation.

These metrics describe performance in the source holdout split and one fresh
NSDUH year. They do not establish performance in other populations, countries,
workplaces, clinical settings, or future time periods.

## Limitations

- Outcome is suicidal ideation, not suicide attempt, death, or imminent risk.
- Data are U.S. survey data and may not transfer to other contexts.
- Several inputs are sensitive or proxy-sensitive.
- The `age` input uses NSDUH 2020 categorical age codes, not raw age in years.
- The model should be recalibrated and validated locally before any applied use.
- The packaged 0.17 threshold was not portable to NSDUH 2024; it produced a
  materially higher flag rate and should not be treated as an operational
  threshold.
- Retrospective discrimination is not evidence of clinical utility.
- The model may perform differently across subgroups and over time.

## Ethical Considerations

Any applied use is high stakes. The model should only be considered in a
consent-based, support-oriented, human-reviewed workflow with privacy,
calibration, fairness, legal, and clinical governance controls.
