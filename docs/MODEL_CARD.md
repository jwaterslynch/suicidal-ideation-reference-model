# Model Card

## Model Details

- Name: Suicidal Ideation Reference Model
- Version: 0.1.0
- Default artifact: `si_xgb_full_2020_v0_1_0.joblib`
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
- 70/30 stratified train-test split.
- Random seed 42.
- Median imputation.
- Standard scaling.
- XGBoost classifier with 150 estimators and max depth 3.
- Sigmoid calibration with 5-fold cross-validation on the training split.

## Validation Metrics

Packaged holdout validation:

| Metric | Value |
|---|---:|
| Total analytic N | 12,458 |
| Training N | 8,720 |
| Test N | 3,738 |
| Test positive cases | 206 |
| Test prevalence | 5.51% |
| Test AUC | 0.872 |
| Test Brier score | 0.0437 |

High-specificity reference operating point:

| Quantity | Value |
|---|---:|
| Threshold | 0.17 |
| Sensitivity | 0.510 |
| Specificity | 0.928 |
| PPV | 0.292 |
| NPV | 0.970 |
| TP / FP / FN / TN | 105 / 254 / 101 / 3,278 |

The training-fold F1-selected threshold was 0.231. On the heldout test split it
produced sensitivity 0.379 and specificity 0.958.

These metrics describe performance in the source holdout split. They do not
establish performance in other populations, countries, workplaces, clinical
settings, or time periods.

## Limitations

- Outcome is suicidal ideation, not suicide attempt, death, or imminent risk.
- Data are U.S. survey data and may not transfer to other contexts.
- Several inputs are sensitive or proxy-sensitive.
- The model should be recalibrated and validated locally before any applied use.
- Retrospective discrimination is not evidence of clinical utility.
- The model may perform differently across subgroups and over time.

## Ethical Considerations

Any applied use is high stakes. The model should only be considered in a
consent-based, support-oriented, human-reviewed workflow with privacy,
calibration, fairness, legal, and clinical governance controls.
