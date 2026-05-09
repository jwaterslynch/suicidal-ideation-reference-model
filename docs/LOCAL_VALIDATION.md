# Local Validation Guide

Any external use should begin with a validation study before operational use.

## Minimum Validation Report

Report:

- Data source, population, and inclusion criteria.
- How each local variable was mapped to the expected feature schema.
- Missingness by feature.
- Outcome definition and how it differs from NSDUH past-year suicidal ideation.
- Sample size, outcome prevalence, and observation period.
- AUC, precision-recall curve, Brier score, calibration intercept, and
  calibration slope.
- Sensitivity, specificity, PPV, and NPV across candidate thresholds.
- Subgroup metrics for legally, clinically, or operationally relevant groups.
- Comparison with a simple baseline model.
- Threshold workload: number and share of people flagged at each threshold.
- Human-review and support workflow for any threshold considered.

## Thresholds

The package documents a high-specificity reference threshold from the 2020
holdout set. Do not treat that threshold as portable. Thresholds should be
chosen locally based on prevalence, calibration, available support capacity,
false-positive and false-negative harms, and governance review.

## Drift And Maintenance

If used repeatedly, monitor:

- Input distributions.
- Missingness.
- Score distributions.
- Outcome prevalence.
- Calibration.
- Subgroup performance.
- Workload at the selected threshold.

Review drift at least annually and after any major population, policy,
measurement, or data-system change.
