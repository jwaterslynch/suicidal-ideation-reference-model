# NSDUH 2024 Fresh-Data Validation

Frozen model: packaged suicidal-ideation reference model v0.1.1.

The 2024 public-use file used here does not expose a sexual-orientation variable, so the `lgbtq` predictor is missing for all scored rows and handled by the model's packaged median imputer.

## Main Results

- Public-use respondents: 58,633
- Employed respondents: 20,781
- Analytic sample: 20,588 employed adults with valid outcome
- Outcome prevalence: 6.276%
- Weighted outcome prevalence: 4.995%
- AUC: 0.830
- AUPRC: 0.304
- Brier score: 0.0513
- Mean predicted probability: 7.951%
- Calibration intercept: -0.342
- Calibration slope: 0.972

## Operating Point at Threshold 0.17

- Sensitivity: 0.721
- Specificity: 0.828
- PPV: 0.219
- NPV: 0.978
- Flag rate: 20.648%
- Weighted flag rate: 15.960%
- Confusion matrix: TP=931, FP=3320, FN=361, TN=15976

## Source Variable Mapping

- `outcome` -> `irsuicthnk`
- `k6_score` -> `ksslr6max`
- `employment` -> `wrkstatwk2`
- `work_hours` -> `wrkdhrswk2`
- `sex` -> `irsex`
- `age` -> `catage`
- `marital` -> `irmarit`
- `military` -> `milstat`
- `drug_use` -> `illyr`
- `mental_health_tx` -> `mhtrtpy2`
- `stratum` -> `vestr_c`
- `psu` -> `verep`
- `race` -> `newrace2`
- `weight` -> `analwt2_c`
- `sexual_orientation` -> `not_available_in_2024_public_use_file`

## Threshold Table

- 0.05: sens=0.834, spec=0.696, ppv=0.155, flag=33.709%
- 0.10: sens=0.792, spec=0.764, ppv=0.184, flag=27.069%
- 0.15: sens=0.742, spec=0.809, ppv=0.207, flag=22.552%
- 0.17: sens=0.721, spec=0.828, ppv=0.219, flag=20.648%
- 0.20: sens=0.666, spec=0.856, ppv=0.236, flag=17.666%
- 0.30: sens=0.368, spec=0.957, ppv=0.363, flag=6.368%
