# Data Dictionary

The reference model expects a CSV or dataframe with the following input
features. Extra columns are ignored by the scorer. Observed values must match
the source coding below; missing values are allowed and are median-imputed.

| Feature | Type | Source coding |
|---|---|---|
| `k6_score` | numeric | Kessler psychological distress score extracted from the NSDUH K6/KSSLR6 variable family |
| `male` | binary | 1 when source survey sex code is male, 0 otherwise |
| `age` | numeric category | NSDUH 2020 `CATAGE` categorical age code (`1`-`4`), not raw age in years |
| `married` | binary | 1 when marital-status source code indicates married, 0 otherwise |
| `lgbtq` | binary | 1 for sexual-minority category under the source survey coding used in the paper pipeline |
| `veteran` | binary | 1 for veteran/military-service category under the source survey coding |
| `drug_use` | binary | 1 for past-year illicit drug-use indicator under source survey coding |
| `mental_health_tx` | binary | 1 for mental-health treatment/help indicator under source survey coding |
| `work_hours` | numeric | usual weekly work hours, after recoding NSDUH sentinel values to missing |

## Valid Observed Values

- `k6_score`: 0-24.
- `age`: 1-4 under the 2020 NSDUH `CATAGE` coding.
- Binary inputs (`male`, `married`, `lgbtq`, `veteran`, `drug_use`,
  `mental_health_tx`): 0 or 1.
- `work_hours`: 0-168 after missing/sentinel values are recoded.

The scorer refuses rows in which all model inputs are missing. It also raises a
clear error when observed values fall outside these ranges.

## Missing Values

The packaged model applies median imputation using the medians learned from the
2020 training split. The scorer appends `si_n_missing` so users can audit how
much imputation each row required. Missing values may still degrade
performance, and local validation should report missingness by feature and by
relevant subgroup.

Training-split imputation medians:

| Feature | Median |
|---|---:|
| `k6_score` | 4 |
| `male` | 1 |
| `age` | 4 |
| `married` | 1 |
| `lgbtq` | 0 |
| `veteran` | 0 |
| `drug_use` | 0 |
| `mental_health_tx` | 0 |
| `work_hours` | 40 |

## NSDUH Sentinel Codes

Version 0.1.1 recodes extended NSDUH work-hours sentinel values `985`, `989`,
`994`, `997`, `998`, and `999` to missing before fitting the reference model.
External NSDUH validation work should apply the same recoding before scoring.
Non-NSDUH users should not encode refusals, skips, or unknowns as large numeric
values; use missing values instead.

## Important Coding Warning

The `age` feature uses the paper pipeline's NSDUH categorical age code. It is
not raw age in years. External users should map their local age field to an
equivalent categorical coding before scoring, then document the mapping in their
validation report.
