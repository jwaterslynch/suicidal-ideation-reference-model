# Data Dictionary

The reference model expects a CSV or dataframe with exactly the following input
features. Extra columns are ignored by the scorer.

| Feature | Type | Source coding |
|---|---|---|
| `k6_score` | numeric | Kessler psychological distress score extracted from the NSDUH K6/KSSLR6 variable family |
| `male` | binary | 1 when source survey sex code is male, 0 otherwise |
| `age` | numeric category | NSDUH categorical age code, not raw age in years |
| `married` | binary | 1 when marital-status source code indicates married, 0 otherwise |
| `lgbtq` | binary | 1 for sexual-minority category under the source survey coding used in the paper pipeline |
| `veteran` | binary | 1 for veteran/military-service category under the source survey coding |
| `drug_use` | binary | 1 for past-year illicit drug-use indicator under source survey coding |
| `mental_health_tx` | binary | 1 for mental-health treatment/help indicator under source survey coding |
| `work_hours` | numeric | usual weekly work hours |

## Missing Values

The packaged model applies median imputation using the medians learned from the
2020 training split. Missing values may still degrade performance, and local
validation should report missingness by feature and by relevant subgroup.

## Important Coding Warning

The `age` feature uses the paper pipeline's NSDUH categorical age code. It is
not raw age in years. External users should map their local age field to an
equivalent categorical coding before scoring, then document the mapping in their
validation report.
