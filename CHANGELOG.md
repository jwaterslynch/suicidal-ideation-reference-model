# Changelog

## 0.1.3 - 2026-05-10

- Hardened the NSDUH 2024 validation downloader with `.partial` file writes,
  byte-count validation, and SHA-256 checks for the pinned public parquet files.
- Kept the fitted model artifact and validation metrics unchanged.

## 0.1.2 - 2026-05-10

- Added the reproducible NSDUH 2024 fresh-data validation workflow.
- Added aggregate 2024 validation reports under `validation/results/`.
- Documented the 2024 partial-feature caveat for unavailable sexual-orientation
  variables in the public-use file.

## 0.1.1 - 2026-05-10

- Rebuilt the packaged artifact after recoding extended NSDUH work-hours
  sentinel values (`985`, `989`, `994`, `997`, `998`, `999`) to missing before
  fitting.
- Added input-range validation, threshold validation, all-missing-row refusal,
  and row-level missing-input counts.
- Improved CLI error messages for common input failures.
- Tightened dependency bounds and package metadata.
- Expanded tests for schema, missingness, thresholds, CLI behavior,
  deterministic predictions, and artifact metadata.
- Updated model-card and data-dictionary documentation for version 0.1.1.

## 0.1.0 - 2026-05-09

- Initial reference-model repository scaffold.
- Added fitted calibrated XGBoost reference model artifact.
- Added Python API and CSV scoring CLI.
- Added synthetic example input data.
- Added model card, local-validation guide, and governance documentation.
- Added citation metadata for GitHub.
