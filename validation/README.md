# NSDUH 2024 Fresh-Data Validation

This folder contains a lightweight validation workflow for scoring the frozen
v0.1.1 reference model on a fresh public NSDUH wave. It was added in package
release v0.1.2 and hardened in v0.1.3 with partial-file download handling and
SHA-256 validation.

The raw 2024 parquet files are public data, but they are not committed to this
repository. They are downloaded into `validation/raw/`, which is ignored by git.
The current Baselight mirror of the SAMHSA public-use file is about 258 MB.

Run:

```bash
uv run --extra validation python validation/validate_nsduh_2024.py --download
```

The script:

1. downloads the row-level public-use parquet and data dictionary if absent;
2. pivots only the small set of validation variables out of the long public-use
   table;
3. maps NSDUH 2024 variables to the model's nine-feature schema;
4. filters to employed adults using the same employment convention as the paper
   pipeline;
5. scores the frozen packaged reference model;
6. writes aggregate validation reports to `validation/results/`.

It does not save row-level predictions by default.

## Reproduced Result

The included aggregate reports were generated from:

- `nsduh_2024_data.parquet` SHA256:
  `95ad20cb919186c304c8b442aa060b279ad61dc29bf252bcb43dfe8274b56e86`
- `nsduh_2024_data_dictionary.parquet` SHA256:
  `ecbecdbf9be2794c5c82ac4b1171e203f393ae2a8a7b6e0d1b44bcd830dedc77`

An independent rerun in a fresh environment reproduced
`validation/results/nsduh_2024_validation_report.md` byte-for-byte and matched
the JSON metrics to ordinary floating-point tolerance.
