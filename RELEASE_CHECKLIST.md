# Release Checklist

## Before Public Release

- [ ] Confirm the artifact was generated from the intended reproduction commit.
- [ ] Confirm no raw NSDUH files or identifiable data are included.
- [ ] Confirm the example CSV is synthetic.
- [ ] Run the package tests.
- [ ] Run CLI scoring on `examples/example_input.csv`.
- [ ] Check `CITATION.cff`.
- [ ] Review the README for deployment overclaiming.
- [ ] Review `docs/GOVERNANCE.md`.
- [ ] Review `docs/LOCAL_VALIDATION.md`.
- [ ] Review `docs/MODEL_CARD.md`.

## GitHub

- [ ] Create `jwaterslynch/suicidal-ideation-reference-model`.
- [ ] Push initial release.
- [ ] Tag `v0.1.0`.
- [ ] Confirm GitHub citation panel is populated.
- [ ] Add repository link to `julianwaterslynch.com/tools.html`.

## Website

- [ ] Add the model card to Research Tools.
- [ ] Link the paper entry to both the reproduction repo and the model repo.
- [ ] Consider a separate note explaining reproduction repo vs model repo.
