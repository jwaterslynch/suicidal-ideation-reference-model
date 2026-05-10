# Website Card Draft

## Card Label

Research Tool - Open source model

## Product Name

Suicidal Ideation Reference Model

## Short Line

An open suicidal-ideation reference model with independently reproduced
fresh-year validation on NSDUH 2024.

## Product Description

An open-source model-release repository containing a fitted suicidal-ideation
prediction model, feature schema, CSV scorer, Python API, synthetic examples,
model card, validation checklist, governance guidance, and an independently
reproduced fresh-data validation workflow on NSDUH 2024.

Built from the machine-learning suicide-prediction paper and separated from the
paper reproduction repo. The tool supports researcher-led local validation
studies and methodological prototyping; it is not a clinical diagnostic system,
an employer screening system, or an automated employment decision tool. The
2024 validation reproduced on 20,588 employed respondents and shows useful
temporal transportability (AUC 0.830), but the packaged threshold requires
local recalibration and one predictor (`lgbtq`) was not available in the 2024
public-use file.

## Tags

Open source - Python - Fitted model - Fresh-year validation - Model governance -
Mental-health methods - MIT license

## Links

- GitHub: `https://github.com/jwaterslynch/suicidal-ideation-reference-model`
- Paper reproduction repo: `https://github.com/jwaterslynch/Workplace-SI-ML-Pipeline`
- Related paper anchor: `index.html#paper-suicidal-ideation`
