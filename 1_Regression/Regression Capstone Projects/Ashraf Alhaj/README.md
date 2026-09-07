# Insurance Regression Notebook

## Introduction

Health insurers need to price premiums before a claim ever happens, which means estimating how much a given policyholder is likely to cost based on attributes known up front. This project tackles that as a regression problem: given a person's demographic and lifestyle attributes, predict their annual medical insurance charges.

The dataset, `insurance.csv`, contains 1,338 records with the following features:

- `age` — age of the primary beneficiary
- `sex` — policyholder's gender (`male` / `female`)
- `bmi` — body mass index
- `children` — number of dependents covered by the plan
- `smoker` — smoking status (`yes` / `no`)
- `region` — residential area in the US (`northeast`, `northwest`, `southeast`, `southwest`)
- `charges` — individual medical costs billed by health insurance (the prediction target)

Charges vary widely across individuals, and features like smoking status, BMI, and age are known to interact in non-linear ways (e.g., a high BMI combined with smoking tends to drive costs up disproportionately). The goal of this pipeline is to learn those relationships from the data and produce a model that accurately predicts `charges` for new policyholders — useful for tasks like premium estimation, risk scoring, and identifying which factors most influence cost.

## Pipeline

Standalone regression pipeline for `insurance.csv` (target: `charges`), covering EDA, feature engineering, preprocessing, model selection, and evaluation in one notebook.

Run `uv sync`, then `uv run jupyter lab` and open `insurance_regression_pipeline.ipynb`.

Each run saves `best_model.pkl`, `pipeline.pkl`, `metadata.json`, and
`model_card.json` (model details, intended use, metrics, limitations) to a
timestamped folder under `models/`.

## Try it

`app.py` is a Gradio UI that loads the most recently trained model from
`models/` and serves live predictions:

```
uv run python app.py
```
