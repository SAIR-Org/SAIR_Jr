"""Gradio UI for the insurance charges regression pipeline.

Loads the most recently trained `pipeline.pkl` / `best_model.pkl` /
`metadata.json` from `models/` (produced by
`insurance_regression_pipeline.ipynb`) and serves interactive predictions
for the six raw policyholder attributes.

Run with: uv run python app.py
"""

import glob
import json
import os
import sys

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

MODEL_ROOT = "models"
RAW_FIELDS = ["age", "sex", "bmi", "children", "smoker", "region"]


# `pipeline.pkl` was pickled from the notebook, where these three
# transformers are defined inline in the kernel's `__main__` module (the
# notebook is intentionally self-contained — see its intro cell). joblib
# resolves a pickled class by module + name, so the same classes must be
# reachable as `__main__.<ClassName>` here too, however this file is run.
# Definitions kept byte-for-byte identical to insurance_regression_pipeline.ipynb §8.
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Fill gaps without knowing the dataset in advance.

    Numeric columns take the training median, everything else the training
    mode. Columns that were entirely empty during fit fall back to 0 / "missing"
    so transform never emits NaN.
    """

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.fill_values_ = {}

        for col in df.columns:
            if col in self.numeric_cols_:
                value = df[col].median()
                self.fill_values_[col] = 0 if pd.isna(value) else value
            else:
                mode = df[col].mode(dropna=True)
                self.fill_values_[col] = mode.iloc[0] if not mode.empty else "missing"

        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for col, value in self.fill_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)

        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Cap numeric columns to their IQR fences.

    Columns are discovered at fit time, so this works on any dataset.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for col in self.numeric_cols_:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                self.lower_bounds_[col] = -np.inf
                self.upper_bounds_[col] = np.inf
            else:
                self.lower_bounds_[col] = Q1 - (self.factor * IQR)
                self.upper_bounds_[col] = Q3 + (self.factor * IQR)

        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for col in self.numeric_cols_:
            if col in df.columns:
                df[col] = df[col].clip(
                    lower=self.lower_bounds_[col], upper=self.upper_bounds_[col]
                )

        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class LocalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode whichever columns happen to be categorical.

    Columns are discovered at fit time from their dtype, so no dataset-specific
    column names are baked in. Unseen categories at predict time are ignored
    rather than raising.
    """

    def __init__(self):
        self.encoder = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )
        self.encoder.set_output(transform="pandas")
        self.categorical_cols_ = None
        self.numeric_cols_ = None

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        self.categorical_cols_ = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.numeric_cols_ = df.select_dtypes(
            exclude=["object", "category"]
        ).columns.tolist()

        if self.categorical_cols_:
            self.encoder.fit(df[self.categorical_cols_])

        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if not self.categorical_cols_:
            return df

        encoded = self.encoder.transform(df[self.categorical_cols_])
        return df.drop(columns=self.categorical_cols_).join(encoded)

    def get_feature_names_out(self, input_features=None):
        if not self.categorical_cols_:
            return list(self.numeric_cols_)
        return list(self.numeric_cols_) + list(self.encoder.get_feature_names_out())


for _cls in (MissingValueHandler, OutlierHandler, LocalEncoder):
    setattr(sys.modules["__main__"], _cls.__name__, _cls)


def latest_model_dir(root=MODEL_ROOT):
    dirs = sorted(glob.glob(os.path.join(root, "*")), key=os.path.getmtime)
    if not dirs:
        raise FileNotFoundError(
            f"No trained model found under '{root}/'. Run the notebook first."
        )
    return dirs[-1]


VERSION_DIR = latest_model_dir()
pipeline = joblib.load(os.path.join(VERSION_DIR, "pipeline.pkl"))
model = joblib.load(os.path.join(VERSION_DIR, "best_model.pkl"))

with open(os.path.join(VERSION_DIR, "metadata.json")) as handle:
    metadata = json.load(handle)

FEATURE_ORDER = metadata["feature_names"]
SCHEMA_BY_NAME = {field["name"]: field for field in metadata["input_schema"]}


def engineer_features(age, sex, bmi, children, smoker, region):
    """Reproduce the notebook's §5 feature engineering for one raw record."""
    smoker_flag = 1 if smoker == "yes" else 0
    row = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_flag,
        "region": region,
        "age_smoker": age * smoker_flag,
        "bmi_smoker": bmi * smoker_flag,
        "age_bmi": age * bmi,
        "is_obese": int(bmi >= 30),
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def predict_charges(age, sex, bmi, children, smoker, region):
    raw = engineer_features(age, sex, bmi, children, smoker, region)
    encoded = pipeline.transform(raw)
    charge = float(model.predict(encoded)[0])
    return f"${charge:,.2f}"


def make_input_component(name):
    """Build a Gradio component for a raw field from metadata's input_schema."""
    if name == "smoker":
        return gr.Radio(["no", "yes"], value="no", label="Smoker")

    field = SCHEMA_BY_NAME[name]
    if field["type"] == "categorical":
        return gr.Dropdown(field["categories"], value=field["default"], label=name.capitalize())

    step = 1 if field["is_integer"] else 0.1
    return gr.Slider(field["min"], field["max"], value=field["default"], step=step, label=name.upper() if name == "bmi" else name.capitalize())


MODEL_CARD_MD = f"""
**Model:** {metadata['model_name']}  ·  trained on `{metadata['dataset']}` ({metadata['n_rows']} rows)

**Validation** — R²: {metadata['val_metrics']['r2']:.3f}  ·  RMSE: ${metadata['val_metrics']['rmse']:,.0f}
**Test** — R²: {metadata['test_metrics']['r2']:.3f}  ·  RMSE: ${metadata['test_metrics']['rmse']:,.0f}

Predicts annual medical insurance charges from demographic and lifestyle
attributes. See `model_card.json` in `{VERSION_DIR}` for the full write-up,
including limitations and ethical considerations.
"""

demo = gr.Interface(
    fn=predict_charges,
    inputs=[make_input_component(name) for name in RAW_FIELDS],
    outputs=gr.Textbox(label="Predicted annual charges"),
    title="Insurance Charges Estimator",
    description=MODEL_CARD_MD,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
