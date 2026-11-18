
"""Reusable components for the Laptop Price Prediction project.
 
Used by ``notebook.ipynb`` (experiments) and ``app.py`` (Streamlit app).
"""
 
import json
import platform
import re
import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    RepeatedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
 
# =============================================================================
# 1. Paths, dataset schema and project configuration
# =============================================================================
 
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
 
TARGET = "Price_euros"
ID_COLUMN = "laptop_ID"
RAW_FEATURE_COLUMNS = [
    "Company", "Product", "TypeName", "Inches", "ScreenResolution",
    "Cpu", "Ram", "Memory", "Gpu", "OpSys", "Weight",
]
TB_IN_GB = 1024
 
 
@dataclass(frozen=True)
class ProjectConfig:
    """Settings shared by the train/test split, the cross-validation and the encoders."""
 
    random_state: int = 42
    test_size: float = 0.2
    stratify_bins: int = 10        # price quantile bins used to stratify the train/test split
    n_splits: int = 5
    n_repeats: int = 3
    min_category_count: int = 10   # rarer categories are grouped by the one-hot encoder
 
 
# =============================================================================
# 2. Data preparation
# =============================================================================
 
def load_data(filename: str) -> pd.DataFrame:
    """Load a CSV dataset from the project's raw data directory."""
    file_path = RAW_DATA_DIR / filename
 
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            f"Please place '{filename}' inside: {RAW_DATA_DIR}"
        )
 
    df = pd.read_csv(file_path, encoding="latin-1")
 
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
 
    return df
 
 
def remove_duplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that repeat the same specifications and price under a different ID."""
    subset = [c for c in df.columns if c != ID_COLUMN]
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
 
 
def split_data(df: pd.DataFrame, config: ProjectConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split; strata are price quantile bins."""
    config = config or ProjectConfig()
    strata = pd.qcut(df[TARGET], q=config.stratify_bins, labels=False, duplicates="drop")
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=strata,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
 
 
# =============================================================================
# 3. Feature extraction
# =============================================================================
 
_RAM = re.compile(r"\s*(\d+)\s*GB\s*")
_WEIGHT = re.compile(r"\s*([\d.]+)\s*kg\s*")
_RESOLUTION = re.compile(r"(\d+)x(\d+)\s*$")
_STORAGE_PART = re.compile(r"\s*([\d.]+)\s*(GB|TB)\s+(SSD|HDD|Hybrid|Flash Storage)\s*")
_STORAGE_COLUMN = {
    "SSD": "ssd_gb",
    "HDD": "hdd_gb",
    "Hybrid": "hybrid_gb",
    "Flash Storage": "flash_gb",
}
_CPU_CLOCK = re.compile(r"([\d.]+)GHz\s*$")
_CPU_GENERATION = re.compile(r"Intel Core (?:i[357]|M) (\d)(?:\d{3}[A-Z]{0,2}|Y\d{2}) ")
_CPU_SUFFIX = re.compile(r"Intel Core i[357] \d{4}([A-Z]{1,2}) ")
_CPU_Y_SUFFIX = re.compile(r"Intel Core (?:i[357]|M) .*\dY\d{2} ")
 
 
def _clean_text(value: object) -> str:
    """Collapse repeated whitespace and strip the ends."""
    return re.sub(r"\s+", " ", str(value)).strip()
 
 
def _fullmatch(pattern: re.Pattern, value: object, column: str) -> re.Match:
    match = pattern.fullmatch(str(value))
    if match is None:
        raise ValueError(f"Cannot parse {column} value: {value!r}")
    return match
 
 
def _parse_storage(value: object) -> dict[str, float]:
    """'256GB SSD + 1TB HDD' -> capacity in GB per storage type (1 TB = 1,024 GB)."""
    totals = dict.fromkeys(_STORAGE_COLUMN.values(), 0.0)
    for part in str(value).split("+"):
        match = _fullmatch(_STORAGE_PART, part, "Memory")
        size = float(match.group(1)) * (TB_IN_GB if match.group(2) == "TB" else 1)
        totals[_STORAGE_COLUMN[match.group(3)]] += size
    return totals
 
 
def _parse_resolution(value: object) -> tuple[int, int]:
    match = _RESOLUTION.search(str(value))
    if match is None:
        raise ValueError(f"Cannot parse ScreenResolution value: {value!r}")
    return int(match.group(1)), int(match.group(2))
 
 
def _cpu_family(text: str) -> str:
    if re.match(r"Intel Core M", text, re.I):
        return "Core M"
    match = re.match(r"Intel Core (i[357])", text, re.I)
    if match:
        return "Core " + match.group(1).lower()
    match = re.match(r"Intel (Celeron|Pentium|Atom|Xeon)", text, re.I)
    if match:
        return match.group(1).capitalize()
    if re.match(r"AMD A\d+-Series", text, re.I):
        return "AMD A-Series"
    match = re.match(r"AMD (E-Series|FX|Ryzen)", text, re.I)
    if match:
        return "AMD " + match.group(1)
    return "Other"
 
 
def _cpu_suffix(text: str) -> str:
    match = _CPU_SUFFIX.search(text)
    if match:
        return match.group(1)
    if _CPU_Y_SUFFIX.search(text):
        return "Y"
    return "none"
 
 
def _cpu_generation(text: str) -> str:
    match = _CPU_GENERATION.search(text)
    return match.group(1) if match else "none"
 
 
def _cpu_clock(text: str) -> float:
    match = _CPU_CLOCK.search(text)
    if match is None:
        raise ValueError(f"Cannot parse CPU clock speed: {text!r}")
    return float(match.group(1))
 
 
def _gpu_normalize(value: object) -> str:
    """Clean a GPU name: whitespace, the '<U+039C>' export artifact, and 'GTX1080' -> 'GTX 1080'."""
    text = _clean_text(value).replace("<U+039C>", "M").replace("Μ", "M")
    return re.sub(r"GTX(\d)", r"GTX \1", text)
 
 
def _gpu_family(text: str) -> str:
    if text.startswith("Nvidia"):
        if "Quadro" in text:
            return "Nvidia Quadro"
        if "GTX" in text:
            return "Nvidia GTX"
        if "MX" in text:
            return "Nvidia MX"
        if "GeForce" in text:
            return "Nvidia GeForce other"
        return "Nvidia other"
    if text.startswith("AMD"):
        if "FirePro" in text:
            return "AMD FirePro"
        if "Radeon" in text:
            return "AMD Radeon"
        return "AMD other"
    if text.startswith("Intel"):
        return "Intel integrated"
    return "Other"
 
 
def _product_line(value: object) -> str:
    """First word of the product name, lower-cased ('Inspiron 5570' -> 'inspiron')."""
    return _clean_text(value).split(" ")[0].lower()
 
 
class LaptopFeatureExtractor(BaseEstimator, TransformerMixin):
    """Stateless conversion of the raw laptop columns into candidate model features.
 
    Parsing is strict: a value that cannot be parsed raises ``ValueError`` instead of
    becoming NaN. ``merge_mac`` maps 'Mac OS X' to 'macOS'. The extractor learns nothing
    from the data, so it can sit inside a pipeline without any leakage risk.
    """
 
    OUTPUT_COLUMNS = [
        "inches", "ram_gb", "weight_kg",
        "ssd_gb", "hdd_gb", "hybrid_gb", "flash_gb",
        "touchscreen", "ips", "res_w", "res_h", "ppi",
        "cpu_ghz",
        "company", "type_name", "opsys",
        "cpu_family", "cpu_suffix", "cpu_gen",
        "gpu_family", "gpu_model", "product_line",
    ]
 
    def __init__(self, merge_mac: bool = True):
        self.merge_mac = merge_mac
 
    def fit(self, X: pd.DataFrame, y=None):
        self._check_columns(X)
        return self
 
    def get_feature_names_out(self, input_features=None):
        return np.array(self.OUTPUT_COLUMNS, dtype=object)
 
    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
 
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        raw = X[RAW_FEATURE_COLUMNS]
        out = pd.DataFrame(index=raw.index)
 
        out["inches"] = raw["Inches"].astype(float)
        out["ram_gb"] = raw["Ram"].map(lambda v: int(_fullmatch(_RAM, v, "Ram").group(1)))
        out["weight_kg"] = raw["Weight"].map(lambda v: float(_fullmatch(_WEIGHT, v, "Weight").group(1)))
 
        storage = pd.DataFrame(raw["Memory"].map(_parse_storage).tolist(), index=raw.index)
        for column in _STORAGE_COLUMN.values():
            out[column] = storage[column]
 
        screen = raw["ScreenResolution"].astype(str)
        out["touchscreen"] = screen.str.contains("Touchscreen").astype(int)
        out["ips"] = screen.str.contains("IPS").astype(int)
        resolution = pd.DataFrame(raw["ScreenResolution"].map(_parse_resolution).tolist(),
                                  index=raw.index, columns=["w", "h"])
        out["res_w"] = resolution["w"]
        out["res_h"] = resolution["h"]
        out["ppi"] = np.hypot(resolution["w"], resolution["h"]) / out["inches"]
 
        cpu = raw["Cpu"].map(_clean_text)
        out["cpu_ghz"] = cpu.map(_cpu_clock)
        out["cpu_family"] = cpu.map(_cpu_family)
        out["cpu_suffix"] = cpu.map(_cpu_suffix)
        out["cpu_gen"] = cpu.map(_cpu_generation)
 
        gpu = raw["Gpu"].map(_gpu_normalize)
        out["gpu_model"] = gpu
        out["gpu_family"] = gpu.map(_gpu_family)
 
        opsys = raw["OpSys"].map(_clean_text)
        if self.merge_mac:
            opsys = opsys.replace({"Mac OS X": "macOS"})
        out["opsys"] = opsys
 
        out["company"] = raw["Company"].map(_clean_text)
        out["type_name"] = raw["TypeName"].map(_clean_text)
        out["product_line"] = raw["Product"].map(_product_line)
 
        return out[self.OUTPUT_COLUMNS]
 
 
# =============================================================================
# 4. Feature sets, pipeline and model registry
# =============================================================================
 
BASE_NUMERIC = [
    "inches", "ram_gb", "weight_kg",
    "ssd_gb", "hdd_gb", "hybrid_gb", "flash_gb",
    "touchscreen", "ips", "res_w",
]
BASE_CATEGORICAL = ["company", "type_name", "opsys", "cpu_family", "gpu_family"]
 
# name -> (numeric columns, categorical columns); each set adds candidate columns to the base set
FEATURE_SETS = {
    "base": (BASE_NUMERIC, BASE_CATEGORICAL),
    "cpu_suffix": (BASE_NUMERIC, BASE_CATEGORICAL + ["cpu_suffix"]),
    "cpu_suffix_gen": (BASE_NUMERIC, BASE_CATEGORICAL + ["cpu_suffix", "cpu_gen"]),
    "cpu_suffix_gen_ghz": (BASE_NUMERIC + ["cpu_ghz"], BASE_CATEGORICAL + ["cpu_suffix", "cpu_gen"]),
    "gpu_model": (BASE_NUMERIC, BASE_CATEGORICAL + ["gpu_model"]),
    "product_line": (BASE_NUMERIC, BASE_CATEGORICAL + ["product_line"]),
    "gpu_product": (BASE_NUMERIC, BASE_CATEGORICAL + ["gpu_model", "product_line"]),
    "gpu_product_cpu": (
        BASE_NUMERIC + ["cpu_ghz"],
        BASE_CATEGORICAL + ["gpu_model", "product_line", "cpu_suffix", "cpu_gen"],
    ),
}
 
 
def build_pipeline(model: BaseEstimator, numeric: list[str], categorical: list[str],
                   scale: bool = False, merge_mac: bool = True, min_category_count: int = 10) -> Pipeline:
    """Feature extractor -> ColumnTransformer -> model, taking the raw dataset columns as input.
 
    The scaler (only when ``scale`` is True) and the one-hot encoder are learned inside the
    pipeline, so cross-validation fits them on each training fold only. Categories with fewer
    than ``min_category_count`` training rows, and categories unseen at prediction time,
    share one "infrequent" bucket.
    """
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler() if scale else "passthrough", numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_count,
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", LaptopFeatureExtractor(merge_mac=merge_mac)),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
 
 
def log_target(estimator: BaseEstimator) -> TransformedTargetRegressor:
    """Train on log1p(price); predictions are converted back to euros."""
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)
 
 
def candidate_models(random_state: int = 42) -> dict[str, tuple[BaseEstimator, bool]]:
    """name -> (estimator, needs_scaling); untuned, sensible starting settings."""
    return {
        "LinearRegression": (LinearRegression(), True),
        "Ridge": (Ridge(alpha=1.0), True),
        "Lasso": (Lasso(alpha=0.1, max_iter=20000), True),
        "ElasticNet": (ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000), True),
        "DecisionTree": (DecisionTreeRegressor(min_samples_leaf=5, random_state=random_state), False),
        "RandomForest": (RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=random_state), False),
        "ExtraTrees": (ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=random_state), False),
        "GradientBoosting": (
            GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=random_state),
            False,
        ),
        "HistGradientBoosting": (HistGradientBoostingRegressor(random_state=random_state), False),
    }
 
 
MIN_FREQUENCY_CHOICES = [5, 10, 20]
 
 
def search_space(name: str) -> dict:
    """Search space for a model inside the pipeline made by build_pipeline (its step is named 'model')."""
    shared = {"preprocess__cat__min_frequency": MIN_FREQUENCY_CHOICES}
    spaces = {
        "GradientBoosting": {
            "model__loss": ["squared_error", "absolute_error", "huber"],
            "model__n_estimators": randint(150, 601),
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_depth": randint(2, 6),
            "model__min_samples_leaf": randint(1, 16),
            "model__subsample": uniform(0.6, 0.4),
            "model__max_features": uniform(0.3, 0.7),
        },
        "RandomForest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 10, 15, 20],
            "model__min_samples_leaf": randint(1, 9),
            "model__max_features": uniform(0.2, 0.8),
        },
        "HistGradientBoosting": {
            "model__loss": ["squared_error", "absolute_error"],
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_iter": randint(100, 601),
            "model__max_leaf_nodes": randint(8, 41),
            "model__max_depth": [None, 3, 5, 8],
            "model__min_samples_leaf": randint(5, 41),
            "model__l2_regularization": loguniform(1e-3, 10),
        },
    }
    return {**spaces[name], **shared}
 
 
# =============================================================================
# 5. Evaluation and tuning
# =============================================================================
 
CV_SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "medae": "neg_median_absolute_error",
}
CV_METRIC_NAMES = ["mae", "rmse", "r2", "medae", "fit_seconds"]
 
 
def make_cv(config: ProjectConfig, X: pd.DataFrame, groups: pd.Series | None = None):
    """Repeated 5-fold splits, identical for every model that uses the same config.
 
    Without ``groups`` it returns a ``RepeatedKFold``; with ``groups`` it returns a list of
    (train, test) index pairs from shuffled ``GroupKFold`` (needs scikit-learn >= 1.6), so all
    rows of a group stay in one fold.
    """
    if groups is None:
        return RepeatedKFold(
            n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.random_state
        )
    return [
        split
        for repeat in range(config.n_repeats)
        for split in GroupKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.random_state + repeat
        ).split(X, groups=groups)
    ]
 
 
def cross_validate_model(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                         config: ProjectConfig | None = None, groups: pd.Series | None = None) -> pd.DataFrame:
    """Repeated CV; one row per fold, errors measured on the original euro scale."""
    config = config or ProjectConfig()
    cv = make_cv(config, X, groups)
    raw = cross_validate(estimator, X, y, cv=cv, scoring=CV_SCORING)
    return pd.DataFrame(
        {
            "mae": -raw["test_mae"],
            "rmse": -raw["test_rmse"],
            "r2": raw["test_r2"],
            "medae": -raw["test_medae"],
            "fit_seconds": raw["fit_time"],
        }
    )
 
 
def cv_metrics(folds: pd.DataFrame) -> dict[str, float]:
    """Mean and standard deviation over the folds of one model, e.g. {'mae_mean': ..., 'mae_std': ...}."""
    return {
        f"{metric}_{stat}": float(getattr(folds[metric], stat)())
        for metric in CV_METRIC_NAMES
        for stat in ["mean", "std"]
    }
 
 
def summarize_cv(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per model; results maps a model name to the DataFrame from cross_validate_model."""
    return pd.DataFrame({name: cv_metrics(folds) for name, folds in results.items()}).T
 
 
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "mape_pct": 100 * mean_absolute_percentage_error(y_true, y_pred),
    }
 
 
PRICE_BINS = [0, 500, 1000, 1500, 2500, np.inf]
 
 
def error_by_price_range(y_true, y_pred, bins: list[float] = PRICE_BINS) -> pd.DataFrame:
    """Errors per actual-price range; bias = mean(predicted - actual)."""
    frame = pd.DataFrame({"actual": np.asarray(y_true), "pred": np.asarray(y_pred)})
    frame["error"] = frame["pred"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    frame["pct_error"] = 100 * frame["abs_error"] / frame["actual"]
    frame["range"] = pd.cut(frame["actual"], bins=bins, right=False)
    return frame.groupby("range", observed=True).agg(
        n=("error", "size"),
        mae=("abs_error", "mean"),
        medae=("abs_error", "median"),
        bias=("error", "mean"),
        mape_pct=("pct_error", "mean"),
    )
 
 
def tune_model(pipeline: Pipeline, param_distributions: dict, X: pd.DataFrame, y: pd.Series,
               config: ProjectConfig, groups: pd.Series | None = None, n_iter: int = 40,
               n_repeats: int = 1, n_jobs: int | None = None) -> tuple[RandomizedSearchCV, dict]:
    """Randomized search minimizing MAE; one repeat of the folds by default.
 
    Returns the fitted search and a summary of the run (best parameters, best CV MAE,
    number of candidates and duration).
    """
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=make_cv(replace(config, n_repeats=n_repeats), X, groups),
        random_state=config.random_state,
        n_jobs=n_jobs,
    )
    start = time.perf_counter()
    search.fit(X, y)
    summary = {
        "best_params": {k: v.item() if isinstance(v, np.generic) else v for k, v in search.best_params_.items()},
        "best_cv_mae": -search.best_score_,
        "n_candidates": n_iter,
        "search_seconds": time.perf_counter() - start,
    }
    return search, summary
 
 
# =============================================================================
# 6. Experiment tracking (MLflow)
# =============================================================================
 
MLFLOW_EXPERIMENT = "laptop-price-prediction"
 
 
def _describe_model(estimator: BaseEstimator) -> tuple[str, dict[str, str]]:
    """Return (model type, hyperparameters and preprocessing choices as strings)."""
    if hasattr(estimator, "named_steps"):
        deep = estimator.get_params()
        keep = ("preprocess__num", "preprocess__cat__min_frequency", "features__merge_mac")
        params = {k: v for k, v in deep.items() if k.startswith("model__") or k in keep}
        model = estimator.named_steps["model"]
    else:
        model, params = estimator, estimator.get_params(deep=False)
    return type(model).__name__, {k: str(v)[:250] for k, v in params.items()}
 
 
class ExperimentTracker:
    """Logs runs to a local MLflow store: a SQLite database and an artifact folder in ``experiments/``.
 
    ``common_params`` are logged with every run, so settings shared by all runs (feature set,
    seed, CV protocol) are written once. MLflow is imported here, not at module level, so the
    Streamlit app does not need it at inference time.
    """
 
    def __init__(self, experiment_name: str = MLFLOW_EXPERIMENT, common_params: dict | None = None):
        import mlflow
 
        self._mlflow = mlflow
        self.common_params = dict(common_params or {})
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{EXPERIMENTS_DIR / 'mlflow.db'}")
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                experiment_name, artifact_location=(EXPERIMENTS_DIR / "mlartifacts").as_uri()
            )
        mlflow.set_experiment(experiment_name)
 
    def log_run(self, run_name: str, estimator: BaseEstimator, metrics: dict[str, float],
                tags: dict[str, str] | None = None, extra_params: dict | None = None,
                artifacts=()) -> None:
        """One MLflow run: model type and hyperparameters, shared and extra parameters, metrics, files."""
        mlflow = self._mlflow
        model_type, params = _describe_model(estimator)
        extra = {**self.common_params, **(extra_params or {})}
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(tags or {})
            mlflow.log_param("model_type", model_type)
            mlflow.log_params({**params, **{k: str(v) for k, v in extra.items()}})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            for path in artifacts:
                mlflow.log_artifact(str(path))
 
 
# =============================================================================
# 7. Saved model artifacts and Streamlit app support
# =============================================================================
 
MODEL_FILENAME = "laptop_price_model.joblib"
OPTIONS_FILENAME = "input_options.json"
METADATA_FILENAME = "model_metadata.json"
 
_INTEGER_OPTIONS = ["ram_gb", "ssd_gb", "hdd_gb", "flash_gb"]
 
 
def _plain(value):
    """numpy scalar -> Python scalar (JSON-serialisable)."""
    return value.item() if isinstance(value, np.generic) else value
 
 
def build_input_options(df: pd.DataFrame, min_product_rows: int = 10) -> dict:
    """Selectable values (and defaults) for the app, derived from the cleaned data."""
    features = LaptopFeatureExtractor().transform(df)
    cpu = df["Cpu"].map(_clean_text).map(lambda t: _CPU_CLOCK.sub(lambda m: f"{float(m.group(1)):g}GHz", t))
    line_counts = features["product_line"].value_counts()
    lines = features["product_line"][features["product_line"].map(line_counts) >= min_product_rows]
    resolution = features["res_w"].astype(int).astype(str) + "x" + features["res_h"].astype(int).astype(str)
 
    sources = {
        "company": features["company"],
        "type_name": features["type_name"],
        "opsys": features["opsys"],
        "cpu": cpu,
        "gpu": features["gpu_model"],
        "product_line": lines,
        "resolution": resolution,
        "ram_gb": features["ram_gb"],
        "ssd_gb": features["ssd_gb"],
        "hdd_gb": features["hdd_gb"],
        "flash_gb": features["flash_gb"],
    }
 
    # Choices: sorted unique values; resolutions are ordered by pixel count.
    options = {key: sorted(_plain(v) for v in series.unique()) for key, series in sources.items()}
    options["resolution"].sort(key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))
    for key in _INTEGER_OPTIONS:
        options[key] = [int(v) for v in options[key]]
 
    # Defaults: the most common value; if that would leave the laptop without any drive,
    # fall back to the most common non-zero SSD size.
    options["defaults"] = {key: _plain(series.mode().iloc[0]) for key, series in sources.items()}
    for key in _INTEGER_OPTIONS:
        options["defaults"][key] = int(options["defaults"][key])
    if not any(options["defaults"][key] for key in ["ssd_gb", "hdd_gb", "flash_gb"]):
        options["defaults"]["ssd_gb"] = int(features.loc[features["ssd_gb"] > 0, "ssd_gb"].mode().iloc[0])
 
    # Numeric inputs: allowed range and median default.
    for key in ["inches", "weight_kg"]:
        options[key] = {
            "min": float(features[key].min()),
            "max": float(features[key].max()),
            "default": round(float(features[key].median()), 2),
        }
    return options
 
 
def build_laptop_row(company: str, type_name: str, product_line: str, opsys: str, inches: float,
                     resolution: str, touchscreen: bool, ips: bool, weight_kg: float, cpu: str, gpu: str,
                     ram_gb: int, ssd_gb: int, hdd_gb: int, flash_gb: int) -> pd.DataFrame:
    """One-row DataFrame in the raw dataset format, ready for the saved pipeline."""
    drives = [f"{int(size)}GB {kind}"
              for size, kind in [(ssd_gb, "SSD"), (hdd_gb, "HDD"), (flash_gb, "Flash Storage")] if size > 0]
    if not drives:
        raise ValueError("Select at least one storage drive (SSD, HDD or flash).")
    screen = " ".join(part for part in ["IPS Panel" if ips else "", "Touchscreen" if touchscreen else "", resolution] if part)
    return pd.DataFrame([{
        "Company": company,
        "Product": product_line,
        "TypeName": type_name,
        "Inches": float(inches),
        "ScreenResolution": screen,
        "Cpu": cpu,
        "Ram": f"{int(ram_gb)}GB",
        "Memory": " + ".join(drives),
        "Gpu": gpu,
        "OpSys": opsys,
        "Weight": f"{float(weight_kg):.2f}kg",
    }])
 
 
def build_model_metadata(model: Pipeline, n_rows: int, test_mae: float, cv_mae: dict[str, float],
                         oof_true, oof_pred) -> dict:
    """Facts about the saved model that the app displays, as a JSON-ready dict.
 
    ``cv_mae`` maps "random" and "grouped" to the cross-validated MAE. ``error_by_predicted_price``
    holds the out-of-fold MAE per *predicted*-price band, so the app can quote a typical error
    for the estimate it has just produced.
    """
    oof_pred = np.asarray(oof_pred)
    frame = pd.DataFrame({"pred": oof_pred, "abs_error": np.abs(oof_pred - np.asarray(oof_true))})
    frame["band"] = pd.cut(frame["pred"], bins=PRICE_BINS, right=False)
    bands = [
        {"from": float(interval.left), "to": None if np.isinf(interval.right) else float(interval.right),
         "mae": round(float(group["abs_error"].mean()), 1), "n": int(len(group))}
        for interval, group in frame.groupby("band", observed=True)
    ]
    return {
        "model_type": type(model.named_steps["model"]).__name__,
        "trained_on_rows": n_rows,
        "trained_on": date.today().isoformat(),
        "test_mae": round(test_mae, 1),
        "cv_mae_random": round(cv_mae["random"], 1),
        "cv_mae_grouped": round(cv_mae["grouped"], 1),
        "error_by_predicted_price": bands,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
 
 
def save_artifacts(model: Pipeline, options: dict, metadata: dict, directory: Path = MODELS_DIR) -> None:
    """Write the three files app.py loads: the model, the input options and the metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / MODEL_FILENAME)
    (directory / OPTIONS_FILENAME).write_text(json.dumps(options, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
 