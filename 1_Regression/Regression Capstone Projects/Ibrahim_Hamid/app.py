import os
import glob
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


class DemandDomainEngineer(BaseEstimator, TransformerMixin):
    """Custom domain feature transformer required in scope for joblib unpickling."""

    def fit(self, X, y=None):
        if "RRP_lag_1" in X.columns:
            q1, q3 = np.percentile(X["RRP_lag_1"].dropna(), [25, 75])
            self.rrp_fence_ = float(q3 + 1.5 * (q3 - q1))
        else:
            self.rrp_fence_ = 180.0
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        dt = pd.to_datetime(X_out["date"])

        # Calendar & weekend flags
        X_out["month"] = dt.dt.month.astype(str)
        X_out["day_of_week"] = dt.dt.dayofweek.astype(str)
        X_out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

        # Cyclical calendar signals
        X_out["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
        X_out["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)

        # U-shaped thermal features (heating & cooling degree proxies)
        X_out["temp_mean"] = (X_out["max_temperature"] + X_out["min_temperature"]) / 2.0
        X_out["temp_range"] = X_out["max_temperature"] - X_out["min_temperature"]
        X_out["heating_degree"] = np.maximum(0.0, 18.0 - X_out["temp_mean"])
        X_out["cooling_degree"] = np.maximum(0.0, X_out["temp_mean"] - 22.0)
        X_out["temp_mean_sq"] = X_out["temp_mean"] ** 2
        X_out["solar_temp_interaction"] = X_out["solar_exposure"] * X_out["max_temperature"]

        # Regime interactions
        is_hol = (X_out["holiday"].astype(str).str.upper() == "Y").astype(int)
        is_sch = (X_out["school_day"].astype(str).str.upper() == "Y").astype(int)
        X_out["holiday_x_temp"] = is_hol * X_out["temp_mean"]
        X_out["school_x_temp"] = is_sch * X_out["temp_mean"]
        X_out["weekend_x_demand_lag1"] = X_out["is_weekend"] * X_out["demand_lag_1"]

        # Cap prior-day spot price at training fence
        if "RRP_lag_1" in X_out.columns:
            X_out["RRP_lag_1"] = np.clip(X_out["RRP_lag_1"], -10.0, self.rrp_fence_)

        return X_out.drop(columns=["date"])


st.set_page_config(
    page_title="Victoria Electricity Demand Forecaster",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource
def load_deployment_bundle():
    default_path = 'models/v1_20261116_204849/best_model.pkl'
    model_file = default_path

    if os.path.isdir("models"):
        versions = sorted(glob.glob(os.path.join("models", "v1_*")))
        if versions:
            latest_model = os.path.join(versions[-1], "best_model.pkl")
            if os.path.exists(latest_model):
                model_file = latest_model

    pipe = joblib.load(model_file)
    card_file = os.path.join(os.path.dirname(model_file), "model_card.json")
    card = {}
    if os.path.exists(card_file):
        with open(card_file, "r", encoding="utf-8") as f:
            card = json.load(f)
    return pipe, model_file, card


pipeline, loaded_path, model_card = load_deployment_bundle()

st.title("⚡ Victoria Daily Electricity Demand Forecaster")
if model_card:
    perf = model_card.get("performance", {})
    st.caption(
        f"Model: {model_card.get('model_name', 'Voting Ensemble')} | "
        f"Version: {model_card.get('model_version', 'v1')} | "
        f"Val R²: {perf.get('val_r2', 0.9051):.4f} | "
        f"Test R²: {perf.get('test_r2', 0.8647):.4f} | "
        f"Test RMSE: {perf.get('test_rmse', 5063.05):,.2f} MWh"
    )
else:
    st.caption(f"Loaded artifact: {loaded_path}")

# Preset scenarios for fast interactive testing
preset = st.selectbox(
    "Quick Scenario Preset",
    options=[
        "Custom Input",
        "Winter Weekday (Heating Demand)",
        "Summer Heatwave Weekday (Cooling Demand)",
        "Weekend Mild Day (Low Demand Regime)",
    ],
    index=0,
)

defaults = {
    "date": date(2020, 7, 15),
    "min_temperature": 8.5,
    "max_temperature": 16.0,
    "solar_exposure": 10.5,
    "rainfall": 0.0,
    "school_day": "Y",
    "holiday": "N",
    "demand_lag_1": 125000.0,
    "demand_lag_2": 123000.0,
    "demand_lag_7": 126500.0,
    "demand_lag_14": 124000.0,
    "demand_roll_mean_7": 122000.0,
    "demand_roll_std_7": 8500.0,
    "RRP_lag_1": 68.5,
    "frac_neg_RRP_lag_1": 0.0,
    "max_temp_lag_1": 15.0,
    "min_temp_lag_1": 7.5,
}

if preset == "Winter Weekday (Heating Demand)":
    defaults.update({
        "date": date(2020, 7, 15),
        "min_temperature": 4.0,
        "max_temperature": 11.5,
        "solar_exposure": 7.5,
        "rainfall": 2.0,
        "school_day": "Y",
        "holiday": "N",
        "demand_lag_1": 134000.0,
        "demand_lag_2": 132000.0,
        "demand_lag_7": 135000.0,
        "demand_lag_14": 131000.0,
        "demand_roll_mean_7": 131500.0,
        "demand_roll_std_7": 7800.0,
        "RRP_lag_1": 82.0,
        "max_temp_lag_1": 12.0,
        "min_temp_lag_1": 5.0,
    })
elif preset == "Summer Heatwave Weekday (Cooling Demand)":
    defaults.update({
        "date": date(2020, 1, 29),
        "min_temperature": 22.0,
        "max_temperature": 38.5,
        "solar_exposure": 29.5,
        "rainfall": 0.0,
        "school_day": "N",
        "holiday": "N",
        "demand_lag_1": 142000.0,
        "demand_lag_2": 136000.0,
        "demand_lag_7": 13000.0,
        "demand_lag_14": 125000.0,
        "demand_roll_mean_7": 131000.0,
        "demand_roll_std_7": 11000.0,
        "RRP_lag_1": 165.0,
        "max_temp_lag_1": 36.0,
        "min_temp_lag_1": 20.5,
    })
elif preset == "Weekend Mild Day (Low Demand Regime)":
    defaults.update({
        "date": date(2020, 4, 18),  # Saturday
        "min_temperature": 12.0,
        "max_temperature": 21.0,
        "solar_exposure": 16.0,
        "rainfall": 0.0,
        "school_day": "N",
        "holiday": "N",
        "demand_lag_1": 121000.0,
        "demand_lag_2": 122500.0,
        "demand_lag_7": 107500.0,
        "demand_lag_14": 108000.0,
        "demand_roll_mean_7": 118500.0,
        "demand_roll_std_7": 9200.0,
        "RRP_lag_1": 55.0,
        "max_temp_lag_1": 20.5,
        "min_temp_lag_1": 11.5,
    })

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📅 Calendar & Weather")
    forecast_date = st.date_input("Forecast Date", value=defaults["date"])
    min_temp = st.number_input("Min Temperature (°C)", value=float(defaults["min_temperature"]), step=0.5)
    max_temp = st.number_input("Max Temperature (°C)", value=float(defaults["max_temperature"]), step=0.5)
    solar_exposure = st.number_input("Solar Exposure (MJ/m²)", value=float(defaults["solar_exposure"]), step=0.5)
    rainfall = st.number_input("Rainfall (mm)", value=float(defaults["rainfall"]), step=0.5)
    school_day = st.selectbox("School Day", options=["Y", "N"], index=0 if defaults["school_day"] == "Y" else 1)
    holiday = st.selectbox("Public Holiday", options=["N", "Y"], index=0 if defaults["holiday"] == "N" else 1)

with col2:
    st.subheader("📉 Historical Demand Lags (MWh)")
    demand_lag_1 = st.number_input("Yesterday Demand (t-1)", value=float(defaults["demand_lag_1"]), step=500.0)
    demand_lag_2 = st.number_input("2 Days Ago Demand (t-2)", value=float(defaults["demand_lag_2"]), step=500.0)
    demand_lag_7 = st.number_input("Last Week Same Day (t-7)", value=float(defaults["demand_lag_7"]), step=500.0)
    demand_lag_14 = st.number_input("2 Weeks Ago Same Day (t-14)", value=float(defaults["demand_lag_14"]), step=500.0)
    demand_roll_mean_7 = st.number_input("7-Day Rolling Mean Demand", value=float(defaults["demand_roll_mean_7"]), step=500.0)
    demand_roll_std_7 = st.number_input("7-Day Rolling Std Demand", value=float(defaults["demand_roll_std_7"]), step=250.0)

with col3:
    st.subheader("💲 Prior-Day Market & Temp Lags")
    rrp_lag_1 = st.number_input("Yesterday Spot Price RRP (AUD/MWh)", value=float(defaults["RRP_lag_1"]), step=5.0)
    frac_neg_rrp_lag_1 = st.slider(
        "Yesterday Fraction at Negative RRP",
        min_value=0.0,
        max_value=1.0,
        value=float(defaults["frac_neg_RRP_lag_1"]),
        step=0.01,
    )
    max_temp_lag_1 = st.number_input("Yesterday Max Temp (°C)", value=float(defaults["max_temp_lag_1"]), step=0.5)
    min_temp_lag_1 = st.number_input("Yesterday Min Temp (°C)", value=float(defaults["min_temp_lag_1"]), step=0.5)

st.divider()

if st.button("🚀 Predict Daily Electricity Demand", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "date": pd.to_datetime(forecast_date),
        "min_temperature": min_temp,
        "max_temperature": max_temp,
        "solar_exposure": solar_exposure,
        "rainfall": rainfall,
        "school_day": school_day,
        "holiday": holiday,
        "demand_lag_1": demand_lag_1,
        "demand_lag_2": demand_lag_2,
        "demand_lag_7": demand_lag_7,
        "demand_lag_14": demand_lag_14,
        "demand_roll_mean_7": demand_roll_mean_7,
        "demand_roll_std_7": demand_roll_std_7,
        "RRP_lag_1": rrp_lag_1,
        "frac_neg_RRP_lag_1": frac_neg_rrp_lag_1,
        "max_temp_lag_1": max_temp_lag_1,
        "min_temp_lag_1": min_temp_lag_1,
    }])

    pred_mwh = float(pipeline.predict(input_df)[0])
    diff_t1 = pred_mwh - demand_lag_1
    diff_t7 = pred_mwh - demand_lag_7
    temp_mean_now = (max_temp + min_temp) / 2.0

    st.success(f"### Forecasted Daily Demand: **{pred_mwh:,.1f} MWh**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Demand", f"{pred_mwh:,.0f} MWh")
    c2.metric("vs. Yesterday (t-1)", f"{pred_mwh:,.0f} MWh", f"{diff_t1:+,.0f} MWh")
    c3.metric("vs. Last Week (t-7)", f"{pred_mwh:,.0f} MWh", f"{diff_t7:+,.0f} MWh")
    c4.metric("Mean Temperature", f"{temp_mean_now:.1f} °C")

    with st.expander("Inspect Engineered Input Payload"):
        st.dataframe(input_df, use_container_width=True)
