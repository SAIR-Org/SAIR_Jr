# python -m streamlit run app.py to start the app
import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.set_page_config(page_title="Insurance Prediction", layout="centered")

st.title("💰 Medical Insurance Cost Prediction")
st.write("Enter customer details:")


age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Children", 0, 5, 0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "southeast", "southwest"])

# =========================
# TRANSFORM
# =========================

sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    input_data = np.array([[
        age,
        sex,
        bmi,
        children,
        smoker,
        region_northwest,
        region_southeast,
        region_southwest
    ]])

    prediction = model.predict(input_data)

    st.success(f"💵 Estimated Insurance Cost: ${prediction[0]:.2f}")