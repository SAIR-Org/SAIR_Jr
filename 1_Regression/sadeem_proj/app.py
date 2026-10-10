import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Energy Efficiency Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f7fbf9;
    }

    .title {
        text-align: center;
        color: #176b55;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #5f746c;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .card {
        background-color: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #dceee7;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .result {
        background-color: #e8f7f1;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        border: 2px solid #b9e4d5;
        margin-top: 20px;
    }

    .result-title {
        color: #176b55;
        font-size: 20px;
        font-weight: 600;
    }

    .result-value {
        color: #124c3e;
        font-size: 38px;
        font-weight: 700;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        background-color: #2e9d7b;
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 12px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #238565;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Load model
model = joblib.load("energy_model.joblib")


# Header
st.markdown(
    '<div class="title">🏠 Energy Efficiency Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the heating load of a building using a trained Gradient Boosting model.'
    '</div>',
    unsafe_allow_html=True
)


# Input section
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🏗️ Building Information")

col1, col2 = st.columns(2)

with col1:
    relative_compactness = st.number_input(
        "Relative Compactness",
        min_value=0.0,
        value=0.75
    )

    surface_area = st.number_input(
        "Surface Area",
        min_value=0.0,
        value=500.0
    )

    wall_area = st.number_input(
        "Wall Area",
        min_value=0.0,
        value=300.0
    )

    roof_area = st.number_input(
        "Roof Area",
        min_value=0.0,
        value=100.0
    )

    overall_height = st.number_input(
        "Overall Height",
        min_value=0.0,
        value=3.5
    )

with col2:
    orientation = st.number_input(
        "Orientation",
        min_value=0.0,
        value=2.0,
        step=1.0
    )

    glazing_area = st.number_input(
        "Glazing Area",
        min_value=0.0,
        value=0.2
    )

    glazing_area_distribution = st.number_input(
        "Glazing Area Distribution",
        min_value=0.0,
        value=2.0,
        step=1.0
    )

    total_surface_area = st.number_input(
        "Total Surface Area",
        min_value=0.0,
        value=900.0
    )

    height_to_surface_ratio = st.number_input(
        "Height to Surface Ratio",
        min_value=0.0,
        value=0.004
    )

st.markdown('</div>', unsafe_allow_html=True)


# Prediction
if st.button("🔮 Predict Heating Load"):

    input_data = pd.DataFrame({
        "Relative_Compactness": [relative_compactness],
        "Surface_Area": [surface_area],
        "Wall_Area": [wall_area],
        "Roof_Area": [roof_area],
        "Overall_Height": [overall_height],
        "Orientation": [orientation],
        "Glazing_Area": [glazing_area],
        "Glazing_Area_Distribution": [glazing_area_distribution],
        "Total_Surface_Area": [total_surface_area],
        "Height_to_Surface_Ratio": [height_to_surface_ratio]
    })

    prediction = model.predict(input_data)[0]
    st.markdown(
        f"""
        <div class="result">
            <div class="result-title">🔥 Predicted Heating Load</div>
            <div class="result-value">{prediction:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()

    st.success("Prediction completed successfully! 🎉")