# app.py
import streamlit as st
import joblib
import pandas as pd
import os
import glob
from utils import AdvancedFeatureEngineer, OutlierHandler

# ===========================
# 1️⃣ Auto-detect latest model version
# ===========================
model_folders = glob.glob("models/v1_20261019_152903")
if not model_folders:
    st.error("❌ No model folders found in 'models/'. Please train and save a model first.")
    st.stop()

# Get the latest folder by modification time
MODEL_DIR = max(model_folders, key=os.path.getmtime)
model_path = os.path.join(MODEL_DIR, "best_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Safety check
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at {model_path}")
    st.stop()
if not os.path.exists(preprocessor_path):
    st.error(f"❌ Preprocessor file not found at {preprocessor_path}")
    st.stop()

# ✅ Make sure utils.py is imported before unpickling
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_model()

# ===========================
# 2️⃣ Streamlit UI
# ===========================
st.set_page_config(page_title=" Prediection of Hospital Length of Stay ", layout="wide")
st.title(" Prediection of Hospital Length of Stay")
st.write(
    """
    Enter the features of a patient to predict the NumbStay in the Hospital.
    This model is a **Gradient Boosting Regressor** trained on Hospital Length of Stay dataset.
    """
)

st.sidebar.header(" Input Patient Data")
gender = st.radio("Gender", ["M" , "F"], horizontal = True)
gender_val = 0 if gender =="M" else 1
#------------Diagnoses : checkboxes    (0/1)-----------
st.subheader("Diagnoses")
col1, col2 = st.columns(2)

with col1:
    rcount = st.slider("Readmission Count(last 180 days )", 0, 5, 0 ,help=" 5 means 5 or more")
    dialysisrenalendstage = st.checkbox("End_Stage Renal Disease")
    asthma = st.checkbox("Asthma")
    irondef  = st.checkbox("Iron Deficiency (Anemia)")
    pneum   = st.checkbox("pneumonia)")                      
    substancedependence  = st.checkbox("substance Dependence)")       
    psychologicaldisordermajor  = st.checkbox("Major Psychological Disorder")  

with col2:

    depress    = st.checkbox("Depression")     
    psychother    = st.checkbox("other Psychological Disorde ")    
    fibrosisandother  = st.checkbox("cystic Fibrosis and other Related conditions")     
    malnutrition   = st.checkbox("Malnutrition")    
    hemo= st.checkbox("Hemophilia")


#----------------Lab values & vitals: sliders#
st.subheader("Lab values & vitals")
hematocrit= st.slider("Hematocrit(red blood cell precentage in blood)", 4.4, 24.1, 11.9, 0.1)                  
neutrophils= st.slider("Neutrophils(a type of white blood cell)", 0.1, 30.0, 9.4, 0.1)                
sodium= st.slider("serum Sodium", 125.0, 151.0, 137.9, 0.1)                     
glucose= st.slider("blood Glucose", 60.0, 272.0, 142.0, 0.1)                     
bloodureanitro= st.slider("blood Urea Nitrrogen", 1.0, 100.0, 12.0, 0.5)              
creatinine= st.slider("Serum Greatinine", 0.2, 2.1, 1.1, 0.01)                  
bmi= st.slider("Body Mass Index", 22.0, 39.0, 29.8, 0.1)                        
pulse= st.slider("Heart Rate", 20, 130, 73)                      
respiration= st.slider("Respiratory Rate", 0.2, 10.0, 6.5, 0.1)                  
secondarydiagnosisnonicd9 = st.slider("Number of Secondary Diagnoses(non_ICD-9)", 0, 10, 1 )    



    
# ===========================
# 3️⃣ Prepare input DataFrame
# ===========================
feature_names = ["num__rcount", 'num__dialysisrenalendstage', 'num__asthma',
       'num__irondef', 'num__pneum', 'num__substancedependence',
       'num__psychologicaldisordermajor', 'num__depress', 'num__psychother',
       'num__fibrosisandother', 'num__malnutrition', 'num__hemo',
       'num__hematocrit', 'num__neutrophils', 'num__sodium', 'num__glucose',
       'num__bloodureanitro', 'num__creatinine', 'num__bmi', 'num__pulse',
       'num__respiration', 'num__secondarydiagnosisnonicd9', 'cat__gender_M'
]

input_df = pd.DataFrame([[ rcount, dialysisrenalendstage ,asthma ,irondef, pneum , 
                          substancedependence , psychologicaldisordermajor, depress 
                          ,psychother ,fibrosisandother, malnutrition , hemo, hematocrit,neutrophils,sodium,glucose,
                              bloodureanitro , creatinine ,  bmi ,   pulse , respiration , secondarydiagnosisnonicd9, gender_val
    
]], columns=feature_names)

# ===========================
# 4️⃣ Preprocess & Predict
# ===========================
if st.button("predict Length of stay"):
    try:
        X_input = preprocessor.transform(input_df)
        prediction = model.predict(X_input)[0]
        predicted_Days= prediction  

        st.success(f"### Predicted Days: {int(round(predicted_Days))}")
        st.write("#### Input Features") 
        st.write(input_df)
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")

# ===========================
# 5️⃣ Sidebar Info
# ===========================
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Model Information")
st.sidebar.write(f"**Loaded from:** `{MODEL_DIR}`")
st.sidebar.write("**Model:** Gradient Boosting Regressor")
st.sidebar.write("**Dataset:** LengthofStay")
st.sidebar.write("**Validation R²:** 0.9411")
st.sidebar.write("**Test R²:** 0.9403 | RMSE: 0.5725 | MAE: 0.4160")