
import gradio as gr
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("best_polynomial_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prediction function
def predict_cost(age, sex, bmi, children, smoker, region):
    # Prepare the 10 features in the exact correct order matching training
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "high_bmi_obese": 1 if bmi > 30 else 0,
        "smoker_bmi_interaction": bmi * (1 if smoker == "yes" else 0),
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0
    }])
   
    # Apply the scaler to the input data just like during training
    input_scaled = scaler.transform(input_data)
   
    # Predict the cost using the scaled input data
    prediction = model.predict(input_scaled)[0]
    return f"Estimated Insurance Cost: ${prediction:,.2f}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 Medical Insurance Cost Predictor")
    gr.Markdown("### Enter your information below to estimate insurance charges.")
   
    with gr.Row():
        age = gr.Number(label="Age", value=25, minimum=18, maximum=100)
        sex = gr.Dropdown(choices=["female", "male"], label="Sex", value="female")
       
    with gr.Row():
        bmi = gr.Number(label="BMI", value=25.0, minimum=10, maximum=60)
        children = gr.Number(label="Number of Children", value=0, minimum=0, maximum=10)
       
    with gr.Row():
        smoker = gr.Dropdown(choices=["yes", "no"], label="Smoker", value="no")
        region = gr.Dropdown(choices=["southwest", "southeast", "northwest", "northeast"], label="Region", value="southwest")

    predict_button = gr.Button("Predict Insurance Cost", variant="primary")
    result = gr.Markdown(label="Prediction Result")

    predict_button.click(
        fn=predict_cost,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=result
    )

if __name__ == "__main__":
    demo.launch(share=True)
    