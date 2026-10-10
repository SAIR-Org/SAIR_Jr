# 🏠 Energy Efficiency Prediction

A machine learning project for predicting the heating load of buildings based on their energy efficiency characteristics.

This project combines data analysis, feature engineering, data preprocessing, machine learning model comparison, hyperparameter tuning, MLflow experiment tracking, and an interactive Streamlit web application for heating load prediction.

## 📌 Project Overview

Building energy efficiency is closely related to the physical characteristics of a building and its design.

The objective of this project is to use building characteristics to develop a machine learning model that predicts heating load.

The project workflow includes:

- Data preparation and exploration
- Exploratory Data Analysis (EDA)
- Feature engineering
- Outlier detection
- Train-test splitting
- Feature scaling
- Regression model comparison
- Hyperparameter tuning
- Model evaluation
- MLflow experiment tracking
- Saving the trained model
- Interactive prediction using Streamlit

## ⚙️ Input Features

The application uses the following building characteristics:

- Relative_Compactness
- Surface_Area
- Wall_Area
- Roof_Area
- Overall_Height
- Orientation
- Glazing_Area
- Glazing_Area_Distribution
- Total_Surface_Area
- Height_to_Surface_Ratio

## 🎯 Target Variable

Heating_Load — the heating load of the building.

## 🧠 Machine Learning Models

Several regression models were trained and compared during the project, including:

- Linear Regression
- Ridge Regression
- Elastic Net
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)
- Voting Ensemble
- Polynomial Regression

After comparing the models, the best-performing models were selected for hyperparameter tuning.

The final model was selected based on its performance on the test data.

### 🏆 Final Model

Tuned Gradient Boosting Regressor

The final model was optimized using GridSearchCV with 5-fold cross-validation.

The best parameters found were:

- n_estimators = 200
- learning_rate = 0.1
- max_depth = 3

The final trained model is saved as:

energy_model.joblib

## 🔬 Feature Engineering

Additional features were created from the original building characteristics to provide the models with useful information about the relationships between the variables.

The engineered features include:

- Total_Surface_Area
- Height_to_Surface_Ratio

These features were included in the final model training.

## 📊 Model Performance

The final model was evaluated using:

R²                      0.998032
RMSE                     0.45294
MAE                     0.350228

## 🖥️ Streamlit Application

The project includes an interactive web application called:

Energy Efficiency Predictor

The application allows users to enter building characteristics and obtain a predicted heating load.

The application:

- Loads the trained Gradient Boosting model.
- Accepts the required building characteristics.
- Creates the required input features.
- Generates a heating load prediction.
- Displays the predicted value through an interactive interface.

## 📁 Project Structure

The main project files include:

- app.py — Streamlit prediction application
- energy_model.joblib — trained Gradient Boosting model
- scaler.joblib — saved feature scaler
- model_card.json — model information and evaluation results
- Regression notebook — machine learning development and experimentation
- experiments/ — MLflow experiment tracking files

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- MLflow
- Jupyter Notebook
- Git
- GitHub

## 💾 Model Files

The trained final model is stored in:

energy_model.joblib

The project also includes:

scaler.joblib

and:

model_card.json

The model card contains information about the final model, evaluation metrics, features, and target variable.

## 📓 Jupyter Notebook

The notebook contains the machine learning workflow used for developing the project, including:
- Data exploration
- EDA
- Feature engineering
- Model training
- Model comparison
- Hyperparameter tuning
- Model evaluation
- Final model selection

## 📈 Prediction Workflow

The prediction workflow follows these main steps:

Building Characteristics  
→ Data Preparation  
→ Feature Engineering  
→ Train/Test Split  
→ Model Training  
→ Hyperparameter Tuning  
→ Gradient Boosting Regression  
→ Heating Load Prediction

## 🎯 Project Objective

The main objective of this project is to demonstrate the use of machine learning for building energy efficiency prediction and to provide an interactive tool for estimating heating load from building characteristics.


## By Engineer :
Sadeem Mohamed