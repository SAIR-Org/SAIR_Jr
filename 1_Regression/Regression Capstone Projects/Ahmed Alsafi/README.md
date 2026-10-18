# 🚀 Employee Attrition Prediction (End-to-End ML Project)

## 📌 Overview

This project aims to predict employee attrition using machine learning techniques.
The goal is to build a robust model that can accurately identify employees likely to leave the company, with a strong focus on handling class imbalance.

---

## 🎯 Objectives

- Analyze employee data and identify key patterns
- Handle class imbalance effectively
- Build and compare multiple machine learning models
- Optimize performance using hyperparameter tuning
- Evaluate models using appropriate metrics (F1-score)

---

## 📊 Dataset

The dataset contains employee-related features such as:

- Demographics (Age, Gender, Education)
- Job-related information (Department, Job Role, Income)
- Work behavior (OverTime, Years at Company)

Target variable:

- **Attrition** (0 = No, 1 = Yes)

---

## ⚠️ Challenges

- Highly imbalanced dataset
- Mixed feature types (categorical & numerical)
- Potential noisy and redundant features

---

## 🛠️ Approach

### 1️⃣ Data Preprocessing

- Removed irrelevant and low-variance features
- Handled missing values
- Encoded categorical variables using One-Hot Encoding

---

### 2️⃣ Handling Class Imbalance

Tested multiple techniques:

- Random Upsampling
- SMOTE
- ✅ **SMOTE + Tomek Links (Final Choice)**

This combination improved class separation and reduced noise.

---

### 3️⃣ Feature Engineering

- Created new features (ratios & interactions)
- Reduced dimensionality
- Improved model learning capability

---

### 4️⃣ Modeling

Trained multiple models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

---

### 5️⃣ Hyperparameter Tuning

- Used **RandomizedSearchCV**
- Optimized model performance
- Reduced overfitting

---

### 6️⃣ Evaluation Metrics

Due to class imbalance, the primary metric used:

- ✅ **F1-Score (Weighted)**

Also evaluated using:

- Accuracy
- ROC-AUC

---

## 📈 Results

- Best performance achieved using:
  - **SMOTE + Tomek Links**
  - Tuned ensemble models

- Significant improvement in detecting the minority class
- Balanced trade-off between precision and recall

---

## 🧠 Key Insights

- Employees working overtime are more likely to leave
- Some demographic features have limited predictive power
- Feature selection improves model stability and performance

---

## ⚖️ Trade-offs

- Resampling increases computational cost
- Feature removal may drop useful information
- Model performance depends on dataset quality

---

## 🚀 Future Improvements

- Apply advanced optimization (Optuna / Bayesian tuning)
- Use SHAP for model interpretability
- Try advanced models (LightGBM, CatBoost)
- Deploy using Streamlit or FastAPI

---

## 🏁 Conclusion

This project demonstrates the importance of:

- Handling imbalanced data properly
- Thoughtful feature engineering
- Careful model evaluation

A well-structured ML pipeline can significantly improve real-world predictive performance.

---

## 📎 Kaggle Notebook

👉 https://www.kaggle.com/code/ahmedalsafiadlan/employee-attrition-prediction-with-smote-tomek

---

## 👨‍💻 Author

Ahmed Alsafi
Data Science Student
