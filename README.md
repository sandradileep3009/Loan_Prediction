# Explainable Credit Risk Scoring System

## AI-powered loan approval prediction platform with model explainability and risk transparency

A machine learning based credit risk scoring system that predicts loan approval decisions while providing interpretable insights into the factors influencing each decision.

Unlike traditional black-box loan prediction models, this project focuses on **explainable AI (XAI)** by showing why a loan application receives a specific prediction.

The system allows a loan officer to enter applicant details through a Streamlit interface and receive:

- Loan approval prediction
- Approval probability score
- Model-based risk assessment
- Explainable decision factors


---

# Project Overview

Financial institutions need accurate risk prediction models, but regulatory requirements demand transparency.

This project builds an end-to-end credit risk assessment pipeline:

Applicant Data  
↓  
Data Processing  
↓  
Multiple ML Models  
↓  
Best Model Selection  
↓  
Loan Decision Prediction  
↓  
Explainable AI Analysis  


---

# Features

## Machine Learning Pipeline

Implemented and compared multiple classification models:

- Logistic Regression (baseline model)
- Random Forest
- XGBoost


Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC


The best performing model is automatically selected and saved.


---

# Explainable AI (XAI)

Integrated SHAP (SHapley Additive exPlanations) to improve model transparency.

The system explains:

- Which features increase approval probability
- Which features increase rejection risk
- Contribution of individual applicant factors


Example explanations:

High credit history → increases approval chance

Large loan amount → increases risk


---

# Streamlit Application

Interactive web interface for loan officers.

Users can input:

- Gender
- Education
- Employment status
- Applicant income
- Co-applicant income
- Loan amount
- Loan term
- Credit history
- Property area


Output:


