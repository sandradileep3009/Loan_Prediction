# Explainable Credit Risk Scoring System

## AI-Powered Loan Approval Prediction with Explainable Machine Learning

A machine learning based credit risk scoring system that predicts loan approval decisions while explaining the factors behind each prediction.

This project focuses on building a transparent lending model instead of a black-box classifier. The system allows a user to enter applicant details through a Streamlit interface and receive a loan decision with a risk score.

## Features

- Loan approval prediction system
- Multiple machine learning model comparison
- Automated best model selection
- Explainable AI integration
- Credit risk scoring interface
- Fairness evaluation for responsible AI


## Machine Learning Models

Implemented and compared:

- Logistic Regression (baseline model)
- Random Forest
- XGBoost


Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC


The best performing model is selected automatically based on ROC-AUC performance.


## Explainable AI (SHAP)

Integrated SHAP (SHapley Additive exPlanations) to improve model transparency.

The system focuses on answering:

"Why was this loan approved or rejected?"


The model analyzes important risk factors including:

- Credit history
- Applicant income
- Loan amount
- Applicant profile information


This helps create more transparent and interpretable lending decisions.


## Streamlit Application

Built an interactive web interface where users can input:

- Gender
- Marital status
- Education
- Employment status
- Dependents
- Applicant income
- Co-applicant income
- Loan amount
- Loan term
- Credit history
- Property area


The application provides:

- Loan approval/rejection decision
- Prediction probability
- Credit risk score


## Fairness Analysis

Added fairness evaluation using demographic parity analysis.

The system checks possible bias in predictions across demographic groups and supports responsible AI development.


## Technology Stack

Python

Machine Learning:
- Scikit-learn
- XGBoost

Explainability:
- SHAP

Data Processing:
- Pandas
- NumPy

Visualization:
- Matplotlib
- Seaborn

Deployment:
- Streamlit


## Project Workflow

Applicant Data

↓

Data Cleaning and Feature Engineering

↓

Train Multiple Machine Learning Models

↓

Compare Model Performance

↓

Select Best Model

↓

Predict Credit Risk

↓

Provide Explainable Decision


## Project Structure

Explainable-Credit-Risk-Scoring-System/

- loan.csv
- code.py
- app.py
- loan_model.pkl
- scaler.pkl
- columns.pkl
- requirements.txt
- README.md


## How To Run

Install dependencies:

pip install -r requirements.txt


Train model:

python code.py


Run application:

streamlit run app.py


## Resume Description

Built an explainable credit risk scoring system using Logistic Regression, Random Forest, and XGBoost with SHAP-based interpretability and a Streamlit interface that provides transparent loan approval predictions while highlighting key risk factors behind each decision.
