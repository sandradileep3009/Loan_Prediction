# 🏦 Loan Approval Prediction using Machine Learning

A Machine Learning project that predicts whether a loan application will be approved based on applicant details such as income, education, marital status, credit history, and property area.

## 📌 Project Overview

Financial institutions receive thousands of loan applications every year. Evaluating these applications manually can be time-consuming and prone to human bias.

This project uses Machine Learning algorithms to automate the loan approval prediction process by analyzing applicant information and predicting whether a loan is likely to be approved.

---

## 🚀 Features

- Data Cleaning and Preprocessing
- Missing Value Handling
- Exploratory Data Analysis (EDA)
- One-Hot Encoding of Categorical Variables
- Feature Scaling using StandardScaler
- Model Training and Comparison
- Loan Approval Prediction
- Streamlit Web Application
- Model Serialization using Pickle

---

## 📊 Dataset Features

| Feature | Description |
|----------|------------|
| Gender | Applicant Gender |
| Married | Marital Status |
| Dependents | Number of Dependents |
| Education | Education Level |
| Self_Employed | Self Employment Status |
| ApplicantIncome | Applicant Income |
| CoapplicantIncome | Co-applicant Income |
| LoanAmount | Loan Amount Requested |
| Loan_Amount_Term | Loan Repayment Term |
| Credit_History | Credit History Status |
| Property_Area | Property Location |
| Loan_Status | Target Variable |

---

## 🔍 Exploratory Data Analysis

The following analyses were performed:

- Loan Approval Distribution
- Missing Value Analysis
- Income Distribution
- Loan Amount Distribution
- Correlation Heatmap
- Credit History vs Loan Approval
- Education vs Loan Approval
- Property Area vs Loan Approval
- Marital Status vs Loan Approval

### Key Insights

- Applicants with a positive credit history had significantly higher approval rates.
- Income and loan amount influence approval probability.
- Education and property area showed moderate impact on loan approval.
- Credit History emerged as one of the most important features.

---

## 🤖 Machine Learning Models

The following models were trained and compared:

### 1. Logistic Regression
A linear classification model commonly used for binary classification problems.

### 2. Random Forest Classifier
An ensemble learning method that combines multiple decision trees.

### 3. XGBoost Classifier
A gradient boosting algorithm known for high predictive performance.

---

## 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- ROC-AUC Score
- Confusion Matrix

