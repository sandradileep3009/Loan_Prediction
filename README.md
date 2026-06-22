💳 CrediShield AI — Explainable Credit Risk Scoring & Bias Audit Platform
Overview
CrediShield AI is an end-to-end machine learning platform designed to predict loan default risk and provide transparent, regulatory-compliant risk assessments.

Because modern lenders require transparency rather than black-box decisions, this system benchmarks multiple classification models, evaluates fairness across demographic groups, and utilizes SHAP explainability to isolate the top 3 risk drivers for individual applicants within an interactive Streamlit UI.

⚠️ This project is an AI prototype for research, automated lending simulation, and educational purposes.

Key Features
🧠 Multi-Model Benchmarking & Credit Scoring
Implemented and evaluated three distinct models: Logistic Regression (baseline), Random Forest, and XGBoost.

Automated selection of the optimal champion model based on the highest ROC-AUC score.

Handled end-to-end data pipelines including categorical one-hot encoding, missing value mitigation, and feature scaling.

🔍 Regulatory Transparency & Explainable AI (XAI)
Integrated SHAP (SHapley Additive exPlanations) to break down complex machine learning decisions.

Generates SHAP waterfall plots dynamically for individual applicants to reveal exactly why a loan application was approved or denied.

Directly surfaces the top 3 risk factors influencing a lender's decision.

⚖️ Algorithmic Fairness & Bias Auditing
Incorporated Fairlearn to audit predictions for systematic bias.

Evaluates the Demographic Parity Difference across sensitive attributes (such as Gender) to ensure ethical and compliant AI lending practices.

📈 Advanced Model Evaluation
Evaluated and visualized via:

Accuracy, Precision, Recall, and F1-Score

Confusion Matrix analytics

Comparative Multi-Model ROC Curve plotting

🖥️ Streamlit Loan Officer Dashboard
An interactive UI tailored for financial institutions:

Interactive input forms for loan applicants' details.

Real-time credit decisions (Approved/Denied).

Visual, local feature contribution graphs for clear compliance auditing.

Tech Stack
Languages

Python

Machine Learning & Frameworks

Scikit-learn

XGBoost

Fairlearn (Fairness Metric Auditing)

Explainability

SHAP (TreeExplainer)

Data Science & Visualization

Pandas

NumPy

Matplotlib

Seaborn

Deployment

Streamlit

Project Workflow
Loan Applicant Data (loan.csv)

↓

Data Cleaning & One-Hot Encoding

↓

Standard Feature Scaling

↓

Multi-Model Benchmarking (LogReg, RF, XGBoost)

↓

Champion Model Selection (Highest ROC-AUC)

↓

Fairness & Bias Audit (Demographic Parity)

↓

SHAP Waterfall Generation (Local Risk Factors)

↓

Streamlit Interactive Dashboard UI
Project Structure
Credit_risk_scoring/
│
├── code.py               # Model training, benchmarking, bias auditing, and serialization
├── app.py                # Streamlit web application frontend
├── loan.csv              # Raw credit applicant dataset
├── loan_model.pkl        # Serialized champion model (XGBoost/RF)
├── scaler.pkl            # Saved StandardScaler instance
├── columns.pkl           # Saved feature column mapping for alignment
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
Running the Project
Install dependencies:

Bash
pip install -r requirements.txt
Train and evaluate models (saves artifacts):

Bash
python code.py
Run the interactive dashboard application:

Bash
streamlit run app.py
Resume Highlight
Built an explainable credit risk scoring system (XGBoost + SHAP) with a Streamlit interface that surfaces the top risk drivers per applicant — designed for regulatory transparency.

Author
Sandra Dileep Computer Science Student | AI/ML Enthusiast
