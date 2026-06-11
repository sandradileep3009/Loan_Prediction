import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Loan Predictor")

st.title("🏦 Loan Application Predictor")

# Load files
model = pickle.load(open("loan.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Inputs

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0.0,
    max_value=100000.0,
    value=5000.0,
    step=100.0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    max_value=100000.0,
    value=0.0,
    step=100.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=1.0
)

loan_amount_term = st.selectbox(
    "Loan Amount Term",
    [12, 36, 60, 84, 120, 180, 240, 300, 360]
)

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)

# Prediction

if st.button("Predict"):

    input_data = {
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,

        "Gender_Male": 1 if gender == "Male" else 0,

        "Married_Yes": 1 if married == "Yes" else 0,

        "Dependents_1": 1 if dependents == "1" else 0,
        "Dependents_2": 1 if dependents == "2" else 0,
        "Dependents_3+": 1 if dependents == "3+" else 0,

        "Education_Not Graduate":
            1 if education == "Not Graduate" else 0,

        "Self_Employed_Yes":
            1 if self_employed == "Yes" else 0,

        "Property_Area_Semiurban":
            1 if property_area == "Semiurban" else 0,

        "Property_Area_Urban":
            1 if property_area == "Urban" else 0,
    }

    # Create dataframe
    data = pd.DataFrame([input_data])

    # Match training columns
    data = data.reindex(columns=columns, fill_value=0)

    # Scale
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    probability = model.predict_proba(data_scaled)

    approval_probability = probability[0][1] * 100

    st.metric(
        "Approval Probability",
        f"{approval_probability:.2f}%"
    )

    if prediction[0] == 1:
        st.success("✅ Loan Application Approved")
    else:
        st.error("❌ Loan Application Rejected")

    # Debug section (remove later)
    with st.expander("Show Input Features"):
        st.write(data)