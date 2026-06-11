import streamlit as st
import pickle
import pandas as pd

st.title("🏦 Loan Application Predictor")

model = pickle.load(open("loan.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0.0,
    value=5000.0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=0.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=100.0
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0,
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)

if st.button("Predict"):

    # Encode categories

    gender_male = 1 if gender == "Male" else 0
    married_yes = 1 if married == "Yes" else 0
    education_not_graduate = 1 if education == "Not Graduate" else 0
    self_employed_yes = 1 if self_employed == "Yes" else 0

    dependents_1 = 1 if dependents == "1" else 0
    dependents_2 = 1 if dependents == "2" else 0
    dependents_3plus = 1 if dependents == "3+" else 0

    property_area_semiurban = 1 if property_area == "Semiurban" else 0
    property_area_urban = 1 if property_area == "Urban" else 0

    data = pd.DataFrame([[
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        gender_male,
        married_yes,
        dependents_1,
        dependents_2,
        dependents_3plus,
        education_not_graduate,
        self_employed_yes,
        property_area_semiurban,
        property_area_urban
    ]])
    data = data.reindex(columns=columns, fill_value=0)
    
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Loan Application Approved")
    else:
        st.error("❌ Loan Application Rejected")