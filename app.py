import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💰 Loan Eligibility Prediction App")

# User input fields
gender = st.radio("Gender", ["Male", "Female"])
married = st.radio("Married", ["Yes", "No"])
education = st.radio("Education", ["Graduate", "Not Graduate"])
self_employed = st.radio("Self Employed", ["Yes", "No"])
dependents = st.slider("Number of Dependents", 0, 3, 0)
applicant_income = st.slider("Applicant Income ($)", 1000, 100000, 1000, step=1000)
coapplicant_income = st.slider("Coapplicant Income ($)", 0, 50000, 0, step=1000)
loan_amount = st.slider("Loan Amount ($1000s)", 5, 500, 50)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 60, 120, 180, 240, 360])
credit_history = st.radio("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode user input
user_data = np.array([[ 
    1 if gender == "Male" else 0,
    1 if married == "Yes" else 0,
    1 if education == "Graduate" else 0,
    1 if self_employed == "Yes" else 0,
    dependents,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
]])

# Predict
if st.button("Check Eligibility"):
    prediction = model.predict(user_data)[0]
    result = "✅ You are Eligible" if prediction == 1 else "❌ You are not Eligible"
    st.success(result)
