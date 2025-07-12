import streamlit as st
import pandas as pd
import pickle

# 🎯 Load the trained model
model = pickle.load(open("loan_model.pkl", "rb"))

# 🧭 Sidebar
st.sidebar.title("ℹ️ App Info")
st.sidebar.info("""
This is a **Loan Eligibility Prediction** app built using a trained **XGBoost** model.
- Data Source: Kaggle Loan Prediction Dataset
- Built with: Streamlit
- Model: XGBoost Classifier
""")

# 🏦 Title
st.title("🏦 Loan Eligibility Prediction App")

# 📋 Form input fields
st.header("📝 Enter Applicant Details:")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Amount Term (in days)", [360, 180, 120, 240, 300, 480, 60])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# 🧮 Preprocess input
def preprocess_input():
    return pd.DataFrame([{
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': 3 if dependents == "3+" else int(dependents),
        'Education': 0 if education == "Graduate" else 1,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0)
    }])

# 🧾 Predict and show result
if st.button("📊 Predict Loan Eligibility"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    st.subheader("🔍 Prediction Result:")
    st.success(result)

    # 📥 Add download button
    input_data["Loan_Status_Predicted"] = result
    csv = input_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Result as CSV",
        data=csv,
        file_name="loan_prediction_result.csv",
        mime='text/csv'
    )
