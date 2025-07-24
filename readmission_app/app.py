import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature list
model = joblib.load('readmission_model.pkl')
feature_list = joblib.load('feature_list.pkl')


# Page title
st.title("🏥 Hospital Readmission Risk Predictor")

# Sidebar for user input
st.sidebar.header("Patient Information")

# Input form
def user_input():
    age = st.sidebar.slider("Age", 0, 100, 50)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    stay_length = st.sidebar.slider("Hospital Stay Length (days)", 1, 60, 5)
    admission_month = st.sidebar.slider("Admission Month", 1, 12, 6)
    admission_type = st.sidebar.selectbox("Admission Type", ['Elective', 'Emergency', 'Urgent'])
    medical_condition = st.sidebar.selectbox("Medical Condition", ['Diabetes', 'Cancer', 'Obesity', 'Hypertension'])
    medication = st.sidebar.selectbox("Medication", ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Penicillin'])
    test_result = st.sidebar.selectbox("Test Result", ['Normal', 'Abnormal', 'Inconclusive'])
    insurance = st.sidebar.selectbox("Insurance Provider", ['Medicare', 'Aetna', 'Blue Cross'])

    data = {
        'Age': age,
        'Gender': 1 if gender == 'Male' else 0,
        'Stay_Length': stay_length,
        'Admission Month': admission_month,
        # One-hot encoded fields below
        'Admission Type_Emergency': 1 if admission_type == 'Emergency' else 0,
        'Admission Type_Urgent': 1 if admission_type == 'Urgent' else 0,
        'Medical Condition_Cancer': 1 if medical_condition == 'Cancer' else 0,
        'Medical Condition_Diabetes': 1 if medical_condition == 'Diabetes' else 0,
        'Medical Condition_Hypertension': 1 if medical_condition == 'Hypertension' else 0,
        'Medical Condition_Obesity': 1 if medical_condition == 'Obesity' else 0,
        'Medication_Ibuprofen': 1 if medication == 'Ibuprofen' else 0,
        'Medication_Paracetamol': 1 if medication == 'Paracetamol' else 0,
        'Medication_Penicillin': 1 if medication == 'Penicillin' else 0,
        'Test Results_Inconclusive': 1 if test_result == 'Inconclusive' else 0,
        'Test Results_Normal': 1 if test_result == 'Normal' else 0,
        'Insurance Provider_Medicare': 1 if insurance == 'Medicare' else 0,
        'Insurance Provider_Blue Cross': 1 if insurance == 'Blue Cross' else 0
    }

    return pd.DataFrame([data])

# Predict
input_df = user_input()
# Ensure all 28 columns are present (even if defaulted to 0)
for col in feature_list:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with default 0

# Reorder columns to match training
input_df = input_df[feature_list]

# Make prediction
prediction = model.predict(input_df)[0]
pred_prob = model.predict_proba(input_df)[0][prediction]

st.subheader("Prediction")

# Display the prediction result with larger font
prediction_text = "🔁 Readmission Risk" if prediction == 1 else "✅ No Readmission Risk"
st.markdown(f"<h2 style='font-size:28px'>{prediction_text}</h2>", unsafe_allow_html=True)

# Display the confidence with larger font
st.markdown(f"<p style='font-size:24px'>Confidence: {pred_prob*100:.2f}%</p>", unsafe_allow_html=True)
