import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App")

# User inputs
Pregnancies = st.number_input("Pregnancies", 0, 20)
Glucose = st.number_input("Glucose Level", 50, 200)
BloodPressure = st.number_input("Blood Pressure", 30, 150)
SkinThickness = st.number_input("Skin Thickness", 0, 100)
Insulin = st.number_input("Insulin", 0, 900)
BMI = st.number_input("BMI", 10.0, 70.0)
DPF = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
Age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                             SkinThickness, Insulin, BMI, DPF, Age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"High Diabetes Risk (Probability: {probability:.2f})")
    else:
        st.success(f"Low Diabetes Risk (Probability: {probability:.2f})")
