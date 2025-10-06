import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load(r"E:\intern\Diabetes\hello\diabetes_model.pkl")
scaler = joblib.load(r"E:\intern\Diabetes\hello\scaler.pkl")

st.title("🩺 Diabetes Prediction App")

# Inputs
gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
age = st.number_input("Age", min_value=1, max_value=120, step=1)
urea = st.number_input("Urea")
cr = st.number_input("Creatinine")
hba1c = st.number_input("HbA1c")
chol = st.number_input("Cholesterol")
tg = st.number_input("Triglycerides")
hdl = st.number_input("HDL")
ldl = st.number_input("LDL")
vldl = st.number_input("VLDL")
bmi = st.number_input("BMI")

if st.button("Predict"):
    # numeric 10 features (jo scaler pehle se trained hai)
    numeric_features = np.array([[age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])
    scaled_numeric = scaler.transform(numeric_features)

    # gender ko alag se add karo (scaling ke bagair)
    final_input = np.hstack(([[gender]], scaled_numeric))  # shape (1,11)

    # prediction
    prediction = model.predict(final_input)[0]

    if prediction == 0:
        st.success(" No Diabetes Detected")
    elif prediction == 1:
        st.warning(" Type 1 Diabetes Detected")
    elif prediction == 2 :
        st.error(" Type 2 Diabetes Detected")
        
