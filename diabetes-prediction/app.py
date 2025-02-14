import streamlit as st
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('Daibetes_model')

model = load_model()

# App Title
st.title("Diabetes Prediction App")

# Input fields for user
st.header("Enter the Patient's Details:")
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0.0, step=0.1)
blood_pressure = st.number_input("BloodPressure", min_value=0.0, step=0.1)
skin_thickness = st.number_input("SkinThickness", min_value=0.0, step=0.1)
insulin = st.number_input("Insulin", min_value=0.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    # Create an input array
    input_features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Make a prediction
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    # Display result
    if prediction == 1:
        st.write(f"The model predicts: **Diabetic** with a probability of {probability:.2%}.")
    else:
        st.write(f"The model predicts: **Non-Diabetic** with a probability of {probability:.2%}.")
