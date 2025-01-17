import streamlit as st
import numpy as np
import joblib

# 1. Load the model, scaler, and encoder
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Prototype Prediction: Will the User Make a Purchase?")

# User input
age = st.number_input("Enter Age:", min_value=0, max_value=100, step=1)
income = st.number_input("Enter Income (RM):", min_value=0.0, step=100.0)
gender = st.selectbox("Select Gender:", ['male', 'female'])

# 3. Process user input
if st.button("Predict"):
    # Encode 'gender'
    gender_encoded = label_encoder.transform([gender])[0]
    
    # Combine user inputs
    user_data = np.array([[age, income, gender_encoded]])
    
    # Scale the user input data
    scaled_user_data = scaler.transform(user_data)
    
    # Prediction
    prediction = model.predict(scaled_user_data)
    prediction_proba = model.predict_proba(scaled_user_data)
    
    # Display the results
    if prediction[0] == 1:
        st.success(f"Prediction: The user is highly likely to make a purchase (Confidence: {prediction_proba[0][1]:.2f}).")
    else:
        st.warning(f"Prediction: The user is unlikely to make a purchase (Confidence: {prediction_proba[0][0]:.2f}).")
