
import streamlit as st
import pandas as pd
import joblib

# Load Model
model_path = "model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Ensure pipeline has trained and saved the model.")
    st.stop()

st.title("Tourism Package Prediction")
st.write("Enter customer details to predict if they will purchase a package.")

# User Inputs (Categorical and Numerical matches train set)
age = st.number_input("Age", min_value=18, max_value=90, value=35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.selectbox("Gender", ['Male', 'Female'])
num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
num_trips = st.number_input("Number of Trips", min_value=1.0, max_value=25.0, value=2.0)
passport = st.selectbox("Has Passport? (1=Yes, 0=No)", [0, 1])
pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.selectbox("Owns Car? (1=Yes, 0=No)", [0, 1])
num_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=20000.0)
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
duration_of_pitch = st.number_input("Duration of Pitch (mins)", min_value=1.0, max_value=120.0, value=15.0)
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
num_followups = st.number_input("Number of Followups", min_value=1.0, max_value=10.0, value=3.0)

if st.button("Predict Purchase"):
    # Create DataFrame from inputs
    input_data = pd.DataFrame({
        'Age': [age], 'TypeofContact': [type_of_contact], 'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch], 'Occupation': [occupation], 'Gender': [gender],
        'NumberOfPersonVisiting': [num_person], 'NumberOfFollowups': [num_followups],
        'ProductPitched': [product_pitched], 'PreferredPropertyStar': [preferred_star],
        'MaritalStatus': [marital_status], 'NumberOfTrips': [num_trips], 'Passport': [passport],
        'PitchSatisfactionScore': [pitch_satisfaction], 'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [num_children], 'Designation': [designation], 'MonthlyIncome': [monthly_income]
    })
    
    # Predict
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Result: The customer is **LIKELY** to purchase the package.")
    else:
        st.warning("Result: The customer is **UNLIKELY** to purchase the package.")
