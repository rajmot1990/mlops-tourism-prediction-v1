
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load Model dynamically
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Ensure pipeline has trained and saved the model.")
    st.stop()
    
st.title("Tourism Package Prediction")
st.write("Enter customer details to predict if they will purchase a package.")

# User Inputs (Categorical and Numerical matches train set)
age = st.number_input("Age", min_value=18.0, max_value=90.0, value=35.0)
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.selectbox("Gender", ['Male', 'Female'])
num_person = st.number_input("Number of Persons Visiting", min_value=1.0, max_value=10.0, value=2.0)
preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
num_trips = st.number_input("Number of Trips", min_value=1.0, max_value=25.0, value=2.0)
passport = st.selectbox("Has Passport? (1=Yes, 0=No)", [0, 1])
pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1.0, 5.0, 3.0)
own_car = st.selectbox("Owns Car? (1=Yes, 0=No)", [0, 1])
num_children = st.number_input("Number of Children", min_value=0.0, max_value=10.0, value=0.0)
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=200000.0, value=20000.0)
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
duration_of_pitch = st.number_input("Duration of Pitch (mins)", min_value=1.0, max_value=120.0, value=15.0)
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
num_followups = st.number_input("Number of Followups", min_value=1.0, max_value=10.0, value=3.0)

if st.button("Predict Purchase"):
    # Create DataFrame from inputs, enforcing exact data types
    input_data = pd.DataFrame({
        'Age': [float(age)], 
        'TypeofContact': [type_of_contact], 
        'CityTier': [int(city_tier)],
        'DurationOfPitch': [float(duration_of_pitch)], 
        'Occupation': [occupation], 
        'Gender': [gender],
        'NumberOfPersonVisiting': [float(num_person)], 
        'NumberOfFollowups': [float(num_followups)],
        'ProductPitched': [product_pitched], 
        'PreferredPropertyStar': [float(preferred_star)],
        'MaritalStatus': [marital_status], 
        'NumberOfTrips': [float(num_trips)], 
        'Passport': [int(passport)],
        'PitchSatisfactionScore': [float(pitch_satisfaction)], 
        'OwnCar': [int(own_car)],
        'NumberOfChildrenVisiting': [float(num_children)], 
        'Designation': [designation], 
        'MonthlyIncome': [float(monthly_income)]
    })
    
    try:
        # Predict
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.success("Result: The customer is **LIKELY** to purchase the package.")
        else:
            st.warning("Result: The customer is **UNLIKELY** to purchase the package.")
    
    # This block ensures any pipeline mismatch errors are printed visibly on the UI
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
