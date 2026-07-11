import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="agarvish/tourism_project_prediction_model", filename="best_tourism_project_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Project Prediction App")
st.write("The Tourism Project Prediction App is an internal tool for 'Visit with Us' travel company's staff that predicts whether customers will take the product or not based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to take.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact?", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Category?", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation?", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender?", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip.", value=2)
PreferredPropertyStar = st.number_input("Preferred hotel rating by the customer", min_value=3.0, max_value=5.0, value=3)
MaritalStatus = st.selectbox("Marital status of the customer?", ["Single", "Married", "Unmarried", "Divorced"])
NumberOfTrips = st.number_input("Average number of trips the customer takes annually.", min_value=1, value=5)
Passport = st.selectbox("Customer holds a valid passport?", ["Yes", "No"])
OwnCar = st.selectbox("Customer owns a car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer.", min_value=1, value=5)
Designation = st.text_input("Customer's designation in their current organization.")
MonthlyIncome = st.number_input("Customer's monthly income.", min_value=0)

ProductPitched = st.selectbox("Type of product pitched to the customer", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
PitchSatisfactionScore = st.number_input("Score indicating the customer's satisfaction with the sales pitch.", min_value=1, max_value=5, value=4)
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch.", min_value=1, value=4)
DurationOfPitch = st.number_input("Duration of the sales pitch delivered to the customer.", min_value=1, value=4)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([
    {
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': 1 if CityTier == "Tier 1" else (2 if CityTier == "Tier 2" else 3),
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'ProductPitched': ProductPitched,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "take Product" if prediction == 1 else "not take Product"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
