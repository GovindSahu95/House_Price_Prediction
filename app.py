import pandas as pd
import streamlit as st

# Load model
import os
import pickle

# Get absolute path of current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "models\House_predict_model.pkl")

with open('models\House_predict_model.pkl', "rb") as f:
    model = pickle.load(f)

# App Header
st.header('House Prices Predict')

# Load dataset
data = pd.read_csv('Bengaluru_House_Data.csv')

# User Inputs
loc = st.selectbox('Choose the location', data['location'].unique())
sqft = st.number_input('Enter Total Sqft', min_value=0.0)
beds = st.number_input('Enter No of Bedrooms', min_value=0.0)
bath = st.number_input('Enter No of Bathrooms', min_value=0.0)
balc = st.number_input('Enter No of Balconies', min_value=0.0)

# Create input DataFrame using user inputs
input = pd.DataFrame([[loc, sqft, bath, balc, beds]], columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

# Prediction
if st.button("Predict Price"):
    output = model.predict(input)
    st.write(f"Price of House is : ₹{output[0] * 100000:.2f}")