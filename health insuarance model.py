import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("C:\\Users\\mwiri\\Downloads\\reg_model.joblib")


# Streamlit app UI
st.title('Health Insurance Cost Prediction')

# User Inputs
age = st.number_input('Age', min_value=0, max_value=100, value=25)
sex = st.selectbox('Sex', ['Male', 'Female'])
bmi = st.number_input('BMI', min_value=0.0, value=22.5)
children = st.number_input('Number of Children', min_value=0, value=1)
smoker = st.selectbox('Smoker', ['Yes', 'No'])
region = st.selectbox('Region', ['Southwest', 'Southeast', 'Northwest', 'Northeast'])

# Map categorical inputs to numerical values
sex = 1 if sex == 'Male' else 0
smoker = 1 if smoker == 'Yes' else 0
region_mapping = {'Southwest': 0, 'Southeast': 1, 'Northwest': 2, 'Northeast': 3}
region = region_mapping[region]

# Prepare the input features for prediction
features = np.array([[age, sex, bmi, children, smoker, region]])

# Predict the cost
if st.button('Predict'):
    prediction = model.predict(features)
    st.write(f'Predicted Health Insurance Cost: ${prediction[0]:,.2f}')


 