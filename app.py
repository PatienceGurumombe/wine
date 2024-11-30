import streamlit as st
import joblib
import pandas as pd
import numpy as np
import random

# Load the trained model
model = joblib.load('model.pkl')

# Define the mean values based on the dataset statistics
mean_values = {
    'alcohol': 10.514267,
    'fixed_acidity': 6.854788,
    'volatile_acidity': 0.278241,
    'citric_acid': 0.334192,
    'residual_sugar': 6.391415,
    'chlorides': 0.045772,
    'free_sulfur_dioxide': 35.308085,
    'total_sulfur_dioxide': 138.360657,
    'density': 0.994027,
    'pH': 3.188267,
    'sulphates': 0.489847,
    'quality': 5.877909
}

# Add subtitle with your name and ID
st.subheader("Patience Gurumombe: R219486B")

# Sidebar navigation for pages
page = st.sidebar.radio("Choose a page", ("Manual Input", "Simulation"))

# Function to generate random input values within the respective ranges
def generate_random_input():
    random_input = {
        'alcohol': random.uniform(8.0, 14.2),
        'fixed_acidity': random.uniform(0.1, 1.5),
        'volatile_acidity': random.uniform(0.08, 1.1),
        'citric_acid': random.uniform(0.0, 1.66),
        'residual_sugar': random.uniform(0.6, 65.8),
        'chlorides': random.uniform(0.009, 0.346),
        'free_sulfur_dioxide': random.uniform(2.0, 289.0),
        'total_sulfur_dioxide': random.uniform(9.0, 440.0),
        'density': random.uniform(0.9871, 1.03898),
        'pH': random.uniform(2.72, 3.82),
        'sulphates': random.uniform(0.22, 1.08),
        'quality': random.randint(3, 9)
    }
    return pd.DataFrame([list(random_input.values())], columns=random_input.keys())

# Manual Input Page (Page 1)
if page == "Manual Input":
    # Create columns for manual input with default values set to the mean
    col1, col2, col3 = st.columns(3)
    with col1:
        alcohol = st.number_input('Alcohol', min_value=0.0, max_value=20.0, step=0.1, value=mean_values['alcohol'])
    with col2:
        fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, step=0.1, value=mean_values['fixed_acidity'])
    with col3:
        volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, step=0.01, value=mean_values['volatile_acidity'])

    # Another row for more inputs
    col4, col5, col6 = st.columns(3)
    with col4:
        citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=2.0, step=0.01, value=mean_values['citric_acid'])
    with col5:
        residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=20.0, step=0.1, value=mean_values['residual_sugar'])
    with col6:
        chlorides = st.number_input('Chlorides', min_value=0.0, max_value=1.0, step=0.01, value=mean_values['chlorides'])

    # More inputs for remaining features
    col7, col8, col9 = st.columns(3)
    with col7:
        free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, step=0.1, value=mean_values['free_sulfur_dioxide'])
    with col8:
        total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, max_value=200.0, step=0.1, value=mean_values['total_sulfur_dioxide'])
    with col9:
        density = st.number_input('Density', min_value=0.0, max_value=2.0, step=0.0001, value=mean_values['density'])

    # Inputs for pH, sulphates, and quality
    col10, col11, col12 = st.columns(3)
    with col10:
        pH = st.number_input('pH', min_value=0.0, max_value=14.0, step=0.01, value=mean_values['pH'])
    with col11:
        sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, step=0.01, value=mean_values['sulphates'])
    with col12:
        quality = st.number_input('Quality', min_value=0.0001, max_value=10.0, step=1.0, value=mean_values['quality'])

    # Create a dataframe with the user inputs
    user_input = pd.DataFrame([[alcohol, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, 
                                sulphates, quality]], 
                              columns=['alcohol', 'fixed_acidity', 'volatile_acidity', 'citric_acid', 
                                       'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 
                                       'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'quality'])

    # Prediction button
    if st.button('Predict Wine Type'):
        try:
            # Make prediction (0 for white wine, 1 for red wine)
            prediction = model.predict(user_input)
            
            # Display the predicted wine type
            wine_type = "Red Wine" if prediction[0] == 1 else "White Wine"
            st.write(f'Predicted Wine Type: {wine_type}')
            
        except Exception as e:
            st.error(f"Error: {e}")

# Simulation Page (Page 2)
elif page == "Simulation":
    st.subheader("Wine Type Prediction - Simulation")
    
    if st.button("Generate Random Inputs and Predict"):
        # Generate random inputs
        random_input = generate_random_input()

        # Make prediction (0 for white wine, 1 for red wine)
        prediction = model.predict(random_input)
        
        # Display the predicted wine type at the top
        wine_type = "Red Wine" if prediction[0] == 1 else "White Wine"
        st.write(f'Predicted Wine Type: {wine_type}')

        # Use custom CSS to fill the table width and height
        st.markdown("""
            <style>
                .full-width-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .full-width-table td, .full-width-table th {
                    padding: 8px;
                    border: 1px solid #ddd;
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)

        # Display the transposed random input in a table with full width and height
        st.write(random_input.T.to_html(classes="full-width-table"), unsafe_allow_html=True)

st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        <p>Patience Gurumombe - R219486B | Wine Classification App</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
