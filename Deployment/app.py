import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Diabetes Type Classifier", layout="wide")

# Title and description
st.title("**Diabetes Type Classifier**")
st.markdown("""
Enter values for the features below to predict the type of diabetes.
The model uses a pre-trained Random Forest classifier based on the diabetes dataset.
""")

# Load dataset to fit scaler
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_dataset00.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file 'diabetes_dataset00.csv' not found.")
    st.stop()

# Load model
try:
    model = pickle.load(open("random_forest.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file 'random_forest.pkl' not found. Please ensure the model is trained and saved.")
    st.stop()

# Define selected features (added Insulin Levels)
selected_features = [
    'Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels', 
    'Waist Circumference', 'Blood Glucose Levels', 'Weight Gain During Pregnancy', 
    'Pancreatic Health', 'Pulmonary Function', 'Neurological Assessments', 
    'Digestive Enzyme Levels'
]

# Fit scaler on selected features
try:
    X = df[selected_features]
    scaler = StandardScaler()
    scaler.fit(X)
except KeyError as e:
    st.error(f"Feature missing in dataset: {e}. Please verify the dataset columns.")
    st.stop()

# Input fields with realistic ranges (based on dataset statistics)
st.subheader("Input Features")
insulin_levels = st.number_input('Insulin Levels (μU/mL)', min_value=5, max_value=50, value=25, step=1)
age = st.number_input('Age (years)', min_value=0, max_value=79, value=40, step=1)
bmi = st.number_input('BMI (kg/m²)', min_value=12, max_value=39, value=25, step=1)
blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=60, max_value=149, value=110, step=1)
cholesterol_levels = st.number_input('Cholesterol Levels (mg/dL)', min_value=100, max_value=299, value=200, step=1)
waist_circumference = st.number_input('Waist Circumference (cm)', min_value=20, max_value=54, value=35, step=1)
blood_glucose = st.number_input('Blood Glucose Levels (mg/dL)', min_value=80, max_value=299, value=150, step=1)
weight_gain_pregnancy = st.number_input('Weight Gain During Pregnancy (kg)', min_value=0, max_value=39, value=15, step=1)
pancreatic_health = st.number_input('Pancreatic Health (arbitrary units)', min_value=0, max_value=100, value=50, step=1)
pulmonary_function = st.number_input('Pulmonary Function (%)', min_value=0, max_value=100, value=80, step=1)
neurological_assessments = st.number_input('Neurological Assessments (score)', min_value=0, max_value=5, value=2, step=1)
digestive_enzyme_levels = st.number_input('Digestive Enzyme Levels (units)', min_value=0, max_value=100, value=50, step=1)

# Predict button
if st.button("Predict"):
    # Prepare input features
    features = np.array([[
        insulin_levels, age, bmi, blood_pressure, cholesterol_levels,
        waist_circumference, blood_glucose, weight_gain_pregnancy,
        pancreatic_health, pulmonary_function, neurological_assessments,
        digestive_enzyme_levels
    ]])
    
    # Scale features
    try:
        features_scaled = scaler.transform(features)
    except ValueError as e:
        st.error(f"Error scaling features: {e}. Ensure the model is trained on the correct features.")
        st.stop()
    
    # Make prediction
    try:
        prediction = model.predict(features_scaled)
    except ValueError as e:
        st.error(f"Prediction error: {e}. The model may expect a different number of features.")
        st.stop()
    
    # Define class labels (from dataset)
    classes = [
    'Cystic Fibrosis-Related Diabetes (CFRD)',  # 0
    'Gestational Diabetes',                    # 1
    'LADA',                                   # 2
    'MODY',                                   # 3
    'Neonatal Diabetes Mellitus (NDM)',       # 4
    'Prediabetic',                            # 5
    'Secondary Diabetes',                     # 6
    'Steroid-Induced Diabetes',               # 7
    'Type 1 Diabetes',                        # 8
    'Type 2 Diabetes',                        # 9
    'Type 3c Diabetes (Pancreatogenic Diabetes)',  # 10
    'Wolcott-Rallison Syndrome',              # 11
    'Wolfram Syndrome'                        # 12
    ]
    
    # Display result
    st.success(f"Predicted Diabetes Type: **{classes[prediction[0]]}**")

# Footer
st.markdown("---\nCreated with Streamlit by xAI. Dataset: `diabetes_dataset00.csv`")