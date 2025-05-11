import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
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
    df = pd.read_csv(r"C:\Users\saher\Desktop\workshop\Disease_analysis_prediction\Data\diabetes_dataset00.csv")
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

# Define selected features
selected_features = [
    'Insulin Levels', 'Age', 'BMI', 'Blood Pressure', 'Cholesterol Levels', 
    'Waist Circumference', 'Blood Glucose Levels', 'Weight Gain During Pregnancy', 
    'Pancreatic Health', 'Pulmonary Function', 'Neurological Assessments', 
    'Digestive Enzyme Levels'
]

# Fit scaler
try:
    X = df[selected_features]
    scaler = StandardScaler()
    scaler.fit(X)
except KeyError as e:
    st.error(f"Feature missing in dataset: {e}. Please verify the dataset columns.")
    st.stop()

# Input form
st.subheader("Input Features")
insulin_levels = st.number_input('Insulin Levels (μU/mL)', min_value=5, max_value=50, value=25)
age = st.number_input('Age (years)', min_value=0, max_value=79, value=40)
bmi = st.number_input('BMI (kg/m²)', min_value=12, max_value=39, value=25)
blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=60, max_value=149, value=110)
cholesterol_levels = st.number_input('Cholesterol Levels (mg/dL)', min_value=100, max_value=299, value=200)
waist_circumference = st.number_input('Waist Circumference (cm)', min_value=20, max_value=54, value=35)
blood_glucose = st.number_input('Blood Glucose Levels (mg/dL)', min_value=80, max_value=299, value=150)
weight_gain_pregnancy = st.number_input('Weight Gain During Pregnancy (kg)', min_value=0, max_value=39, value=15)
pancreatic_health = st.number_input('Pancreatic Health (arbitrary units)', min_value=0, max_value=100, value=50)
pulmonary_function = st.number_input('Pulmonary Function (%)', min_value=0, max_value=100, value=80)
neurological_assessments = st.number_input('Neurological Assessments (score)', min_value=0, max_value=5, value=2)
digestive_enzyme_levels = st.number_input('Digestive Enzyme Levels (units)', min_value=0, max_value=100, value=50)

# Predict
if st.button("Predict"):
    features = np.array([[insulin_levels, age, bmi, blood_pressure, cholesterol_levels,
                            waist_circumference, blood_glucose, weight_gain_pregnancy,
                            pancreatic_health, pulmonary_function, neurological_assessments,
                            digestive_enzyme_levels]])
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    classes = [
        'Cystic Fibrosis-Related Diabetes (CFRD)', 'Gestational Diabetes', 'LADA', 'MODY',
        'Neonatal Diabetes Mellitus (NDM)', 'Prediabetic', 'Secondary Diabetes',
        'Steroid-Induced Diabetes', 'Type 1 Diabetes', 'Type 2 Diabetes',
        'Type 3c Diabetes (Pancreatogenic Diabetes)', 'Wolcott-Rallison Syndrome', 'Wolfram Syndrome'
    ]
    st.success(f"Predicted Diabetes Type: **{classes[prediction[0]]}**")

# Dashboard
st.markdown("---")
st.header("📊 Diabetes Classification Dashboard")

# Ensure target column exists
if 'Diabetes Type' in df.columns:

    # Class distribution
    st.subheader("📌 Diabetes Type Distribution")
    class_counts = df['Diabetes Type'].value_counts().reset_index()
    class_counts.columns = ['Diabetes Type', 'Count']
    fig_class_dist = px.bar(class_counts, x='Diabetes Type', y='Count', color='Diabetes Type',
                            title="Diabetes Type Frequency", labels={'Count': 'Number of Cases'},
                            height=500)
    st.plotly_chart(fig_class_dist, use_container_width=True)

    # Feature importance
    st.subheader("📈 Feature Importance from Random Forest")
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values(by="Importance", ascending=True)
        fig_importance = go.Figure(go.Barh(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h'
        ))
        fig_importance.update_layout(title="Feature Importance", height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
    except:
        st.warning("Feature importance not available for this model.")

    # Parallel categories
    st.subheader("🔀 Feature Distribution Across Diabetes Types")
    df_sample = df.sample(n=500) if len(df) > 500 else df
    fig_parallel = px.parallel_categories(df_sample,
                                            dimensions=['Diabetes Type', 'Age', 'BMI', 'Blood Pressure'],
                                            color_continuous_scale=px.colors.sequential.Inferno)
    st.plotly_chart(fig_parallel, use_container_width=True)

    # Correlation heatmap
    st.subheader("🧪 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[selected_features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Column 'Diabetes Type' not found in dataset.")

# Footer
st.markdown("---\nCreated with Streamlit by xAI. Dataset: `diabetes_dataset00.csv`")
