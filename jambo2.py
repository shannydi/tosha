import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def load_data(url):
    """
    Load CSV data from the provided URL into a DataFrame.
    
    Args:
    url (str): The URL of the raw CSV file.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    """
    Preprocess the data by creating the target variable and attributes,
    splitting it into training and testing datasets, and standardizing the features.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the dataset.
    
    Returns:
    tuple: A tuple containing X_train, X_test, y_train, and y_test.
    """
    X = df.drop('Population', axis=1).values
    y = df['Population'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    
    return X_scaled_train, X_scaled_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Random Forest regressor model on the provided training data.
    
    Args:
    X_train (ndarray): The standardized features of the training dataset.
    y_train (ndarray): The target variable of the training dataset.
    
    Returns:
    RandomForestRegressor: The trained Random Forest regressor model.
    """
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    return rf_regressor

# URL of the raw CSV file in the GitHub repository
url = "https://raw.githubusercontent.com/shannydi/tosha/main/algae2.csv"

# Load data
df = load_data(url)

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train model
rf_model = train_model(X_train, y_train)

# Set up page configurations
st.set_page_config(page_title="Algae Population Predictor", layout="wide")

# Custom CSS for green theme
st.markdown("""
    <style>
    .main {
        background-color: #e8f5e9;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a Page:", ["Home Page", "Analysis Page", "Prediction Page"])

# Page 1: Home Page
if page == "Home Page":
    st.title("Understanding Algal Bloom")
    st.markdown("""
    Algal blooms are rapid increases in the population of algae in aquatic systems. 
    These events can significantly impact water quality, ecosystem stability, 
    and biodiversity. Factors like light availability, nutrient levels, and water 
    temperature play crucial roles in the proliferation of algae.
    """)


# Page 2: Analysis Page
elif page == "Analysis Page":
    st.title("Data Analysis of Environmental Factors")
    st.markdown("Explore the relationships between different environmental factors that influence algal populations.")
    
    # Dropdown for variable selection
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Select the first variable", df.columns)
    with col2:
        var2 = st.selectbox("Select the second variable", df.columns, index=1)
    
    # Button to generate plot
    if st.button("Generate Plot"):
       fig, ax = plt.subplots()
       sns.kdeplot(x=var1, y=var2, data=df, ax=ax, fill=True, thresh=0, levels=100, cmap="viridis")
       plt.xlabel(var1)
       plt.ylabel(var2)
       plt.title(f"Density Plot of {var1} vs {var2}")
       st.pyplot(fig)

# Page 3: Prediction Page
elif page == "Prediction Page":
    st.title("Predict Algal Population")
    st.markdown("Input environmental variables to predict algal population.")

    # Input fields for features
    inputs = {feature: st.number_input(f"{feature}:", float(df[feature].min()), float(df[feature].max()), step=0.01) for feature in df.columns if feature != 'Population'}

    # Predict button
    if st.button("Predict"):
        # Make prediction
        features = np.array([list(inputs.values())]).reshape(1, -1)
        prediction = rf_model.predict(features)
        st.success(f"The predicted algae population is {prediction[0]:.2f}")