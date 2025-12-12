import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

@st.cache_data
def load_data():
    """Loads the final cleaned dataset efficiently."""
    try:
        df = pd.read_csv('./data/air_quality_data_clean.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("❌ Error: Cleaned data file not found at './data/air_quality_data_clean.csv'.")
        st.warning("Please ensure you have executed Task 2 and saved the final DataFrame.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Loads the trained Random Forest model efficiently."""
    try:
        with open('./model/best_rfr_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Error: Trained model file not found at './model/best_rfr_model.pkl'.")
        st.warning("Please ensure you have executed Task 3 and saved the 'best_rfr' model.")
        return None

st.set_page_config(
    page_title="India AQI Forecast App",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏭"
)

air_quality_df = load_data()
best_model = load_model()
