import streamlit as st
import pandas as pd
import os

#Configuration
st.set_page_config(
    page_title="AQI Forecasting Application",
    layout="wide"
)

#Use st.cache_data to load the heavy dataset only once
@st.cache_data
def load_data():
    try:
        #Load the cleaned dataset saved in Task 2
        df = pd.read_csv('air_quality_cleaned_merged.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Error: 'air_quality_cleaned_merged.csv' not found. Please ensure it is in the same directory.")
        return pd.DataFrame()

#Main Page Content
st.title("Air Quality Index (AQI) Forecasting Application")
st.markdown("---")

st.header("Project Overview")
st.markdown(
    """
    This application, developed as part of **Task 4**, demonstrates the full pipeline of data analysis and machine learning
    for predicting the daily Air Quality Index (AQI) across various Indian cities.

    Use the navigation menu on the left (the three tabs: Data Overview, EDA, Modeling) to explore the project.

    ### Goal
    To provide a data-driven system for understanding and forecasting air pollution levels using historical pollutant data and temporal features.

    ### Dataset
    The application utilizes the cleaned, merged dataset containing daily records of pollutants (PM2.5, NOX, SO2, etc.) and the calculated AQI from 2015 to 2020.
    """
)

#Load data once to make it available to all pages
df = load_data()
if not df.empty:
    st.success(f"Data Loaded: {len(df):,} records available for analysis.")
