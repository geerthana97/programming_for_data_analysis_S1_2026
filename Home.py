import streamlit as st
import pandas as pd
import numpy as np
import os

#Configuration
st.set_page_config(
    page_title="1. Data Loading & Overview",
    layout="wide"
)

st.sidebar.header("About This App ℹ️")
st.sidebar.markdown(
    """
    This application provides an **end-to-end analysis pipeline** for
    forecasting air quality in India (AQI). Use the navigation above
    to explore each stage of the analysis.
    """
)
st.sidebar.markdown("---") 

st.sidebar.markdown("### Project Workflow")
st.sidebar.markdown(
    """
1. **Data Loading & Overview 📁:** Explore the structure and initial quality of the cleaned dataset.

2. **Exploratory Analysis 📊:** Gain crucial insights into seasonal and pollutant trends via interactive charts (Task 2).

3. **Modeling & Prediction 🧠:** View the performance and features of the optimized Random Forest Regressor (Task 3).
    """
)
st.sidebar.markdown("---") 

#Data Loading 
@st.cache_data
def load_data():
    """Loads and caches the cleaned dataset."""
    try:
        df = pd.read_csv('air_quality_cleaned_merged.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Error: 'air_quality_cleaned_merged.csv' not found. Please ensure it is in the same directory.")
        return pd.DataFrame()

#Main Page 
st.title("1. Data Loading & Overview 📊")
st.markdown("""
    Goal: Before we dive into analysis, let's get acquainted with the dataset.
    This initial overview helps identify data structure and quality issues, setting a foundation for further steps.
""")

df = load_data()

if not df.empty:
    st.success(f"Data Loaded: {len(df):,} records available for analysis.")

    st.header("Data Preview 🚀")
    num_rows = st.slider("Select Number of Rows to Preview", min_value=5, max_value=len(df), value=10)
    st.dataframe(df.head(num_rows))

    st.header("Dataset Structure")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Shape and Types")
        st.info(f"Total Rows: {len(df):,}")
        st.info(f"Total Columns: {df.shape[1]}")
    
    with col2:
        st.subheader("Data Types Summary")
        dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index().rename(columns={'index': 'Column'})
        st.dataframe(dtypes_df)

    st.header("Missing Value Summary (Percentage)")
    missing_info = pd.DataFrame({
        'Missing %': (df.isnull().sum() / len(df)) * 100
    }).sort_values(by='Missing %', ascending=False)
    
    st.dataframe(missing_info[missing_info['Missing %'] > 0])
    st.markdown("---")
    st.write("Interpretation: The dataset is clean and pre-processed, showing minimal missing data.")
