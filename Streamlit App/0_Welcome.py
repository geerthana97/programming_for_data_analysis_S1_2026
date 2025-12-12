import streamlit as st
from streamlit_app import air_quality_df, best_model  

st.title("Data-Driven Air Quality Analysis & Forecasting")
st.subheader("Final Project Assessment: Programming of Data Analysis")

st.markdown("""
Welcome! This multi-page application successfully implements the full data science pipeline, 
from secure data ingestion to predictive modeling, using the India Air Quality dataset (2015-2020).
""")

st.header("About This App ℹ️")

st.markdown("""
This application provides an end-to-end analysis pipeline for India's air quality data:

**Data Loading:** 📁 Explore the raw dataset and verify structure.
**Data Preprocessing:** 🧹 Clean and enhance the dataset (Task 1 & 2).
**Data Visualization:** 🧠 Gain insights through interactive charts (Task 2).
**Data Modeling and Evaluation:** 🔮 Build, tune, and evaluate predictive models (Task 3).
""")

st.markdown("Use the navigation above to explore each stage of the analysis. Enjoy discovering insights! ✨")

st.sidebar.header("Application Status")
if not air_quality_df.empty:
    st.sidebar.success(f"Data Loaded: {air_quality_df.shape[0]:,} records")
else:
    st.sidebar.error("Data Not Loaded.")
    
if best_model is not None:
    st.sidebar.success("Model Loaded: Ready for Prediction")
else:
    st.sidebar.error("Model Not Loaded.")
