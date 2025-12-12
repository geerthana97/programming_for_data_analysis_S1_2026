import streamlit as st
import pandas as pd
import numpy as np

#Import the load_data function from the main script
from main_app import load_data

st.set_page_config(layout="wide")

st.title("📊 Data Overview")
st.markdown("---")

df = load_data()

if not df.empty:
    st.header("Dataset Structure")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Shape and Memory")
        st.info(f"Total Rows: {len(df):,}")
        st.info(f"Total Columns: {df.shape[1]}")
    
    with col2:
        st.subheader("Data Head (First 5 Rows)")
        st.dataframe(df.head())

    st.header("Data Types and Missing Values")
    
    #Data Types Table
    st.subheader("Column Data Types")
    dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index().rename(columns={'index': 'Column'})
    st.dataframe(dtypes_df)

    #Missing Value Table
    st.subheader("Missing Value Summary (Percentage)")
    missing_info = pd.DataFrame({
        'Missing %': (df.isnull().sum() / len(df)) * 100
    }).sort_values(by='Missing %', ascending=False)
    
    #Only show columns with any missing values
    st.dataframe(missing_info[missing_info['Missing %'] > 0])
    
    st.markdown("---")
    st.write("The dataset is already cleaned and imputed (as per Task 2), so minimal missing data remains.")
