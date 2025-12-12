import streamlit as st
import pandas as pd
from streamlit_app import air_quality_df  

#Function to render a DataFrame using simple HTML
def display_html_dataframe(df):
    """Converts a DataFrame to a simple, unstyled HTML table and renders it."""
    html_table = df.to_html(index=True, bold_rows=False) 
    st.markdown(html_table, unsafe_allow_html=True)

st.title("📁 Data Loading & Preprocessing Summary")

st.info("""
**Goal:** Before we dive into analysis, let's get acquainted with the dataset.
This section verifies the data loading and summarizes the cleaning steps (Task 1 & 2).
""")

st.header("Data Preview 📊")

if air_quality_df.empty:
    st.warning("Data not loaded. Please return to the main page and ensure the file paths are correct.")
else:
    st.markdown("""
    On this page, you can:
    * Preview the final cleaned data
    * View descriptive statistics
    * Review the data types and final null checks
    """)
    
    #Simple slider to control the number of rows displayed
    preview_rows = st.slider("Select Number of Rows to Preview", 5, 20, 5)
    
    st.subheader(f"Data Sample (First {preview_rows} Rows)")
    display_html_dataframe(air_quality_df.head(preview_rows).reset_index(drop=True))

    st.header("Cleaning and Structure Summary 🧹")
    
    st.markdown("### Dataset Dimensions")
    st.metric(label="Total Records (Rows)", value=f"{air_quality_df.shape[0]:,}")
    st.metric(label="Total Features (Columns)", value=air_quality_df.shape[1])
    st.markdown(f"The final cleaned dataset covers daily records from **{air_quality_df['Date'].min().year}** to **{air_quality_df['Date'].max().year}**.")
    
    st.subheader("Final Missing Value Check (Post-Imputation)")
    final_missing_check = air_quality_df.isnull().sum().sort_values(ascending=False)

    if final_missing_check[final_missing_check > 0].empty:
         st.success("✅ No missing values remain after the cleaning and imputation process.")
    else:
        display_html_dataframe(final_missing_check[final_missing_check > 0].to_frame(name='Remaining Missing Count'))
    
    st.subheader("Data Types")
    dtypes_df = air_quality_df.dtypes.to_frame(name='Data Type')
    dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)
    display_html_dataframe(dtypes_df)

    st.subheader("Comprehensive Statistical Summary (All Features)")
    display_html_dataframe(air_quality_df.describe().T)
