import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Import the load_data
from Home import load_data 

st.set_page_config(page_title="2. Exploratory Analysis", layout="wide")

st.title("2. Exploratory Analysis 📈")
st.markdown("---")

df = load_data()

if not df.empty:
    sns.set_style("whitegrid")
    
    #Goals and Checklist Structure 
    st.header("Data Visualization 🎨")
    st.markdown("Goal: Gain insights into data patterns and relationships through visual exploration. 🧐")
    
    st.markdown(
        """
        On this page, you can:
        * Examine temporal trends and seasonality 📅
        * Compare key pollutant correlations 🔗
        * Explore the distribution of the target variable (AQI) 🌈
        """
    )
    
    st.subheader("Interactive Charts")
    
    #Use checkboxes to control chart visibility
    show_monthly_trend = st.checkbox("Show Monthly Average AQI (Seasonal Variation)", value=True)
    show_yearly_trend = st.checkbox("Show Yearly Average AQI Trend (2015-2020)")
    show_aqi_distribution = st.checkbox("Show AQI Distribution Histogram")
    show_correlation = st.checkbox("Show Pollutant Correlation Heatmap")
    
    st.markdown("---")

    #Monthly Average AQI (Seasonal)
    if show_monthly_trend:
        st.subheader("Monthly Average AQI (Seasonal)")
        monthly_avg = df.groupby('Month')['AQI'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Month', y='AQI', data=monthly_avg, palette='viridis', hue='Month', legend=False, ax=ax)
        ax.set_title('AQI Seasonal Variation')
        st.pyplot(fig)
        st.markdown("*Interpretation: Pollution peaks sharply during winter months (Oct-Jan).*")
        st.markdown("---")


    #Yearly Average AQI Trend
    if show_yearly_trend:
        st.subheader("Yearly Average AQI Trend")
        yearly_avg = df.groupby('Year')['AQI'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x='Year', y='AQI', data=yearly_avg, marker='o', color='darkblue', ax=ax)
        ax.set_title('AQI Trend Over the Years')
        st.pyplot(fig)
        st.markdown("*Interpretation: Baseline AQI remains consistently high across the period.*")
        st.markdown("---")


    #AQI Distribution Histogram
    if show_aqi_distribution:
        st.subheader("AQI Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['AQI'], bins=50, kde=True, color='skyblue', ax=ax)
        ax.set_title('Distribution of Air Quality Index (AQI)')
        ax.set_xlabel('AQI Value')
        ax.set_ylabel('Frequency (Days)')
        st.pyplot(fig)
        st.markdown("*Interpretation: The distribution is heavily right-skewed, confirming frequent moderate days and severe pollution outliers.*")
        st.markdown("---")


    #Correlation Heatmap
    if show_correlation:
        st.subheader("Pollutant Correlation with AQI")
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
        corr_matrix = df[pollutants].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('Pollutant Correlation Heatmap')
        st.pyplot(fig)
        st.markdown("*Interpretation: PM2.5 and PM10 show the strongest correlation (r > 0.9) with AQI, making them key features.*")
