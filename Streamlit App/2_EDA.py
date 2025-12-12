import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_app import air_quality_df 

st.title("2️⃣ Exploratory Data Analysis (EDA)")

if air_quality_df.empty:
    st.warning("Data not loaded. Please check the main app page for errors.")
else:
    #Temporal Trend (AQI Line Plot)
    st.header("📈 Temporal Trend: AQI Over Time")
    
    monthly_avg_aqi = air_quality_df.groupby('Date')['AQI'].mean().resample('ME').mean().reset_index()
    
    fig_line = px.line(
        monthly_avg_aqi, 
        x='Date', 
        y='AQI', 
        title='Monthly Average AQI Trend (2015-2020)',
        labels={'AQI': 'Average AQI', 'Date': 'Time'}
    )

    st.plotly_chart(fig_line, width='stretch')

    #City Comparison (Box Plot)
    st.header("📍 City-Level Variability and Outliers")
    top_cities = air_quality_df['City'].value_counts().nlargest(10).index
    
    fig_box = px.box(
        air_quality_df[air_quality_df['City'].isin(top_cities)], 
        x='City', 
        y='AQI', 
        title='AQI Distribution and Outliers Across Major Cities',
        color='City'
    )
    st.plotly_chart(fig_box, width='stretch')

    #Correlation Heatmap 
    st.header("🔗 Pollutant Relationships (Correlation)")
    corr_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'SO2', 'O3', 'Benzene', 'AQI']
    correlation_matrix = air_quality_df[corr_cols].corr().round(2)
    
    fig_heatmap = px.imshow(
        correlation_matrix, 
        text_auto=True, 
        aspect="auto",
        title='Correlation Matrix: Pollutants vs. AQI',
        color_continuous_scale='RdYlBu_r'
    )
    st.plotly_chart(fig_heatmap, width='stretch')