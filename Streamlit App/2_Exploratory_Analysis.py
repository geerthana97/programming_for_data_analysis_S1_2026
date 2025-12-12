import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from main_app import load_data

st.set_page_config(layout="wide")

st.title("📈 Exploratory Data Analysis (EDA)")
st.markdown("---")

df = load_data()

if not df.empty:
    sns.set_style("whitegrid")
    
    #Temporal Trend Analysis
    st.header("1. Seasonal and Temporal Trends")
    monthly_avg = df.groupby('Month')['AQI'].mean().reset_index()
    yearly_avg = df.groupby('Year')['AQI'].mean().reset_index()
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Monthly Average AQI")
        fig, ax = plt.subplots(figsize=(7, 4))
  
        sns.barplot(x='Month', y='AQI', data=monthly_avg, palette='viridis', hue='Month', legend=False, ax=ax)
        ax.set_title('AQI Seasonal Variation')
        st.pyplot(fig)
        st.markdown("*Interpretation: Pollution peaks sharply during winter months (Oct-Jan).*")

    with colB:
        st.subheader("Yearly Average AQI Trend")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(x='Year', y='AQI', data=yearly_avg, marker='o', color='darkblue', ax=ax)
        ax.set_title('AQI Trend Over the Years')
        st.pyplot(fig)
        st.markdown("*Interpretation: Baseline AQI remains consistently high across the period.*")


    #Geospatial and Correlation Analysis
    st.header("2. Correlation and Geospatial Insights")
    colC, colD = st.columns(2)

    with colC:
        st.subheader("Top 10 Cities by Median AQI")
        city_ranking = df.groupby('City')['AQI'].median().sort_values(ascending=False).head(10).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
      
        sns.barplot(x='AQI', y='City', data=city_ranking, palette='Reds_r', hue='City', legend=False, ax=ax)
        ax.set_title('Top 10 Most Polluted Cities')
        ax.set_xlabel('Median AQI')
        st.pyplot(fig)
        st.markdown("*Interpretation: Pollution is geographically concentrated in specific urban centers.*")
        
    with colD:
        st.subheader("Pollutant Correlation with AQI")
        #Select key numerical columns for correlation
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']
        corr_matrix = df[pollutants].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('Pollutant Correlation Heatmap')
        st.pyplot(fig)
        st.markdown("*Interpretation: PM2.5 and PM10 show the strongest correlation (r > 0.9) with AQI.*")
