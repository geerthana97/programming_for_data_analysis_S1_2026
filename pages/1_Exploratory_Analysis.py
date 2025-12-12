import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

#Import the load_data function from the Home script
from Home import load_data 

st.set_page_config(page_title="2. Exploratory Analysis", layout="wide")

st.title("2. Exploratory Analysis 📈")
st.markdown("---")

df = load_data()

if not df.empty:
    sns.set_style("whitegrid")
    
    st.header("Temporal Trend Analysis")
    monthly_avg = df.groupby('Month')['AQI'].mean().reset_index()
    yearly_avg = df.groupby('Year')['AQI'].mean().reset_index()
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Monthly Average AQI (Seasonal)")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x='Month', y='AQI', data=monthly_avg, palette='viridis', hue='Month', legend=False, ax=ax)
        ax.set_title('AQI Seasonal Variation')
        st.pyplot(fig)
        st.markdown("*Interpretation: Pollution peaks sharply during winter months (Oct-Jan).*")

    with colB:
        st.subheader("Pollutant Correlation with AQI")
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'AQI']
        corr_matrix = df[pollutants].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('Pollutant Correlation Heatmap')
        st.pyplot(fig)
        st.markdown("*Interpretation: PM2.5 and PM10 show the strongest correlation (r > 0.9) with AQI.*")
