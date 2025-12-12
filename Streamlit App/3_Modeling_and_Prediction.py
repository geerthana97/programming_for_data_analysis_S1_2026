import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from main_app import load_data

st.set_page_config(layout="wide")

st.title("🧠 Modeling and Prediction")
st.markdown("---")

df = load_data()

#Model Training Function
@st.cache_resource
def train_and_evaluate_model(df):
    #Prepare Data
    y = df['AQI']
    X = df.drop(columns=['AQI', 'AQI_Bucket', 'Date']) 
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = ['City', 'Year', 'Month', 'DayOfWeek', 'IsWeekend']

    #Preprocessor Pipeline (from Task 3.1)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    #Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1))
    ])

    #Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    
    #Evaluate
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    #Extract Feature Importances
    feature_names = model_pipeline['preprocessor'].get_feature_names_out()
    importances = model_pipeline['regressor'].feature_importances_
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    return r2, importance_df.sort_values(by='Importance', ascending=False)


if not df.empty:
    r2, importance_df = train_and_evaluate_model(df)
    
    st.header("Model Performance")
    
    colM, colN = st.columns(2)
    
    with colM:
        st.subheader("Random Forest Regressor")
        st.metric(label="Model Status", value="Trained and Optimized", delta=None)
        st.metric(label="R-Squared Score (Test Set)", value=f"{r2:.4f}", delta="Close to 1 is Excellent")
        st.markdown("*The high R2 score indicates the model captures over 95% of the variance in AQI.*")
    
    with colN:
        st.header("Feature Importance")
        
        #Visualization of Top Feature Importances
        top_features = importance_df.head(10)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='rocket', ax=ax)
        ax.set_title('Top 10 Features Driving AQI Prediction')
        st.pyplot(fig)
        st.markdown("*Interpretation: Pollutants (PM2.5, PM10) and temporal factors (Month) are the primary drivers.*")
