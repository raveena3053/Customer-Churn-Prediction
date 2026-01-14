import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

try:
    from src.data_loader import load_data
    from src.feature_engineering import preprocess_data
except ImportError:
   
    import sys
    import os
    sys.path.append(os.path.abspath("src"))
    from data_loader import load_data
    from feature_engineering import preprocess_data

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Telco Customer Churn Prediction System")
st.markdown("""Please select a customer from the dropdown to know the prediction""")

@st.cache_resource
def load_resources():
    try:
        with open('models/best_model.txt', 'r') as f:
            model_name_raw = f.read().strip()
            model_file_name = model_name_raw.replace(" ", "_").lower()
            
        model = joblib.load(f'models/{model_file_name}.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/features.pkl')
        
        return model, scaler, feature_names, model_name_raw
    except FileNotFoundError:
        st.error("Models not found. Please run model training first.")
        return None, None, None, None

model, scaler, feature_names, model_name = load_resources()

@st.cache_data
def get_data():
    df_processed = load_data() 
    try:
        df_full = pd.read_excel('data/telco_churn.csv', engine='openpyxl')
    except:
        df_full = df_processed 
        
    if df_processed is not None:
        X_train, X_test, y_train, y_test, _, _ = preprocess_data(df_processed)
        return df_full, X_test, y_test
    return None, None, None

raw_df, X_test, y_test = get_data()

if model and raw_df is not None:
    st.sidebar.header("Customer Selection")
   
    customer_ids = X_test.index[:100].tolist() 
    selected_id = st.sidebar.selectbox("Select Customer Index", customer_ids)

    customer_data = X_test.loc[[selected_id]] 
  
    customer_raw = raw_df.loc[selected_id]
    
    actual_churn = y_test.loc[selected_id]
    
    st.sidebar.write(f"**Actual Status:** {'Churned' if actual_churn == 1 else 'Retained'}")
   
    prob = model.predict_proba(customer_data)[0][1]
    prediction = 1 if prob > 0.5 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Churn Probability", f"{prob:.2%}", delta_color="inverse")
    with col2:
        st.metric("Predicted Label", "Churn" if prediction == 1 else "Retain")
 
    st.markdown("---")
    st.subheader("👤 Customer Details")
  
    
    priority = [
        "CustomerID", "Gender", "Contract", "Tenure Months", 
        "Payment Method", "Streaming Movies", "Streaming TV"
    ]
    
    bottom = ["Latitude", "Longitude"]

    all_cols = list(customer_raw.index)

    priority_exists = [c for c in priority if c in all_cols]
    bottom_exists = [c for c in bottom if c in all_cols]

    rest = [c for c in all_cols if c not in priority_exists and c not in bottom_exists]

    final_order = priority_exists + rest + bottom_exists
    
    details_col1, details_col2 = st.columns(2)
    
    mid_point = len(final_order) // 2
    
    with details_col1:
        for attr in final_order[:mid_point]:
            st.markdown(f"**{attr}:** {customer_raw[attr]}")
            
    with details_col2:
        for attr in final_order[mid_point:]:
            st.markdown(f"**{attr}:** {customer_raw[attr]}")

    st.markdown("---")
    st.subheader("Prediction Percentage:")
 
    fig, ax = plt.subplots(figsize=(5, 1.5))

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    categories = ['Churn', 'Retain']
    values = [prob, 1 - prob]
    colors = ['#FF4B4B', '#1C83E1']

    bars = ax.barh(categories, values, color=colors)
  
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='y', colors='white', labelsize=12) 
    
    ax.set_yticks([]) 
    
    for bar, label, val in zip(bars, categories, values):
        width = bar.get_width()
       
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.1%} {label}", 
                ha='left', va='center', color='white', fontweight='bold', fontsize=7)
    
    st.pyplot(fig)

else:
    st.warning("Please ensure data is available and models are trained.")

