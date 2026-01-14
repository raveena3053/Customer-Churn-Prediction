import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Import Modules
# Try importing from 'src' package (Standard for structured apps)
try:
    from src.data_loader import load_data
    from src.feature_engineering import preprocess_data
except ImportError:
    # Fallback: Add src to path if package import fails (e.g. running from inside src or weird paths)
    import sys
    import os
    sys.path.append(os.path.abspath("src"))
    from data_loader import load_data
    from feature_engineering import preprocess_data

# Page Config
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊", layout="wide")

# Title
st.title("📊 Telco Customer Churn Prediction System")
st.markdown("""Please select a customer from the dropdown to know the prediction""")

# Load Models & Data
@st.cache_resource
def load_resources():
    try:
        # Load best model name
        with open('models/best_model.txt', 'r') as f:
            model_name_raw = f.read().strip()
            model_file_name = model_name_raw.replace(" ", "_").lower()
            
        model = joblib.load(f'models/{model_file_name}.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/features.pkl')
        
        return model, scaler, feature_names, model_name_raw
    except FileNotFoundError:
        st.error("Models not found. Please run model training first.")
        st.warning("Debug Info:")
        import os
        st.write(f"Current Directory: `{os.getcwd()}`")
        try:
            st.write(f"Files in root: `{os.listdir('.')}`")
            if 'models' in os.listdir('.'):
                st.write(f"Files in models: `{os.listdir('models')}`")
            else:
                st.error("The 'models' folder is MISSING from the uploaded files.")
        except Exception as e:
            st.write(f"Error listing files: {e}")
        return None, None, None, None

model, scaler, feature_names, model_name = load_resources()

# Load Data for Simulation
@st.cache_data
def get_data():
    # Load processed data for the model
    df_processed = load_data() 
    
    # Load RAW data for display (preserving CustomerID and other dropped columns)
    # We use the same file logic
    try:
        # We know it ended up being an xlsx in the end check
        df_full = pd.read_excel('data/telco_churn.csv', engine='openpyxl')
    except:
        df_full = df_processed # Fallback
        
    if df_processed is not None:
        X_train, X_test, y_train, y_test, _, _ = preprocess_data(df_processed)
        return df_full, X_test, y_test
    return None, None, None

raw_df, X_test, y_test = get_data()

if model and raw_df is not None:
    st.sidebar.header("Customer Selection")
    
    # Select a customer from Test Set
    customer_ids = X_test.index[:100].tolist() 
    selected_id = st.sidebar.selectbox("Select Customer Index", customer_ids)
    
    # Get Customer Data for Model
    customer_data = X_test.loc[[selected_id]] 
    
    # Get Customer Data for Display (from raw full df)
    # X_test index corresponds to original df index
    customer_raw = raw_df.loc[selected_id]
    
    actual_churn = y_test.loc[selected_id]
    
    st.sidebar.write(f"**Actual Status:** {'Churned' if actual_churn == 1 else 'Retained'}")
    
    # Prediction
    prob = model.predict_proba(customer_data)[0][1]
    prediction = 1 if prob > 0.5 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Churn Probability", f"{prob:.2%}", delta_color="inverse")
    with col2:
        st.metric("Predicted Label", "Churn" if prediction == 1 else "Retain")
    
    # Customer Details (Vertical Display)
    st.markdown("---")
    st.subheader("👤 Customer Details")
    
    # Ordering Logic
    # 1. CustomerID
    # 2. Gender
    # 3. Contract
    # 4. Tenure Months
    # 5. Payment Method
    # 6. Streaming Movies
    # 7. Streaming TV
    # ... Rest ...
    # Last: Latitude, Longitude
    
    priority = [
        "CustomerID", "Gender", "Contract", "Tenure Months", 
        "Payment Method", "Streaming Movies", "Streaming TV"
    ]
    
    bottom = ["Latitude", "Longitude"]
    
    # Get available columns
    all_cols = list(customer_raw.index)
    
    # Filter what exists
    priority_exists = [c for c in priority if c in all_cols]
    bottom_exists = [c for c in bottom if c in all_cols]
    
    # The rest
    rest = [c for c in all_cols if c not in priority_exists and c not in bottom_exists]
    
    # Final Order
    final_order = priority_exists + rest + bottom_exists
    
    details_col1, details_col2 = st.columns(2)
    
    mid_point = len(final_order) // 2
    
    with details_col1:
        for attr in final_order[:mid_point]:
            st.markdown(f"**{attr}:** {customer_raw[attr]}")
            
    with details_col2:
        for attr in final_order[mid_point:]:
            st.markdown(f"**{attr}:** {customer_raw[attr]}")

    # Prediction Visualization
    st.markdown("---")
    st.subheader("Prediction Percentage:")
    
    # Horizontal Bar Chart for Churn vs Retain
    fig, ax = plt.subplots(figsize=(5, 1.5))
    
    # Transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    # Data
    categories = ['Churn', 'Retain']
    values = [prob, 1 - prob]
    colors = ['#FF4B4B', '#1C83E1']
    
    # Plot Separate Bars
    bars = ax.barh(categories, values, color=colors)
    
    # Remove axes and spines
    ax.set_xticks([])
    # ax.set_yticks([]) # Keep y-ticks to see labels? Or put labels on bars?
    # User said "two separate bars", usually implies labels are needed.
    # Let's clean it up:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Move y-labels inside or just use standard
    ax.tick_params(axis='y', colors='white', labelsize=12) # White labels for transparency
    # But Streamlit usually has white bg? 
    # Wait, the user asked for transparent background earlier.
    # If the user is in light mode, white text is invisible. 
    # We should default to black if transparency is ON but background is unknown?
    # Actually, previous charts had "color='white'" explicitly for stacked bar text. 
    # I'll stick to black text for axes labels for safety, or white if bold/inside.
    # Let's put text *next* to the bars like the butterfly chart.
    
    ax.set_yticks([]) # Hide default ticks
    
    for bar, label, val in zip(bars, categories, values):
        width = bar.get_width()
        
        # Value and Label at the end (e.g. "45.0% Churn")
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{val:.1%} {label}", 
                ha='left', va='center', color='white', fontweight='bold', fontsize=7)
    
    st.pyplot(fig)

else:
    st.warning("Please ensure data is available and models are trained.")
