import pandas as pd
import joblib
import shap
import os
import sys

sys.path.append(os.path.abspath("src"))
from data_loader import load_data
from feature_engineering import preprocess_data

def test_logic():
    print("Loading resources...")
    try:
        with open('models/best_model.txt', 'r') as f:
            model_name = f.read().strip().replace(" ", "_").lower()
        model = joblib.load(f'models/{model_name}.pkl')
        print(f"Model {model_name} loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading data...")
    df = load_data()
    if df is None:
        print("Data load failed.")
        return
        
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df)
    
    customer_data = X_test.iloc[[0]]
    print("Predicting...")
    prob = model.predict_proba(customer_data)[0][1]
    print(f"Probability: {prob}")
    
    print("Generating SHAP explanation...")
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(customer_data)
        print("SHAP values generated successfully.")
    except Exception as e:
        print(f"SHAP failed: {e}")

if __name__ == "__main__":
    test_logic()

