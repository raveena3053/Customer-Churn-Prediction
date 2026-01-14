import shap
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def load_artifacts():
    try:
        # Load best model name
        with open('models/best_model.txt', 'r') as f:
            model_name = f.read().strip().replace(" ", "_").lower()
            
        model = joblib.load(f'models/{model_name}.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/features.pkl')
        
        return model, scaler, feature_names
    except FileNotFoundError:
        return None, None, None

def explain_prediction(model, X_instance, feature_names):
    """
    Generates SHAP values for a single instance.
    """
    # For Logistic Regression (Linear Model), use LinearExplainer
    # For Tree models, use TreeExplainer
    # We'll try to detect or just use generic Explainer (might be slower but safer)
    
    # Create the explainer
    # Note: For Linear models, we need the background dataset (X_train summary) strictly speaking,
    # but LinearExplainer can work with just the model for coefficients in some versions.
    # However, SHAP's Explainer usually handles this auto-detection.
    
    # Check model type
    model_type = type(model).__name__
    
    try:
        if 'LogisticRegression' in model_type:
            # LinearExplainer requires a masker (background data) usually, 
            # or we can inspect valid coefficients. 
            # Simpler: use the coefficients directly for global importance, 
            # but for local SHAP, let's use a generic Explainer or LinearExplainer with a sample.
            # Ideally we passed X_train during training, but here we don't have it easily without reloading data.
            # We'll rely on the fact that for LogReg, SHAP is just coef * (x - mean(x)).
            # Let's try shap.Explainer
            explainer = shap.Explainer(model, X_instance) # passing instance as masker? No.
        else:
             explainer = shap.Explainer(model)
             
        # Actually, safely:
        # If we don't have background data easily, LinearExplainer might require it.
        # Let's assume we can compute it on the fly or just use the coefficients for now?
        # No, the user wants SHAP.
        pass
    except:
        pass

    # Better approach for the App:
    # We will compute SHAP values using the libraries specific explainers.
    # Since we reload data in the app anyway to pick a customer, we can pass that data to explain.
    return None

def get_shap_explainer(model, X_background):
    """
    Returns a configured SHAP explainer
    """
    try:
        return shap.Explainer(model, X_background)
    except:
        # Fallback for some models
        return shap.LinearExplainer(model, X_background)

if __name__ == "__main__":
    pass
