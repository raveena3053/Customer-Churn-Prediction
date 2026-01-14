import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from data_loader import load_data
from feature_engineering import preprocess_data
import os

def train_models():
    # Load and Preprocess
    df = load_data()
    if df is None:
        return
        
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
    
    print(f"Final Training Data Shape: {X_train.shape}")
    
    # Save feature names and scaler for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(features, 'models/features.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_roc_auc = 0
    
    results = {}
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        }
        
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save Model
        joblib.dump(model, f'models/{name.replace(" ", "_").lower()}.pkl')
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = name

    print(f"\nBest Performing Model: {best_model} with ROC-AUC: {best_roc_auc:.4f}")
    
    # Save Best Model metadata
    with open('models/best_model.txt', 'w') as f:
        f.write(best_model)
        
    return results

if __name__ == "__main__":
    train_models()
