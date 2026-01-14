import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, target_col='Churn Value'):
    """
    Performs feature engineering:
    - Splits into X and y
    - Encodes categorical variables (One-Hot Encoding)
    - Scales numerical variables
    - Splits into Train/Test sets
    """
    
    # Separate Target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify Categorical and Numerical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"Categorical Columns: {cat_cols}")
    print(f"Numerical Columns: {num_cols}")
    
    # One-Hot Encoding for categorical variables
    # drop_first=True to avoid dummy variable trap
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling Numerical Features
    # We fit scaler only on training data to avoid data leakage
    scaler = StandardScaler()
    
    # Filter num_cols to only those present in the encoded set (some might have been dropped or changed?)
    # Actually, num_cols are untouched by get_dummies, so we can use them directly.
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, scaler, X_encoded.columns.tolist()

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
        print(f"Training Features Shape: {X_train.shape}")
        print(f"Test Features Shape: {X_test.shape}")
