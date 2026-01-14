import pandas as pd
import numpy as np

def load_data(filepath='data/telco_churn.csv'):
    """
    Loads the churn dataset. 
    Handles the edge case where an .xlsx file is named .csv
    """
    try:
        # Try reading as Excel first because we know it's likely an Excel file despite extension
        # or just let pandas figure it out? No, pandas trusts extension by default.
        if filepath.endswith('.csv'):
            try:
                # Try as CSV first, but if it fails (as we saw), try as Excel
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                print("CSV read failed, trying as Excel...")
                df = pd.read_excel(filepath, engine='openpyxl')
            except Exception as e:
                # Could be other csv error, but let's try excel just in case
                 print(f"CSV read error: {e}, trying as Excel...")
                 df = pd.read_excel(filepath, engine='openpyxl')
        else:
            df = pd.read_excel(filepath)
            
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Drop non-feature columns
    # 'Churn Reason' is target-leakage (only known after churn)
    # 'Churn Score' could be a target or a pre-calculated score, filtering it out to build our own.
    # 'CLTV' might be useful but 'Churn Score' is definitely competing with our goal.
    # 'Count', 'Country', 'State' are single-value or redundant.
    # 'City' has high cardinality (1129 unique values) -> High dimensionality with OHE.
    # 'Zip Code' is redundant if we have 'City' or 'Lat Long', but since we drop City, we might keep Zip? 
    # Actually, Zip Code as numeric is bad. As categorical it's high dimensionality.
    # Best to drop both and rely on Lat/Long or just general service features.
    
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'Lat Long', 
        'Churn Reason', 'Churn Score', 'City', 'Zip Code'
    ]
    
    # We will use 'Churn Value' as the target (0/1). 'Churn Label' is the string version.
    if 'Churn Value' in df.columns:
        cols_to_drop.append('Churn Label')
    
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Ensure Total Charges is numeric
    # Sometimes Total Charges has empty strings for new customers (tenure 0)
    # We coerce errors to NaN and then fill with 0 or drop.
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df['Total Charges'].fillna(0, inplace=True)

    print(f"Data cleaned. New shape: {df.shape}")
    return df

if __name__ == "__main__":
    load_data()
