import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans

def run_automated_ml(csv_path, target_column=None):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() == len(df):
            print(f"Dropping irrelevant ID column: {col}")
            df = df.drop(columns=[col])
            

    if target_column is None:
        task = "clustering"
    elif df[target_column].dtype in ['float64', 'int64'] and df[target_column].nunique() > 20:
        task = "regression"
    else:
        task = "classification"
        
    print(f"Task detected: {task.upper()}")
    
    if task in ['classification', 'regression']:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    if task == 'classification':
        model = RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1)
    elif task == 'regression':
        model = RandomForestRegressor(random_state=42, n_estimators=50, n_jobs=-1)
    else:
        model = MiniBatchKMeans(n_clusters=5, random_state=42) 

    master_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    print("Training model...")
    if task in ['classification', 'regression']:
        master_pipeline.fit(X, y)
        predictions = master_pipeline.predict(X)
    else:
        predictions = master_pipeline.fit_predict(X)
    
    profile = {
        "Rows": len(df),
        "Features Used": len(X.columns),
        "Missing Values": int(df.isnull().sum().sum()),
        "Task": task
    }
    
    print("\n--- DATA PROFILE ---")
    for k, v in profile.items():
        print(f"{k}: {v}")
        
    print(f"\nSuccessfully generated {len(predictions)} predictions.")
    return predictions, profile
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_to_process = sys.argv[1]
    else:
        file_to_process = "telco_churn.csv"
        
    final_predictions, final_profile = run_automated_ml(file_to_process)
    
    pd.DataFrame(final_predictions, columns=["prediction"]).to_csv("predictions.csv", index=False)
    print("Execution complete. Output saved to predictions.csv.")