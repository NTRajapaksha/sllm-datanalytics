import pandas as pd
import numpy as np
import time
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from config import RAW_DATA_PATH, RESULTS_DIR, MODELS

# We use Payment Method as classification target for the test as requested.
# The assumption is this exists or the LLM didn't drop it. 
# It might get renamed, so we will try to find it dynamically or fallback to default.
TARGET_COL_HINTS = ["Payment Method", "payment_method", "payment", "method"]

def evaluate_dataset(df_path, name):
    if not os.path.exists(df_path):
        print(f"Dataset {name} not found at {df_path}")
        return
        
    df = pd.read_csv(df_path)
    
    # Find target column
    target_col = None
    for col in df.columns:
        if any(hint.lower() in col.lower() for hint in TARGET_COL_HINTS):
            target_col = col
            break
            
    if target_col is None:
        print(f"Could not find target column for {name}. Available columns: {df.columns.tolist()}")
        return
        
    print(f"\nEvaluating dataset: {name} (Target: {target_col})")
    
    df.dropna(subset=[target_col], inplace=True)
    
    if len(df) == 0:
        print("Dataset empty after dropping target NaNs.")
        return
    
    # Simple feature encoding
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    
    # Determine classification type
    num_classes = len(np.unique(y))
    if num_classes < 2:
        print("Target has less than 2 classes. Cannot evaluate.")
        return
    elif num_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'auc'
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'
    
    # Handle bool features for XGBoost
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        
    # Handle categorical features for XGBoost (XGBoost can handle categoricals if correctly typed)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype(str).astype('category')
        
    # Exclude datetime columns
    for col in X.select_dtypes(include=['datetime64']).columns:
        X.drop(columns=[col], inplace=True)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = xgb.XGBClassifier(enable_categorical=True, random_state=42, objective=objective, tree_method='hist')
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        preds = clf.predict(X_test)
        
        if num_classes == 2:
            preds_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds_proba)
            f1 = f1_score(y_test, preds, average='binary')
        else:
            preds_proba = clf.predict_proba(X_test)
            auc = roc_auc_score(y_test, preds_proba, multi_class='ovr')
            f1 = f1_score(y_test, preds, average='weighted')
            
        print(f"Results for {name}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC:      {auc:.4f}")
        print(f"Time:     {train_time:.2f} seconds")
        
    except Exception as e:
        print(f"Evaluation failed for {name}: {e}")

if __name__ == "__main__":
    evaluate_dataset(RAW_DATA_PATH, "RAW DATA")
    for model in MODELS:
        safe_model_name = model.replace(":", "_")
        cleaned_path = os.path.join(RESULTS_DIR, f"{safe_model_name}_cleaned.csv")
        evaluate_dataset(cleaned_path, f"CLEANED BY {model}")
