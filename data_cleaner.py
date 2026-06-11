import pandas as pd
import json
import os
import numpy as np
from config import RAW_DATA_PATH, RESULTS_DIR, MODELS

pd.set_option('future.no_silent_downcasting', True)

def clean_data_with_llm_suggestions(model_name):
    """
    Applies the JSON suggestions from the LLM to the raw CSV data.
    """
    safe_model_name = model_name.replace(":", "_")
    suggestion_file = os.path.join(RESULTS_DIR, f"{safe_model_name}_suggestions.json")
    
    if not os.path.exists(suggestion_file):
        print(f"Suggestion file for {model_name} not found. Skipping cleaning.")
        return None
        
    try:
        with open(suggestion_file, 'r') as f:
            content = f.read()
            # Try to extract JSON if it is wrapped in markdown code blocks or has trailing text
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                content = match.group(0)
            suggestions = json.loads(content)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for {model_name}. The LLM might have output invalid JSON. Skipping.")
        return None
        
    df = pd.read_csv(RAW_DATA_PATH)
    
    if "columns" not in suggestions:
        print(f"Invalid suggestion format for {model_name}. Skipping.")
        return None
        
    cols_to_drop = []
    rename_mapping = {}
    
    for col_info in suggestions["columns"]:
        orig_name = col_info.get("original_name")
        if orig_name not in df.columns:
            continue
            
        role = col_info.get("role", "").lower()
        if role == "drop-candidate":
            cols_to_drop.append(orig_name)
            continue
            
        sug_name = col_info.get("suggested_name")
        if sug_name and sug_name != orig_name:
            rename_mapping[orig_name] = sug_name
            
        # Basic dtype conversion (attempt to parse based on suggestion)
        # Note: In a real-world scenario, you might add more robust coercion.
        dtype_sug = col_info.get("suggested_dtype", "").lower()
        try:
            if "numeric" in dtype_sug or "int" in dtype_sug or "float" in dtype_sug:
                df[orig_name] = pd.to_numeric(df[orig_name], errors='coerce')
            elif "date" in dtype_sug:
                df[orig_name] = pd.to_datetime(df[orig_name], errors='coerce')
            elif "bool" in dtype_sug or "boolean" in dtype_sug:
                # Advanced boolean coercion
                bool_map = {'true': True, '1': True, 'yes': True, 't': True, 'y': True,
                            'false': False, '0': False, 'no': False, 'f': False, 'n': False}
                df[orig_name] = df[orig_name].astype(str).str.strip().str.lower().map(bool_map)
            elif "category" in dtype_sug or "categorical" in dtype_sug:
                # Clean strings for categories
                if df[orig_name].dtype == object:
                    df[orig_name] = df[orig_name].astype(str).str.strip()
        except Exception as e:
            print(f"Could not convert {orig_name} to {dtype_sug}: {e}")
            

                
        # Outlier auto-capping logic
        has_outliers = col_info.get("has_outliers", False)
        # Robust check in case LLM outputs a string "true" instead of boolean true
        if str(has_outliers).lower() == "true":
            if pd.api.types.is_numeric_dtype(df[orig_name]):
                try:
                    q_low = df[orig_name].quantile(0.01)
                    q_hi  = df[orig_name].quantile(0.99)
                    df[orig_name] = df[orig_name].clip(lower=q_low, upper=q_hi)
                    print(f"  -> [{model_name}] Auto-capped outliers for: {orig_name}")
                except Exception as e:
                    pass
            
    # Drop and rename
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.rename(columns=rename_mapping, inplace=True)
    
    # Save cleaned data
    out_file = os.path.join(RESULTS_DIR, f"{safe_model_name}_cleaned.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved cleaned data for {model_name} to {out_file}")
    return out_file

if __name__ == "__main__":
    for model in MODELS:
        clean_data_with_llm_suggestions(model)
