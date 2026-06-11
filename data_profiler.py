import pandas as pd
import json

def get_dataset_metadata(file_path):
    """
    Reads the CSV and extracts metadata for the LLM.
    Returns a dictionary of metadata.
    """
    df = pd.read_csv(file_path)
    metadata = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": {}
    }
    
    for col in df.columns:
        col_meta = {}
        col_meta["current_dtype"] = str(df[col].dtype)
        col_meta["missing_percentage"] = float(df[col].isnull().mean() * 100)
        
        # Get a sample of up to 5 unique values
        unique_vals = df[col].dropna().unique()
        col_meta["unique_sample"] = [str(x) for x in unique_vals[:5]]
        
        # If numeric, provide min, max, mean
        if pd.api.types.is_numeric_dtype(df[col]):
            col_meta["min"] = float(df[col].min())
            col_meta["max"] = float(df[col].max())
            col_meta["mean"] = float(df[col].mean())
            
        metadata["columns"][col] = col_meta
        
    return metadata

if __name__ == "__main__":
    from config import RAW_DATA_PATH
    meta = get_dataset_metadata(RAW_DATA_PATH)
    print(json.dumps(meta, indent=2))
