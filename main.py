import os
from data_profiler import get_dataset_metadata
from llm_copilot import run_llm_copilot
from data_cleaner import clean_data_with_llm_suggestions
from model_evaluator import evaluate_dataset
from config import RAW_DATA_PATH, RESULTS_DIR, MODELS

def main():
    print("="*50)
    print(" SMALL LLM PREPROCESSING & BI COPILOT PIPELINE ")
    print("="*50)
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data not found at {RAW_DATA_PATH}")
        print("Please place the 'retail_store_sales.csv' in the 'data/' directory.")
        return
        
    print("\n[1/4] Running Data Profiler...")
    metadata = get_dataset_metadata(RAW_DATA_PATH)
    print(f"Successfully extracted metadata for {metadata['num_cols']} columns and {metadata['num_rows']} rows.")
    
    print("\n[2/4] Running LLM Copilot (This may take some time depending on your hardware)...")
    try:
        run_llm_copilot()
    except Exception as e:
        print(f"Error during LLM Copilot execution: {e}")
        print("Please make sure Ollama is running and the models are pulled.")
        
    print("\n[3/4] Cleaning Data based on LLM suggestions...")
    for model in MODELS:
        print(f"Applying suggestions from {model}...")
        clean_data_with_llm_suggestions(model)
        
    print("\n[4/4] Evaluating Models (XGBoost Classification)...")
    evaluate_dataset(RAW_DATA_PATH, "RAW DATA")
    for model in MODELS:
        safe_model_name = model.replace(":", "_")
        cleaned_path = os.path.join(RESULTS_DIR, f"{safe_model_name}_cleaned.csv")
        evaluate_dataset(cleaned_path, f"CLEANED BY {model}")
        
    print("\n" + "="*50)
    print(" PIPELINE COMPLETE ")
    print("="*50)
    print(f"Check the '{RESULTS_DIR}/' directory for LLM suggestions and cleaned CSV files.")

if __name__ == "__main__":
    main()
