import os

# Define the local models to use via Ollama
MODELS = ["qwen2.5:0.5b", "llama3.2:3b", "phi3:mini"]

# File paths
DATA_DIR = "data"
RESULTS_DIR = "results"
RAW_DATA_PATH = os.path.join(DATA_DIR, "retail_store_sales.csv")

# Prompt template for data preprocessing and dashboard suggestions
LLM_PROMPT_TEMPLATE = """
You are a Data Preprocessing and Business Intelligence (BI) Copilot.
I am providing you with metadata about a dataset. I need you to analyze the metadata and provide recommendations.

Dataset Metadata:
{metadata}

Based on this metadata, please return a strict JSON output with the following structure. Do not output anything other than JSON.
{{
  "columns": [
    {{
      "original_name": "...",
      "suggested_name": "...", 
      "suggested_dtype": "...", // e.g., 'numeric', 'categorical', 'date', 'boolean', 'id'
      "role": "...", // e.g., 'identifier', 'feature', 'target', 'drop-candidate'
      "has_outliers": true, // true or false boolean based on if suspicious outliers exist in min/max
      "recommended_imputation": "..." // e.g., 'mean', 'median', 'mode', 'constant:Unknown', 'drop', or 'none' (for documentation only)
    }},
    ...
  ],
  "dashboard_suggestions": {{
    "suggested_kpis": ["...", "..."],
    "suggested_charts": [
      {{"chart_type": "...", "columns": ["..."], "description": "..."}}
    ],
    "suggested_filters": ["...", "..."],
    "dashboard_pages": ["...", "..."]
  }}
}}
"""
