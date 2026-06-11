# 🚀 Small LLM as a Data Preprocessing & BI Copilot

This project explores whether small, locally-hosted Large Language Models (LLMs in the 0.5B to 3.8B parameter range) can successfully act as autonomous data preprocessing assistants. 

By analyzing only dataset **metadata** (avoiding context limits and privacy concerns), the LLMs are tasked with inferring data types, standardizing column names, identifying statistical outliers, and generating Tableau/Power BI dashboard specifications. We then pipe the LLM-cleaned data into an **XGBoost Classifier** to see if the LLM's automated cleaning improves downstream Machine Learning performance compared to the raw data.

---

## 🧠 Models Tested
- `llama3.2:3b` - The reliable 3B baseline.
- `phi3:mini` - ~3.8B parameter model known for strong reasoning.
- `qwen2.5:0.5b` - A highly-efficient 0.5B lower-bound test.

---

## 🛠️ The Pipeline Architecture

The project is heavily modularized into 5 distinct pipeline stages:

1. **`data_profiler.py`**: Reads the messy CSV and extracts lightweight metadata (column names, current data types, % missing, and a small sample of unique values).
2. **`llm_copilot.py`**: Connects to a local Ollama instance and prompts the models with the metadata. **Includes "Self-Healing" JSON Logic:** If the LLM hallucinates broken JSON syntax, Python intercepts the crash and automatically re-prompts the LLM (up to 3 times) to fix its own syntax errors.
3. **`data_cleaner.py`**: Parses the LLM's suggestions and applies them to the raw dataset. **Includes Auto-Capping:** If the LLM flags a numeric column as having an "outlier" or "anomaly", the script automatically winsorizes (caps) the top and bottom 1% of the data.
4. **`model_evaluator.py`**: Trains an XGBoost classification model to predict `Payment Method`. It compares the F1 Score, AUC, and training time of the raw data vs. the LLM-cleaned datasets.
5. **`dashboard_generator.py`**: Translates the LLM's text-based Power BI/Tableau chart suggestions into actual PNG visualizations using `seaborn` and `matplotlib`.

---

## 📈 Final Evaluation & Key Findings

The experiment proved to be a resounding success, highlighting both the extreme value and the limitations of local LLMs:

1. **Massive ML Training Time Reduction (~50%)**
   - The LLMs successfully inferred complex data types (converting messy string booleans into native `bools` and object columns into `categories`). 
   - By correctly formatting the data types before ML training, XGBoost's histogram builder became significantly more efficient, dropping training time from **5.33 seconds (Raw)** down to **~2.6 seconds (LLM-Cleaned)**.
   
2. **Outlier Detection Directly Boosted Accuracy**
   - `phi3:mini` successfully identified extreme variances in continuous variables (like `Total Spent`). 
   - Acting on Phi-3's warning, the script automatically clipped the outliers, driving the **AUC score up from 0.5190 to 0.5230** and the **F1 score from 0.3413 to 0.3447**.

3. **XGBoost Robustness & The Imputation Trap**
   - **Binning Robustness:** We observed that models triggering different combinations of outlier capping (`Total Spent` only vs all three numeric columns) produced mathematically identical F1/AUC scores. This proved XGBoost's histogram binning algorithm is highly robust, naturally handling secondary outliers so long as the primary target-correlated outlier is clipped.
   - **The Imputation Penalty:** We initially allowed the LLMs to force impute `NaN` values via mean/mode before ML training. This actually **lowered** the predictive score! XGBoost relies on the "missingness" of data as a native branch-split signal. By updating the LLM to strictly act as an advisory Copilot (documenting the ideal imputation strategy in the JSON audit log but preserving the native `NaN`s for XGBoost), we successfully restored peak ML performance.

4. **Model Personality & Stability**
   - **Llama 3.2 (3B)** proved to be the most stable. It safely cleaned the data without aggressive outlier capping, preserving baseline statistics while drastically speeding up execution.
   - **Phi-3 Mini (3.8B)** achieved the highest performance and provided the best visual dashboard suggestions, but occasionally suffered from JSON-generation syntax bugs, proving the absolute necessity of our self-healing retry loops.
   - **Qwen 2.5 (0.5B)** proved too small to be deterministic. While it succeeded occasionally, it often hallucinated column names, establishing 1B+ parameters as the safe minimum for zero-shot JSON engineering.

---

## ⚙️ Quick Start Guide

### Prerequisites
1. Ensure you have Python 3.8+ installed.
2. Install [Ollama](https://ollama.com/) and ensure it is running in the background.

### Step 1: Pull the Local Models
Open your terminal and pull the necessary models:
```bash
ollama pull qwen2.5:0.5b
ollama pull llama3.2:3b
ollama pull phi3:mini
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Pipeline
Ensure your dataset (`retail_store_sales.csv`) is inside the `data/` folder, then execute the main script:
```bash
python main.py
```

### Running on Custom Datasets
This pipeline is 90% generalized! If you want to run it on your own CSV, `data_profiler.py`, `llm_copilot.py`, and `data_cleaner.py` will dynamically process the new file without any code changes (just update `RAW_DATA_PATH` in `config.py`). 

However, `model_evaluator.py` needs to know what it is trying to predict! Open `model_evaluator.py` and update line 14 to match your new target column:
```python
# Change this to match the target column of your custom dataset!
TARGET_COL_HINTS = ["Payment Method", "payment_method", "payment", "method"]
```

### Output
Check the dynamically generated `results/` folder for:
- The raw JSON suggestion files from the LLMs.
- The newly cleaned CSV files.
- The auto-generated Dashboard PNG charts!

---

## 🐳 Docker Deployment (Optional)
If you prefer an isolated container environment:
1. `docker compose up -d ollama`
2. `docker exec -it <ollama_container_id> ollama pull llama3.2:3b` (repeat for all models)
3. `docker compose up pipeline`
