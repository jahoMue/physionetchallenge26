# PhysioNet/Computing in Cardiology Challenge 2026 - Team IBMT

This repository contains the source code for Team IBMT's submission to the PhysioNet/Computing in Cardiology Challenge 2026: **Screening for Cognitive Impairment During Sleep Studies**.

---

## 📂 Data Storage & Configuration

The project expects the dataset to follow the standard PhysioNet Challenge structure:
```text
data_folder/
├── demographics.csv
├── physiological_data/
├── algorithmic_annotations/
└── human_annotations/
```

**How to configure the data path:**
1. **Default via config:** Open `config.py` and set `DATA_DIR` to the root of your physiological data, and `DEMOGRAPHICS_FILE` to your `.csv` file. 
2. **Override via CLI:** You can override these paths on the fly when running local experiments (e.g., `python main.py --data-dir path/to/data`).
3. **Challenge Interface:** The official scripts (`train_model.py`, `run_model.py`) automatically handle paths through the `-d` CLI argument.

---

## 🚀 Local Development (`main.py`)

`main.py` is the primary orchestrator for local development, testing, and debugging. It runs the entire pipeline—from data loading to preprocessing, feature extraction, training, and evaluation—while allowing you to run specific steps individually.

### Basic Usage
Run the full pipeline (preprocess -> extract -> train -> evaluate) using default settings in `config.py`:
```bash
python main.py
```

### Advanced CLI Options
You can control exactly what the pipeline does using CLI arguments:

```bash
# Run specific pipeline steps
python main.py --step preprocess     # Run only data loading and feature extraction
python main.py --step features       # Only load saved features and display summary
python main.py --step train          # Train a model on pre-extracted features
python main.py --step evaluate       # Evaluate the trained model

# Override data and output directories
python main.py --data-dir ./data/train --output-dir ./results

# Control processing subset and hyperparameters
python main.py --patients 10         # Process only the first 10 patients
python main.py --segment-length 300  # Change sliding window segment length to 300s
python main.py --models xgboost      # Train specifically an XGBoost model
python main.py --tune                # Enable Optuna hyperparameter tuning
python main.py --no-plots            # Disable visualization generation
python main.py --verbose             # Enable debug-level logging
```

---

## 🏆 Official Challenge Interface

The project includes the mandatory standard scripts required by the PhysioNet challenge organizers. **These scripts must not be modified in their signature or usage.** 

They integrate seamlessly into our custom codebase via the **`team_code.py`** bridge. When the challenge scripts run, `team_code.py` intercepts the `-d` and `-m` parameters and *dynamically patches `config.py`* in memory so our custom modules (`main.py`, `preprocessing`, etc.) read and write strictly to the challenge-designated directories.

### 1. Training the Model (`train_model.py`)
Trains the model on the provided dataset and saves the artifacts to the model folder.
```bash
python train_model.py -d path/to/training_data -m path/to/save_model -v
```
* **Integration:** Calls `team_code.train_model()`, which utilizes a `ProcessPoolExecutor` to run the identical preprocessing steps defined in `main.py` (via `process_single_patient`), aggregates the features, trains the configured ensemble model, and saves the trained classifier along with its scaler and imputer into `model.sav`.

### 2. Running the Model (`run_model.py`)
Executes the trained model against a new (unseen) dataset.
```bash
python run_model.py -d path/to/test_data -m path/to/saved_model -o path/to/outputs -v
```
* **Integration:** 
  1. Calls `team_code.load_model()` to load the `.sav` file into memory.
  2. Calls `team_code.run_model()` iteratively for every patient in `demographics.csv`.
  3. To maximize efficiency during testing, the very first call to `run_model` fires a background prefetching thread pool that preprocesses all patients in parallel (mirroring the training behavior).
  4. Outputs (binary predictions and probabilities) are mapped back to an updated `demographics.csv` saved in the `-o` output folder.

### 3. Evaluating the Model (`evaluate_model.py`)
Evaluates the model's performance on the test predictions compared to ground truth labels.
```bash
python evaluate_model.py -d labels.csv -o predictions.csv -s scores.csv
```
* **Integration:** This script calculates standard Challenge metrics: Area Under the Receiver Operating Characteristic curve (AUROC), Area Under the Precision-Recall Curve (AUPRC), Accuracy, and F-measure.

---

## 🧩 Architecture Summary

* **`train_model.py`, `run_model.py`, `evaluate_model.py`**: Standard challenge entry points.
* **`team_code.py`**: The "adapter" connecting the standard challenge entry points to the Team IBMT custom code. Handles dynamic config overrides and parallel processing orchestration.
* **`main.py`**: Developer orchestration script. Defines high-level routines (`run_preprocessing_pipeline`, `run_training_pipeline`, etc.).
* **`config.py`**: Central configuration registry for file paths, biological parameters (e.g., filters, segment overlap), and toggles.
* **`preprocessing/`**: Raw EDF signal (EEG, ECG, Respiration) and annotation cleaning.
* **`feature_extraction/`**: Extracting statistical, morphological, and spectral features on isolated overlapping windows.
* **`classification/`**: Building, tuning, and evaluating the final predictive models.