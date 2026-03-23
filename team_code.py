#!/usr/bin/env python

"""
team_code.py
=============
Bridge between the PhysioNet Challenge 2026 interface (train_model.py / run_model.py)
and our custom pipeline (main.py and submodules).

Required functions (signatures MUST NOT change):
    train_model(data_folder, model_folder, verbose)
    load_model(model_folder, verbose)
    run_model(model, record, data_folder, verbose)
    save_model(model_folder, model_dict)
"""

import os
import sys
import gc
import time
import traceback
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 0.  HELPER: Monkey-patch config paths BEFORE any pipeline module is imported
# ---------------------------------------------------------------------------

def _patch_config(data_folder: str, model_folder: str):
    """
    Override config.py path variables so that every downstream module
    sees the challenge-provided folders instead of the hardcoded ones.
    """
    import config

    data_path = Path(data_folder).resolve()
    model_path = Path(model_folder).resolve()

    # Training-set root
    config.TRAINING_SET_DIR = data_path

    # Sub-directories that mirror the challenge layout
    config.PHYSIOLOGICAL_DATA_DIR = data_path / "physiological_data"
    config.ALGORITHMIC_ANNOTATIONS_DIR = data_path / "algorithmic_annotations"
    config.HUMAN_ANNOTATIONS_DIR = data_path / "human_annotations"
    config.DATA_DIR = config.PHYSIOLOGICAL_DATA_DIR

    # Demographics
    config.DEMOGRAPHICS_FILE = data_path / "demographics.csv"

    # Output directories → all go under model_folder so they are persisted
    config.OUTPUT_DIR = model_path
    config.FEATURE_DIR = model_path / "features"
    config.MODEL_DIR = model_path / "models"
    config.LOG_DIR = model_path / "logs"
    config.PLOT_DIR = model_path / "plots"

    for d in [model_path, config.FEATURE_DIR, config.MODEL_DIR,
              config.LOG_DIR, config.PLOT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Disable plotting in challenge environment
    config.PLOT_ENABLED = False


def _setup_logger(verbose: bool):
    """
    Configure loguru: stdout sink when verbose, otherwise suppress.
    """
    from loguru import logger
    logger.remove()  # remove default stderr sink
    if verbose:
        logger.add(sys.stdout, level="INFO",
                    format="{time:HH:mm:ss} | {level:<7} | {message}")
    return logger


# ---------------------------------------------------------------------------
# 1.  TRAIN MODEL  (called by the challenge's train_model.py)
# ---------------------------------------------------------------------------

def train_model(data_folder, model_folder, verbose):
    """
    Train the model on *all* patients found in data_folder.
    Save everything needed for inference into model_folder.
    """
    # --- 0. Patch paths & logger -------------------------------------------
    _patch_config(data_folder, model_folder)

    logger = _setup_logger(verbose)

    # Now safe to import pipeline modules (they read config at import time)
    import config
    from helper_code import find_patients, DEMOGRAPHICS_FILE, HEADERS

    # Our pipeline imports
    from main import process_single_patient
    from feature_table.build_feature_table import (
        build_cohort_feature_table,
    )
    from classification.train_model import (
        train_model as classification_train_model,
    )

    if verbose:
        print("Finding the Challenge data...")

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError("No data were provided.")

    if verbose:
        print(f"Found {num_records} records. Starting preprocessing...")

    # --- 1. Run preprocessing pipeline (parallel) --------------------------
    feature_dir = config.FEATURE_DIR
    feature_dir.mkdir(parents=True, exist_ok=True)

    successful_results = {}  # patient_id -> {seg_path, pat_path, sleep_summary}
    max_workers = min(config.NUM_WORKERS, num_records) if num_records > 0 else 1

    total_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(num_records):
            record = patient_metadata_list[i]
            patient_id_bids = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            # Build the patient_id string our pipeline expects:
            #   "site_id/sub-XXX_ses-N"
            record_name = f"{patient_id_bids}_ses-{session_id}"
            pipeline_patient_id = f"{site_id}/{record_name}"

            patient_dir = config.PHYSIOLOGICAL_DATA_DIR / site_id

            if not patient_dir.exists():
                if verbose:
                    print(f"  ! Directory not found: {patient_dir}. Skipping.")
                continue

            future = executor.submit(
                process_single_patient,
                patient_id=pipeline_patient_id,
                patient_dir=patient_dir,
                record_name=record_name,
                segment_length_sec=config.SEGMENT_LENGTH_SEC,
                overlap_sec=config.SEGMENT_OVERLAP_SEC,
                feature_output_dir=feature_dir,
            )
            futures[future] = pipeline_patient_id

        for i, future in enumerate(as_completed(futures)):
            pid = futures[future]
            try:
                result = future.result()
                if result and result["success"]:
                    successful_results[pid] = {
                        "seg_features_path": result.get("seg_features_path"),
                        "pat_features_path": result.get("pat_features_path"),
                        "sleep_summary": result.get("sleep_summary"),
                    }
            except Exception as e:
                if verbose:
                    print(f"  !!! Error processing {pid}: {e}")

            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - total_start
                print(f"  Progress: {i+1}/{len(futures)} "
                      f"({elapsed:.0f}s elapsed)")

    if verbose:
        print(f"Preprocessing done: {len(successful_results)}/{num_records} "
              f"successful in {time.time()-total_start:.0f}s")

    # --- 2. Load features from disk & build cohort table -------------------
    patient_segment_tables = {}
    patient_sleep_summaries = {}

    for pid, paths in successful_results.items():
        seg_path = paths.get("seg_features_path")
        if seg_path and Path(seg_path).exists():
            try:
                patient_segment_tables[pid] = pd.read_parquet(seg_path)
            except Exception as e:
                if verbose:
                    print(f"  ! Failed to load {seg_path}: {e}")

        if paths.get("sleep_summary"):
            patient_sleep_summaries[pid] = paths["sleep_summary"]

    del successful_results
    gc.collect()

    if not patient_segment_tables:
        if verbose:
            print("ERROR: No features extracted. Training a fallback model.")
        # Save a trivial fallback model so load_model / run_model don't crash
        _save_fallback_model(model_folder)
        return

    segment_level, patient_level = build_cohort_feature_table(
        patient_segment_tables=patient_segment_tables,
        patient_sleep_summaries=patient_sleep_summaries,
        demographics_path=config.DEMOGRAPHICS_FILE,
        output_dir=feature_dir,
        save=True,
        logger=logger if verbose else None,
    )

    del patient_segment_tables, patient_sleep_summaries
    gc.collect()

    if patient_level is None or len(patient_level) == 0:
        if verbose:
            print("ERROR: Patient-level table is empty. Saving fallback model.")
        _save_fallback_model(model_folder)
        return

    # --- 3. Train the ML model ---------------------------------------------
    if verbose:
        print(f"Training model on {len(patient_level)} patients, "
              f"{len(patient_level.columns)} features...")

    model_result = classification_train_model(
        feature_table=patient_level,
        model_type="ensemble",
        feature_selection=True,
        handle_imbalance="class_weight",
        output_dir=config.MODEL_DIR,
        logger=logger if verbose else None,
    )

    # --- 4. Save to model_folder -------------------------------------------
    save_model(model_folder, model_result)

    if verbose:
        cv = model_result.get("cv_results", {})
        print(f"Training complete. "
              f"CV AUROC={cv.get('auroc_mean', 'N/A')}")
        print("Done.")


# ---------------------------------------------------------------------------
# 2.  LOAD MODEL  (called by the challenge's run_model.py)
# ---------------------------------------------------------------------------

def load_model(model_folder, verbose):
    """
    Load everything saved by train_model / save_model.
    Returns a dict that will be passed as the first arg to run_model().
    """
    model_path = os.path.join(model_folder, 'model.sav')

    if not os.path.exists(model_path):
        if verbose:
            print(f"WARNING: {model_path} not found. Using fallback.")
        return {"fallback": True}

    model_dict = joblib.load(model_path)

    if verbose:
        mt = model_dict.get("training_config", {}).get("model_type", "unknown")
        nf = len(model_dict.get("selected_features", []))
        print(f"Model loaded: type={mt}, features={nf}")

    return model_dict


# ---------------------------------------------------------------------------
# 3.  RUN MODEL  (called by the challenge's run_model.py, once per patient)
# ---------------------------------------------------------------------------

def run_model(model, record, data_folder, verbose):
    """
    Run the trained model on a single patient record.

    Parameters
    ----------
    model : dict
        The object returned by load_model().
    record : dict
        Contains BidsFolder, SiteID, SessionID.
    data_folder : str
        Root of the challenge data.
    verbose : bool

    Returns
    -------
    (binary_output, probability_output)
    """
    from helper_code import HEADERS

    patient_id_bids = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # ---- Fallback model ---------------------------------------------------
    if model.get("fallback", False):
        return 0, 0.5

    # ---- Patch config for this data_folder --------------------------------
    # Use a temporary model folder (we won't save anything during inference)
    import tempfile
    tmp_model_dir = tempfile.mkdtemp(prefix="physionet_run_")
    _patch_config(data_folder, tmp_model_dir)

    logger = _setup_logger(verbose)

    import config
    from main import process_single_patient

    # ---- Build paths our pipeline expects ---------------------------------
    record_name = f"{patient_id_bids}_ses-{session_id}"
    pipeline_patient_id = f"{site_id}/{record_name}"
    patient_dir = config.PHYSIOLOGICAL_DATA_DIR / site_id

    if not patient_dir.exists():
        if verbose:
            print(f"  ! Patient dir not found: {patient_dir}")
        return 0, 0.5

    # ---- Run the full pipeline on this single patient ---------------------
    try:
        feature_output_dir = Path(tmp_model_dir) / "features"
        feature_output_dir.mkdir(parents=True, exist_ok=True)

        # Parallel preprocessing using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
            future = executor.submit(
                process_single_patient,
                patient_id=pipeline_patient_id,
                patient_dir=patient_dir,
                record_name=record_name,
                segment_length_sec=config.SEGMENT_LENGTH_SEC,
                overlap_sec=config.SEGMENT_OVERLAP_SEC,
                feature_output_dir=feature_output_dir,
            )
            result = future.result()

        if result is None or not result["success"]:
            if verbose:
                print(f"  ! Pipeline failed for {pipeline_patient_id}")
            return 0, 0.5

        # ---- Load the patient-level features from disk --------------------
        pat_path = result.get("pat_features_path")
        if pat_path is None or not Path(pat_path).exists():
            # Try building patient-level from segment-level
            seg_path = result.get("seg_features_path")
            if seg_path and Path(seg_path).exists():
                from feature_table.build_feature_table import (
                    aggregate_to_patient_level, add_demographics,
                )
                seg_df = pd.read_parquet(seg_path)
                sleep_summary = result.get("sleep_summary")
                patient_features = aggregate_to_patient_level(
                    seg_df, sleep_summary
                )
                patient_features = add_demographics(
                    patient_features,
                    demographics_path=config.DEMOGRAPHICS_FILE,
                )
            else:
                if verbose:
                    print(f"  ! No features for {pipeline_patient_id}")
                return 0, 0.5
        else:
            patient_features = pd.read_parquet(pat_path)
            # Add demographics for consistency
            from feature_table.build_feature_table import add_demographics
            patient_features = add_demographics(
                patient_features,
                demographics_path=config.DEMOGRAPHICS_FILE,
            )

        if patient_features is None or len(patient_features) == 0:
            return 0, 0.5

        # ---- Apply the trained model pipeline -----------------------------
        binary_output, probability_output = _predict_single_patient(
            model, patient_features, verbose
        )

        return binary_output, probability_output

    except Exception as e:
        if verbose:
            print(f"  !!! run_model error for {pipeline_patient_id}: {e}")
            traceback.print_exc()
        return 0, 0.5

    finally:
        gc.collect()
        # Clean up temp dir (best effort)
        try:
            import shutil
            shutil.rmtree(tmp_model_dir, ignore_errors=True)
        except Exception:
            pass


def _predict_single_patient(
    model_dict: dict,
    patient_features: pd.DataFrame,
    verbose: bool
) -> Tuple[int, float]:
    """
    Apply imputer → feature selector → scaler → model.predict on one patient.
    """
    ml_model = model_dict["model"]
    scaler = model_dict["scaler"]
    imputer = model_dict["imputer"]
    feature_selector = model_dict.get("feature_selector")
    feature_names = model_dict["feature_names"]
    selected_features = model_dict["selected_features"]

    # Ensure all expected features exist (fill missing with NaN)
    for f in feature_names:
        if f not in patient_features.columns:
            patient_features[f] = np.nan

    X = patient_features[feature_names].values.astype(np.float64)

    # Impute
    X_imp = imputer.transform(X)

    # Feature selection
    if feature_selector is not None:
        try:
            X_sel = feature_selector.transform(X_imp)
        except Exception:
            sel_idx = [feature_names.index(f) for f in selected_features
                       if f in feature_names]
            X_sel = X_imp[:, sel_idx]
    else:
        X_sel = X_imp

    # Scale
    X_scaled = scaler.transform(X_sel)

    # Predict
    binary_output = int(ml_model.predict(X_scaled)[0])
    probability_output = float(ml_model.predict_proba(X_scaled)[0][1])

    return binary_output, probability_output


# ---------------------------------------------------------------------------
# 4.  SAVE MODEL
# ---------------------------------------------------------------------------

def save_model(model_folder, model_dict):
    """
    Save the full model dictionary (model + scaler + imputer + selector +
    feature names + config) into model_folder/model.sav.
    """
    os.makedirs(model_folder, exist_ok=True)
    filename = os.path.join(model_folder, 'model.sav')

    # Build a serialisable dict
    to_save = {}
    keys_to_save = [
        "model", "scaler", "imputer", "feature_selector",
        "feature_names", "selected_features",
        "training_config", "cv_results",
    ]
    for k in keys_to_save:
        if k in model_dict:
            to_save[k] = model_dict[k]

    joblib.dump(to_save, filename, protocol=2)


def _save_fallback_model(model_folder):
    """
    Save a trivial fallback model that always predicts 0 / 0.5.
    Used when preprocessing fails for all patients.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    dummy = DummyClassifier(strategy="constant", constant=0)
    dummy.fit(np.zeros((2, 1)), np.array([0, 1]))

    model_dict = {
        "model": dummy,
        "scaler": StandardScaler().fit(np.zeros((2, 1))),
        "imputer": SimpleImputer(strategy="median").fit(np.zeros((2, 1))),
        "feature_selector": None,
        "feature_names": ["dummy_feature"],
        "selected_features": ["dummy_feature"],
        "training_config": {"model_type": "fallback"},
        "cv_results": {},
        "fallback": True,
    }
    save_model(model_folder, model_dict)
