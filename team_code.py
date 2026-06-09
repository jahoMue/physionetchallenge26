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

def _prior_probability_shift(
    p: np.ndarray,
    train_prevalence: float,
    target_prevalence: Optional[float],
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Adjust predicted probabilities from the training prior to the expected
    deployment/test prior.

    odds_target = odds_model *
        [pi_target / (1 - pi_target)] / [pi_train / (1 - pi_train)]
    """
    if target_prevalence is None:
        return p

    train_prevalence = float(np.clip(train_prevalence, eps, 1.0 - eps))
    target_prevalence = float(np.clip(target_prevalence, eps, 1.0 - eps))

    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)

    odds = p / (1.0 - p)

    train_prior_odds = train_prevalence / (1.0 - train_prevalence)
    target_prior_odds = target_prevalence / (1.0 - target_prevalence)

    correction = target_prior_odds / train_prior_odds
    adjusted_odds = odds * correction

    return adjusted_odds / (1.0 + adjusted_odds)


def _safe_predict_proba_positive(ml_model, X_scaled: np.ndarray) -> np.ndarray:
    """
    Return P(class=1) robustly, including for degenerate/dummy models.
    """
    if not hasattr(ml_model, "predict_proba"):
        pred = ml_model.predict(X_scaled)
        return np.asarray(pred, dtype=float)

    proba = ml_model.predict_proba(X_scaled)

    if proba.ndim == 1:
        return proba.astype(float)

    if proba.shape[1] == 1:
        classes = getattr(ml_model, "classes_", np.array([0]))
        only_class = int(classes[0]) if len(classes) > 0 else 0
        if only_class == 1:
            return proba[:, 0].astype(float)
        return np.zeros(proba.shape[0], dtype=float)

    classes = getattr(ml_model, "classes_", np.array([0, 1]))
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = 1

    return proba[:, pos_idx].astype(float)


def _predict_feature_table(
    model_dict: dict,
    feature_table: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply the exact saved preprocessing chain:

        raw feature table
        -> column alignment
        -> imputer
        -> feature selector
        -> scaler
        -> model probabilities
        -> prior-probability correction
        -> binary threshold

    This function is used for both:
      1. internal holdout evaluation during training
      2. challenge inference in run_model()
    """
    if model_dict.get("fallback", False):
        patient_ids = (
            feature_table["patient_id"].values
            if "patient_id" in feature_table.columns
            else np.arange(len(feature_table))
        )
        return pd.DataFrame({
            "patient_id": patient_ids,
            "prediction": np.zeros(len(feature_table), dtype=int),
            "probability": np.full(len(feature_table), 0.10, dtype=float),
        })

    ml_model = model_dict["model"]
    scaler = model_dict["scaler"]
    imputer = model_dict["imputer"]
    feature_selector = model_dict.get("feature_selector")
    feature_names = model_dict["feature_names"]
    selected_features = model_dict.get("selected_features", feature_names)

    df = feature_table.copy()

    patient_ids = (
        df["patient_id"].values
        if "patient_id" in df.columns
        else np.arange(len(df))
    )

    # Ensure all expected training features exist.
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features and verbose:
        print(
            f"  ! Missing features at prediction: "
            f"{len(missing_features)}/{len(feature_names)}"
        )

    for f in missing_features:
        df[f] = np.nan

    # Match training feature order exactly.
    X = df[feature_names].values.astype(np.float32)

    # Saved preprocessing.
    X_imp = imputer.transform(X).astype(np.float32)

    if feature_selector is not None:
        try:
            X_sel = feature_selector.transform(X_imp)
        except Exception:
            sel_idx = [
                feature_names.index(f)
                for f in selected_features
                if f in feature_names
            ]
            X_sel = X_imp[:, sel_idx]
    else:
        X_sel = X_imp

    X_scaled = scaler.transform(X_sel).astype(np.float32)

    # Raw model probability.
    raw_probabilities = _safe_predict_proba_positive(ml_model, X_scaled)

    # Prior correction using training metadata if available.
    config = model_dict.get("training_config", {})
    train_prev = config.get("training_prevalence", None)
    target_prev = config.get("expected_test_prevalence", None)
    threshold = float(config.get("decision_threshold", 0.5))

    if train_prev is not None and target_prev is not None:
        probabilities = _prior_probability_shift(
            raw_probabilities,
            train_prevalence=float(train_prev),
            target_prevalence=float(target_prev),
        )
    else:
        probabilities = raw_probabilities

    probabilities = np.clip(probabilities, 0.0, 1.0)
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame({
        "patient_id": patient_ids,
        "prediction": predictions.astype(int),
        "probability": probabilities.astype(float),
    })

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

    logger.info("Finding the Challenge data...")

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError("No data were provided.")

    logger.info(f"Found {num_records} records. Starting preprocessing...")

    # --- 1. Run preprocessing pipeline (parallel, pool-recycling) -----------
    feature_dir = config.FEATURE_DIR
    feature_dir.mkdir(parents=True, exist_ok=True)

    successful_results = {}  # patient_id -> {seg_path, pat_path, sleep_summary}
    max_workers = min(config.NUM_WORKERS, num_records) if num_records > 0 else 1

    total_start = time.time()

    # Process in batches, recycling the pool after each batch
    # so that worker process memory is fully released between batches
    batch_size = max_workers  # One subject per worker, then recycle

    for batch_start in range(0, num_records, batch_size):
        batch_end = min(batch_start + batch_size, num_records)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(batch_start, batch_end):
                record = patient_metadata_list[i]
                patient_id_bids = record[HEADERS['bids_folder']]
                site_id = record[HEADERS['site_id']]
                session_id = record[HEADERS['session_id']]

                record_name = f"{patient_id_bids}_ses-{session_id}"
                pipeline_patient_id = f"{site_id}/{record_name}"
                patient_dir = config.PHYSIOLOGICAL_DATA_DIR / site_id

                if not patient_dir.exists():
                    logger.warning(f"Directory not found: {patient_dir}. Skipping.")
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

            for future in as_completed(futures):
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
                    logger.error(f"Error processing {pid}: {e}")

        # Pool is destroyed here — all worker processes are killed and RAM is freed
        gc.collect()

        elapsed = time.time() - total_start
        logger.info(
            f"Batch progress: {batch_end}/{num_records} patients processed "
            f"({len(successful_results)} successful, {elapsed:.0f}s elapsed)"
        )

    logger.info(
        f"Preprocessing done: {len(successful_results)}/{num_records} "
        f"successful in {time.time() - total_start:.0f}s"
    )

    # --- 2. Load features from disk & build cohort table -------------------
    patient_segment_tables = {}
    patient_level_tables = {}
    patient_sleep_summaries = {}

    for pid, paths in successful_results.items():
        seg_path = paths.get("seg_features_path")
        if seg_path and Path(seg_path).exists():
            try:
                patient_segment_tables[pid] = pd.read_parquet(seg_path)
            except Exception as e:
                logger.warning(f"Failed to load {seg_path}: {e}")

        pat_path = paths.get("pat_features_path")
        if pat_path and Path(pat_path).exists():
            try:
                patient_level_tables[pid] = pd.read_parquet(pat_path)
            except Exception as e:
                logger.warning(f"Failed to load {pat_path}: {e}")

        if paths.get("sleep_summary"):
            patient_sleep_summaries[pid] = paths["sleep_summary"]

    segment_level, patient_level = build_cohort_feature_table(
        patient_segment_tables=patient_segment_tables,
        patient_features_tables=patient_level_tables,
        patient_sleep_summaries=patient_sleep_summaries,
        demographics_path=config.DEMOGRAPHICS_FILE,
        output_dir=feature_dir,
        save=True,
        logger=logger if verbose else None,
    )

    del patient_segment_tables, patient_sleep_summaries
    gc.collect()

    if patient_level is None or len(patient_level) == 0:
        logger.error("Patient-level table is empty. Saving fallback model.")
        _save_fallback_model(model_folder)
        return

    # --- 3. Train the ML model ---------------------------------------------
    from sklearn.model_selection import train_test_split
    from classification.train_model import train_model as classification_train_model

    logger.info(
        f"Training model on {len(patient_level)} patients, "
        f"{len(patient_level.columns)} features..."
    )

    # --- 3a. Reserve a 20% held-out internal validation set ---
    # This gives us ONE truly unbiased AUROC estimate that has never
    # been seen by feature selection, hyperparameters, or model fitting.
    target_col = next(
        (c for c in ["target", config.TARGET_COLUMN, "Cognitive_Impairment"]
        if c in patient_level.columns), None
    )
    if target_col is None:
        logger.error("No target column. Saving fallback.")
        _save_fallback_model(model_folder)
        return

    valid_mask = patient_level[target_col].notna()
    patient_level_valid = patient_level.loc[valid_mask].reset_index(drop=True)

    train_df, holdout_df = train_test_split(
        patient_level_valid,
        test_size=0.20,
        stratify=patient_level_valid[target_col].astype(int),
        random_state=config.RANDOM_SEED,
    )
    logger.info(
        f"Internal split: train={len(train_df)} (prev={train_df[target_col].mean():.3f}), "
        f"holdout={len(holdout_df)} (prev={holdout_df[target_col].mean():.3f})"
    )

    # --- 3b. Train (CV runs inside, all preprocessing inside pipeline) ---
    model_result = classification_train_model(
        feature_table=train_df,
        output_dir=config.MODEL_DIR,
        expected_test_prevalence=0.10,
        logger=logger if verbose else None,
    )


    if not model_result:
        logger.error("Training failed. Saving fallback.")
        _save_fallback_model(model_folder)
        return

        # --- 3c. Honest evaluation on the untouched 20% holdout ---
    # Important:
    # Use the exact same preprocessing + probability correction as inference.
    # Do NOT call model.predict_proba() directly on raw holdout features.
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        f1_score,
        balanced_accuracy_score,
    )

    y_hold = holdout_df[target_col].astype(int).values

    if len(np.unique(y_hold)) == 2:
        holdout_pred_df = _predict_feature_table(
            model_dict=model_result,
            feature_table=holdout_df,
            verbose=verbose,
        )

        proba_hold = holdout_pred_df["probability"].values.astype(float)
        binary_hold = holdout_pred_df["prediction"].values.astype(int)

        holdout_auroc = roc_auc_score(y_hold, proba_hold)
        holdout_auprc = average_precision_score(y_hold, proba_hold)
        holdout_accuracy = accuracy_score(y_hold, binary_hold)
        holdout_f1 = f1_score(y_hold, binary_hold, zero_division=0)
        holdout_bal_acc = balanced_accuracy_score(y_hold, binary_hold)

        cv_auroc = model_result.get("cv_results", {}).get("auroc_mean", np.nan)
        gap = (
            float(cv_auroc) - float(holdout_auroc)
            if np.isfinite(cv_auroc)
            else np.nan
        )

        logger.info("=" * 60)
        logger.info("HONEST HOLDOUT EVALUATION (20% never-seen)")
        logger.info(f"  CV AUROC:       {cv_auroc:.4f}" if np.isfinite(cv_auroc) else "  CV AUROC:       N/A")
        logger.info(f"  Holdout AUROC:  {holdout_auroc:.4f}")
        logger.info(f"  Holdout AUPRC:  {holdout_auprc:.4f}")
        logger.info(f"  Holdout Acc:    {holdout_accuracy:.4f}")
        logger.info(f"  Holdout F1:     {holdout_f1:.4f}")
        logger.info(f"  Holdout BalAcc: {holdout_bal_acc:.4f}")

        if np.isfinite(gap):
            logger.info(
                f"  CV-Holdout gap: {gap:+.4f}  "
                f"{'⚠ overfitting' if gap > 0.05 else '✓ healthy'}"
            )
        logger.info("=" * 60)

        model_result.setdefault("cv_results", {}).update({
            "holdout_auroc": float(holdout_auroc),
            "holdout_auprc": float(holdout_auprc),
            "holdout_accuracy": float(holdout_accuracy),
            "holdout_f1": float(holdout_f1),
            "holdout_balanced_accuracy": float(holdout_bal_acc),
            "cv_holdout_gap": float(gap) if np.isfinite(gap) else np.nan,
        })
    else:
        logger.warning(
            "Holdout evaluation skipped because holdout set contains "
            "only one class."
        )


    # --- 3d. Optional: refit on 100% of training data for final submission ---
        # Trade-off: more data vs. having an honest estimate. With ~600 patients,
    # refitting can still help, while the holdout estimate above remains the
    # unbiased diagnostic number.
    logger.info("Refitting final model on 100% of training data...")
    final_result = classification_train_model(
        feature_table=patient_level_valid,
        output_dir=config.MODEL_DIR,
        expected_test_prevalence=0.10,
        logger=logger if verbose else None,
    )

    if final_result:
        # Preserve honest holdout diagnostics from the pre-refit model.
        holdout_keys = [
            "holdout_auroc",
            "holdout_auprc",
            "holdout_accuracy",
            "holdout_f1",
            "holdout_balanced_accuracy",
            "cv_holdout_gap",
        ]
        final_result.setdefault("cv_results", {}).update({
            k: model_result.get("cv_results", {}).get(k)
            for k in holdout_keys
            if k in model_result.get("cv_results", {})
        })
        model_result = final_result


    # --- 4. Save to model_folder -------------------------------------------
    save_model(model_folder, model_result)

    cv = model_result.get("cv_results", {})

    logger.info(
        f"Training complete. CV AUROC={cv.get('auroc_mean', 'N/A')}"
    )
    logger.info("Done.")




# ---------------------------------------------------------------------------
# 2.  LOAD MODEL
# ---------------------------------------------------------------------------

def load_model(model_folder, verbose):
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

    # No prefetch pool — process each patient synchronously in run_model()
    model_dict["_config_patched"] = False
    return model_dict


# ---------------------------------------------------------------------------
# WORKER: runs ONE patient in a child process, then exits
# ---------------------------------------------------------------------------

def _preprocess_one_patient_worker(
    data_folder,
    tmp_model_dir,
    pipeline_patient_id,
    patient_dir_str,
    record_name,
    segment_length_sec,
    overlap_sec,
    feature_output_dir_str,
):
    """
    Runs process_single_patient in a fresh child process.
    The child exits when done → all memory (YASA, MNE, numpy arrays) is
    fully reclaimed by the OS before the next patient starts.
    """
    _patch_config(data_folder, tmp_model_dir)
    from main import process_single_patient

    return process_single_patient(
        patient_id=pipeline_patient_id,
        patient_dir=Path(patient_dir_str),
        record_name=record_name,
        segment_length_sec=segment_length_sec,
        overlap_sec=overlap_sec,
        feature_output_dir=Path(feature_output_dir_str),
    )


# ---------------------------------------------------------------------------
# 3.  RUN MODEL  — one patient at a time, fresh worker each call
# ---------------------------------------------------------------------------

def run_model(model, record, data_folder, verbose):
    """
    Run the trained model on a single patient record.

    Processes the patient in a *single-use* ProcessPoolExecutor
    (max_workers=1). The worker exits after each patient, fully freeing
    memory and avoiding the BrokenProcessPool that occurs when too many
    heavy preprocessing jobs are queued at once.
    """
    from helper_code import HEADERS

    patient_id_bids = record[HEADERS['bids_folder']]
    site_id         = record[HEADERS['site_id']]
    session_id      = record[HEADERS['session_id']]

    # ---- Fallback model ---------------------------------------------------
    if model.get("fallback", False):
        return 0, 0.10


    # ---- Patch config in main process (once) ------------------------------
    import tempfile
    tmp_main = tempfile.mkdtemp(prefix="physionet_run_main_")
    _patch_config(data_folder, tmp_main)

    import config
    logger = _setup_logger(verbose)

    # ---- Build patient identifier -----------------------------------------
    record_name         = f"{patient_id_bids}_ses-{session_id}"
    pipeline_patient_id = f"{site_id}/{record_name}"
    patient_dir         = config.PHYSIOLOGICAL_DATA_DIR / site_id

    if not patient_dir.exists():
        if verbose:
            print(f"  ! Patient dir not found: {patient_dir}")
        _cleanup_dir(tmp_main)
        return 0, 0.1

    # ---- Per-patient temp dir for the worker ------------------------------
    tmp_worker_dir = tempfile.mkdtemp(prefix=f"physionet_run_worker_")
    feature_output_dir = Path(tmp_worker_dir) / "features"
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    result = None
    try:
        # Single-use pool: one worker, one patient, then tear down.
        # This mirrors the batch-recycling pattern in train_model().
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _preprocess_one_patient_worker,
                data_folder=data_folder,
                tmp_model_dir=tmp_worker_dir,
                pipeline_patient_id=pipeline_patient_id,
                patient_dir_str=str(patient_dir),
                record_name=record_name,
                segment_length_sec=config.SEGMENT_LENGTH_SEC,
                overlap_sec=config.SEGMENT_OVERLAP_SEC,
                feature_output_dir_str=str(feature_output_dir),
            )
            try:
                result = future.result()
            except Exception as e:
                # Covers BrokenProcessPool and any in-worker exception [[8]]
                if verbose:
                    print(f"  ! Worker failed for {pipeline_patient_id}: "
                          f"{type(e).__name__}: {e}")
                    traceback.print_exc()
                result = None
        # Pool is destroyed here → all worker memory released

        gc.collect()

        if result is None or not result.get("success"):
            if verbose:
                print(f"  ! Pipeline unsuccessful for {pipeline_patient_id}")
            return 0, 0.1

        # ---- Load patient-level features ----------------------------------
        pat_path = result.get("pat_features_path")
        seg_path = result.get("seg_features_path")

        if pat_path and Path(pat_path).exists():
            patient_features = pd.read_parquet(pat_path)
            from feature_table.build_feature_table import add_demographics
            patient_features = add_demographics(
                patient_features,
                demographics_path=config.DEMOGRAPHICS_FILE,
            )
        elif seg_path and Path(seg_path).exists():
            from feature_table.build_feature_table import (
                aggregate_to_patient_level, add_demographics,
            )
            seg_df = pd.read_parquet(seg_path)
            patient_features = aggregate_to_patient_level(
                seg_df, result.get("sleep_summary")
            )
            patient_features = add_demographics(
                patient_features,
                demographics_path=config.DEMOGRAPHICS_FILE,
            )
        else:
            if verbose:
                print(f"  ! No features on disk for {pipeline_patient_id}")
            return 0, 0.1
        
        if patient_features is None or len(patient_features) == 0:
            return 0, 0.1

        # ---- Predict ------------------------------------------------------
        return _predict_single_patient(model, patient_features, verbose)

    except Exception as e:
        if verbose:
            print(f"  !!! run_model error for {pipeline_patient_id}: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
        return 0, 0.1

    finally:
        gc.collect()
        _cleanup_dir(tmp_worker_dir)
        _cleanup_dir(tmp_main)


def _cleanup_dir(path):
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _predict_single_patient(
    model_dict: dict,
    patient_features: pd.DataFrame,
    verbose: bool
) -> Tuple[int, float]:
    """
    Predict one patient using the same preprocessing and prior correction
    used for holdout evaluation.
    """
    pred_df = _predict_feature_table(
        model_dict=model_dict,
        feature_table=patient_features,
        verbose=verbose,
    )

    if pred_df is None or len(pred_df) == 0:
        return 0, 0.10

    binary_output = int(pred_df["prediction"].iloc[0])
    probability_output = float(pred_df["probability"].iloc[0])

    if not np.isfinite(probability_output):
        probability_output = 0.10
        binary_output = 0

    probability_output = float(np.clip(probability_output, 0.0, 1.0))

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
        "training_config", "cv_results", "fallback",
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
