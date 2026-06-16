#!/usr/bin/env python

"""
team_code.py
============

Bridge between the PhysioNet Challenge 2026 interface
(train_model.py / run_model.py) and our custom pipeline.

Required Challenge functions; signatures MUST NOT change:
    train_model(data_folder, model_folder, verbose)
    load_model(model_folder, verbose)
    run_model(model, record, data_folder, verbose)

Optional helper:
    save_model(model_folder, model_dict)

This version is adapted for the updated official PhysioNet 2026 code base:
  - train_model.py still calls train_model(data_folder, model_folder, verbose)
  - run_model.py loads the model once and calls run_model(...) per patient
  - run_model.py now asserts that the binary output is boolean-like or NaN
    and that the probability output is numeric
  - labels are expected in demographics.csv as Cognitive_Impairment, which
    may have been produced by the official create_labels.py script
"""

import gc
import os
import sys
import time
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Constants used without importing helper_code globally.
# helper_code is safe to import globally, but keeping these literals here makes
# early path/label handling independent of challenge helper changes.
# ---------------------------------------------------------------------------

DEMOGRAPHICS_BASENAME = "demographics.csv"
LABEL_COLUMN = "Cognitive_Impairment"
TIME_TO_EVENT_COLUMN = "Time_to_Event"

# ---------------------------------------------------------------------------
# Preprocessing cache / manifest helpers
# ---------------------------------------------------------------------------

PREPROCESS_MANIFEST_FILENAME = "preprocessing_manifest.json"


def _safe_cache_key(s: str) -> str:
    """
    Convert a patient identifier into a filesystem/JSON-friendly key.
    """
    return (
        str(s)
        .replace("\\", "/")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _preprocess_manifest_path(feature_output_dir) -> Path:
    return Path(feature_output_dir) / PREPROCESS_MANIFEST_FILENAME


def _load_preprocess_manifest(feature_output_dir) -> Dict:
    """
    Load preprocessing manifest from feature_output_dir.
    """
    manifest_path = _preprocess_manifest_path(feature_output_dir)

    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    return {}


def _save_preprocess_manifest(feature_output_dir, manifest: Dict):
    """
    Atomically save preprocessing manifest.
    """
    feature_output_dir = Path(feature_output_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = _preprocess_manifest_path(feature_output_dir)
    tmp_path = manifest_path.with_suffix(".json.tmp")

    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)

    os.replace(tmp_path, manifest_path)


def _parquet_file_looks_valid(path) -> bool:
    """
    Check that a cached parquet file exists and is readable.
    This is intentionally conservative: corrupt or empty files are ignored.
    """
    if path is None:
        return False

    try:
        p = Path(path)

        if not p.exists() or not p.is_file():
            return False

        if p.stat().st_size == 0:
            return False

        # Patient-level files are small; segment files are usually manageable.
        # Reading validates that the parquet is not truncated/corrupt.
        df = pd.read_parquet(p)

        return df is not None and len(df) > 0

    except Exception:
        return False


def _normalise_cached_preprocessing_entry(entry: Dict) -> Optional[Dict]:
    """
    Convert a manifest entry into the same result-like dictionary returned by
    process_single_patient(), but only if at least one feature parquet is valid.
    """
    if not isinstance(entry, dict):
        return None

    seg_path = entry.get("seg_features_path")
    pat_path = entry.get("pat_features_path")

    seg_ok = _parquet_file_looks_valid(seg_path)
    pat_ok = _parquet_file_looks_valid(pat_path)

    if not seg_ok and not pat_ok:
        return None

    return {
        "success": True,
        "seg_features_path": str(Path(seg_path)) if seg_ok else None,
        "pat_features_path": str(Path(pat_path)) if pat_ok else None,
        "sleep_summary": entry.get("sleep_summary", None),
        "from_cache": True,
    }


def _find_cached_preprocessing(
    feature_output_dir,
    pipeline_patient_id: str,
    record_name: Optional[str] = None,
    logger=None,
) -> Optional[Dict]:
    """
    Look for cached preprocessing results for one subject.

    Primary method:
      - preprocessing_manifest.json

    Fallback method:
      - simple filename/path heuristic for older preprocessed files created
        before the manifest existed.
    """
    feature_output_dir = Path(feature_output_dir)

    if not feature_output_dir.exists():
        return None

    manifest = _load_preprocess_manifest(feature_output_dir)

    possible_keys = [
        pipeline_patient_id,
        _safe_cache_key(pipeline_patient_id),
    ]

    if record_name:
        possible_keys.append(record_name)
        possible_keys.append(_safe_cache_key(record_name))

    for key in possible_keys:
        if key in manifest:
            cached = _normalise_cached_preprocessing_entry(manifest[key])
            if cached is not None:
                if logger:
                    logger.info(f"Using cached preprocessing for {pipeline_patient_id}")
                return cached

    # ------------------------------------------------------------------
    # Fallback heuristic for existing parquet files without manifest.
    # This is intentionally conservative.
    # ------------------------------------------------------------------
    try:
        patient_token = _safe_cache_key(pipeline_patient_id).lower()
        record_token = _safe_cache_key(record_name).lower() if record_name else None

        parquet_files = list(feature_output_dir.rglob("*.parquet"))

        matches = []
        for p in parquet_files:
            haystack = _safe_cache_key(str(p.relative_to(feature_output_dir))).lower()

            if patient_token in haystack or (record_token and record_token in haystack):
                matches.append(p)

        if not matches:
            return None

        seg_candidates = []
        pat_candidates = []

        for p in matches:
            name = p.name.lower()
            full = str(p).lower()

            if "segment" in name or "segments" in name or "seg" in name:
                seg_candidates.append(p)
            elif "patient" in name or "pat" in name:
                pat_candidates.append(p)
            elif "segment" in full or "segments" in full:
                seg_candidates.append(p)
            elif "patient" in full:
                pat_candidates.append(p)

        seg_path = None
        pat_path = None

        for p in sorted(seg_candidates, key=lambda x: x.stat().st_mtime, reverse=True):
            if _parquet_file_looks_valid(p):
                seg_path = str(p)
                break

        for p in sorted(pat_candidates, key=lambda x: x.stat().st_mtime, reverse=True):
            if _parquet_file_looks_valid(p):
                pat_path = str(p)
                break

        if seg_path is None and pat_path is None:
            return None

        cached = {
            "success": True,
            "seg_features_path": seg_path,
            "pat_features_path": pat_path,
            "sleep_summary": None,
            "from_cache": True,
        }

        if logger:
            logger.info(
                f"Found cached preprocessing by file scan for {pipeline_patient_id}"
            )

        return cached

    except Exception as e:
        if logger:
            logger.debug(f"Cache scan failed for {pipeline_patient_id}: {e}")

    return None


def _remember_preprocessing_result(
    feature_output_dir,
    pipeline_patient_id: str,
    result: Dict,
    record_name: Optional[str] = None,
    logger=None,
):
    """
    Store successful preprocessing paths in preprocessing_manifest.json.
    """
    if not result or not result.get("success"):
        return

    feature_output_dir = Path(feature_output_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    seg_path = result.get("seg_features_path")
    pat_path = result.get("pat_features_path")

    if not seg_path and not pat_path:
        return

    entry = {
        "pipeline_patient_id": pipeline_patient_id,
        "record_name": record_name,
        "seg_features_path": str(Path(seg_path).resolve()) if seg_path else None,
        "pat_features_path": str(Path(pat_path).resolve()) if pat_path else None,
        "sleep_summary": result.get("sleep_summary", None),
        "cached_at_utc": datetime.utcnow().isoformat() + "Z",
    }

    manifest = _load_preprocess_manifest(feature_output_dir)

    manifest[pipeline_patient_id] = entry
    manifest[_safe_cache_key(pipeline_patient_id)] = entry

    if record_name:
        manifest[record_name] = entry
        manifest[_safe_cache_key(record_name)] = entry

    try:
        _save_preprocess_manifest(feature_output_dir, manifest)
    except Exception as e:
        if logger:
            logger.warning(
                f"Could not update preprocessing manifest for "
                f"{pipeline_patient_id}: {e}"
            )


# ---------------------------------------------------------------------------
# 0. Logging
# ---------------------------------------------------------------------------

class _SimpleLogger:
    """Fallback logger if loguru is unavailable."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _print(self, level: str, msg: str):
        if self.verbose:
            print(f"{level:<7} | {msg}")

    def info(self, msg: str):
        self._print("INFO", msg)

    def warning(self, msg: str):
        self._print("WARNING", msg)

    def error(self, msg: str):
        self._print("ERROR", msg)

    def debug(self, msg: str):
        self._print("DEBUG", msg)


def _setup_logger(verbose: bool):
    """
    Configure loguru when available. Otherwise, use a minimal stdout logger.
    """
    try:
        from loguru import logger
        logger.remove()
        if verbose:
            logger.add(
                sys.stdout,
                level="INFO",
                format="{time:HH:mm:ss} | {level:<7} | {message}",
            )
        return logger
    except Exception:
        return _SimpleLogger(verbose=verbose)


# ---------------------------------------------------------------------------
# 1. Config patching
# ---------------------------------------------------------------------------

def _patch_config(data_folder: str, model_folder: str):
    """
    Override config.py path variables so that downstream pipeline modules use
    the Challenge-provided folders instead of hardcoded local paths.

    This function must run before importing custom pipeline modules that read
    config.py at import time.
    """
    import config

    data_path = Path(data_folder).resolve()
    model_path = Path(model_folder).resolve()

    # Training/test data root.
    config.TRAINING_SET_DIR = data_path

    # Challenge subdirectories.
    config.PHYSIOLOGICAL_DATA_DIR = data_path / "physiological_data"
    config.ALGORITHMIC_ANNOTATIONS_DIR = data_path / "algorithmic_annotations"
    config.HUMAN_ANNOTATIONS_DIR = data_path / "human_annotations"
    config.DATA_DIR = config.PHYSIOLOGICAL_DATA_DIR

    # Demographics file.
    config.DEMOGRAPHICS_FILE = data_path / DEMOGRAPHICS_BASENAME

    # Persist all custom-pipeline outputs in model_folder during training.
    # During inference, callers pass a temporary folder here.
    config.OUTPUT_DIR = model_path
    config.FEATURE_DIR = model_path / "features"
    config.MODEL_DIR = model_path / "models"
    config.LOG_DIR = model_path / "logs"
    config.PLOT_DIR = model_path / "plots"

    for d in [
        model_path,
        config.FEATURE_DIR,
        config.MODEL_DIR,
        config.LOG_DIR,
        config.PLOT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Disable plotting in the Challenge environment.
    config.PLOT_ENABLED = False

    # Robust defaults if these are absent in config.py.
    if not hasattr(config, "NUM_WORKERS"):
        config.NUM_WORKERS = max(1, min(os.cpu_count() or 1, 4))
    if not hasattr(config, "SEGMENT_LENGTH_SEC"):
        config.SEGMENT_LENGTH_SEC = 30
    if not hasattr(config, "SEGMENT_OVERLAP_SEC"):
        config.SEGMENT_OVERLAP_SEC = 0
    if not hasattr(config, "RANDOM_SEED"):
        config.RANDOM_SEED = 56
    if not hasattr(config, "TARGET_COLUMN"):
        config.TARGET_COLUMN = LABEL_COLUMN


# ---------------------------------------------------------------------------
# 2. Label handling for updated official create_labels.py compatibility
# ---------------------------------------------------------------------------

def _sanitize_label_value(x):
    """
    Convert common boolean/numeric/string label representations to 0/1.
    Return NaN if the value cannot be interpreted.
    """
    if x is None:
        return np.nan

    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass

    if isinstance(x, (bool, np.bool_)):
        return int(x)

    if isinstance(x, (int, np.integer)):
        if int(x) in (0, 1):
            return int(x)

    if isinstance(x, (float, np.floating)):
        if np.isfinite(float(x)) and float(x) in (0.0, 1.0):
            return int(float(x))

    s = str(x).strip().casefold()
    if s in ("true", "t", "yes", "y", "1", "1.0"):
        return 1
    if s in ("false", "f", "no", "n", "0", "0.0"):
        return 0

    return np.nan


def _maybe_create_labelled_demographics(
    data_folder: str,
    model_folder: str,
    logger,
) -> Path:
    """
    Use data_folder/demographics.csv if it already contains Cognitive_Impairment.

    If labels are missing but an ICD file is available, try to call the official
    create_labels.py function and store the labelled demographics inside
    model_folder. This is optional robustness; official training data should
    already include the labels.
    """
    demographics_path = Path(data_folder) / DEMOGRAPHICS_BASENAME

    if not demographics_path.exists():
        raise FileNotFoundError(f"Missing demographics file: {demographics_path}")

    try:
        demo_head = pd.read_csv(demographics_path, nrows=5)
        if LABEL_COLUMN in demo_head.columns:
            return demographics_path
    except Exception as e:
        logger.warning(f"Could not inspect demographics labels: {e}")
        return demographics_path

    # Try common ICD filename locations.
    candidates = [
        Path(data_folder) / "icd_codes_CI.csv",
        Path(data_folder) / "ICD_codes_CI.csv",
        Path(data_folder) / "icd_codes_ci.csv",
        Path(data_folder).parent / "icd_codes_CI.csv",
        Path(data_folder).parent / "ICD_codes_CI.csv",
    ]
    icd_path = next((p for p in candidates if p.exists()), None)

    if icd_path is None:
        logger.warning(
            f"{LABEL_COLUMN} is not present in {demographics_path}, and no "
            "ICD file was found. Training will continue, but model training "
            "will fall back if labels cannot be found later."
        )
        return demographics_path

    labelled_path = Path(model_folder) / "demographics_with_CI.csv"
    try:
        from create_labels import create_labels

        logger.info(
            f"{LABEL_COLUMN} missing from demographics.csv. Creating labels "
            f"with official create_labels.py using {icd_path}..."
        )
        create_labels(
            str(demographics_path),
            str(icd_path),
            str(labelled_path),
        )
        return labelled_path
    except Exception as e:
        logger.warning(f"Could not create labels with create_labels.py: {e}")
        return demographics_path


def _coerce_target_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Return a copy of df with target_col converted to numeric 0/1/NaN.
    """
    out = df.copy()
    out[target_col] = out[target_col].apply(_sanitize_label_value)
    return out

# ---------------------------------------------------------------------------
# 2b. Official age-specific prevalence handling
# ---------------------------------------------------------------------------

AGE_COLUMN_CANDIDATES = ["Age", "age", "demo_age"]


def _extract_age_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return numeric age Series from common age columns.
    """
    for col in AGE_COLUMN_CANDIDATES:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def _compute_prevalence_reference(
    demographics_path: Path,
    target_col: str = LABEL_COLUMN,
    logger=None,
) -> Dict:
    """
    Store labelled training ages and labels as the prevalence reference.

    This mirrors the official evaluator, which computes prevalence for each
    evaluated age from prevalence-reference patients within +/-2 years.
    """
    info = {
        "global_prevalence": None,
        "prevalence_reference_ages": [],
        "prevalence_reference_labels": [],
        "prevalence_age_gap": 2,
        "age_prevalence_enabled": False,
    }

    try:
        df = pd.read_csv(demographics_path)
    except Exception as e:
        if logger:
            logger.warning(f"Could not read prevalence demographics: {e}")
        return info

    if target_col not in df.columns:
        if logger:
            logger.warning(
                f"{target_col} not found in demographics; "
                "cannot compute prevalence reference."
            )
        return info

    ages = _extract_age_series(df)
    if ages is None:
        if logger:
            logger.warning("Age column not found; cannot compute age prevalence.")
        return info

    labels = df[target_col].apply(_sanitize_label_value)

    valid = labels.notna() & ages.notna()
    if int(valid.sum()) == 0:
        if logger:
            logger.warning("No valid age/label pairs for prevalence reference.")
        return info

    ref_labels = labels.loc[valid].astype(int).to_numpy()
    ref_ages = ages.loc[valid].astype(float).to_numpy()

    info["global_prevalence"] = float(
        np.clip(np.mean(ref_labels), 1e-6, 1.0 - 1e-6)
    )
    info["prevalence_reference_ages"] = [float(x) for x in ref_ages]
    info["prevalence_reference_labels"] = [int(x) for x in ref_labels]
    info["age_prevalence_enabled"] = True

    if logger:
        logger.info(
            f"Prevalence reference: n={len(ref_labels)}, "
            f"global={info['global_prevalence']:.4f}, "
            f"age gap=±{info['prevalence_age_gap']} years"
        )

    return info


def _official_age_prevalence_from_config(
    age,
    training_config: Dict,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """
    Compute age-specific prevalence using the same rule as official
    evaluate_model.compute_prevalence():

        labels within abs(age - prevalence_age) <= gap
        p = max(sum(labels), 0.5) / n
    """
    if fallback is None:
        fallback = training_config.get("expected_test_prevalence", None)

    try:
        age = float(age)
    except Exception:
        return fallback

    if not np.isfinite(age):
        return fallback

    ref_ages = np.asarray(
        training_config.get("prevalence_reference_ages", []),
        dtype=float,
    )
    ref_labels = np.asarray(
        training_config.get("prevalence_reference_labels", []),
        dtype=float,
    )
    gap = float(training_config.get("prevalence_age_gap", 2))

    if len(ref_ages) == 0 or len(ref_labels) == 0 or len(ref_ages) != len(ref_labels):
        return fallback

    mask = np.isfinite(ref_ages) & (np.abs(ref_ages - age) <= gap)
    n = int(mask.sum())

    if n == 0:
        return fallback

    positives = float(np.nansum(ref_labels[mask]))
    prevalence = max(positives, 0.5) / n

    return float(np.clip(prevalence, 1e-6, 1.0 - 1e-6))


def _rowwise_age_prior_shift_and_thresholds(
    raw_probabilities: np.ndarray,
    feature_table: pd.DataFrame,
    training_config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply age-specific prior correction and return age-specific decision
    thresholds.

    For the official reward, if q is the calibrated risk and p is the
    age-specific prevalence, the expected reward is maximized by predicting
    positive when q >= p.
    """
    raw_probabilities = np.asarray(raw_probabilities, dtype=float)

    train_prev = training_config.get("training_prevalence", None)
    global_prev = training_config.get("expected_test_prevalence", None)

    ages = _extract_age_series(feature_table)

    if ages is None:
        if train_prev is not None and global_prev is not None:
            probabilities = _prior_probability_shift(
                raw_probabilities,
                train_prevalence=float(train_prev),
                target_prevalence=float(global_prev),
            )
            thresholds = np.full(
                len(probabilities),
                float(np.clip(global_prev, 1e-6, 1.0 - 1e-6)),
                dtype=float,
            )
            return probabilities, thresholds

        return raw_probabilities, np.full(
            len(raw_probabilities),
            float(training_config.get("decision_threshold", 0.5)),
            dtype=float,
        )

    probabilities = raw_probabilities.copy()
    thresholds = np.full(len(probabilities), 0.5, dtype=float)

    for i in range(len(probabilities)):
        age_i = ages.iloc[i] if i < len(ages) else np.nan

        p_age = _official_age_prevalence_from_config(
            age_i,
            training_config=training_config,
            fallback=global_prev,
        )

        if p_age is None:
            p_age = float(training_config.get("decision_threshold", 0.5))

        p_age = float(np.clip(p_age, 1e-6, 1.0 - 1e-6))
        thresholds[i] = p_age

        if train_prev is not None:
            probabilities[i] = _prior_probability_shift(
                np.asarray([raw_probabilities[i]], dtype=float),
                train_prevalence=float(train_prev),
                target_prevalence=p_age,
            )[0]

    return probabilities, thresholds

def _attach_prevalence_info_to_model(
    model_result: Optional[dict],
    prevalence_info: Dict,
    expected_global_prevalence: float,
) -> Optional[dict]:
    """
    Attach official-style prevalence metadata to a model dictionary.

    This is needed both for:
      - the internal holdout model, so holdout evaluation uses the same
        age-specific reward-aware thresholding as inference; and
      - the final model saved for Challenge inference.
    """
    if model_result is None:
        return model_result

    model_result.setdefault("training_config", {})
    tc = model_result["training_config"]

    tc["expected_test_prevalence"] = float(expected_global_prevalence)
    tc["expected_test_prevalence_source"] = (
        "training_demographics_global_prevalence"
    )
    tc["age_prevalence_enabled"] = bool(
        prevalence_info.get("age_prevalence_enabled", False)
    )
    tc["prevalence_reference_ages"] = prevalence_info.get(
        "prevalence_reference_ages", []
    )
    tc["prevalence_reference_labels"] = prevalence_info.get(
        "prevalence_reference_labels", []
    )
    tc["prevalence_age_gap"] = int(
        prevalence_info.get("prevalence_age_gap", 2)
    )
    tc["decision_threshold_strategy"] = "age_specific_official_prevalence"

    return model_result
def _compute_official_style_reward(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_table: pd.DataFrame,
    training_config: Dict,
) -> float:
    """
    Compute the official-style age-specific reward for internal validation.

    For each patient:
      p_age = prevalence among reference patients within +/- prevalence_age_gap
      TP reward = 1 / p_age - 1
      FP reward = -1
      FN reward = -1
      TN reward = 1 / (1 - p_age) - 1
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    ages = _extract_age_series(feature_table)
    if ages is None:
        return np.nan

    rewards = []

    global_prev = training_config.get("expected_test_prevalence", None)

    for i in range(len(y_true)):
        age_i = ages.iloc[i] if i < len(ages) else np.nan

        p = _official_age_prevalence_from_config(
            age_i,
            training_config=training_config,
            fallback=global_prev,
        )

        if p is None or not np.isfinite(float(p)):
            continue

        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

        if y_true[i] == 1 and y_pred[i] == 1:
            reward = 1.0 / p - 1.0
        elif y_true[i] == 0 and y_pred[i] == 0:
            reward = 1.0 / (1.0 - p) - 1.0
        else:
            reward = -1.0

        rewards.append(reward)

    if len(rewards) == 0:
        return np.nan

    return float(np.mean(rewards))

# ---------------------------------------------------------------------------
# 3. Probability and prediction helpers
# ---------------------------------------------------------------------------

def _prior_probability_shift(
    p: np.ndarray,
    train_prevalence: float,
    target_prevalence: Optional[float],
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Adjust predicted probabilities from the training prior to an expected
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
    Return P(class=1) robustly, including degenerate/dummy models.
    """
    if not hasattr(ml_model, "predict_proba"):
        pred = ml_model.predict(X_scaled)
        return np.asarray(pred, dtype=float)

    proba = ml_model.predict_proba(X_scaled)
    proba = np.asarray(proba)

    if proba.ndim == 1:
        return proba.astype(float)

    if proba.shape[1] == 1:
        classes = getattr(ml_model, "classes_", np.array([0]))
        only_class = int(classes[0]) if len(classes) > 0 else 0
        if only_class == 1:
            return proba[:, 0].astype(float)
        return np.zeros(proba.shape[0], dtype=float)

    classes = getattr(ml_model, "classes_", np.array([0, 1]))
    classes = np.asarray(classes)

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
    Apply the saved preprocessing chain:

        raw feature table
        -> column alignment
        -> imputer
        -> feature selector
        -> scaler
        -> model probabilities
        -> optional prior-probability correction
        -> binary threshold

    Used for both internal holdout evaluation and Challenge inference.
    """
    if model_dict.get("fallback", False):
        patient_ids = (
            feature_table["patient_id"].values
            if "patient_id" in feature_table.columns
            else np.arange(len(feature_table))
        )
        return pd.DataFrame(
            {
                "patient_id": patient_ids,
                "prediction": np.zeros(len(feature_table), dtype=int),
                "probability": np.full(len(feature_table), 0.10, dtype=float),
            }
        )

    required_keys = ["model", "scaler", "imputer", "feature_names"]
    if any(k not in model_dict for k in required_keys):
        if verbose:
            print(
                "  ! Loaded model does not contain the expected custom "
                "pipeline keys. Using fallback prediction."
            )
        patient_ids = (
            feature_table["patient_id"].values
            if "patient_id" in feature_table.columns
            else np.arange(len(feature_table))
        )
        return pd.DataFrame(
            {
                "patient_id": patient_ids,
                "prediction": np.zeros(len(feature_table), dtype=int),
                "probability": np.full(len(feature_table), 0.10, dtype=float),
            }
        )

    ml_model = model_dict["model"]
    scaler = model_dict["scaler"]
    imputer = model_dict["imputer"]
    feature_selector = model_dict.get("feature_selector")
    feature_names = list(model_dict["feature_names"])
    selected_features = model_dict.get("selected_features", feature_names)

    df = feature_table.copy()

    patient_ids = (
        df["patient_id"].values if "patient_id" in df.columns else np.arange(len(df))
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

    # Prior correction and binary decision.
    # Prefer the official age-specific prevalence reference if available.
    training_config = model_dict.get("training_config", {})

    if training_config.get("age_prevalence_enabled", False):
        probabilities, thresholds = _rowwise_age_prior_shift_and_thresholds(
            raw_probabilities=raw_probabilities,
            feature_table=df,
            training_config=training_config,
        )
    else:
        train_prev = training_config.get("training_prevalence", None)
        target_prev = training_config.get("expected_test_prevalence", None)

        if train_prev is not None and target_prev is not None:
            probabilities = _prior_probability_shift(
                raw_probabilities,
                train_prevalence=float(train_prev),
                target_prevalence=float(target_prev),
            )
            thresholds = np.full(
                len(probabilities),
                float(np.clip(target_prev, 1e-6, 1.0 - 1e-6)),
                dtype=float,
            )
        else:
            probabilities = raw_probabilities
            thresholds = np.full(
                len(probabilities),
                float(training_config.get("decision_threshold", 0.5)),
                dtype=float,
            )

    probabilities = np.clip(probabilities, 0.0, 1.0)
    predictions = (probabilities >= thresholds).astype(int)


    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "prediction": predictions.astype(int),
            "probability": probabilities.astype(float),
        }
    )


# ---------------------------------------------------------------------------
# 4. TRAIN MODEL
# ---------------------------------------------------------------------------

def train_model(data_folder, model_folder, verbose):
    """
    Train the model on all labelled patients found in data_folder.
    Save everything needed for inference into model_folder.
    """
    # Patch paths before importing custom pipeline modules.
    _patch_config(data_folder, model_folder)
    logger = _setup_logger(verbose)

    import config
    from helper_code import find_patients, DEMOGRAPHICS_FILE, HEADERS

    # If official create_labels.py has not yet been run, optionally create a
    # labelled demographics copy in model_folder when an ICD file is available.
    labelled_demographics = _maybe_create_labelled_demographics(
        data_folder=data_folder,
        model_folder=model_folder,
        logger=logger,
    )

    # IMPORTANT:
    # If create_labels.py created a labelled demographics copy, all downstream
    # code must use that file. The previous version computed prevalence and
    # built the feature table from config.DEMOGRAPHICS_FILE, which still pointed
    # to the original demographics.csv.
    config.DEMOGRAPHICS_FILE = Path(labelled_demographics).resolve()

    prevalence_info = _compute_prevalence_reference(
        demographics_path=config.DEMOGRAPHICS_FILE,
        target_col=LABEL_COLUMN,
        logger=logger,
    )



    expected_global_prevalence = prevalence_info.get("global_prevalence", None)

    if expected_global_prevalence is None:
        expected_global_prevalence = 0.10
        logger.warning(
            "Could not compute training prevalence from demographics. "
            "Falling back to 0.10."
        )


    # Import custom pipeline modules only after config patching.
    from main import process_single_patient
    from feature_table.build_feature_table import build_cohort_feature_table
    from classification.train_model import train_model as classification_train_model

    logger.info("Finding the Challenge data...")

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError("No data were provided.")

    logger.info(f"Found {num_records} records. Starting preprocessing...")

        # -----------------------------------------------------------------------
    # 4.1 Run preprocessing pipeline, with cache/skip support.
    # -----------------------------------------------------------------------
    feature_dir = Path(config.FEATURE_DIR)
    feature_dir.mkdir(parents=True, exist_ok=True)

    successful_results = {}
    records_to_process = []
    cached_subjects = []

    segment_length_sec = getattr(config, "SEGMENT_LENGTH_SEC", 30)
    overlap_sec = getattr(config, "SEGMENT_OVERLAP_SEC", 0)

    logger.info("Checking preprocessing cache for training subjects...")

    for i, record in enumerate(patient_metadata_list):
        patient_id_bids = record[HEADERS["bids_folder"]]
        site_id = record[HEADERS["site_id"]]
        session_id = record[HEADERS["session_id"]]

        record_name = f"{patient_id_bids}_ses-{session_id}"
        pipeline_patient_id = f"{site_id}/{record_name}"
        patient_dir = Path(config.PHYSIOLOGICAL_DATA_DIR) / site_id

        cached = _find_cached_preprocessing(
            feature_output_dir=feature_dir,
            pipeline_patient_id=pipeline_patient_id,
            record_name=record_name,
            logger=logger if verbose else None,
        )

        if cached is not None:
            successful_results[pipeline_patient_id] = {
                "seg_features_path": cached.get("seg_features_path"),
                "pat_features_path": cached.get("pat_features_path"),
                "sleep_summary": cached.get("sleep_summary"),
            }
            cached_subjects.append(pipeline_patient_id)
            continue

        records_to_process.append(
            {
                "index": i,
                "record": record,
                "patient_id_bids": patient_id_bids,
                "site_id": site_id,
                "session_id": session_id,
                "record_name": record_name,
                "pipeline_patient_id": pipeline_patient_id,
                "patient_dir": patient_dir,
            }
        )

    logger.info(
        f"Preprocessing cache status: "
        f"{len(cached_subjects)}/{num_records} already preprocessed, "
        f"{len(records_to_process)} still need preprocessing."
    )

    if verbose and records_to_process:
        pending_preview = [
            x["pipeline_patient_id"] for x in records_to_process[:50]
        ]
        logger.info(
            "Subjects still needing preprocessing"
            + (" first 50" if len(records_to_process) > 50 else "")
            + ": "
            + ", ".join(pending_preview)
        )

    if verbose and cached_subjects:
        cached_preview = cached_subjects[:50]
        logger.info(
            "Subjects skipped because cached"
            + (" first 50" if len(cached_subjects) > 50 else "")
            + ": "
            + ", ".join(cached_preview)
        )

    max_workers = min(int(getattr(config, "NUM_WORKERS", 1)), max(1, len(records_to_process)))
    max_workers = max(1, max_workers)

    total_start = time.time()
    batch_size = max_workers

    for batch_start in range(0, len(records_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(records_to_process))
        batch = records_to_process[batch_start:batch_end]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for item in batch:
                pipeline_patient_id = item["pipeline_patient_id"]
                patient_dir = item["patient_dir"]
                record_name = item["record_name"]

                if not patient_dir.exists():
                    logger.warning(f"Directory not found: {patient_dir}. Skipping.")
                    continue

                future = executor.submit(
                    process_single_patient,
                    patient_id=pipeline_patient_id,
                    patient_dir=patient_dir,
                    record_name=record_name,
                    segment_length_sec=segment_length_sec,
                    overlap_sec=overlap_sec,
                    feature_output_dir=feature_dir,
                )

                futures[future] = item

            for future in as_completed(futures):
                item = futures[future]
                pid = item["pipeline_patient_id"]
                record_name = item["record_name"]

                try:
                    result = future.result()

                    if result and result.get("success"):
                        successful_results[pid] = {
                            "seg_features_path": result.get("seg_features_path"),
                            "pat_features_path": result.get("pat_features_path"),
                            "sleep_summary": result.get("sleep_summary"),
                        }

                        _remember_preprocessing_result(
                            feature_output_dir=feature_dir,
                            pipeline_patient_id=pid,
                            record_name=record_name,
                            result=result,
                            logger=logger if verbose else None,
                        )

                    else:
                        logger.warning(f"Preprocessing failed for {pid}")

                except Exception as e:
                    logger.error(f"Error processing {pid}: {type(e).__name__}: {e}")

        # Recycle process pool and release memory after each batch.
        gc.collect()

        elapsed = time.time() - total_start
        processed_now = batch_end
        logger.info(
            f"Batch progress: {processed_now}/{len(records_to_process)} newly "
            f"processed subjects, {len(cached_subjects)} cached, "
            f"{len(successful_results)} usable total, "
            f"{elapsed:.0f}s elapsed."
        )

    logger.info(
        f"Preprocessing done: {len(successful_results)}/{num_records} usable "
        f"subjects. Skipped cached={len(cached_subjects)}, "
        f"newly processed={len(records_to_process)}. "
        f"Elapsed={time.time() - total_start:.0f}s"
    )


    logger.info(
        f"Preprocessing done: {len(successful_results)}/{num_records} "
        f"successful in {time.time() - total_start:.0f}s"
    )

    # -----------------------------------------------------------------------
    # 4.2 Load saved per-patient features and build cohort table.
    # -----------------------------------------------------------------------
    patient_segment_tables = {}
    patient_level_tables = {}
    patient_sleep_summaries = {}

    for pid, paths in successful_results.items():
        seg_path = paths.get("seg_features_path")
        if seg_path and Path(seg_path).exists():
            try:
                patient_segment_tables[pid] = pd.read_parquet(seg_path)
            except Exception as e:
                logger.warning(f"Failed to load segment features {seg_path}: {e}")

        pat_path = paths.get("pat_features_path")
        if pat_path and Path(pat_path).exists():
            try:
                patient_level_tables[pid] = pd.read_parquet(pat_path)
            except Exception as e:
                logger.warning(f"Failed to load patient features {pat_path}: {e}")

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

    # -----------------------------------------------------------------------
    # 4.3 Prepare labels.
    # -----------------------------------------------------------------------
    target_candidates = [
        "target",
        getattr(config, "TARGET_COLUMN", LABEL_COLUMN),
        LABEL_COLUMN,
    ]

    target_col = next((c for c in target_candidates if c in patient_level.columns), None)

    if target_col is None:
        logger.error(
            f"No target column found. Expected one of {target_candidates}. "
            "Saving fallback model."
        )
        _save_fallback_model(model_folder)
        return

    patient_level = _coerce_target_column(patient_level, target_col)
    patient_level_valid = patient_level.loc[
        patient_level[target_col].notna()
    ].reset_index(drop=True)

    if len(patient_level_valid) == 0:
        logger.error("No valid labels after sanitization. Saving fallback model.")
        _save_fallback_model(model_folder)
        return

    unique_labels = sorted(patient_level_valid[target_col].dropna().unique().tolist())
    if len(unique_labels) < 2:
        logger.error(
            f"Only one class present in training labels: {unique_labels}. "
            "Saving fallback model."
        )
        _save_fallback_model(model_folder)
        return

    patient_level_valid[target_col] = patient_level_valid[target_col].astype(int)

    logger.info(
        f"Training data after label filtering: {len(patient_level_valid)} patients, "
        f"prevalence={patient_level_valid[target_col].mean():.3f}, "
        f"{len(patient_level_valid.columns)} columns."
    )

    # -----------------------------------------------------------------------
    # 4.4 Optional honest internal holdout, then final refit on all data.
    # -----------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        f1_score,
        balanced_accuracy_score,
    )

    class_counts = patient_level_valid[target_col].value_counts()
    can_holdout = (
        len(patient_level_valid) >= 10
        and len(class_counts) == 2
        and int(class_counts.min()) >= 2
    )

    model_result = None

    if can_holdout:
        try:
            train_df, holdout_df = train_test_split(
                patient_level_valid,
                test_size=0.20,
                stratify=patient_level_valid[target_col].astype(int),
                random_state=int(getattr(config, "RANDOM_SEED", 56)),
            )

            logger.info(
                f"Internal split: train={len(train_df)} "
                f"(prev={train_df[target_col].mean():.3f}), "
                f"holdout={len(holdout_df)} "
                f"(prev={holdout_df[target_col].mean():.3f})"
            )

            model_result = classification_train_model(
                feature_table=train_df,
                output_dir=config.MODEL_DIR,
                expected_test_prevalence=expected_global_prevalence,
                logger=logger if verbose else None,
            )

            model_result = _attach_prevalence_info_to_model(
                model_result=model_result,
                prevalence_info=prevalence_info,
                expected_global_prevalence=expected_global_prevalence,
            )


            if model_result:
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

                    holdout_reward = _compute_official_style_reward(
                        y_true=y_hold,
                        y_pred=binary_hold,
                        feature_table=holdout_df,
                        training_config=model_result.get("training_config", {}),
                    )

                    cv_auroc = model_result.get("cv_results", {}).get(
                        "auroc_mean", np.nan
                    )
                    gap = (
                        float(cv_auroc) - float(holdout_auroc)
                        if np.isfinite(cv_auroc)
                        else np.nan
                    )

                    logger.info("=" * 60)
                    logger.info("HONEST HOLDOUT EVALUATION")
                    if np.isfinite(cv_auroc):
                        logger.info(f"  CV AUROC:       {cv_auroc:.4f}")
                    else:
                        logger.info("  CV AUROC:       N/A")
                    logger.info(f"  Holdout AUROC:  {holdout_auroc:.4f}")
                    logger.info(f"  Holdout AUPRC:  {holdout_auprc:.4f}")
                    logger.info(f"  Holdout Acc:    {holdout_accuracy:.4f}")
                    logger.info(f"  Holdout F1:     {holdout_f1:.4f}")
                    logger.info(f"  Holdout BalAcc: {holdout_bal_acc:.4f}")

                    if np.isfinite(holdout_reward):
                        logger.info(f"  Holdout Reward: {holdout_reward:.4f}")
                    else:
                        logger.info("  Holdout Reward: N/A")


                    if np.isfinite(gap):
                        logger.info(
                            f"  CV-Holdout gap: {gap:+.4f}  "
                            f"{'overfitting?' if gap > 0.05 else 'healthy'}"
                        )
                    logger.info("=" * 60)

                    model_result.setdefault("cv_results", {}).update(
                        {
                            "holdout_auroc": float(holdout_auroc),
                            "holdout_auprc": float(holdout_auprc),
                            "holdout_accuracy": float(holdout_accuracy),
                            "holdout_f1": float(holdout_f1),
                            "holdout_balanced_accuracy": float(holdout_bal_acc),
                            "cv_holdout_gap": (
                                float(gap) if np.isfinite(gap) else np.nan
                            ),
                            "holdout_reward": (
                                float(holdout_reward)
                                if np.isfinite(holdout_reward)
                                else np.nan
                            ),

                        }
                    )

        except Exception as e:
            logger.warning(
                f"Internal holdout training/evaluation failed: "
                f"{type(e).__name__}: {e}. Continuing with full-data fit."
            )
            model_result = None
    else:
        logger.info(
            "Skipping internal holdout because the dataset/class counts are too small."
        )

    # Final model for Challenge submission: train on all valid labelled data.
    logger.info("Fitting final model on 100% of valid labelled training data...")

    try:
        final_result = classification_train_model(
            feature_table=patient_level_valid,
            output_dir=config.MODEL_DIR,
            expected_test_prevalence=expected_global_prevalence,
            logger=logger if verbose else None,
        )
    except Exception as e:
        logger.error(f"Final training failed: {type(e).__name__}: {e}")
        final_result = None

    if final_result:
        # Preserve honest holdout diagnostics from pre-refit model.
        if model_result and model_result.get("cv_results"):
            holdout_keys = [
                "holdout_auroc",
                "holdout_auprc",
                "holdout_accuracy",
                "holdout_f1",
                "holdout_balanced_accuracy",
                "holdout_reward",
                "cv_holdout_gap",
            ]

            final_result.setdefault("cv_results", {}).update(
                {
                    k: model_result.get("cv_results", {}).get(k)
                    for k in holdout_keys
                    if k in model_result.get("cv_results", {})
                }
            )

        model_result = final_result

    if not model_result:
        logger.error("Training failed. Saving fallback model.")
        _save_fallback_model(model_folder)
        return

    # -----------------------------------------------------------------------
    # 4.5 Save final model.
    # -----------------------------------------------------------------------
    # Store official-style prevalence reference for inference-time calibration
    # and prevalence-aware binary decisions.
    model_result = _attach_prevalence_info_to_model(
        model_result=model_result,
        prevalence_info=prevalence_info,
        expected_global_prevalence=expected_global_prevalence,
    )



    save_model(model_folder, model_result)

    cv = model_result.get("cv_results", {})
    logger.info(f"Training complete. CV AUROC={cv.get('auroc_mean', 'N/A')}")
    logger.info("Done.")


# ---------------------------------------------------------------------------
# 5. LOAD MODEL
# ---------------------------------------------------------------------------

def load_model(model_folder, verbose):
    """
    Load the trained model from model_folder/model.sav.
    """
    model_path = os.path.join(model_folder, "model.sav")

    if not os.path.exists(model_path):
        if verbose:
            print(f"WARNING: {model_path} not found. Using fallback.")
        return {"fallback": True}

    try:
        model_dict = joblib.load(model_path)
    except Exception as e:
        if verbose:
            print(f"WARNING: Could not load {model_path}: {e}. Using fallback.")
        return {"fallback": True}

    if not isinstance(model_dict, dict):
        if verbose:
            print("WARNING: Loaded model is not a dictionary. Using fallback.")
        return {"fallback": True}
    
    # Runtime-only metadata. Do not save this inside model.sav.
    model_dict["__model_folder"] = os.path.abspath(model_folder)


    if verbose:
        mt = model_dict.get("training_config", {}).get("model_type", "unknown")
        nf = len(model_dict.get("selected_features", []))
        fb = model_dict.get("fallback", False)
        print(f"Model loaded: type={mt}, features={nf}, fallback={fb}")

    return model_dict


# ---------------------------------------------------------------------------
# 6. Inference worker
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
    Run process_single_patient in a fresh child process.

    The child exits after each patient, so memory held by MNE/YASA/numpy arrays
    is reclaimed by the OS before the next patient.
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


def _manual_add_demographics(
    patient_features: pd.DataFrame,
    demographics_path: Path,
) -> pd.DataFrame:
    """
    Fallback demographic merge if the custom add_demographics() function fails,
    e.g., because hidden holdout demographics do not contain labels.

    It creates a pipeline-compatible patient_id:
        SiteID / (BidsFolder_ses-SessionID)
    and merges demographics columns onto patient_features.
    """
    from helper_code import HEADERS

    if patient_features is None or len(patient_features) == 0:
        return patient_features

    demo = pd.read_csv(demographics_path)

    required = [
        HEADERS["site_id"],
        HEADERS["bids_folder"],
        HEADERS["session_id"],
    ]
    if any(c not in demo.columns for c in required):
        return patient_features

    demo = demo.copy()
    demo["patient_id"] = (
        demo[HEADERS["site_id"]].astype(str)
        + "/"
        + demo[HEADERS["bids_folder"]].astype(str)
        + "_ses-"
        + demo[HEADERS["session_id"]].astype(str)
    )

    if "patient_id" not in patient_features.columns:
        return patient_features

    # Avoid duplicating existing columns except patient_id.
    merge_cols = ["patient_id"] + [
        c for c in demo.columns if c != "patient_id" and c not in patient_features.columns
    ]

    return patient_features.merge(demo[merge_cols], on="patient_id", how="left")


def _add_demographics_safely(
    patient_features: pd.DataFrame,
    demographics_path: Path,
    verbose: bool,
) -> pd.DataFrame:
    """
    Try the custom add_demographics() first; if it fails, use a conservative
    manual merge that does not require labels.
    """
    try:
        from feature_table.build_feature_table import add_demographics

        return add_demographics(
            patient_features,
            demographics_path=demographics_path,
        )
    except Exception as e:
        if verbose:
            print(
                f"  ! add_demographics() failed "
                f"({type(e).__name__}: {e}); using manual merge."
            )

        try:
            return _manual_add_demographics(patient_features, demographics_path)
        except Exception as e2:
            if verbose:
                print(
                    f"  ! Manual demographics merge also failed "
                    f"({type(e2).__name__}: {e2})."
                )
            return patient_features


# ---------------------------------------------------------------------------
# 7. RUN MODEL
# ---------------------------------------------------------------------------

def run_model(model, record, data_folder, verbose):
    """
    Run the trained model on a single patient record.

    The official run_model.py calls this once per patient and expects:
        binary_output, probability_output

    binary_output must be boolean-like, i.e., 0/1 or False/True.
    probability_output must be numeric.
    """
    from helper_code import HEADERS

    patient_id_bids = record[HEADERS["bids_folder"]]
    site_id = record[HEADERS["site_id"]]
    session_id = record[HEADERS["session_id"]]

    # Fallback model.
    if model is None or model.get("fallback", False):
        return 0, 0.10

    tmp_main = None
    tmp_worker_dir = None

    try:
        # Patch config in main process.
        tmp_main = tempfile.mkdtemp(prefix="physionet_run_main_")
        _patch_config(data_folder, tmp_main)

        import config

        logger = _setup_logger(verbose)

        record_name = f"{patient_id_bids}_ses-{session_id}"
        pipeline_patient_id = f"{site_id}/{record_name}"
        patient_dir = Path(config.PHYSIOLOGICAL_DATA_DIR) / site_id

        if not patient_dir.exists():
            if verbose:
                print(f"  ! Patient dir not found: {patient_dir}")
            return 0, 0.10

        tmp_worker_dir = tempfile.mkdtemp(prefix="physionet_run_worker_")

        # Best-effort persistent test cache.
        # If model_folder is writable, cached test features survive repeated
        # local run_model.py executions. If not, fall back to a temp cache.
        runtime_model_folder = model.get("__model_folder", None)

        if runtime_model_folder is not None:
            try:
                feature_output_dir = (
                    Path(runtime_model_folder)
                    / "preprocessed_test_cache"
                    / "features"
                )
                feature_output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                feature_output_dir = Path(tmp_worker_dir) / "features"
                feature_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            feature_output_dir = Path(tmp_worker_dir) / "features"
            feature_output_dir.mkdir(parents=True, exist_ok=True)

        result = _find_cached_preprocessing(
            feature_output_dir=feature_output_dir,
            pipeline_patient_id=pipeline_patient_id,
            record_name=record_name,
            logger=logger if verbose else None,
        )

        if result is not None:
            if verbose:
                print(f"  - Using cached preprocessing for {pipeline_patient_id}")

        else:
            if verbose:
                print(f"  - No cached preprocessing for {pipeline_patient_id}; processing...")

            # Single-use pool: one worker, one patient, then tear down.
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _preprocess_one_patient_worker,
                    data_folder=data_folder,
                    tmp_model_dir=tmp_worker_dir,
                    pipeline_patient_id=pipeline_patient_id,
                    patient_dir_str=str(patient_dir),
                    record_name=record_name,
                    segment_length_sec=getattr(config, "SEGMENT_LENGTH_SEC", 30),
                    overlap_sec=getattr(config, "SEGMENT_OVERLAP_SEC", 0),
                    feature_output_dir_str=str(feature_output_dir),
                )

                try:
                    result = future.result()

                    if result and result.get("success"):
                        _remember_preprocessing_result(
                            feature_output_dir=feature_output_dir,
                            pipeline_patient_id=pipeline_patient_id,
                            record_name=record_name,
                            result=result,
                            logger=logger if verbose else None,
                        )

                except Exception as e:
                    if verbose:
                        print(
                            f"  ! Worker failed for {pipeline_patient_id}: "
                            f"{type(e).__name__}: {e}"
                        )
                        traceback.print_exc()
                    result = None

        gc.collect()


        if result is None or not result.get("success"):
            if verbose:
                print(f"  ! Pipeline unsuccessful for {pipeline_patient_id}")
            return 0, 0.10

        # Load patient-level features.
        pat_path = result.get("pat_features_path")
        seg_path = result.get("seg_features_path")

        if pat_path and Path(pat_path).exists():
            patient_features = pd.read_parquet(pat_path)
            patient_features = _add_demographics_safely(
                patient_features,
                demographics_path=Path(config.DEMOGRAPHICS_FILE),
                verbose=verbose,
            )

        elif seg_path and Path(seg_path).exists():
            from feature_table.build_feature_table import aggregate_to_patient_level

            seg_df = pd.read_parquet(seg_path)
            patient_features = aggregate_to_patient_level(
                seg_df,
                result.get("sleep_summary"),
            )
            patient_features = _add_demographics_safely(
                patient_features,
                demographics_path=Path(config.DEMOGRAPHICS_FILE),
                verbose=verbose,
            )

        else:
            if verbose:
                print(f"  ! No features on disk for {pipeline_patient_id}")
            return 0, 0.10

        if patient_features is None or len(patient_features) == 0:
            return 0, 0.10

        return _predict_single_patient(model, patient_features, verbose)

    except Exception as e:
        if verbose:
            print(
                f"  !!! run_model error for {patient_id_bids}: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
        return 0, 0.10

    finally:
        gc.collect()

        if tmp_worker_dir is not None:
            _cleanup_dir(tmp_worker_dir)

        if tmp_main is not None:
            _cleanup_dir(tmp_main)


def _predict_single_patient(
    model_dict: dict,
    patient_features: pd.DataFrame,
    verbose: bool,
) -> Tuple[int, float]:
    """
    Predict one patient using the same preprocessing and prior correction used
    during internal holdout evaluation.
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
    binary_output = int(1 if binary_output else 0)

    return binary_output, probability_output


# ---------------------------------------------------------------------------
# 8. SAVE MODEL
# ---------------------------------------------------------------------------

def save_model(model_folder, model_dict):
    """
    Save the full model dictionary into model_folder/model.sav.
    """
    os.makedirs(model_folder, exist_ok=True)
    filename = os.path.join(model_folder, "model.sav")

    keys_to_save = [
        "model",
        "scaler",
        "imputer",
        "feature_selector",
        "feature_names",
        "selected_features",
        "training_config",
        "cv_results",
        "fallback",
    ]

    to_save = {}
    for k in keys_to_save:
        if k in model_dict:
            to_save[k] = model_dict[k]

    joblib.dump(to_save, filename, protocol=2)


def _save_fallback_model(model_folder):
    """
    Save a trivial fallback model that always predicts 0 / 0.10.
    Used when preprocessing or training fails.
    """
    os.makedirs(model_folder, exist_ok=True)

    model_dict = {
        "feature_names": ["dummy_feature"],
        "selected_features": ["dummy_feature"],
        "training_config": {
            "model_type": "fallback",
            "decision_threshold": 0.5,
            "expected_test_prevalence": None,
            "expected_test_prevalence_source": "fallback",
            "age_prevalence_enabled": False,
            "prevalence_reference_ages": [],
            "prevalence_reference_labels": [],
            "prevalence_age_gap": 2,
            "decision_threshold_strategy": "fallback",
        },

        "cv_results": {},
        "fallback": True,
    }

    save_model(model_folder, model_dict)


def _cleanup_dir(path):
    """
    Best-effort recursive directory cleanup.
    """
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
