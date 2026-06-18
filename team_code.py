#!/usr/bin/env python

"""
team_code.py
============

Bridge between the PhysioNet Challenge 2026 interface
(train_model.py / run_model.py) and the custom pipeline.

Required Challenge functions; signatures MUST NOT change:
    train_model(data_folder, model_folder, verbose)
    load_model(model_folder, verbose)
    run_model(model, record, data_folder, verbose)
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


DEMOGRAPHICS_BASENAME = "demographics.csv"
LABEL_COLUMN = "Cognitive_Impairment"
TIME_TO_EVENT_COLUMN = "Time_to_Event"

PREPROCESS_MANIFEST_FILENAME = "preprocessing_manifest.json"
HOLDOUT_PREPROCESS_MARKER_FILENAME = "holdout_preprocessing_complete.json"


# =============================================================================
# Cache helpers
# =============================================================================

def _safe_cache_key(s: str) -> str:
    return (
        str(s)
        .replace("\\", "/")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _legacy_single_underscore_cache_key(s: str) -> str:
    return (
        str(s)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _preprocess_manifest_path(feature_output_dir) -> Path:
    return Path(feature_output_dir) / PREPROCESS_MANIFEST_FILENAME


def _load_preprocess_manifest(feature_output_dir) -> Dict:
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
    feature_output_dir = Path(feature_output_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = _preprocess_manifest_path(feature_output_dir)
    tmp_path = manifest_path.with_suffix(".json.tmp")

    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)

    os.replace(tmp_path, manifest_path)


def _parquet_file_looks_valid(path) -> bool:
    if path is None:
        return False

    try:
        p = Path(path)

        if not p.exists() or not p.is_file():
            return False

        if p.stat().st_size == 0:
            return False

        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(p)
            md = pf.metadata

            return md is not None and md.num_rows > 0
        except Exception:
            return True

    except Exception:
        return False


def _normalise_cached_preprocessing_entry(entry: Dict) -> Optional[Dict]:
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

    This version intentionally searches only inside the fixed cache directory:

        <model_folder>/preprocessed_test_cache/features

    It also supports older filenames created by process_single_patient(), which
    use single underscores, e.g.:

        S0001_sub-xxx_ses-1_patient_features.parquet

    while the manifest-safe cache key uses double underscores for slashes.
    """
    feature_output_dir = Path(feature_output_dir)

    if not feature_output_dir.exists():
        return None

    manifest = _load_preprocess_manifest(feature_output_dir)

    possible_keys = [
        pipeline_patient_id,
        _safe_cache_key(pipeline_patient_id),
        _legacy_single_underscore_cache_key(pipeline_patient_id),
    ]

    if record_name:
        possible_keys.extend(
            [
                record_name,
                _safe_cache_key(record_name),
                _legacy_single_underscore_cache_key(record_name),
            ]
        )

    for key in possible_keys:
        if key in manifest:
            cached = _normalise_cached_preprocessing_entry(manifest[key])
            if cached is not None:
                if logger:
                    logger.info(f"Using cached preprocessing for {pipeline_patient_id}")
                return cached

    # Fallback scan for old parquet files without a manifest.
    try:
        tokens = set()

        tokens.add(_safe_cache_key(pipeline_patient_id).lower())
        tokens.add(_legacy_single_underscore_cache_key(pipeline_patient_id).lower())
        tokens.add(str(pipeline_patient_id).replace("\\", "/").lower())

        if record_name:
            tokens.add(_safe_cache_key(record_name).lower())
            tokens.add(_legacy_single_underscore_cache_key(record_name).lower())
            tokens.add(str(record_name).lower())

        tokens = {t for t in tokens if t}

        parquet_files = list(feature_output_dir.rglob("*.parquet"))

        matches = []

        for p in parquet_files:
            rel = str(p.relative_to(feature_output_dir))
            haystack_raw = rel.replace("\\", "/").lower()
            haystack_safe = _safe_cache_key(rel).lower()
            haystack_single = _legacy_single_underscore_cache_key(rel).lower()

            if any(
                token in haystack_raw
                or token in haystack_safe
                or token in haystack_single
                for token in tokens
            ):
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

    keys = [
        pipeline_patient_id,
        _safe_cache_key(pipeline_patient_id),
        _legacy_single_underscore_cache_key(pipeline_patient_id),
    ]

    if record_name:
        keys.extend(
            [
                record_name,
                _safe_cache_key(record_name),
                _legacy_single_underscore_cache_key(record_name),
            ]
        )

    for key in keys:
        manifest[key] = entry

    try:
        _save_preprocess_manifest(feature_output_dir, manifest)
    except Exception as e:
        if logger:
            logger.warning(
                f"Could not update preprocessing manifest for "
                f"{pipeline_patient_id}: {e}"
            )


# =============================================================================
# Logging
# =============================================================================

class _SimpleLogger:
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


# =============================================================================
# Config patching
# =============================================================================

def _patch_config(data_folder: str, model_folder: str):
    import config

    data_path = Path(data_folder).resolve()
    model_path = Path(model_folder).resolve()

    config.TRAINING_SET_DIR = data_path

    config.PHYSIOLOGICAL_DATA_DIR = data_path / "physiological_data"
    config.ALGORITHMIC_ANNOTATIONS_DIR = data_path / "algorithmic_annotations"
    config.HUMAN_ANNOTATIONS_DIR = data_path / "human_annotations"
    config.DATA_DIR = config.PHYSIOLOGICAL_DATA_DIR

    config.DEMOGRAPHICS_FILE = data_path / DEMOGRAPHICS_BASENAME

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

    config.PLOT_ENABLED = False

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


# =============================================================================
# Labels
# =============================================================================

def _sanitize_label_value(x):
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
    out = df.copy()
    out[target_col] = out[target_col].apply(_sanitize_label_value)
    return out


# =============================================================================
# Prevalence handling
# =============================================================================

AGE_COLUMN_CANDIDATES = ["Age", "age", "demo_age"]


def _extract_age_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in AGE_COLUMN_CANDIDATES:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def _compute_prevalence_reference(
    demographics_path: Path,
    target_col: str = LABEL_COLUMN,
    logger=None,
) -> Dict:
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


def _prior_probability_shift(
    p: np.ndarray,
    train_prevalence: float,
    target_prevalence: Optional[float],
    eps: float = 1e-7,
) -> np.ndarray:
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


def _rowwise_age_prior_shift_and_thresholds(
    raw_probabilities: np.ndarray,
    feature_table: pd.DataFrame,
    training_config: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
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


# =============================================================================
# Prediction helpers
# =============================================================================

def _safe_predict_proba_positive(ml_model, X_scaled: np.ndarray) -> np.ndarray:
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

    # Remove duplicate columns and create a compact frame.
    df = df.loc[:, ~df.columns.duplicated()].copy()

    patient_ids = (
        df["patient_id"].values if "patient_id" in df.columns else np.arange(len(df))
    )

    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features and verbose:
        print(
            f"  ! Missing features at prediction: "
            f"{len(missing_features)}/{len(feature_names)}"
        )

    # Important:
    # Do not insert missing columns one by one with df[f] = np.nan.
    # That causes DataFrame fragmentation and the repeated pandas warning.
    # reindex creates all missing columns at once.
    X_df = df.reindex(columns=feature_names)
    X = X_df.to_numpy(dtype=np.float32, copy=False)

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
    raw_probabilities = _safe_predict_proba_positive(ml_model, X_scaled)

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


# =============================================================================
# Compact feature loading for training
# =============================================================================

def _keep_patient_feature_column(col: str) -> bool:
    col = str(col)

    if col in {
        "patient_id",
        "target",
        LABEL_COLUMN,
        "Cognitive_Impairment",
        "Age",
        "Sex",
        "Race",
        "Ethnicity",
        "BMI",
        "SiteID",
        "BDSPPatientID",
        "BidsFolder",
        "SessionID",
        "CreationTime",
    }:
        return True

    if col.startswith("demo_"):
        return True

    if col.startswith(
        (
            "sleep_",
            "stage_",
            "missingness_",
            "n_",
            "pct_",
        )
    ):
        return True

    if (
        col.startswith("cap_")
        or col.startswith("ORP_")
        or col
        in {
            "CSI",
            "Delta_Power_Entropy",
            "ORP_Mean",
            "Artifact_Fraction",
            "ORP_NREM",
            "ORP_std_NREM",
            "ORP_REM",
            "ORP_Wake",
            "ORP_APeak",
            "ORP_A9",
        }
    ):
        return True

    key_patterns = [
        "hr_mean",
        "hr_cv",
        "hrv_sdnn",
        "hrv_rmssd",
        "hrv_pnn50",
        "hrv_pnn20",
        "hrv_lf_power",
        "hrv_hf_power",
        "hrv_lf_hf_ratio",
        "hrv_sd1",
        "hrv_sd2",
        "hrv_sample_entropy",
        "hrv_dfa_alpha1",
        "rsa_p2t_mean",
        "rsa_coupling_strength",
        "delta_power",
        "delta_power_rel",
        "theta_power",
        "theta_power_rel",
        "alpha_power",
        "alpha_power_rel",
        "sigma_power",
        "sigma_power_rel",
        "beta_power",
        "beta_power_rel",
        "swa_power",
        "spindle_power_total",
        "slow_spindle_power",
        "fast_spindle_power",
        "theta_alpha_ratio",
        "delta_alpha_ratio",
        "slowing_ratio",
        "dar",
        "spectral_entropy",
        "sample_entropy",
        "permutation_entropy",
        "hjorth_complexity",
        "_sp_density",
        "_sp_amplitude_mean",
        "_sp_duration_mean",
        "_sp_frequency_mean",
        "_sp_slow_density",
        "_sp_fast_density",
        "_sp_fast_slow_ratio",
        "_sp_rms_mean",
        "_sp_rel_power_mean",
        "_so_density",
        "_so_ptp_amplitude_mean",
        "_so_slope_mean",
        "_so_neg_peak_mean",
        "_so_frequency_mean",
        "_coup_mrl",
        "_coup_mean_phase_deg",
        "_coup_rate",
        "_coup_pac_mi",
        "_coup_rayleigh_z",
        "ann_sleep_depth",
        "ann_total_event_count",
        "ann_arousal_count",
        "ann_respiratory_event_count",
    ]

    useful_prefixes = (
        "all_",
        "nrem_",
        "rem_",
        "n2_",
        "n3_",
        "third",
        "cycle",
    )

    if col.startswith(useful_prefixes) and any(p in col for p in key_patterns):
        allowed_suffixes = (
            "_mean",
            "_std",
            "_median",
            "_iqr",
        )

        if col.endswith(allowed_suffixes):
            return True

        if (
            "third_diff" in col
            or "third_rel_change" in col
            or "cycle_trend" in col
            or "cycle_diff" in col
            or "cycle_rel_change" in col
        ):
            return True

    return False


def _read_parquet_compact(path: Path, logger=None) -> Optional[pd.DataFrame]:
    path = Path(path)

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        available_cols = list(pf.schema.names)

        keep_cols = [
            c for c in available_cols
            if _keep_patient_feature_column(c)
        ]

        if "patient_id" in available_cols and "patient_id" not in keep_cols:
            keep_cols.insert(0, "patient_id")

        if not keep_cols:
            return None

        return pd.read_parquet(path, columns=keep_cols)

    except Exception as e:
        if logger:
            logger.warning(
                f"Compact parquet read failed for {path}: "
                f"{type(e).__name__}: {e}"
            )
        return None


def _compact_patient_df(patient_df: pd.DataFrame, pid: str) -> Optional[Dict]:
    if patient_df is None or len(patient_df) == 0:
        return None

    patient_df = patient_df.loc[:, ~patient_df.columns.duplicated()]
    patient_df = patient_df.iloc[[0]].copy()
    patient_df["patient_id"] = pid

    keep_cols = [
        c for c in patient_df.columns
        if _keep_patient_feature_column(c)
    ]

    if "patient_id" not in keep_cols:
        keep_cols.insert(0, "patient_id")

    patient_df = patient_df[keep_cols]

    row = {}

    for col, val in patient_df.iloc[0].items():
        if isinstance(val, (np.floating, float)):
            if pd.isna(val):
                row[col] = np.nan
            else:
                row[col] = np.float32(val)
        elif isinstance(val, (np.integer, int)):
            row[col] = int(val)
        else:
            row[col] = val

    return row


def _load_patient_level_features_streaming(
    successful_results: Dict,
    feature_dir: Path,
    demographics_path: Path,
    logger=None,
) -> pd.DataFrame:
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    n_from_patient_cache = 0
    n_from_segment_fallback = 0
    n_failed = 0

    try:
        from feature_table.build_feature_table import (
            aggregate_to_patient_level,
            add_demographics,
            _final_cleanup_patient_level,
        )
    except Exception:
        aggregate_to_patient_level = None
        add_demographics = None
        _final_cleanup_patient_level = None

    total = len(successful_results)

    for i, (pid, paths) in enumerate(successful_results.items(), start=1):
        row = None

        pat_path = paths.get("pat_features_path")

        if pat_path and Path(pat_path).exists():
            try:
                patient_df = _read_parquet_compact(
                    Path(pat_path),
                    logger=logger,
                )

                row = _compact_patient_df(patient_df, pid)

                if row is not None:
                    n_from_patient_cache += 1

                del patient_df
                gc.collect()

            except Exception as e:
                if logger:
                    logger.warning(
                        f"Could not read compact patient-level features "
                        f"for {pid}: {type(e).__name__}: {e}"
                    )
                row = None
                gc.collect()

        if row is None:
            seg_path = paths.get("seg_features_path")

            if (
                aggregate_to_patient_level is not None
                and seg_path
                and Path(seg_path).exists()
            ):
                try:
                    seg_df = pd.read_parquet(seg_path)

                    if seg_df is not None and len(seg_df) > 0:
                        patient_df = aggregate_to_patient_level(
                            seg_df,
                            paths.get("sleep_summary"),
                            logger=None,
                        )

                        row = _compact_patient_df(patient_df, pid)

                        if row is not None:
                            n_from_segment_fallback += 1

                    del seg_df
                    try:
                        del patient_df
                    except Exception:
                        pass
                    gc.collect()

                except Exception as e:
                    if logger:
                        logger.warning(
                            f"Could not aggregate segment features for {pid}: "
                            f"{type(e).__name__}: {e}"
                        )
                    row = None
                    gc.collect()

        if row is None:
            n_failed += 1
            continue

        rows.append(row)

        if logger and i % 100 == 0:
            logger.info(
                f"Loaded compact patient-level features: "
                f"{i}/{total} records scanned, {len(rows)} usable."
            )

    if not rows:
        if logger:
            logger.error("No patient-level feature rows could be loaded.")
        return pd.DataFrame()

    if logger:
        logger.info(
            f"Creating compact patient table from {len(rows)} rows "
            f"(patient cache={n_from_patient_cache}, "
            f"segment fallback={n_from_segment_fallback}, failed={n_failed})."
        )

    patient_level = pd.DataFrame.from_records(rows)

    del rows
    gc.collect()

    if add_demographics is not None:
        try:
            patient_level = add_demographics(
                patient_level,
                demographics_path=demographics_path,
                logger=logger,
            )
        except Exception as e:
            if logger:
                logger.warning(
                    f"add_demographics failed: {type(e).__name__}: {e}"
                )

    if _final_cleanup_patient_level is not None:
        try:
            patient_level = _final_cleanup_patient_level(
                patient_level,
                logger=logger,
            )
        except Exception as e:
            if logger:
                logger.warning(
                    f"Final patient-level cleanup failed: "
                    f"{type(e).__name__}: {e}"
                )

    for col in patient_level.select_dtypes(include=["float64"]).columns:
        patient_level[col] = patient_level[col].astype(np.float32)

    for col in patient_level.select_dtypes(include=["int64"]).columns:
        if col not in ["target", LABEL_COLUMN, "Cognitive_Impairment"]:
            try:
                patient_level[col] = pd.to_numeric(
                    patient_level[col],
                    downcast="integer",
                )
            except Exception:
                pass

    out_parquet = feature_dir / "features_patient_level.parquet"
    out_csv = feature_dir / "features_patient_level.csv"

    try:
        patient_level.to_parquet(out_parquet, index=False)
        if logger:
            logger.info(f"Compact patient-level features saved: {out_parquet}")
    except Exception as e:
        if logger:
            logger.warning(
                f"Could not save compact patient-level parquet "
                f"({type(e).__name__}: {e}); saving CSV instead."
            )
        patient_level.to_csv(out_csv, index=False)

    if logger:
        logger.info(
            f"Compact patient-level table: "
            f"{patient_level.shape[0]} patients x "
            f"{patient_level.shape[1]} columns"
        )

    return patient_level


# =============================================================================
# Training
# =============================================================================

def train_model(data_folder, model_folder, verbose):
    _patch_config(data_folder, model_folder)
    logger = _setup_logger(verbose)

    import config
    from helper_code import find_patients, DEMOGRAPHICS_FILE, HEADERS

    labelled_demographics = _maybe_create_labelled_demographics(
        data_folder=data_folder,
        model_folder=model_folder,
        logger=logger,
    )

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

    from main import process_single_patient
    from classification.train_model import train_model as classification_train_model

    logger.info("Finding the Challenge data...")

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError("No data were provided.")

    logger.info(f"Found {num_records} records. Starting preprocessing...")

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

    configured_workers = int(getattr(config, "NUM_WORKERS", 1))
    configured_workers = max(1, configured_workers)
    max_workers = min(configured_workers, max(1, len(records_to_process)))
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

    patient_level = _load_patient_level_features_streaming(
        successful_results=successful_results,
        feature_dir=feature_dir,
        demographics_path=config.DEMOGRAPHICS_FILE,
        logger=logger if verbose else None,
    )

    gc.collect()

    if patient_level is None or len(patient_level) == 0:
        logger.error("Patient-level table is empty. Saving fallback model.")
        _save_fallback_model(model_folder)
        return

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

    model_result = _attach_prevalence_info_to_model(
        model_result=model_result,
        prevalence_info=prevalence_info,
        expected_global_prevalence=expected_global_prevalence,
    )

    save_model(model_folder, model_result)

    cv = model_result.get("cv_results", {})
    logger.info(f"Training complete. CV AUROC={cv.get('auroc_mean', 'N/A')}")
    logger.info("Done.")


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_folder, verbose):
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

    model_dict["__model_folder"] = os.path.abspath(model_folder)

    if verbose:
        mt = model_dict.get("training_config", {}).get("model_type", "unknown")
        nf = len(model_dict.get("selected_features", []))
        fb = model_dict.get("fallback", False)
        print(f"Model loaded: type={mt}, features={nf}, fallback={fb}")

    return model_dict


# =============================================================================
# Inference preprocessing
# =============================================================================

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


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        with open(test_file, "w") as f:
            f.write("ok")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _get_inference_feature_output_dir(
    model: dict,
    data_folder: str,
    tmp_worker_dir: str,
) -> Path:
    """
    Fixed inference cache directory.

    IMPORTANT:
    This intentionally keeps the cache path unchanged:

        <model_folder>/preprocessed_test_cache/features

    In your Docker container this is expected to be:

        challenge/model/preprocessed_test_cache/features

    No data-key subfolder is used.
    """
    runtime_model_folder = None

    if isinstance(model, dict):
        runtime_model_folder = model.get("__model_folder", None)

    if runtime_model_folder is not None:
        fixed_dir = (
            Path(runtime_model_folder)
            / "preprocessed_test_cache"
            / "features"
        )

        if _is_writable_dir(fixed_dir):
            return fixed_dir

    fallback_dir = Path(tmp_worker_dir) / "features"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir


def _holdout_preprocess_marker_path(feature_output_dir: Path) -> Path:
    return Path(feature_output_dir) / HOLDOUT_PREPROCESS_MARKER_FILENAME


def _load_holdout_preprocess_marker(feature_output_dir: Path) -> Optional[Dict]:
    marker_path = _holdout_preprocess_marker_path(feature_output_dir)

    if not marker_path.exists():
        return None

    try:
        with open(marker_path, "r") as f:
            marker = json.load(f)

        if isinstance(marker, dict):
            return marker
    except Exception:
        pass

    return None


def _save_holdout_preprocess_marker(feature_output_dir: Path, marker: Dict):
    feature_output_dir = Path(feature_output_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    marker_path = _holdout_preprocess_marker_path(feature_output_dir)
    tmp_path = marker_path.with_suffix(".json.tmp")

    with open(tmp_path, "w") as f:
        json.dump(marker, f, indent=2)

    os.replace(tmp_path, marker_path)


def _ensure_holdout_preprocessed_parallel(
    data_folder: str,
    tmp_model_dir: str,
    feature_output_dir: Path,
    verbose: bool,
):
    """
    Preprocess all holdout patients once, in parallel, using config.NUM_WORKERS.

    Unlike the earlier version, this function does NOT blindly trust an old
    holdout_preprocessing_complete.json marker. It always scans the fixed cache
    directory first. If all current holdout subjects are cached, it returns.
    If some are missing, it processes only the missing ones in parallel.
    """
    _patch_config(data_folder, tmp_model_dir)

    import config
    from helper_code import find_patients, DEMOGRAPHICS_FILE, HEADERS

    logger = _setup_logger(verbose)

    feature_output_dir = Path(feature_output_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    marker = _load_holdout_preprocess_marker(feature_output_dir)

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError("No holdout data were provided.")

    # Fast path:
    # If a previous pass already completed for this cache directory and the
    # demographics row count matches, do not scan all cached subjects again.
    if marker is not None:
        marker_num_records = marker.get("num_records", None)
        marker_cache_dir = marker.get("cache_dir", None)

        try:
            marker_num_records_ok = (
                marker_num_records is None
                or int(marker_num_records) == int(num_records)
            )
        except Exception:
            marker_num_records_ok = False

        try:
            marker_cache_dir_ok = (
                marker_cache_dir is None
                or Path(marker_cache_dir).resolve() == feature_output_dir.resolve()
            )
        except Exception:
            marker_cache_dir_ok = True

        if marker_num_records_ok and marker_cache_dir_ok:
            if verbose:
                logger.info(
                    f"Holdout preprocessing marker found for "
                    f"{num_records} records in {feature_output_dir}; "
                    "skipping full cache scan."
                )
            return


    segment_length_sec = getattr(config, "SEGMENT_LENGTH_SEC", 30)
    overlap_sec = getattr(config, "SEGMENT_OVERLAP_SEC", 0)

    records_to_process = []
    cached_subjects = []

    if verbose:
        logger.info(f"Checking fixed holdout preprocessing cache: {feature_output_dir}")

    for i, record in enumerate(patient_metadata_list):
        patient_id_bids = record[HEADERS["bids_folder"]]
        site_id = record[HEADERS["site_id"]]
        session_id = record[HEADERS["session_id"]]

        record_name = f"{patient_id_bids}_ses-{session_id}"
        pipeline_patient_id = f"{site_id}/{record_name}"
        patient_dir = Path(config.PHYSIOLOGICAL_DATA_DIR) / site_id

        cached = _find_cached_preprocessing(
            feature_output_dir=feature_output_dir,
            pipeline_patient_id=pipeline_patient_id,
            record_name=record_name,
            # Do not log every cache hit during the global scan.
            # Otherwise verbose mode prints one line per cached patient per run_model call.
            logger=None,
        )


        if cached is not None:
            cached_subjects.append(pipeline_patient_id)

            # If an old cache was found only by filename scan, ensure the
            # manifest is updated so later lookups are fast.
            #_remember_preprocessing_result(
            #    feature_output_dir=feature_output_dir,
            #    pipeline_patient_id=pipeline_patient_id,
            #    record_name=record_name,
            #    result=cached,
            #    logger=logger if verbose else None,
            #)
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

    configured_workers = int(getattr(config, "NUM_WORKERS", 1))
    configured_workers = max(1, configured_workers)

    if len(records_to_process) == 0:
        if verbose:
            if marker is not None:
                logger.info(
                    f"Holdout marker exists and all subjects are cached: "
                    f"{len(cached_subjects)}/{num_records}."
                )
            else:
                logger.info(
                    f"All holdout subjects already cached without marker: "
                    f"{len(cached_subjects)}/{num_records}."
                )

        _save_holdout_preprocess_marker(
            feature_output_dir,
            {
                "completed_at_utc": datetime.utcnow().isoformat() + "Z",
                "cache_dir": str(feature_output_dir),
                "num_records": num_records,
                "num_cached_initially": len(cached_subjects),
                "num_newly_processed": 0,
                "num_failed": 0,
                "configured_num_workers": configured_workers,
            },
        )
        return

    max_workers = min(configured_workers, len(records_to_process))
    max_workers = max(1, max_workers)
    batch_size = max_workers

    if verbose:
        logger.info(
            f"Preprocessing holdout data in parallel into fixed cache "
            f"{feature_output_dir}: "
            f"{len(records_to_process)} uncached / {num_records} total, "
            f"config.NUM_WORKERS={configured_workers}, using {max_workers} workers."
        )

    total_start = time.time()
    newly_processed = 0
    failed_subjects = []

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
                    if verbose:
                        logger.warning(
                            f"Patient directory not found: {patient_dir}. Skipping."
                        )
                    failed_subjects.append(pipeline_patient_id)
                    continue

                future = executor.submit(
                    _preprocess_one_patient_worker,
                    data_folder=data_folder,
                    tmp_model_dir=tmp_model_dir,
                    pipeline_patient_id=pipeline_patient_id,
                    patient_dir_str=str(patient_dir),
                    record_name=record_name,
                    segment_length_sec=segment_length_sec,
                    overlap_sec=overlap_sec,
                    feature_output_dir_str=str(feature_output_dir),
                )

                futures[future] = item

            for future in as_completed(futures):
                item = futures[future]
                pid = item["pipeline_patient_id"]
                record_name = item["record_name"]

                try:
                    result = future.result()

                    if result and result.get("success"):
                        newly_processed += 1

                        _remember_preprocessing_result(
                            feature_output_dir=feature_output_dir,
                            pipeline_patient_id=pid,
                            record_name=record_name,
                            result=result,
                            logger=logger if verbose else None,
                        )
                    else:
                        failed_subjects.append(pid)
                        if verbose:
                            logger.warning(f"Holdout preprocessing failed for {pid}")

                except Exception as e:
                    failed_subjects.append(pid)
                    if verbose:
                        logger.error(
                            f"Error preprocessing holdout subject {pid}: "
                            f"{type(e).__name__}: {e}"
                        )
                        logger.error(traceback.format_exc())

        gc.collect()

        if verbose:
            elapsed = time.time() - total_start
            logger.info(
                f"Holdout preprocessing progress: "
                f"{batch_end}/{len(records_to_process)} uncached subjects scanned, "
                f"{newly_processed} newly processed, "
                f"{len(cached_subjects)} initially cached, "
                f"{len(failed_subjects)} failed, "
                f"{elapsed:.0f}s elapsed."
            )

    _save_holdout_preprocess_marker(
        feature_output_dir,
        {
            "completed_at_utc": datetime.utcnow().isoformat() + "Z",
            "cache_dir": str(feature_output_dir),
            "num_records": num_records,
            "num_cached_initially": len(cached_subjects),
            "num_newly_processed": newly_processed,
            "num_failed": len(failed_subjects),
            "failed_subjects": failed_subjects,
            "configured_num_workers": configured_workers,
            "used_num_workers": max_workers,
            "elapsed_sec": time.time() - total_start,
        },
    )

    if verbose:
        logger.info(
            f"Parallel holdout preprocessing complete in fixed cache "
            f"{feature_output_dir}: "
            f"{len(cached_subjects)} cached initially, "
            f"{newly_processed} newly processed, "
            f"{len(failed_subjects)} failed, "
            f"elapsed={time.time() - total_start:.0f}s."
        )


# =============================================================================
# Demographics during inference
# =============================================================================

def _manual_add_demographics(
    patient_features: pd.DataFrame,
    demographics_path: Path,
) -> pd.DataFrame:
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

    merge_cols = ["patient_id"] + [
        c for c in demo.columns if c != "patient_id" and c not in patient_features.columns
    ]

    return patient_features.merge(demo[merge_cols], on="patient_id", how="left")


def _add_demographics_safely(
    patient_features: pd.DataFrame,
    demographics_path: Path,
    verbose: bool,
) -> pd.DataFrame:
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


# =============================================================================
# Inference
# =============================================================================

def run_model(model, record, data_folder, verbose):
    from helper_code import HEADERS

    patient_id_bids = record[HEADERS["bids_folder"]]
    site_id = record[HEADERS["site_id"]]
    session_id = record[HEADERS["session_id"]]

    if model is None or model.get("fallback", False):
        return 0, 0.10

    tmp_main = None
    tmp_worker_dir = None

    try:
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

        feature_output_dir = _get_inference_feature_output_dir(
            model=model,
            data_folder=data_folder,
            tmp_worker_dir=tmp_worker_dir,
        )

        # run_model.py loads `model` once and then calls run_model(...) once per patient.
        # Therefore, do the expensive global holdout cache/preprocessing check only once
        # per run_model.py execution.
        preprocess_state_key = (
            f"{Path(data_folder).resolve()}::"
            f"{Path(feature_output_dir).resolve()}"
        )

        if model.get("__holdout_preprocess_state_key") != preprocess_state_key:
            _ensure_holdout_preprocessed_parallel(
                data_folder=data_folder,
                tmp_model_dir=tmp_worker_dir,
                feature_output_dir=feature_output_dir,
                verbose=verbose,
            )
            model["__holdout_preprocess_state_key"] = preprocess_state_key
        else:
            if verbose:
                print("  - Global holdout preprocessing/cache check already done; skipping.")


        result = _find_cached_preprocessing(
            feature_output_dir=feature_output_dir,
            pipeline_patient_id=pipeline_patient_id,
            record_name=record_name,
            logger=logger if verbose else None,
        )

        if result is not None:
            if verbose:
                print(
                    f"  - Using cached preprocessing for {pipeline_patient_id} "
                    f"from {feature_output_dir}"
                )

        else:
            if verbose:
                print(
                    f"  - No cached preprocessing for {pipeline_patient_id}; "
                    f"processing this patient only..."
                )

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


# =============================================================================
# Save model
# =============================================================================

def save_model(model_folder, model_dict):
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
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
