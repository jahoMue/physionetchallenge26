#!/usr/bin/env python
"""
calc_shap_values.py
===================

Calculate SHAP values for a trained PhysioNet Challenge 2026 model.

This version is robust against corrupted/problematic Parquet columns by reading
only the columns needed for SHAP:

    patient_id + model feature columns

It also falls back to:
    1. pyarrow selected-column read
    2. pyarrow column-by-column rescue
    3. fastparquet selected-column read, if installed
    4. CSV fallback, if a matching .csv exists
    5. rebuilding from per-patient *_patient_features.parquet files, if possible

Usage:
    python calc_shap_values.py

Optional environment variables:
    SHAP_MODEL_DIR
    SHAP_FEATURES_PATH
    SHAP_OUTPUT_DIR
    SHAP_MAX_BACKGROUND
    SHAP_MAX_PATIENTS
"""

import os
import sys
import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# These paths are resolved in main() by resolve_paths().
# Environment variables override all defaults.
DEFAULT_LOCAL_MODEL_DIR = Path(
    r"C:\Users\Biosig 3\Documents\Richard\Physionet26Data\model\models\model_lightgbm_20260703_151444"
)
DEFAULT_LOCAL_FEATURES_PATH = Path(
    r"C:\Users\Biosig 3\Documents\Richard\Physionet26Data\model\features\features_patient_level.parquet"
)

FALLBACK_MODEL_DIR = Path("output/models/model_xgboost_20260331_222925")
FALLBACK_FEATURES_PATH = Path("output/features/features_patient_level.parquet")

OUTPUT_DIR = Path(os.environ.get("SHAP_OUTPUT_DIR", "output/shap"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Optional row limit for quick testing/debugging.
# Set e.g. PowerShell:
#   $env:SHAP_MAX_PATIENTS="200"
SHAP_MAX_PATIENTS = os.environ.get("SHAP_MAX_PATIENTS", "").strip()
SHAP_MAX_PATIENTS = int(SHAP_MAX_PATIENTS) if SHAP_MAX_PATIENTS else None


# ==============================================================================
# OPTIONAL PICKLE COMPATIBILITY HELPERS
# ==============================================================================

class CombinedSelector:
    """
    Compatibility class for feature selectors saved by train_model.py.

    If joblib needs to resolve CombinedSelector from __main__ during unpickling,
    defining this class here prevents loading failures.
    """
    def __init__(self, mask, scores, f_scores=None, mi_scores=None):
        self.mask_ = mask
        self.scores_ = scores
        self.f_scores_ = f_scores
        self.mi_scores_ = mi_scores

    def get_support(self):
        return self.mask_

    def transform(self, X):
        return X[:, self.mask_]

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)


class ModelSelector:
    """
    Compatibility class for feature selectors saved by train_model.py.
    """
    def __init__(self, mask, scores):
        self.mask_ = mask
        self.scores_ = scores

    def get_support(self):
        return self.mask_

    def transform(self, X):
        return X[:, self.mask_]

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)


# ==============================================================================
# PATH RESOLUTION
# ==============================================================================

def _resolve_latest_model_dir(base_dir: Path) -> Path | None:
    """
    Try to resolve output/models/latest, output/models/latest.txt, or the newest
    model_* directory.
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        return None

    latest_link = base_dir / "latest"
    latest_txt = base_dir / "latest.txt"

    try:
        if latest_link.exists() or latest_link.is_symlink():
            resolved = latest_link.resolve()
            if resolved.exists():
                return resolved
    except Exception:
        pass

    try:
        if latest_txt.exists():
            p = Path(latest_txt.read_text().strip())
            if p.exists():
                return p
    except Exception:
        pass

    try:
        candidates = [
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith("model_")
        ]
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    except Exception:
        pass

    return None


def resolve_paths():
    """
    Resolve model/features/output paths.

    Priority:
      1. Environment variables
      2. User-specific paths from the observed error
      3. latest model under output/models
      4. fallback paths from the original uploaded script
    """
    env_model = os.environ.get("SHAP_MODEL_DIR", "").strip()
    env_features = os.environ.get("SHAP_FEATURES_PATH", "").strip()
    env_output = os.environ.get("SHAP_OUTPUT_DIR", "").strip()

    if env_output:
        output_dir = Path(env_output)
    else:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    if env_model:
        model_dir = Path(env_model)
    elif DEFAULT_LOCAL_MODEL_DIR.exists():
        model_dir = DEFAULT_LOCAL_MODEL_DIR
    else:
        latest = _resolve_latest_model_dir(Path("output/models"))
        model_dir = latest if latest is not None else FALLBACK_MODEL_DIR

    if env_features:
        features_path = Path(env_features)
    elif DEFAULT_LOCAL_FEATURES_PATH.exists():
        features_path = DEFAULT_LOCAL_FEATURES_PATH
    else:
        features_path = FALLBACK_FEATURES_PATH

    return model_dir, features_path, output_dir


# ==============================================================================
# 1. LOAD MODEL ARTIFACTS
# ==============================================================================

def load_model_artifacts(model_dir: Path) -> dict:
    """
    Load all model artifacts saved during training.
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    artifacts = {}

    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    imputer_path = model_dir / "imputer.joblib"
    feature_names_path = model_dir / "feature_names.json"

    required = [model_path, scaler_path, imputer_path, feature_names_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifact(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    artifacts["model"] = joblib.load(model_path)
    artifacts["scaler"] = joblib.load(scaler_path)
    artifacts["imputer"] = joblib.load(imputer_path)

    selector_path = model_dir / "feature_selector.joblib"
    if selector_path.exists():
        artifacts["feature_selector"] = joblib.load(selector_path)
    else:
        artifacts["feature_selector"] = None

    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_data = json.load(f)

    artifacts["feature_names"] = list(feature_data["all_features"])
    artifacts["selected_features"] = list(feature_data["selected_features"])

    config_path = model_dir / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            artifacts["training_config"] = json.load(f)
    else:
        artifacts["training_config"] = {}

    importance_path = model_dir / "feature_importance.csv"
    if importance_path.exists():
        artifacts["feature_importance"] = pd.read_csv(importance_path)
    else:
        artifacts["feature_importance"] = None

    model_type = artifacts["training_config"].get("model_type", "unknown")

    print(f"Model loaded from: {model_dir}")
    print(f"  Model type: {model_type}")
    print(f"  All features: {len(artifacts['feature_names'])}")
    print(f"  Selected features: {len(artifacts['selected_features'])}")

    return artifacts


# ==============================================================================
# 2. ROBUST PARQUET READING
# ==============================================================================

def _get_parquet_columns(features_path: Path) -> list[str] | None:
    """
    Return Parquet schema column names without reading the full table.
    """
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(features_path)
        return list(pf.schema.names)
    except Exception:
        return None


def _safe_read_parquet_columns_pyarrow(
    features_path: Path,
    columns: list[str],
) -> pd.DataFrame:
    """
    Read selected columns using pyarrow.
    """
    return pd.read_parquet(
        features_path,
        columns=columns,
        engine="pyarrow",
    )


def _safe_read_parquet_columns_fastparquet(
    features_path: Path,
    columns: list[str] | None,
) -> pd.DataFrame:
    """
    Read selected columns using fastparquet, if installed.
    """
    return pd.read_parquet(
        features_path,
        columns=columns,
        engine="fastparquet",
    )


def _read_columns_one_by_one_pyarrow(
    features_path: Path,
    requested_cols: list[str],
) -> pd.DataFrame:
    """
    Rescue mode: read requested columns individually. If a column is corrupted or
    unreadable, skip it and let prepare_data() fill it with NaN later.
    """
    pieces = []
    bad_cols = []

    for i, col in enumerate(requested_cols, start=1):
        try:
            one_col = pd.read_parquet(
                features_path,
                columns=[col],
                engine="pyarrow",
            )
            pieces.append(one_col)
        except Exception as e:
            bad_cols.append((col, type(e).__name__, str(e)))

        if i % 250 == 0:
            print(f"  Column rescue progress: {i}/{len(requested_cols)}")

    if not pieces:
        raise RuntimeError(
            "Column-by-column rescue failed: no requested columns could be read."
        )

    rescued = pd.concat(pieces, axis=1)
    rescued = rescued.loc[:, ~rescued.columns.duplicated()].copy()

    if bad_cols:
        print(f"WARNING: Could not read {len(bad_cols)} requested columns.")
        print("First unreadable columns:")
        for col, err_type, msg in bad_cols[:25]:
            print(f"  - {col}: {err_type}: {msg[:180]}")

    return rescued


def _try_csv_fallback(
    features_path: Path,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """
    Try matching CSV file if it exists.
    """
    csv_path = features_path.with_suffix(".csv")

    if not csv_path.exists():
        return None

    print(f"Trying CSV fallback: {csv_path}")

    wanted = set(feature_names)
    wanted.add("patient_id")

    return pd.read_csv(
        csv_path,
        usecols=lambda c: c in wanted,
    )


def _read_single_patient_feature_file(
    path: Path,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """
    Read one *_patient_features.parquet file in compact mode.
    """
    try:
        available_cols = _get_parquet_columns(path)

        if available_cols is None:
            df = pd.read_parquet(path)
            keep = [c for c in ["patient_id"] + feature_names if c in df.columns]
            if not keep:
                return None
            return df[keep].iloc[[0]].copy()

        requested = []
        if "patient_id" in available_cols:
            requested.append("patient_id")

        requested.extend([
            f for f in feature_names
            if f in available_cols and f not in requested
        ])

        if not requested:
            return None

        try:
            df = pd.read_parquet(path, columns=requested, engine="pyarrow")
        except Exception:
            df = _read_columns_one_by_one_pyarrow(path, requested)

        if df is None or len(df) == 0:
            return None

        return df.iloc[[0]].copy()

    except Exception:
        return None


def _rebuild_from_individual_patient_files(
    features_path: Path,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """
    If features_patient_level.parquet is unreadable, rebuild a SHAP-only compact
    table from individual *_patient_features.parquet files in the same folder.
    """
    feature_dir = features_path.parent

    if not feature_dir.exists():
        return None

    candidates = sorted(
        p for p in feature_dir.glob("*_patient_features.parquet")
        if p.name != features_path.name
    )

    if not candidates:
        return None

    print(
        f"Trying to rebuild SHAP feature table from "
        f"{len(candidates)} individual patient feature files..."
    )

    rows = []
    failed = 0

    for i, p in enumerate(candidates, start=1):
        row = _read_single_patient_feature_file(p, feature_names)

        if row is not None and len(row) > 0:
            if "patient_id" not in row.columns:
                inferred_pid = (
                    p.name
                    .replace("_patient_features.parquet", "")
                    .replace("_", "/")
                )
                row.insert(0, "patient_id", inferred_pid)
            rows.append(row)
        else:
            failed += 1

        if i % 100 == 0:
            print(f"  Rebuild progress: {i}/{len(candidates)} files scanned")

    if not rows:
        print("Rebuild failed: no individual patient feature files could be read.")
        return None

    rebuilt = pd.concat(rows, ignore_index=True)
    rebuilt = rebuilt.loc[:, ~rebuilt.columns.duplicated()].copy()

    out_path = feature_dir / "features_patient_level_shap_rebuilt.parquet"

    try:
        rebuilt.to_parquet(out_path, index=False)
        print(f"Saved rebuilt SHAP feature table: {out_path} {rebuilt.shape}")
    except Exception as e:
        print(f"WARNING: Could not save rebuilt table: {type(e).__name__}: {e}")

    print(
        f"Rebuilt table: {rebuilt.shape[0]} patients, "
        f"{rebuilt.shape[1]} columns; failed files={failed}"
    )

    return rebuilt


def read_patient_features_for_shap(
    features_path: Path,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Robust reader for the SHAP feature table.

    It avoids reading the full Parquet file because the full file may contain
    unrelated problematic columns that are not needed by the model.
    """
    features_path = Path(features_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    print(f"Reading patient feature table: {features_path}")

    available_cols = _get_parquet_columns(features_path)

    requested_cols = None

    if available_cols is not None:
        requested_cols = []

        if "patient_id" in available_cols:
            requested_cols.append("patient_id")

        requested_cols.extend([
            f for f in feature_names
            if f in available_cols and f not in requested_cols
        ])

        print(
            f"Parquet schema detected: {len(available_cols)} columns. "
            f"Requesting {len(requested_cols)} columns "
            f"(patient_id + model features present in file)."
        )

        if not requested_cols:
            raise RuntimeError(
                f"No requested model columns found in Parquet file: {features_path}"
            )

        try:
            df = _safe_read_parquet_columns_pyarrow(features_path, requested_cols)
            print("Selected-column pyarrow read succeeded.")
            return df

        except Exception as e:
            print(
                f"Selected-column pyarrow read failed: "
                f"{type(e).__name__}: {e}"
            )

            # Try fastparquet before the very slow pyarrow column-by-column rescue.
            try:
                print("Trying fastparquet selected-column read before column rescue...")
                df = _safe_read_parquet_columns_fastparquet(
                    features_path,
                    requested_cols,
                )
                print("fastparquet selected-column read succeeded.")
                return df
            except Exception as fp_e:
                print(
                    f"fastparquet selected-column read failed: "
                    f"{type(fp_e).__name__}: {fp_e}"
                )

            print("Trying pyarrow column-by-column rescue...")

            try:
                df = _read_columns_one_by_one_pyarrow(features_path, requested_cols)
                print("Column-by-column rescue succeeded.")
                return df
            except Exception as e2:
                print(
                    f"Column-by-column rescue failed: "
                    f"{type(e2).__name__}: {e2}"
                )


    else:
        print(
            "WARNING: Could not inspect Parquet schema. "
            "Will try alternative readers."
        )

    try:
        print("Trying fastparquet selected-column read...")
        df = _safe_read_parquet_columns_fastparquet(features_path, requested_cols)
        print("fastparquet read succeeded.")
        return df
    except Exception as e:
        print(f"fastparquet read failed: {type(e).__name__}: {e}")

    try:
        df = _try_csv_fallback(features_path, feature_names)
        if df is not None:
            print("CSV fallback succeeded.")
            return df
    except Exception as e:
        print(f"CSV fallback failed: {type(e).__name__}: {e}")

    allow_rebuild = os.environ.get("SHAP_ALLOW_REBUILD", "0").strip() in {
        "1", "true", "True", "yes", "YES"
    }

    if allow_rebuild:
        rebuilt = _rebuild_from_individual_patient_files(features_path, feature_names)
        if rebuilt is not None:
            return rebuilt
    else:
        print(
            "Skipping automatic rebuild from individual patient files. "
            "Set SHAP_ALLOW_REBUILD=1 if you really want this slow fallback."
        )


    raise RuntimeError(
        f"Could not read feature table: {features_path}\n"
        "Tried pyarrow selected-column read, pyarrow column rescue, "
        "fastparquet, CSV fallback, and individual patient-file rebuild."
    )


# ==============================================================================
# 3. PREPARE DATA
# ==============================================================================

def prepare_data(
    features_path: Path,
    artifacts: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fast SHAP data preparation.

    Instead of reading all original training features, this reads only the final
    selected features used by the fitted model. This avoids thousands of Parquet
    column reads and prevents the huge per-patient rebuild loop from taking hours.
    """
    feature_names = list(artifacts["feature_names"])
    selected_features = list(artifacts["selected_features"])
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]

    print(
        f"Fast SHAP mode: reading only selected features "
        f"({len(selected_features)}) instead of all features "
        f"({len(feature_names)})."
    )

    # Read only patient_id + selected model features.
    patient_level = read_patient_features_for_shap(features_path, selected_features)
    patient_level = patient_level.loc[:, ~patient_level.columns.duplicated()].copy()

    if SHAP_MAX_PATIENTS is not None and len(patient_level) > SHAP_MAX_PATIENTS:
        print(f"Limiting patients for debug: {SHAP_MAX_PATIENTS}/{len(patient_level)}")
        patient_level = patient_level.iloc[:SHAP_MAX_PATIENTS].copy()

    print(
        f"Loaded feature table: {patient_level.shape[0]} patients, "
        f"{patient_level.shape[1]} columns"
    )

    # Some selected features may still be missing if the Parquet file is damaged
    # or if a selected column could not be read. Reindex fills those with NaN.
    X_selected_raw_df = patient_level.reindex(columns=selected_features)

    # Convert selected raw features to numeric array.
    X_selected_raw = X_selected_raw_df.to_numpy(dtype=np.float64, copy=False)

    # Impute selected features using the corresponding statistics from the
    # original all-feature imputer.
    selected_indices = []
    effective_selected_features = []

    for f in selected_features:
        if f in feature_names:
            selected_indices.append(feature_names.index(f))
            effective_selected_features.append(f)
        else:
            print(f"WARNING: selected feature not found in all_features: {f}")

    if len(selected_indices) != len(selected_features):
        # Keep only features that exist in the original feature list.
        X_selected_raw_df = X_selected_raw_df[effective_selected_features]
        X_selected_raw = X_selected_raw_df.to_numpy(dtype=np.float64, copy=False)
        selected_features = effective_selected_features

    imputer_statistics = getattr(imputer, "statistics_", None)

    if imputer_statistics is None:
        raise RuntimeError(
            "The saved imputer does not have statistics_. "
            "Cannot do fast selected-feature imputation."
        )

    selected_medians = np.asarray(imputer_statistics)[selected_indices]

    print("Applying fast selected-feature imputation...")
    X_selected_imputed = X_selected_raw.copy()

    nan_mask = np.isnan(X_selected_imputed)

    if nan_mask.any():
        rows, cols = np.where(nan_mask)
        X_selected_imputed[rows, cols] = selected_medians[cols]

    # If any imputer median itself is NaN, replace remaining NaN with zero.
    X_selected_imputed = np.nan_to_num(
        X_selected_imputed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print("Applying scaler...")
    X_scaled = scaler.transform(X_selected_imputed)

    X_df = pd.DataFrame(
        X_scaled.astype(np.float32),
        columns=selected_features,
    )

    if "patient_id" in patient_level.columns:
        X_df.index = patient_level["patient_id"].astype(str).values
    else:
        X_df.index = [f"patient_{i}" for i in range(len(X_df))]

    print(
        f"Preprocessed data: {X_df.shape[0]} patients, "
        f"{X_df.shape[1]} selected features"
    )

    return X_df, patient_level



# ==============================================================================
# 4. SHAP CALCULATION
# ==============================================================================

def _patch_shap_xgboost_base_score_if_needed():
    """
    Patch older SHAP versions that fail on XGBoost base_score strings like '[0.5]'.

    This only modifies the installed SHAP source if the exact vulnerable pattern
    is present. If the pattern is absent, nothing is changed.
    """
    try:
        import shap.explainers._tree as _tree_module

        source_file = Path(_tree_module.__file__)
        print(f"  SHAP tree explainer source: {source_file}")

        lines = source_file.read_text(encoding="utf-8").splitlines(keepends=True)

        patched = False

        for i, line in enumerate(lines):
            if (
                'float(learner_model_param["base_score"])' in line
                and ".strip" not in line
            ):
                original = line
                lines[i] = line.replace(
                    'float(learner_model_param["base_score"])',
                    'float(str(learner_model_param["base_score"]).strip("[]"))',
                )
                print(f"  Patched SHAP source line {i + 1}:")
                print(f"    OLD: {original.rstrip()}")
                print(f"    NEW: {lines[i].rstrip()}")
                patched = True

        if patched:
            source_file.write_text("".join(lines), encoding="utf-8")

            mods_to_remove = [k for k in sys.modules if k == "shap" or k.startswith("shap.")]
            for mod_name in mods_to_remove:
                del sys.modules[mod_name]

            import shap as shap_reloaded
            globals()["shap"] = shap_reloaded

            print("  SHAP source patched and module reloaded.")
        else:
            print("  SHAP XGBoost base_score patch not needed.")

    except Exception as e:
        print(f"  WARNING: SHAP patch check failed: {type(e).__name__}: {e}")


def calculate_shap_values(
    model,
    X_df: pd.DataFrame,
) -> tuple[shap.Explanation, object]:
    """
    Calculate SHAP values with TreeExplainer.
    """
    print("Calculating SHAP values with TreeExplainer...")

    # Harmless for LightGBM; useful for some XGBoost/SHAP version combinations.
    _patch_shap_xgboost_base_score_if_needed()

    explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer(X_df)
    except Exception as e:
        print(
            f"explainer(X_df) failed: {type(e).__name__}: {e}\n"
            "Trying explainer.shap_values(X_df) fallback..."
        )

        raw_values = explainer.shap_values(X_df)

        if isinstance(raw_values, list):
            if len(raw_values) >= 2:
                values = raw_values[1]
            else:
                values = raw_values[0]
        else:
            values = raw_values

        base_values = getattr(explainer, "expected_value", 0.0)

        if isinstance(base_values, (list, tuple, np.ndarray)):
            base_arr = np.asarray(base_values)
            if base_arr.ndim > 0 and len(base_arr) >= 2:
                base_values = base_arr[1]
            else:
                base_values = float(np.ravel(base_arr)[0])

        shap_values = shap.Explanation(
            values=np.asarray(values),
            base_values=np.full(X_df.shape[0], float(base_values)),
            data=X_df.values,
            feature_names=X_df.columns.tolist(),
        )

    if isinstance(shap_values, shap.Explanation):
        explanation = shap_values
    else:
        explanation = shap.Explanation(
            values=np.asarray(shap_values),
            data=X_df.values,
            feature_names=X_df.columns.tolist(),
        )

    values = np.asarray(explanation.values)

    # Binary/multiclass classifiers can produce shape:
    #   n_samples x n_features x n_classes
    if values.ndim == 3:
        class_idx = 1 if values.shape[2] > 1 else 0

        base_values = explanation.base_values

        try:
            base_values_arr = np.asarray(base_values)
            if base_values_arr.ndim == 2 and base_values_arr.shape[1] > class_idx:
                base_values = base_values_arr[:, class_idx]
        except Exception:
            pass

        explanation = shap.Explanation(
            values=values[:, :, class_idx],
            base_values=base_values,
            data=explanation.data,
            feature_names=X_df.columns.tolist(),
        )

    if explanation.feature_names is None:
        explanation.feature_names = X_df.columns.tolist()

    print(f"SHAP values calculated: {np.asarray(explanation.values).shape}")

    try:
        base_mean = float(np.mean(explanation.base_values))
        print(f"Base value / expected value: {base_mean:.6f}")
    except Exception:
        pass

    return explanation, explainer


# ==============================================================================
# 5. HELPERS
# ==============================================================================

def extract_site_ids(patient_ids):
    """
    Extract site ID from patient_id strings.
    Example:
        I0002/sub-I0002150000686_ses-1 -> I0002
    """
    sites = []

    for pid in patient_ids:
        pid_str = str(pid)
        sites.append(pid_str[:5] if len(pid_str) >= 5 else "UNK")

    return np.array(sites)


def _safe_plot_filename(feature_name: str) -> str:
    safe = str(feature_name)
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
        safe = safe.replace(ch, "_")
    return safe[:180]


def _top_feature_names(shap_explanation, max_display: int = 5) -> list[str]:
    values = np.asarray(shap_explanation.values)
    feature_names = list(shap_explanation.feature_names)

    mean_abs_shap = np.abs(values).mean(axis=0)

    order = np.argsort(mean_abs_shap)[::-1]
    top_idx = order[: min(max_display, len(order))]

    return [feature_names[i] for i in top_idx]


# ==============================================================================
# 6. VISUALIZATIONS
# ==============================================================================

def plot_shap_summary_bar(shap_explanation, X_df, output_dir):
    """
    Global feature importance bar plot.
    """
    try:
        plt.figure(figsize=(12, 10))
        shap.plots.bar(shap_explanation, max_display=10, show=False)
        plt.title("SHAP Feature Importance (mean |SHAP value|)")
        plt.tight_layout()

        save_path = output_dir / "shap_summary_bar.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
    except Exception as e:
        plt.close("all")
        print(f"WARNING: Could not create SHAP bar plot: {type(e).__name__}: {e}")


def plot_shap_beeswarm(shap_explanation, X_df, output_dir):
    """
    Standard SHAP beeswarm plot.
    """
    try:
        plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(shap_explanation, max_display=10, show=False)
        plt.title("SHAP Beeswarm Plot")
        plt.tight_layout()

        save_path_svg = output_dir / "shap_beeswarm.svg"
        save_path_png = output_dir / "shap_beeswarm.png"

        plt.savefig(save_path_svg, bbox_inches="tight")
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path_svg}")
        print(f"Saved: {save_path_png}")

    except Exception as e:
        plt.close("all")
        print(f"WARNING: Could not create SHAP beeswarm plot: {type(e).__name__}: {e}")


def plot_shap_beeswarm_by_site(shap_explanation, X_df, output_dir):
    """
    Beeswarm-like plot for top features, colored by site ID instead of feature value.
    """
    try:
        max_display = 5
        feature_names = list(shap_explanation.feature_names)
        top_features = _top_feature_names(shap_explanation, max_display=max_display)

        patient_ids = X_df.index.values
        sites = extract_site_ids(patient_ids)
        unique_sites = sorted(np.unique(sites))

        print(f"\nSites found: {unique_sites}")
        for s in unique_sites:
            print(f"  {s}: {np.sum(sites == s)} patients")

        color_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        site_colors_map = {
            site: color_palette[i % len(color_palette)]
            for i, site in enumerate(unique_sites)
        }

        fig, ax = plt.subplots(figsize=(12, 8))
        n_features = len(top_features)

        for feat_idx_plot, feat_name in enumerate(reversed(top_features)):
            feat_col_idx = feature_names.index(feat_name)
            shap_vals = np.asarray(shap_explanation.values)[:, feat_col_idx]
            y_pos = feat_idx_plot

            rng = np.random.default_rng(RANDOM_SEED + feat_col_idx)
            jitter = rng.normal(0, 0.12, size=len(shap_vals))

            for site in unique_sites:
                mask = sites == site
                ax.scatter(
                    shap_vals[mask],
                    y_pos + jitter[mask],
                    c=site_colors_map[site],
                    s=14,
                    alpha=0.65,
                    edgecolors="none",
                    label=site if feat_idx_plot == 0 else None,
                    rasterized=True,
                )

        ax.set_yticks(range(n_features))
        ax.set_yticklabels(list(reversed(top_features)), fontsize=11)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
        ax.axvline(x=0, color="grey", linewidth=0.8)
        ax.set_title("SHAP Beeswarm Plot — Colored by Site", fontsize=14)

        handles = [
            mpatches.Patch(color=site_colors_map[s], label=s)
            for s in unique_sites
        ]
        ax.legend(
            handles=handles,
            title="Site",
            loc="lower right",
            fontsize=10,
            title_fontsize=11,
            framealpha=0.9,
        )

        plt.tight_layout()

        save_path_svg = output_dir / "shap_beeswarm_by_site.svg"
        save_path_png = output_dir / "shap_beeswarm_by_site.png"

        plt.savefig(save_path_svg, bbox_inches="tight")
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path_svg}")
        print(f"Saved: {save_path_png}")

    except Exception as e:
        plt.close("all")
        print(
            f"WARNING: Could not create site-colored beeswarm: "
            f"{type(e).__name__}: {e}"
        )


def plot_shap_waterfall(shap_explanation, patient_idx, output_dir):
    """
    Waterfall plot for one patient.
    """
    try:
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_explanation[patient_idx], max_display=10, show=False)
        plt.title(f"SHAP Waterfall — Patient {patient_idx}")
        plt.tight_layout()

        save_path = output_dir / f"shap_waterfall_patient_{patient_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

    except Exception as e:
        plt.close("all")
        print(
            f"WARNING: Could not create waterfall for patient {patient_idx}: "
            f"{type(e).__name__}: {e}"
        )


def plot_shap_heatmap(shap_explanation, output_dir):
    """
    Heatmap of SHAP values across patients.
    """
    try:
        plt.figure(figsize=(16, 10))
        shap.plots.heatmap(shap_explanation, max_display=10, show=False)
        plt.title("SHAP Heatmap")
        plt.tight_layout()

        save_path = output_dir / "shap_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

    except Exception as e:
        plt.close("all")
        print(f"WARNING: Could not create heatmap: {type(e).__name__}: {e}")


def plot_shap_dependence(shap_explanation, X_df, feature_name, output_dir):
    """
    Dependence plot for one feature.
    """
    if feature_name not in X_df.columns:
        print(f"Feature '{feature_name}' not found. Skipping dependence plot.")
        return

    try:
        plt.figure(figsize=(10, 6))
        shap.plots.scatter(
            shap_explanation[:, feature_name],
            color=shap_explanation,
            show=False,
        )
        plt.title(f"SHAP Dependence — {feature_name}")
        plt.tight_layout()

        safe_name = _safe_plot_filename(feature_name)
        save_path = output_dir / f"shap_dependence_{safe_name}.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

    except Exception as e:
        plt.close("all")
        print(
            f"WARNING: Could not create dependence plot for {feature_name}: "
            f"{type(e).__name__}: {e}"
        )


# ==============================================================================
# 7. STATISTICAL ANALYSIS BY SITE
# ==============================================================================

def analyze_shap_by_site(shap_explanation, X_df, output_dir):
    """
    Test whether SHAP values for top features differ by site.
    """
    max_display = 5
    feature_names = list(shap_explanation.feature_names)
    top_features = _top_feature_names(shap_explanation, max_display=max_display)

    patient_ids = X_df.index.values
    sites = extract_site_ids(patient_ids)
    unique_sites = sorted(np.unique(sites))
    n_sites = len(unique_sites)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: SHAP values by Site")
    print("=" * 70)
    print(f"Sites: {unique_sites}")
    for s in unique_sites:
        print(f"  {s}: n={np.sum(sites == s)}")
    print()

    n_comparisons = n_sites * (n_sites - 1) // 2

    all_results = []
    report_lines = [
        "=" * 70,
        "STATISTICAL ANALYSIS: SHAP values by Site",
        "=" * 70,
        f"Sites: {unique_sites}",
    ]

    for s in unique_sites:
        report_lines.append(f"  {s}: n={np.sum(sites == s)}")

    report_lines.append(
        f"Bonferroni correction: {n_comparisons} pairwise comparisons per feature"
    )
    report_lines.append("")

    for feat_name in top_features:
        feat_col_idx = feature_names.index(feat_name)
        shap_vals = np.asarray(shap_explanation.values)[:, feat_col_idx]

        report_lines.append("-" * 70)
        report_lines.append(f"Feature: {feat_name}")
        report_lines.append("-" * 70)

        print(f"\n--- Feature: {feat_name} ---")

        groups = {}
        for site in unique_sites:
            mask = sites == site
            groups[site] = shap_vals[mask]

        report_lines.append("  Descriptive statistics:")
        print("  Descriptive statistics:")

        for site in unique_sites:
            vals = groups[site]

            if len(vals) == 0:
                desc = f"    {site}: n=0"
            else:
                desc = (
                    f"    {site}: n={len(vals)}, "
                    f"mean={np.mean(vals):.4f}, "
                    f"median={np.median(vals):.4f}, "
                    f"std={np.std(vals):.4f}, "
                    f"min={np.min(vals):.4f}, "
                    f"max={np.max(vals):.4f}"
                )

            print(desc)
            report_lines.append(desc)

        group_arrays = [groups[s] for s in unique_sites if len(groups[s]) > 0]

        if len(group_arrays) >= 2:
            try:
                kw_stat, kw_p = scipy_stats.kruskal(*group_arrays)
            except Exception:
                kw_stat, kw_p = np.nan, np.nan
        else:
            kw_stat, kw_p = np.nan, np.nan

        significant = bool(np.isfinite(kw_p) and kw_p < 0.05)

        kw_line = (
            f"  Kruskal-Wallis H-test: H={kw_stat:.4f}, "
            f"p={kw_p:.2e}, "
            f"{'*** SIGNIFICANT' if significant else 'not significant'} "
            f"(alpha=0.05)"
        )
        print(kw_line)
        report_lines.append(kw_line)

        result_row = {
            "feature": feat_name,
            "kruskal_wallis_H": kw_stat,
            "kruskal_wallis_p": kw_p,
            "kruskal_wallis_significant": significant,
        }

        for site in unique_sites:
            vals = groups[site]
            result_row[f"{site}_n"] = len(vals)
            result_row[f"{site}_mean_shap"] = np.mean(vals) if len(vals) else np.nan
            result_row[f"{site}_median_shap"] = np.median(vals) if len(vals) else np.nan
            result_row[f"{site}_std_shap"] = np.std(vals) if len(vals) else np.nan

        if significant and n_sites >= 2:
            report_lines.append(
                "  Pairwise Mann-Whitney U tests (Bonferroni corrected):"
            )
            print("  Pairwise Mann-Whitney U tests (Bonferroni corrected):")

            for site_a, site_b in combinations(unique_sites, 2):
                vals_a = groups[site_a]
                vals_b = groups[site_b]

                if len(vals_a) == 0 or len(vals_b) == 0:
                    continue

                try:
                    u_stat, u_p = scipy_stats.mannwhitneyu(
                        vals_a,
                        vals_b,
                        alternative="two-sided",
                    )
                    u_p_corrected = min(u_p * max(n_comparisons, 1), 1.0)

                    n1, n2 = len(vals_a), len(vals_b)
                    r_effect = 1 - (2 * u_stat) / (n1 * n2)

                    if u_p_corrected < 0.001:
                        sig_marker = "***"
                    elif u_p_corrected < 0.01:
                        sig_marker = "**"
                    elif u_p_corrected < 0.05:
                        sig_marker = "*"
                    else:
                        sig_marker = "n.s."

                    pair_line = (
                        f"    {site_a} vs {site_b}: "
                        f"U={u_stat:.1f}, "
                        f"p={u_p:.2e}, "
                        f"p_corrected={u_p_corrected:.2e} {sig_marker}, "
                        f"effect_size_r={r_effect:.4f}"
                    )

                    print(pair_line)
                    report_lines.append(pair_line)

                    result_row[f"mwu_{site_a}_vs_{site_b}_U"] = u_stat
                    result_row[f"mwu_{site_a}_vs_{site_b}_p"] = u_p
                    result_row[f"mwu_{site_a}_vs_{site_b}_p_corrected"] = u_p_corrected
                    result_row[f"mwu_{site_a}_vs_{site_b}_significant"] = (
                        u_p_corrected < 0.05
                    )
                    result_row[f"mwu_{site_a}_vs_{site_b}_effect_r"] = r_effect

                except Exception as e:
                    pair_line = (
                        f"    {site_a} vs {site_b}: failed "
                        f"({type(e).__name__}: {e})"
                    )
                    print(pair_line)
                    report_lines.append(pair_line)

        else:
            report_lines.append(
                "  Pairwise tests skipped (Kruskal-Wallis not significant)."
            )
            print("  Pairwise tests skipped (Kruskal-Wallis not significant).")

        all_results.append(result_row)
        report_lines.append("")

    n_significant = sum(
        1 for r in all_results
        if bool(r.get("kruskal_wallis_significant", False))
    )

    summary_line = (
        f"\nSUMMARY: {n_significant}/{len(all_results)} top features show "
        f"statistically significant differences in SHAP values between sites."
    )
    print(summary_line)
    report_lines.append(summary_line)

    if n_significant > 0:
        warning = (
            "WARNING: Site-dependent SHAP value distributions suggest the model "
            "may be capturing site-specific effects rather than, or in addition to, "
            "generalizable biomarkers."
        )
        print(warning)
        report_lines.append(warning)
    else:
        ok_msg = (
            "OK: No significant site differences detected in SHAP values for the "
            "top features."
        )
        print(ok_msg)
        report_lines.append(ok_msg)

    results_df = pd.DataFrame(all_results)

    csv_path = output_dir / "shap_site_analysis.csv"
    report_path = output_dir / "shap_site_analysis_report.txt"

    results_df.to_csv(csv_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {report_path}")

    return results_df


# ==============================================================================
# 8. EXPORT SHAP VALUES
# ==============================================================================

def export_shap_values(shap_explanation, X_df, output_dir):
    """
    Export SHAP values and global importance.
    """
    values = np.asarray(shap_explanation.values)

    shap_df = pd.DataFrame(
        values,
        columns=X_df.columns,
        index=X_df.index,
    )

    csv_path = output_dir / "shap_values.csv"
    parquet_path = output_dir / "shap_values.parquet"

    shap_df.to_csv(csv_path)

    try:
        shap_df.to_parquet(parquet_path)
        print(f"Saved: {parquet_path}")
    except Exception as e:
        print(
            f"WARNING: Could not save SHAP parquet: "
            f"{type(e).__name__}: {e}"
        )

    print(f"Saved: {csv_path}")

    denom = np.sum(np.mean(np.abs(values), axis=0))

    mean_abs_shap = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": np.mean(np.abs(values), axis=0),
        "mean_shap": np.mean(values, axis=0),
        "std_shap": np.std(values, axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    mean_abs_shap["rank"] = range(1, len(mean_abs_shap) + 1)

    if denom > 0:
        mean_abs_shap["cumulative_importance"] = (
            mean_abs_shap["mean_abs_shap"].cumsum() / denom
        )
    else:
        mean_abs_shap["cumulative_importance"] = np.nan

    importance_path = output_dir / "shap_feature_importance.csv"
    mean_abs_shap.to_csv(importance_path, index=False)

    print(f"Saved: {importance_path}")

    print("\nTop 10 Features by mean |SHAP value|:")
    print("-" * 90)

    for _, row in mean_abs_shap.head(10).iterrows():
        direction = "↑ CI / positive class" if row["mean_shap"] > 0 else "↓ CI / negative class"
        print(
            f"  {int(row['rank']):3d}. "
            f"{row['feature']:<55s} "
            f"|SHAP|={row['mean_abs_shap']:.6f}  "
            f"mean={row['mean_shap']:.6f}  "
            f"({direction})"
        )

    return mean_abs_shap


def compare_with_model_importance(artifacts, mean_abs_shap, output_dir):
    """
    Compare SHAP importance with model's saved feature importance, if available.
    """
    feature_importance = artifacts.get("feature_importance")

    if feature_importance is None:
        return

    if "feature" not in feature_importance.columns:
        return

    print("\nComparing SHAP importance vs. model feature importance...")

    model_imp = feature_importance.copy()

    if "importance_normalized" in model_imp.columns:
        model_imp = model_imp[["feature", "importance_normalized"]].rename(
            columns={"importance_normalized": "model_importance"}
        )
    elif "importance" in model_imp.columns:
        model_imp = model_imp[["feature", "importance"]].rename(
            columns={"importance": "model_importance_raw"}
        )
        total = model_imp["model_importance_raw"].sum()
        if total > 0:
            model_imp["model_importance"] = model_imp["model_importance_raw"] / total
        else:
            model_imp["model_importance"] = 0.0
        model_imp = model_imp[["feature", "model_importance"]]
    else:
        return

    shap_imp = mean_abs_shap[["feature", "mean_abs_shap"]].copy()
    total_shap = shap_imp["mean_abs_shap"].sum()

    if total_shap > 0:
        shap_imp["shap_importance"] = shap_imp["mean_abs_shap"] / total_shap
    else:
        shap_imp["shap_importance"] = 0.0

    comparison = (
        model_imp.merge(
            shap_imp[["feature", "shap_importance"]],
            on="feature",
            how="outer",
        )
        .fillna(0)
        .sort_values("shap_importance", ascending=False)
    )

    out_path = output_dir / "importance_comparison.csv"
    comparison.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")


# ==============================================================================
# 9. MAIN
# ==============================================================================

def main():
    warnings.filterwarnings("default")

    model_dir, features_path, output_dir = resolve_paths()

    print("=" * 70)
    print("SHAP VALUE CALCULATION")
    print(f"Model: {model_dir}")
    print(f"Features: {features_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}\n"
            "Set SHAP_MODEL_DIR to the correct model folder."
        )

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file does not exist: {features_path}\n"
            "Set SHAP_FEATURES_PATH to the correct features_patient_level.parquet."
        )

    artifacts = load_model_artifacts(model_dir)
    model = artifacts["model"]

    X_df, patient_level = prepare_data(features_path, artifacts)

    shap_explanation, explainer = calculate_shap_values(model, X_df)

    mean_abs_shap = export_shap_values(shap_explanation, X_df, output_dir)

    print("\nGenerating SHAP plots...")

    plot_shap_summary_bar(shap_explanation, X_df, output_dir)
    plot_shap_beeswarm(shap_explanation, X_df, output_dir)
    plot_shap_beeswarm_by_site(shap_explanation, X_df, output_dir)

    analyze_shap_by_site(shap_explanation, X_df, output_dir)

    plot_shap_heatmap(shap_explanation, output_dir)

    n_waterfall = min(3, len(X_df))
    for i in range(n_waterfall):
        plot_shap_waterfall(shap_explanation, i, output_dir)

    top_features = mean_abs_shap.head(5)["feature"].tolist()
    for feat in top_features:
        plot_shap_dependence(shap_explanation, X_df, feat, output_dir)

    compare_with_model_importance(artifacts, mean_abs_shap, output_dir)

    print("\n" + "=" * 70)
    print("SHAP analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
