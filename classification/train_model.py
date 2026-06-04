"""
05_classification/train_model.py
==================================
Machine Learning Training für die PhysioNet Challenge 2026:
Vorhersage von Cognitive Impairment aus PSG-Daten.

Dieses Modul:
- Trainiert mehrere ML-Modelle (XGBoost, LightGBM, Random Forest, Logistic Regression)
- Führt Hyperparameter-Tuning durch (Optuna oder GridSearch)
- Behandelt Klassen-Imbalance (SMOTE, Class Weights)
- Implementiert Feature Selection
- Speichert trainierte Modelle und Konfigurationen
- Unterstützt sowohl Patient-Level als auch Segment-Level Training

Referenz: Die PhysioNet Challenge 2026 erwartet train_model.py und
run_model.py Skripte.
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    balanced_accuracy_score, average_precision_score,
    make_scorer
)

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb

# Imbalanced Learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

from config import (
    MODEL_DIR, FEATURE_DIR, RANDOM_SEED,
    TARGET_COLUMN, TIME_TO_EVENT_COLUMN,
    CV_FOLDS, TEST_SIZE
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# CUSTOM SELECTORS (Modul-Ebene für Pickle-Kompatibilität!)
# ==============================================================================

class CombinedSelector:
    """Feature Selector basierend auf kombiniertem ANOVA + MI Score."""
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
    """Feature Selector basierend auf Modell-Importance."""
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

def _prior_probability_shift(
    p: np.ndarray,
    train_prevalence: float,
    target_prevalence: Optional[float],
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Adjust predicted probabilities from the training prior to the expected
    deployment/test prior.

    odds_target = odds_train *
        [pi_target / (1 - pi_target)] / [pi_train / (1 - pi_train)]
    """
    if target_prevalence is None:
        return p

    train_prevalence = float(np.clip(train_prevalence, eps, 1.0 - eps))
    target_prevalence = float(np.clip(target_prevalence, eps, 1.0 - eps))

    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)

    odds = p / (1.0 - p)

    train_odds_prior = train_prevalence / (1.0 - train_prevalence)
    target_odds_prior = target_prevalence / (1.0 - target_prevalence)

    correction = target_odds_prior / train_odds_prior
    adjusted_odds = odds * correction

    return adjusted_odds / (1.0 + adjusted_odds)


def _fit_preprocessor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    feature_selection: bool = True,
    n_features_select: Optional[int] = None,
    logger=None,
):
    """
    Fit imputer, feature selector, and scaler ONLY on the training fold.
    This avoids leakage into validation folds.
    """
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train).astype(np.float32)

    feature_selector = None
    selected_features = feature_names.copy()

    if feature_selection:
        X_sel, selected_features, feature_selector = select_features(
            X_imp,
            y_train,
            feature_names,
            n_features=n_features_select,
            logger=logger,
        )
    else:
        X_sel = X_imp

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel).astype(np.float32)

    return X_scaled, imputer, feature_selector, scaler, selected_features


def _transform_preprocessor(
    X: np.ndarray,
    imputer: SimpleImputer,
    feature_selector: Optional[Any],
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply already-fitted preprocessing to validation/test data.
    """
    X_imp = imputer.transform(X).astype(np.float32)

    if feature_selector is not None:
        X_sel = feature_selector.transform(X_imp)
    else:
        X_sel = X_imp

    X_scaled = scaler.transform(X_sel).astype(np.float32)

    return X_scaled

class PrevalenceAdjustedKFold:
    """
    Custom K-Fold split that downsamples the positive class in the validation set 
    to match an expected prevalence. Unused positive samples are shifted back to 
    the training set so no data is wasted.
    """
    def __init__(self, n_splits=5, expected_prevalence=0.10, random_state=42):
        self.n_splits = n_splits
        self.expected_prevalence = expected_prevalence
        self.random_state = random_state

    def split(self, X, y, groups=None):
        rng = np.random.RandomState(self.random_state)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in skf.split(X, y):
            y_val = y[val_idx]
            pos_idx = val_idx[y_val == 1]
            neg_idx = val_idx[y_val == 0]
            
            # prevalence = P / (P + N) => P = expected_prevalence * N / (1 - expected_prevalence)
            target_pos_count = int(len(neg_idx) * self.expected_prevalence / (1.0 - self.expected_prevalence))
            target_pos_count = max(1, target_pos_count)
            
            if target_pos_count < len(pos_idx):
                selected_pos_idx = rng.choice(pos_idx, target_pos_count, replace=False)
                new_val_idx = np.concatenate([neg_idx, selected_pos_idx])
                
                unused_pos_idx = np.setdiff1d(pos_idx, selected_pos_idx)
                new_train_idx = np.concatenate([train_idx, unused_pos_idx])
            else:
                new_val_idx = val_idx
                new_train_idx = train_idx
                
            rng.shuffle(new_train_idx)
            rng.shuffle(new_val_idx)
            
            yield new_train_idx, new_val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# ==============================================================================
# HAUPT-FUNKTION: MODELL TRAINIEREN
# ==============================================================================

def train_model(
    feature_table: pd.DataFrame,
    model_type: str = "ensemble",
    feature_selection: bool = True,
    handle_imbalance: str = "class_weight",
    expected_test_prevalence: float = 0.10,
    n_features_select: Optional[int] = None,
    output_dir: Path = MODEL_DIR,
    logger=None
) -> Dict:
    if logger:
        logger.info("=" * 60)
        logger.info("MODELL-TRAINING")
        logger.info(f"Modelltyp: {model_type}")
        logger.info(f"Feature Selection: {feature_selection}")
        logger.info(f"Imbalance-Handling: {handle_imbalance}")
        logger.info("=" * 60)

    X, y, feature_names, preparation_info = prepare_training_data(
        feature_table, logger
    )

    if X is None or y is None:
        if logger:
            logger.error("Datenaufbereitung fehlgeschlagen!")
        return {}

    if logger:
        logger.info(f"Trainingsdaten: {X.shape[0]} Patienten, "
                     f"{X.shape[1]} Features")
        logger.info(f"Target-Verteilung: {dict(zip(*np.unique(y, return_counts=True)))}")

    train_prevalence = float(np.mean(y))

    if logger:
        logger.info(f"Trainingsdaten: {X.shape[0]} Patienten, {X.shape[1]} Features")
        logger.info(f"Target-Verteilung: {dict(zip(*np.unique(y, return_counts=True)))}")
        logger.info(f"Trainingsprävalenz: {train_prevalence:.3f}")
        logger.info(f"Erwartete Testprävalenz: {expected_test_prevalence:.3f}")

    # ------------------------------------------------------------------
    # Leakage-safe CV:
    # imputer, selector, and scaler are fit separately inside each fold.
    # ------------------------------------------------------------------
    cv_results = _cross_validate_model_leakage_free(
        model_type=model_type,
        X=X,
        y=y,
        feature_names=feature_names,
        feature_selection=feature_selection,
        handle_imbalance=handle_imbalance,
        expected_test_prevalence=expected_test_prevalence,
        n_features_select=n_features_select,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Final preprocessing on all available training patients
    # ------------------------------------------------------------------
    X_processed, imputer, feature_selector, scaler, selected_features = _fit_preprocessor(
        X_train=X,
        y_train=y,
        feature_names=feature_names,
        feature_selection=feature_selection,
        n_features_select=n_features_select,
        logger=logger,
    )

    if logger:
        logger.info(f"Features nach Selection: {len(selected_features)}")

    # Important:
    # Do not use expected_test_prevalence as class weight.
    # Use it only later for probability prior correction.
    model = _create_model(
        model_type=model_type,
        handle_imbalance=handle_imbalance,
        n_features=X_processed.shape[1],
        y=y,
        logger=logger,
        expected_test_prevalence=None,
    )

    if handle_imbalance == "smote":
        smote = SMOTE(random_state=RANDOM_SEED)
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    elif handle_imbalance == "adasyn":
        adasyn = ADASYN(random_state=RANDOM_SEED)
        X_resampled, y_resampled = adasyn.fit_resample(X_processed, y)
    elif handle_imbalance == "smote_tomek":
        smt = SMOTETomek(random_state=RANDOM_SEED)
        X_resampled, y_resampled = smt.fit_resample(X_processed, y)
    else:
        X_resampled, y_resampled = X_processed, y

    model.fit(X_resampled, y_resampled)

    if logger:
        logger.info("Finales Modell auf allen Daten trainiert.")


    if logger:
        logger.info(f"Features nach Selection: {len(selected_features)}")

    model = _create_model(
        model_type=model_type,
        handle_imbalance=handle_imbalance,
        n_features=X_scaled.shape[1],
        y=y,
        logger=logger,
        expected_test_prevalence=expected_test_prevalence
    )

    cv_results = _cross_validate_model(
        model, X_scaled, y, handle_imbalance, logger, expected_test_prevalence
    )

    if handle_imbalance == "smote":
        smote = SMOTE(random_state=RANDOM_SEED)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    elif handle_imbalance == "adasyn":
        adasyn = ADASYN(random_state=RANDOM_SEED)
        X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y)
    elif handle_imbalance == "smote_tomek":
        smt = SMOTETomek(random_state=RANDOM_SEED)
        X_resampled, y_resampled = smt.fit_resample(X_scaled, y)
    else:
        X_resampled, y_resampled = X_scaled, y

    model.fit(X_resampled, y_resampled)

    if logger:
        logger.info("Finales Modell auf allen Daten trainiert.")

    feature_importance = _get_feature_importance(
        model, selected_features, logger
    )

    training_config = {
        "model_type": model_type,
        "feature_selection": feature_selection,
        "handle_imbalance": handle_imbalance,
        "n_features_original": len(feature_names),
        "n_features_selected": len(selected_features),
        "n_patients": X.shape[0],
        "target_distribution": dict(zip(*np.unique(y, return_counts=True))),
        "training_prevalence": train_prevalence,
        "expected_test_prevalence": expected_test_prevalence,
        "probability_prior_adjustment": True,
        "decision_threshold": 0.5,
        "cv_folds": CV_FOLDS,
        "random_seed": RANDOM_SEED,
        "timestamp": datetime.now().isoformat(),
    }


    result = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_selector": feature_selector,
        "feature_names": feature_names,
        "selected_features": selected_features,
        "cv_results": cv_results,
        "feature_importance": feature_importance,
        "training_config": training_config,
        "preparation_info": preparation_info,
    }

    _save_model(result, output_dir, model_type, logger)

    return result


# ==============================================================================
# DATENAUFBEREITUNG
# ==============================================================================

def prepare_training_data(
    feature_table: pd.DataFrame,
    logger=None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], Dict]:
    info = {
        "n_patients_original": len(feature_table),
        "n_features_original": 0,
        "n_patients_with_target": 0,
        "removed_features": [],
        "removed_patients": 0,
    }

    df = feature_table.copy()

    target_col = None
    for candidate in ["target", TARGET_COLUMN, "Cognitive_Impairment"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        if logger:
            logger.error("Keine Target-Variable gefunden!")
        return None, None, [], info

    df = df.dropna(subset=[target_col])
    y = df[target_col].values.astype(int)
    info["n_patients_with_target"] = len(df)
    info["removed_patients"] = info["n_patients_original"] - len(df)

    if logger:
        logger.info(f"Patienten mit Target: {len(df)}/{info['n_patients_original']}")

    exclude_cols = [
        "patient_id", "segment_idx", target_col,
        TARGET_COLUMN, TIME_TO_EVENT_COLUMN,
        "Cognitive_Impairment", "Time_to_Event",
        "SiteID", "BDSPPatientID", "CreationTime",
        "BidsFolder", "SessionID", "Last_Known_Visit_Date",
        "Time_to_Last_Visit",
        "Age", "Sex", "Race", "Ethnicity", "BMI",
        "demo_time_to_event", "demo_time_to_last_visit",
    ]

    string_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    exclude_cols.extend(string_cols)

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    valid_features = []
    for col in feature_cols:
        nan_ratio = df[col].isna().mean()
        if nan_ratio <= 0.5:
            valid_features.append(col)
        else:
            info["removed_features"].append((col, f"NaN: {nan_ratio:.1%}"))

    final_features = []
    for col in valid_features:
        if df[col].nunique(dropna=True) > 1:
            final_features.append(col)
        else:
            info["removed_features"].append((col, "zero variance"))

    info["n_features_original"] = len(final_features)

    X = df[final_features].values.astype(np.float32)

    if logger:
        logger.info(f"Features: {len(final_features)} "
                     f"(entfernt: {len(info['removed_features'])})")
        if info["removed_features"]:
            for feat, reason in info["removed_features"][:10]:
                logger.debug(f"  Entfernt: {feat} ({reason})")
            if len(info["removed_features"]) > 10:
                logger.debug(f"  ... und {len(info['removed_features'])-10} weitere")

    return X, y, final_features, info


# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features: Optional[int] = None,
    method: str = "combined",
    logger=None
) -> Tuple[np.ndarray, List[str], Any]:
    if n_features is None:
        n_features = min(
            int(np.sqrt(X.shape[0]) * 3),
            min(100, X.shape[1])
        )
        n_features = max(n_features, 10)

    n_features = min(n_features, X.shape[1])

    if logger:
        logger.info(f"Feature Selection: {X.shape[1]} -> {n_features} Features "
                     f"(Methode: {method})")

    if method == "anova":
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        mask = selector.get_support()

    elif method == "mutual_info":
        selector = SelectKBest(
            lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_SEED),
            k=n_features
        )
        X_selected = selector.fit_transform(X, y)
        mask = selector.get_support()

    elif method == "combined":
        selector, mask = _combined_feature_selection(
            X, y, feature_names, n_features, logger
        )
        X_selected = X[:, mask]

    elif method == "model_based":
        selector, mask = _model_based_feature_selection(
            X, y, feature_names, n_features, logger
        )
        X_selected = X[:, mask]

    else:
        raise ValueError(f"Unbekannte Feature Selection Methode: {method}")

    selected_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]

    if logger:
        logger.info(f"Selektierte Features: {len(selected_names)}")
        if hasattr(selector, 'scores_') and selector.scores_ is not None:
            scores = selector.scores_
            top_indices = np.argsort(scores)[::-1][:20]
            logger.info("Top 20 Features:")
            for idx in top_indices:
                if idx < len(feature_names):
                    logger.info(f"  {feature_names[idx]}: {scores[idx]:.4f}")

    return X_selected, selected_names, selector


def _combined_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features: int,
    logger=None
) -> Tuple[Any, np.ndarray]:
    n_total = X.shape[1]

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X).astype(np.float32)

    try:
        f_scores, f_pvalues = f_classif(X_imp, y)
        f_scores = np.nan_to_num(f_scores, nan=0)
    except Exception:
        f_scores = np.zeros(n_total)

    try:
        mi_scores = mutual_info_classif(X_imp, y, random_state=RANDOM_SEED)
        mi_scores = np.nan_to_num(mi_scores, nan=0)
    except Exception:
        mi_scores = np.zeros(n_total)

    f_max = f_scores.max()
    if f_max > 0:
        f_norm = f_scores / f_max
    else:
        f_norm = f_scores

    mi_max = mi_scores.max()
    if mi_max > 0:
        mi_norm = mi_scores / mi_max
    else:
        mi_norm = mi_scores

    combined_scores = 0.5 * f_norm + 0.5 * mi_norm

    top_indices = np.argsort(combined_scores)[::-1][:n_features]
    mask = np.zeros(n_total, dtype=bool)
    mask[top_indices] = True

    selector = CombinedSelector(mask, combined_scores, f_scores, mi_scores)

    if logger:
        logger.debug(f"Combined Selection: ANOVA + MI")
        logger.debug(f"  ANOVA max score: {f_max:.4f}")
        logger.debug(f"  MI max score: {mi_max:.4f}")

    return selector, mask


def _model_based_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features: int,
    logger=None
) -> Tuple[Any, np.ndarray]:
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X).astype(np.float32)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(X_imp, y)
    importances = model.feature_importances_

    top_indices = np.argsort(importances)[::-1][:n_features]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_indices] = True

    selector = ModelSelector(mask, importances)

    return selector, mask



# ==============================================================================
# MODELL-ERSTELLUNG
# ==============================================================================

def _create_model(
    model_type: str,
    handle_imbalance: str,
    n_features: int,
    y: np.ndarray,
    logger=None,
    expected_test_prevalence: Optional[float] = 0.10
) -> Any:
    class_counts = np.bincount(y.astype(int))
    if len(class_counts) >= 2 and class_counts.min() > 0:
        if expected_test_prevalence is not None:
            # Adjust weights to match the expected test set prevalence (Prior Probability Shift)
            target_pos_ratio = expected_test_prevalence / (1.0 - expected_test_prevalence)
            actual_pos_ratio = class_counts[1] / class_counts[0]
            scale_pos_weight = target_pos_ratio / actual_pos_ratio
        else:
            # Standard balancing
            scale_pos_weight = class_counts[0] / class_counts[1]
    else:
        scale_pos_weight = 1.0

    use_class_weight = handle_imbalance == "class_weight"
    
    # Create class_weight dictionary for models that support it (RF, LogReg)
    class_weight_dict = {0: 1.0, 1: scale_pos_weight} if use_class_weight else None


    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.5,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            random_state=RANDOM_SEED,
            eval_metric="auc",
            # use_label_encoder=False,
            n_jobs=-1,
            early_stopping_rounds=None,
        )

    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            metric="auc",
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight=class_weight_dict,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    elif model_type == "logistic_regression":
        model = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight=class_weight_dict,
            random_state=RANDOM_SEED,
        )

    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
        )

    elif model_type == "ensemble":
        model = _create_voting_ensemble(
            use_class_weight, scale_pos_weight, class_weight_dict
        )

    elif model_type == "stacking":
        model = _create_stacking_ensemble(
            use_class_weight, scale_pos_weight, class_weight_dict
        )

    else:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}")

    if logger:
        logger.info(f"Modell erstellt: {model_type}")
        if use_class_weight:
            logger.info(f"  Class Weight / Scale Pos Weight: {scale_pos_weight:.2f}")

    return model


def _create_voting_ensemble(
    use_class_weight: bool,
    scale_pos_weight: float,
    class_weight_dict: Optional[dict] = None
) -> VotingClassifier:
    estimators = [
        ("xgb", xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            random_state=RANDOM_SEED,
            eval_metric="auc",
            n_jobs=-1,
        )),
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            reg_alpha=1.0,
            reg_lambda=2.0,
            metric="auc",
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1,
    )


def _create_stacking_ensemble(
    use_class_weight: bool,
    scale_pos_weight: float,
    class_weight_dict: Optional[dict] = None
) -> StackingClassifier:
    estimators = [
        ("xgb", xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            random_state=RANDOM_SEED,
            eval_metric="auc",
            n_jobs=-1,
        )),
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            scale_pos_weight=scale_pos_weight if use_class_weight else 1.0,
            reg_alpha=1.0,
            reg_lambda=2.0,
            metric="auc",
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=1.0,
            class_weight=class_weight_dict,
            random_state=RANDOM_SEED,
            max_iter=1000,
        ),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )


# ==============================================================================
# CROSS-VALIDATION
# ==============================================================================
def _cross_validate_model_leakage_free(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    feature_selection: bool,
    handle_imbalance: str,
    expected_test_prevalence: Optional[float] = 0.10,
    n_features_select: Optional[int] = None,
    logger=None,
) -> Dict:
    """
    Cross-validation without preprocessing leakage.

    For every fold:
      1. Fit imputer on training fold only.
      2. Fit feature selector on training fold only.
      3. Fit scaler on training fold only.
      4. Train model on processed training fold.
      5. Evaluate on processed validation fold.
    """
    if expected_test_prevalence is not None:
        cv = PrevalenceAdjustedKFold(
            n_splits=CV_FOLDS,
            expected_prevalence=expected_test_prevalence,
            random_state=RANDOM_SEED,
        )
        if logger:
            logger.info(
                f"Leakage-safe CV: {CV_FOLDS}-Fold, "
                f"validation prevalence adjusted to "
                f"{expected_test_prevalence * 100:.1f}%"
            )
    else:
        cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        if logger:
            logger.info(f"Leakage-safe CV: {CV_FOLDS}-Fold Stratified")

    metrics = {
        "auroc": [],
        "average_precision": [],
        "f1": [],
        "balanced_accuracy": [],
        "accuracy": [],
        "auroc_train": [],
        "f1_train": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            if logger:
                logger.warning(
                    f"Fold {fold_idx + 1}: skipped because train or "
                    f"validation has only one class."
                )
            continue

        # Fit preprocessing on training fold only.
        X_train, imputer, selector, scaler, selected_features = _fit_preprocessor(
            X_train=X_train_raw,
            y_train=y_train,
            feature_names=feature_names,
            feature_selection=feature_selection,
            n_features_select=n_features_select,
            logger=None,
        )

        X_val = _transform_preprocessor(
            X=X_val_raw,
            imputer=imputer,
            feature_selector=selector,
            scaler=scaler,
        )

        # Create fresh model for this fold.
        fold_model = _create_model(
            model_type=model_type,
            handle_imbalance=handle_imbalance,
            n_features=X_train.shape[1],
            y=y_train,
            logger=None,
            expected_test_prevalence=None,
        )

        # Optional resampling inside training fold only.
        if handle_imbalance == "smote":
            try:
                resampler = SMOTE(random_state=RANDOM_SEED)
                X_fit, y_fit = resampler.fit_resample(X_train, y_train)
            except Exception:
                X_fit, y_fit = X_train, y_train
        elif handle_imbalance == "adasyn":
            try:
                resampler = ADASYN(random_state=RANDOM_SEED)
                X_fit, y_fit = resampler.fit_resample(X_train, y_train)
            except Exception:
                X_fit, y_fit = X_train, y_train
        elif handle_imbalance == "smote_tomek":
            try:
                resampler = SMOTETomek(random_state=RANDOM_SEED)
                X_fit, y_fit = resampler.fit_resample(X_train, y_train)
            except Exception:
                X_fit, y_fit = X_train, y_train
        else:
            X_fit, y_fit = X_train, y_train

        try:
            fold_model.fit(X_fit, y_fit)

            # Raw model probabilities.
            p_val_raw = fold_model.predict_proba(X_val)[:, 1]
            p_train_raw = fold_model.predict_proba(X_train)[:, 1]

            # Prior-adjusted probabilities for expected deployment prevalence.
            train_prev = float(np.mean(y_train))
            p_val = _prior_probability_shift(
                p_val_raw,
                train_prevalence=train_prev,
                target_prevalence=expected_test_prevalence,
            )
            p_train = _prior_probability_shift(
                p_train_raw,
                train_prevalence=train_prev,
                target_prevalence=expected_test_prevalence,
            )

            y_val_pred = (p_val >= 0.5).astype(int)
            y_train_pred = (p_train >= 0.5).astype(int)

            metrics["auroc"].append(roc_auc_score(y_val, p_val))
            metrics["average_precision"].append(average_precision_score(y_val, p_val))
            metrics["f1"].append(f1_score(y_val, y_val_pred, zero_division=0))
            metrics["balanced_accuracy"].append(
                balanced_accuracy_score(y_val, y_val_pred)
            )
            metrics["accuracy"].append(accuracy_score(y_val, y_val_pred))

            metrics["auroc_train"].append(roc_auc_score(y_train, p_train))
            metrics["f1_train"].append(
                f1_score(y_train, y_train_pred, zero_division=0)
            )

            if logger:
                logger.info(
                    f"Fold {fold_idx + 1}/{CV_FOLDS}: "
                    f"AUROC={metrics['auroc'][-1]:.4f}, "
                    f"AUPRC={metrics['average_precision'][-1]:.4f}, "
                    f"F1={metrics['f1'][-1]:.4f}, "
                    f"features={len(selected_features)}"
                )

        except Exception as e:
            if logger:
                logger.warning(
                    f"Fold {fold_idx + 1}: training/evaluation failed: {e}"
                )
            continue

    cv_results = {}

    for metric_name, scores in metrics.items():
        if len(scores) == 0:
            continue

        cv_results[f"{metric_name}_mean"] = float(np.mean(scores))
        cv_results[f"{metric_name}_std"] = float(np.std(scores))
        cv_results[f"{metric_name}_scores"] = [float(s) for s in scores]

    if "auroc_mean" in cv_results and "auroc_train_mean" in cv_results:
        gap = cv_results["auroc_train_mean"] - cv_results["auroc_mean"]
        cv_results["overfit_gap_auroc"] = float(gap)

        if logger:
            if gap > 0.15:
                logger.warning(
                    f"Possible overfitting: train AUROC - val AUROC = {gap:.4f}"
                )
            else:
                logger.info(f"Overfitting gap AUROC: {gap:.4f}")

    if logger:
        logger.info("Leakage-safe CV results:")
        for key, value in cv_results.items():
            if key.endswith("_mean"):
                std = cv_results.get(key.replace("_mean", "_std"), 0.0)
                logger.info(f"  {key}: {value:.4f} ± {std:.4f}")

    return cv_results

def _cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    handle_imbalance: str,
    logger=None,
    expected_test_prevalence: Optional[float] = 0.10
) -> Dict:
    if expected_test_prevalence is not None:
        cv = PrevalenceAdjustedKFold(n_splits=CV_FOLDS, expected_prevalence=expected_test_prevalence, random_state=RANDOM_SEED)
        if logger:
            logger.info(f"Cross-Validation: {CV_FOLDS}-Fold (Val-Prevalence adjusted to {expected_test_prevalence*100:.0f}%)")
    else:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        if logger:
            logger.info(f"Cross-Validation: {CV_FOLDS}-Fold Stratified")

    scoring = {
        "auroc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
    }

    if handle_imbalance in ["smote", "adasyn", "smote_tomek"]:
        cv_results = _cv_with_resampling(
            model, X, y, cv, handle_imbalance, logger
        )
    else:
        try:
            cv_output = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
            )

            cv_results = {}
            for metric_name in scoring:
                test_key = f"test_{metric_name}"
                train_key = f"train_{metric_name}"

                if test_key in cv_output:
                    scores = cv_output[test_key]
                    cv_results[f"{metric_name}_mean"] = float(np.mean(scores))
                    cv_results[f"{metric_name}_std"] = float(np.std(scores))
                    cv_results[f"{metric_name}_scores"] = scores.tolist()

                if train_key in cv_output:
                    train_scores = cv_output[train_key]
                    cv_results[f"{metric_name}_train_mean"] = float(np.mean(train_scores))

        except Exception as e:
            if logger:
                logger.warning(f"Standard-CV fehlgeschlagen: {e}, "
                               f"verwende manuelle CV")
            cv_results = _cv_with_resampling(
                model, X, y, cv, "none", logger
            )

    if logger:
        logger.info("Cross-Validation Ergebnisse:")
        for key, value in cv_results.items():
            if key.endswith("_mean"):
                metric = key.replace("_mean", "")
                std_key = f"{metric}_std"
                std = cv_results.get(std_key, 0)
                logger.info(f"  {metric}: {value:.4f} ± {std:.4f}")

    return cv_results


def _cv_with_resampling(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    handle_imbalance: str,
    logger=None
) -> Dict:
    from sklearn.base import clone

    metrics = {
        "auroc": [], "f1": [], "balanced_accuracy": [],
        "average_precision": [], "accuracy": [],
        "auroc_train": [], "f1_train": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if handle_imbalance == "smote":
            try:
                resampler = SMOTE(random_state=RANDOM_SEED)
                X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
            except Exception:
                X_train_res, y_train_res = X_train, y_train
        elif handle_imbalance == "adasyn":
            try:
                resampler = ADASYN(random_state=RANDOM_SEED)
                X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
            except Exception:
                X_train_res, y_train_res = X_train, y_train
        elif handle_imbalance == "smote_tomek":
            try:
                resampler = SMOTETomek(random_state=RANDOM_SEED)
                X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
            except Exception:
                X_train_res, y_train_res = X_train, y_train
        else:
            X_train_res, y_train_res = X_train, y_train

        fold_model = clone(model)

        try:
            fold_model.fit(X_train_res, y_train_res)
        except Exception as e:
            if logger:
                logger.warning(f"Fold {fold_idx+1}: Training fehlgeschlagen: {e}")
            continue

        try:
            y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
            y_pred = fold_model.predict(X_val)

            metrics["auroc"].append(roc_auc_score(y_val, y_pred_proba))
            metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            metrics["balanced_accuracy"].append(balanced_accuracy_score(y_val, y_pred))
            metrics["average_precision"].append(average_precision_score(y_val, y_pred_proba))
            metrics["accuracy"].append(accuracy_score(y_val, y_pred))

            y_train_pred_proba = fold_model.predict_proba(X_train_res)[:, 1]
            y_train_pred = fold_model.predict(X_train_res)
            metrics["auroc_train"].append(roc_auc_score(y_train_res, y_train_pred_proba))
            metrics["f1_train"].append(f1_score(y_train_res, y_train_pred, zero_division=0))

        except Exception as e:
            if logger:
                logger.warning(f"Fold {fold_idx+1}: Evaluation fehlgeschlagen: {e}")
            continue

        if logger:
            logger.debug(
                f"Fold {fold_idx+1}/{CV_FOLDS}: "
                f"AUROC={metrics['auroc'][-1]:.4f}, "
                f"F1={metrics['f1'][-1]:.4f}, "
                f"BA={metrics['balanced_accuracy'][-1]:.4f}"
            )

    cv_results = {}
    for metric_name, scores in metrics.items():
        if scores:
            cv_results[f"{metric_name}_mean"] = float(np.mean(scores))
            cv_results[f"{metric_name}_std"] = float(np.std(scores))
            cv_results[f"{metric_name}_scores"] = scores

    if "auroc_mean" in cv_results and "auroc_train_mean" in cv_results:
        overfit_gap = cv_results["auroc_train_mean"] - cv_results["auroc_mean"]
        cv_results["overfit_gap_auroc"] = overfit_gap
        if logger:
            if overfit_gap > 0.15:
                logger.warning(f"Mögliches Overfitting! "
                               f"Train AUROC - Val AUROC = {overfit_gap:.4f}")
            else:
                logger.info(f"Overfitting-Gap (AUROC): {overfit_gap:.4f}")

    return cv_results


# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================

def _get_feature_importance(
    model: Any,
    feature_names: List[str],
    logger=None
) -> Optional[pd.DataFrame]:
    importances = None

    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        elif hasattr(model, 'estimators_'):
            all_importances = []
            for name, estimator in model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    all_importances.append(estimator.feature_importances_)
            if all_importances:
                importances = np.mean(all_importances, axis=0)
    except Exception as e:
        if logger:
            logger.debug(f"Feature Importance Extraktion fehlgeschlagen: {e}")
        return None

    if importances is None or len(importances) != len(feature_names):
        return None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    total = importance_df["importance"].sum()
    if total > 0:
        importance_df["importance_normalized"] = (
            importance_df["importance"] / total
        )
    else:
        importance_df["importance_normalized"] = 0

    importance_df["importance_cumulative"] = (
        importance_df["importance_normalized"].cumsum()
    )

    importance_df["rank"] = range(1, len(importance_df) + 1)

    if logger:
        logger.info("Top 20 Features nach Importance:")
        for _, row in importance_df.head(20).iterrows():
            logger.info(f"  {row['rank']:3d}. {row['feature']}: "
                        f"{row['importance_normalized']:.4f} "
                        f"(cum: {row['importance_cumulative']:.4f})")

        n_90 = (importance_df["importance_cumulative"] <= 0.90).sum() + 1
        logger.info(f"Features für 90% Importance: {n_90}/{len(importance_df)}")

    return importance_df


# ==============================================================================
# HYPERPARAMETER-TUNING
# ==============================================================================

def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "xgboost",
    n_trials: int = 50,
    handle_imbalance: str = "class_weight",
    logger=None,
    expected_test_prevalence: Optional[float] = 0.10
) -> Dict:
    if logger:
        logger.info(f"Hyperparameter-Tuning: {model_type}, {n_trials} Trials")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return _tune_with_optuna(
            X, y, model_type, n_trials, handle_imbalance, logger, expected_test_prevalence
        )
    except ImportError:
        if logger:
            logger.info("Optuna nicht verfügbar, verwende RandomizedSearch")
        return _tune_with_randomized_search(
            X, y, model_type, n_trials, handle_imbalance, logger, expected_test_prevalence
        )


def _tune_with_optuna(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_trials: int,
    handle_imbalance: str,
    logger=None,
    expected_test_prevalence: Optional[float] = 0.10
) -> Dict:
    import optuna

    class_counts = np.bincount(y.astype(int))
    scale_pos_weight = (
        class_counts[0] / class_counts[1]
        if len(class_counts) >= 2 and class_counts.min() > 0
        else 1.0
    )
    
    if expected_test_prevalence is not None and len(class_counts) >= 2 and class_counts.min() > 0:
        target_pos_ratio = expected_test_prevalence / (1.0 - expected_test_prevalence)
        actual_pos_ratio = class_counts[1] / class_counts[0]
        scale_pos_weight = target_pos_ratio / actual_pos_ratio
        
    use_cw = handle_imbalance == "class_weight"
    class_weight_dict = {0: 1.0, 1: scale_pos_weight} if use_cw else None

    def objective(trial):
        if expected_test_prevalence is not None:
            cv = PrevalenceAdjustedKFold(n_splits=CV_FOLDS, expected_prevalence=expected_test_prevalence, random_state=RANDOM_SEED)
        else:
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        if model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
                "scale_pos_weight": scale_pos_weight if use_cw else 1.0,
                "random_state": RANDOM_SEED,
                "eval_metric": "auc",
                # "use_label_encoder": False,
                "n_jobs": -1,
            }
            model = xgb.XGBClassifier(**params)

        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "scale_pos_weight": scale_pos_weight if use_cw else 1.0,
                "metric": "auc",
                "random_state": RANDOM_SEED,
                "verbose": -1,
                "n_jobs": -1,
            }
            model = lgb.LGBMClassifier(**params)

        elif model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
                ),
                "class_weight": class_weight_dict,
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        else:
            raise ValueError(f"Tuning nicht unterstützt für: {model_type}")

        scores = cross_val_score(
            model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return np.mean(scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    if logger:
        logger.info(f"Bestes AUROC: {best_score:.4f}")
        logger.info(f"Beste Parameter: {best_params}")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "n_trials": n_trials,
        "method": "optuna",
    }


def _tune_with_randomized_search(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_trials: int,
    handle_imbalance: str,
    logger=None,
    expected_test_prevalence: Optional[float] = 0.10
) -> Dict:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint, loguniform

    class_counts = np.bincount(y.astype(int))
    scale_pos_weight = (
        class_counts[0] / class_counts[1]
        if len(class_counts) >= 2 and class_counts.min() > 0
        else 1.0
    )
    
    if expected_test_prevalence is not None and len(class_counts) >= 2 and class_counts.min() > 0:
        target_pos_ratio = expected_test_prevalence / (1.0 - expected_test_prevalence)
        actual_pos_ratio = class_counts[1] / class_counts[0]
        scale_pos_weight = target_pos_ratio / actual_pos_ratio
        
    use_cw = handle_imbalance == "class_weight"
    class_weight_dict = {0: 1.0, 1: scale_pos_weight} if use_cw else None

    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight if use_cw else 1.0,
            random_state=RANDOM_SEED,
            eval_metric="auc",
            n_jobs=-1,
        )
        param_distributions = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 6),
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": randint(1, 10),
            "gamma": uniform(0, 1),
            "reg_alpha": loguniform(0.1, 10),
            "reg_lambda": loguniform(0.1, 10),
        }

    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight if use_cw else 1.0,
            metric="auc",
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1,
        )
        param_distributions = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 6),
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_samples": randint(10, 50),
            "num_leaves": randint(15, 63),
            "reg_alpha": loguniform(0.1, 10),
            "reg_lambda": loguniform(0.1, 10),
        }

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            class_weight=class_weight_dict,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        param_distributions = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 8),
            "min_samples_split": randint(5, 20),
            "min_samples_leaf": randint(2, 10),
        }

    else:
        if logger:
            logger.warning(f"RandomizedSearch nicht konfiguriert für {model_type}")
        return {"best_params": {}, "best_score": 0, "method": "none"}

    if expected_test_prevalence is not None:
        cv = PrevalenceAdjustedKFold(n_splits=CV_FOLDS, expected_prevalence=expected_test_prevalence, random_state=RANDOM_SEED)
    else:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_trials,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X, y)

    if logger:
        logger.info(f"Bestes AUROC: {search.best_score_:.4f}")
        logger.info(f"Beste Parameter: {search.best_params_}")

    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "n_trials": n_trials,
        "method": "randomized_search",
    }


# ==============================================================================
# MODELL SPEICHERN / LADEN
# ==============================================================================

def _save_model(
    result: Dict,
    output_dir: Path,
    model_type: str,
    logger=None
):
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{model_type}_{timestamp}"
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(result["model"], model_path)

    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(result["scaler"], scaler_path)

    imputer_path = model_dir / "imputer.joblib"
    joblib.dump(result["imputer"], imputer_path)

    if result.get("feature_selector") is not None:
        selector_path = model_dir / "feature_selector.joblib"
        joblib.dump(result["feature_selector"], selector_path)

    features_path = model_dir / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump({
            "all_features": result["feature_names"],
            "selected_features": result["selected_features"],
        }, f, indent=2)

    config_path = model_dir / "training_config.json"
    config_serializable = {}
    for key, value in result["training_config"].items():
        if isinstance(value, (int, float, str, bool, list)):
            config_serializable[key] = value
        elif isinstance(value, dict):
            config_serializable[key] = {
                str(k): int(v) if isinstance(v, np.integer) else v
                for k, v in value.items()
            }
        elif isinstance(value, np.integer):
            config_serializable[key] = int(value)
        elif isinstance(value, np.floating):
            config_serializable[key] = float(value)
        else:
            config_serializable[key] = str(value)

    with open(config_path, "w") as f:
        json.dump(config_serializable, f, indent=2)

    cv_path = model_dir / "cv_results.json"
    cv_serializable = {}
    for key, value in result["cv_results"].items():
        if isinstance(value, (list, np.ndarray)):
            cv_serializable[key] = [float(v) for v in value]
        elif isinstance(value, (int, float, np.integer, np.floating)):
            cv_serializable[key] = float(value)
        else:
            cv_serializable[key] = str(value)

    with open(cv_path, "w") as f:
        json.dump(cv_serializable, f, indent=2)

    if result.get("feature_importance") is not None:
        importance_path = model_dir / "feature_importance.csv"
        result["feature_importance"].to_csv(importance_path, index=False)

    latest_link = output_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(model_dir.name)
    except OSError:
        with open(output_dir / "latest.txt", "w") as f:
            f.write(str(model_dir))

    if logger:
        logger.info(f"Modell gespeichert: {model_dir}")
        logger.info(f"  model.joblib: {model_path.stat().st_size / 1024:.1f} KB")


def load_model(
    model_dir: Optional[Path] = None,
    logger=None
) -> Optional[Dict]:
    if model_dir is None:
        latest_link = MODEL_DIR / "latest"
        latest_txt = MODEL_DIR / "latest.txt"

        if latest_link.exists() or latest_link.is_symlink():
            model_dir = MODEL_DIR / latest_link.resolve().name
        elif latest_txt.exists():
            with open(latest_txt, "r") as f:
                model_dir = Path(f.read().strip())
        else:
            if logger:
                logger.error("Kein gespeichertes Modell gefunden!")
            return None

    if not model_dir.exists():
        if logger:
            logger.error(f"Modell-Verzeichnis nicht gefunden: {model_dir}")
        return None

    result = {}

    try:
        result["model"] = joblib.load(model_dir / "model.joblib")
        result["scaler"] = joblib.load(model_dir / "scaler.joblib")
        result["imputer"] = joblib.load(model_dir / "imputer.joblib")

        selector_path = model_dir / "feature_selector.joblib"
        if selector_path.exists():
            result["feature_selector"] = joblib.load(selector_path)
        else:
            result["feature_selector"] = None

        with open(model_dir / "feature_names.json", "r") as f:
            feature_data = json.load(f)
            result["feature_names"] = feature_data["all_features"]
            result["selected_features"] = feature_data["selected_features"]

        config_path = model_dir / "training_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                result["training_config"] = json.load(f)

        cv_path = model_dir / "cv_results.json"
        if cv_path.exists():
            with open(cv_path, "r") as f:
                result["cv_results"] = json.load(f)

        importance_path = model_dir / "feature_importance.csv"
        if importance_path.exists():
            result["feature_importance"] = pd.read_csv(importance_path)

        if logger:
            logger.info(f"Modell geladen: {model_dir}")
            if "training_config" in result:
                logger.info(f"  Typ: {result['training_config'].get('model_type', 'unknown')}")
                logger.info(f"  Features: {len(result['selected_features'])}")

        return result

    except Exception as e:
        if logger:
            logger.error(f"Fehler beim Laden des Modells: {e}")
        return None


# ==============================================================================
# MULTI-MODEL TRAINING
# ==============================================================================

def train_multiple_models(
    feature_table: pd.DataFrame,
    model_types: Optional[List[str]] = None,
    handle_imbalance: str = "class_weight",
    feature_selection: bool = True,
    expected_test_prevalence: float = 0.10,
    tune: bool = True,
    n_tune_trials: int = 100,
    output_dir: Path = MODEL_DIR,
    logger=None
) -> Dict[str, Dict]:
    if model_types is None:
        model_types = [
            "xgboost", "lightgbm", "random_forest",
            "logistic_regression", "ensemble"
        ]

    if logger:
        logger.info("=" * 60)
        logger.info(f"MULTI-MODEL TRAINING: {len(model_types)} Modelle")
        logger.info("=" * 60)

    results = {}

    for model_type in model_types:
        if logger:
            logger.info(f"\n{'─'*40}")
            logger.info(f"Training: {model_type}")
            logger.info(f"{'─'*40}")

        try:
            result = train_model(
                feature_table=feature_table,
                model_type=model_type,
                feature_selection=feature_selection,
                handle_imbalance=handle_imbalance,
                expected_test_prevalence=expected_test_prevalence,
                output_dir=output_dir,
                logger=logger,
            )

            if tune and model_type in ["xgboost", "lightgbm", "random_forest"]:
                if logger:
                    logger.info(f"Starte Hyperparameter-Tuning für {model_type}...")

                X, y, feature_names, _ = prepare_training_data(feature_table, logger)
                if X is not None:
                    imputer = SimpleImputer(strategy="median")
                    X_imp = imputer.fit_transform(X).astype(np.float32)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)

                    tune_result = tune_hyperparameters(
                        X_scaled, y, model_type, n_tune_trials,
                        handle_imbalance, logger,
                        expected_test_prevalence=expected_test_prevalence
                    )
                    result["tuning_result"] = tune_result

            results[model_type] = result

        except Exception as e:
            if logger:
                logger.error(f"Training fehlgeschlagen für {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    if logger:
        _log_model_comparison(results, logger)

    return results


def _log_model_comparison(results: Dict[str, Dict], logger):
    logger.info("\n" + "=" * 60)
    logger.info("MODELL-VERGLEICH")
    logger.info("=" * 60)

    comparison_rows = []

    for model_type, result in results.items():
        if "error" in result:
            logger.info(f"  {model_type}: FEHLER - {result['error']}")
            continue

        cv = result.get("cv_results", {})

        row = {
            "model": model_type,
            "auroc": cv.get("auroc_mean", np.nan),
            "auroc_std": cv.get("auroc_std", np.nan),
            "f1": cv.get("f1_mean", np.nan),
            "f1_std": cv.get("f1_std", np.nan),
            "bal_acc": cv.get("balanced_accuracy_mean", np.nan),
            "bal_acc_std": cv.get("balanced_accuracy_std", np.nan),
            "avg_prec": cv.get("average_precision_mean", np.nan),
            "overfit_gap": cv.get("overfit_gap_auroc", np.nan),
            "n_features": len(result.get("selected_features", [])),
        }
        comparison_rows.append(row)

    if not comparison_rows:
        logger.warning("Keine Modelle zum Vergleichen vorhanden.")
        return

    comparison_rows.sort(key=lambda x: x.get("auroc", 0), reverse=True)

    logger.info(f"\n{'Modell':<25} {'AUROC':>12} {'F1':>12} "
                f"{'Bal.Acc':>12} {'Avg.Prec':>12} {'Overfit':>10} {'#Feat':>8}")
    logger.info("-" * 95)

    for row in comparison_rows:
        auroc_str = (
            f"{row['auroc']:.4f}±{row['auroc_std']:.3f}"
            if not np.isnan(row['auroc']) else "N/A"
        )
        f1_str = (
            f"{row['f1']:.4f}±{row['f1_std']:.3f}"
            if not np.isnan(row['f1']) else "N/A"
        )
        ba_str = (
            f"{row['bal_acc']:.4f}±{row['bal_acc_std']:.3f}"
            if not np.isnan(row['bal_acc']) else "N/A"
        )
        ap_str = (
            f"{row['avg_prec']:.4f}"
            if not np.isnan(row['avg_prec']) else "N/A"
        )
        of_str = (
            f"{row['overfit_gap']:.4f}"
            if not np.isnan(row['overfit_gap']) else "N/A"
        )

        logger.info(
            f"{row['model']:<25} {auroc_str:>12} {f1_str:>12} "
            f"{ba_str:>12} {ap_str:>12} {of_str:>10} {row['n_features']:>8}"
        )

    best = comparison_rows[0]
    logger.info(f"\n{'='*60}")
    logger.info(f"BESTES MODELL: {best['model']}")
    logger.info(f"  AUROC: {best['auroc']:.4f} ± {best['auroc_std']:.4f}")
    logger.info(f"  F1:    {best['f1']:.4f} ± {best['f1_std']:.4f}")
    logger.info(f"  Features: {best['n_features']}")

    if not np.isnan(best.get("overfit_gap", np.nan)):
        if best["overfit_gap"] > 0.15:
            logger.warning(f"  ⚠ Overfitting-Risiko: Gap = {best['overfit_gap']:.4f}")
        else:
            logger.info(f"  ✓ Overfitting-Gap: {best['overfit_gap']:.4f}")

    logger.info("=" * 60)

    try:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = MODEL_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Vergleichstabelle gespeichert: {comparison_path}")
    except Exception as e:
        logger.debug(f"Vergleichstabelle konnte nicht gespeichert werden: {e}")


# ==============================================================================
# PREDICT-FUNKTION (für neue Daten)
# ==============================================================================

def predict(
    feature_table: pd.DataFrame,
    model_result: Optional[Dict] = None,
    model_dir: Optional[Path] = None,
    logger=None
) -> pd.DataFrame:
    if model_result is None:
        model_result = load_model(model_dir, logger)

    if model_result is None:
        if logger:
            logger.error("Kein Modell verfügbar für Vorhersage!")
        return pd.DataFrame()

    model = model_result["model"]
    scaler = model_result["scaler"]
    imputer = model_result["imputer"]
    feature_selector = model_result.get("feature_selector")
    feature_names = model_result["feature_names"]
    selected_features = model_result["selected_features"]

    patient_ids = (
        feature_table["patient_id"].values
        if "patient_id" in feature_table.columns
        else None
    )

    available_features = [f for f in feature_names if f in feature_table.columns]
    missing_features = [f for f in feature_names if f not in feature_table.columns]

    if missing_features:
        if logger:
            logger.warning(
                f"Fehlende Features: {len(missing_features)}/{len(feature_names)}"
            )
            for f in missing_features[:10]:
                logger.debug(f"  Fehlend: {f}")

        for f in missing_features:
            feature_table[f] = np.nan

    X = feature_table[feature_names].values.astype(np.float32)

    X_imputed = imputer.transform(X).astype(np.float32)

    if feature_selector is not None:
        try:
            X_selected = feature_selector.transform(X_imputed)
        except Exception:
            selected_indices = [
                feature_names.index(f) for f in selected_features
                if f in feature_names
            ]
            X_selected = X_imputed[:, selected_indices]
    else:
        X_selected = X_imputed

    X_scaled = scaler.transform(X_selected).astype(np.float32)

    try:
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        if logger:
            logger.error(f"Vorhersage fehlgeschlagen: {e}")
        return pd.DataFrame()

    result_df = pd.DataFrame({
        "prediction": predictions.astype(int),
        "probability": probabilities,
    })

    if patient_ids is not None:
        result_df.insert(0, "patient_id", patient_ids)

    result_df["confidence"] = pd.cut(
        result_df["probability"],
        bins=[0, 0.3, 0.45, 0.55, 0.7, 1.0],
        labels=[
            "high_negative", "moderate_negative",
            "uncertain", "moderate_positive", "high_positive"
        ]
    )

    if logger:
        n_positive = (predictions == 1).sum()
        n_total = len(predictions)
        logger.info(
            f"Vorhersagen: {n_positive}/{n_total} positiv "
            f"({n_positive/n_total*100:.1f}%)"
        )
        logger.info(f"Mittlere Wahrscheinlichkeit: {probabilities.mean():.4f}")

        conf_dist = result_df["confidence"].value_counts().to_dict()
        logger.info(f"Konfidenz-Verteilung: {conf_dist}")

    return result_df
