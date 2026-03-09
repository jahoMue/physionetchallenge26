"""
05_classification/evaluate_model.py
=====================================
Modell-Evaluation für die PhysioNet Challenge 2026.

Dieses Modul:
- Berechnet die Challenge-spezifischen Metriken (AUROC, AUPRC, Accuracy, F-Measure)
- Erstellt detaillierte Evaluationsberichte
- Generiert Visualisierungen (ROC, PR-Kurve, Confusion Matrix, etc.)
- Analysiert Feature Importance und Modell-Interpretierbarkeit
- Führt Fehleranalyse durch (welche Patienten werden falsch klassifiziert)
- Berechnet Konfidenzintervalle via Bootstrapping
- Evaluiert Fairness über demographische Gruppen

Referenz: Die PhysioNet Challenge 2026 verwendet AUROC, AUPRC, 
Accuracy und F-Measure als Evaluationsmetriken [[5]] [[6]].
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score,
    precision_score, recall_score,
    balanced_accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    brier_score_loss, log_loss,
    matthews_corrcoef, cohen_kappa_score,
)
from sklearn.calibration import calibration_curve

from sklearn.model_selection import StratifiedKFold

# Visualisierung
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    MODEL_DIR, FEATURE_DIR, OUTPUT_DIR,
    RANDOM_SEED, CV_FOLDS, TARGET_COLUMN
)

warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# HAUPT-FUNKTION: VOLLSTÄNDIGE EVALUATION
# ==============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    feature_table: Optional[pd.DataFrame] = None,
    model_result: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    generate_plots: bool = True,
    logger=None
) -> Dict:
    """
    Vollständige Modell-Evaluation.
    
    Parameters
    ----------
    y_true : np.ndarray
        Wahre Labels (0/1).
    y_pred : np.ndarray
        Vorhergesagte Labels (0/1).
    y_prob : np.ndarray
        Vorhergesagte Wahrscheinlichkeiten (0-1).
    patient_ids : np.ndarray, optional
        Patienten-IDs.
    feature_table : pd.DataFrame, optional
        Feature-Tabelle (für Fehleranalyse).
    model_result : Dict, optional
        Trainiertes Modell (für Feature Importance).
    output_dir : Path, optional
        Ausgabeverzeichnis für Berichte und Plots.
    generate_plots : bool
        Ob Visualisierungen erstellt werden sollen.
    logger : loguru.Logger, optional
    
    Returns
    -------
    Dict mit allen Evaluationsergebnissen.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if logger:
        logger.info("=" * 60)
        logger.info("MODELL-EVALUATION")
        logger.info(f"Samples: {len(y_true)}, Positive: {y_true.sum()}, "
                     f"Negative: {(1-y_true).sum()}")
        logger.info("=" * 60)
    
    results = {
        "timestamp": timestamp,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "prevalence": float(y_true.mean()),
    }
    
    # --- 1. Challenge-Metriken ---
    challenge_metrics = compute_challenge_metrics(y_true, y_pred, y_prob, logger)
    results["challenge_metrics"] = challenge_metrics
    
    # --- 2. Erweiterte Metriken ---
    extended_metrics = compute_extended_metrics(y_true, y_pred, y_prob, logger)
    results["extended_metrics"] = extended_metrics
    
    # --- 3. Konfidenzintervalle ---
    ci_metrics = compute_confidence_intervals(y_true, y_prob, logger=logger)
    results["confidence_intervals"] = ci_metrics
    
    # --- 4. Schwellenwert-Analyse ---
    threshold_analysis = analyze_thresholds(y_true, y_prob, logger)
    results["threshold_analysis"] = threshold_analysis
    
    # --- 5. Kalibrierung ---
    calibration = assess_calibration(y_true, y_prob, logger)
    results["calibration"] = calibration
    
    # --- 6. Fehleranalyse ---
    if patient_ids is not None:
        error_analysis = analyze_errors(
            y_true, y_pred, y_prob, patient_ids,
            feature_table, logger
        )
        results["error_analysis"] = error_analysis
    
    # --- 7. Fairness-Analyse ---
    if feature_table is not None:
        fairness = analyze_fairness(
            y_true, y_pred, y_prob, feature_table, logger
        )
        results["fairness"] = fairness
    
    # --- 8. Feature Importance ---
    if model_result is not None:
        results["feature_importance"] = model_result.get("feature_importance")
    
    # --- 9. Visualisierungen ---
    if generate_plots:
        plot_paths = generate_evaluation_plots(
            y_true, y_pred, y_prob,
            results, model_result,
            output_dir, logger
        )
        results["plot_paths"] = plot_paths
    
    # --- 10. Bericht speichern ---
    _save_evaluation_report(results, output_dir, timestamp, logger)
    
    # --- Zusammenfassung loggen ---
    if logger:
        _log_evaluation_summary(results, logger)
    
    return results


# ==============================================================================
# CHALLENGE-METRIKEN
# ==============================================================================

def compute_challenge_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    logger=None
) -> Dict:
    """
    Berechnet die PhysioNet Challenge 2026 Metriken.
    AUROC, AUPRC, Accuracy, F-Measure [[5]].
    """
    metrics = {}
    
    try:
        # AUROC
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception as e:
        metrics["auroc"] = np.nan
        if logger:
            logger.warning(f"AUROC Berechnung fehlgeschlagen: {e}")
    
    try:
        # AUPRC (Average Precision)
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    except Exception as e:
        metrics["auprc"] = np.nan
        if logger:
            logger.warning(f"AUPRC Berechnung fehlgeschlagen: {e}")
    
    try:
        # Accuracy
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        metrics["accuracy"] = np.nan
    
    try:
        # F-Measure (F1-Score)
        metrics["f_measure"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["f_measure"] = np.nan
    
    # Challenge-Score (gewichtete Kombination)
    # Basierend auf typischen PhysioNet Challenge Scoring
    valid_metrics = [
        v for v in [metrics.get("auroc"), metrics.get("auprc"),
                     metrics.get("accuracy"), metrics.get("f_measure")]
        if v is not None and not np.isnan(v)
    ]
    metrics["challenge_score"] = float(np.mean(valid_metrics)) if valid_metrics else np.nan
    
    if logger:
        logger.info("Challenge-Metriken:")
        logger.info(f"  AUROC:     {metrics['auroc']:.4f}")
        logger.info(f"  AUPRC:     {metrics['auprc']:.4f}")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  F-Measure: {metrics['f_measure']:.4f}")
        logger.info(f"  Score:     {metrics['challenge_score']:.4f}")
    
    return metrics


# ==============================================================================
# ERWEITERTE METRIKEN
# ==============================================================================

def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    logger=None
) -> Dict:
    """
    Berechnet erweiterte Evaluationsmetriken.
    """
    metrics = {}
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    
    # Precision, Recall, Specificity
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["sensitivity"] = metrics["recall"]  # Alias
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    
    # Negative Predictive Value
    metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan
    
    # Positive/Negative Likelihood Ratios
    if metrics["specificity"] is not None and metrics["specificity"] < 1:
        metrics["positive_lr"] = (
            metrics["sensitivity"] / (1 - metrics["specificity"])
            if (1 - metrics["specificity"]) > 0 else np.nan
        )
    else:
        metrics["positive_lr"] = np.nan
    
    if metrics["sensitivity"] is not None and metrics["sensitivity"] < 1:
        metrics["negative_lr"] = (
            (1 - metrics["sensitivity"]) / metrics["specificity"]
            if metrics["specificity"] > 0 else np.nan
        )
    else:
        metrics["negative_lr"] = np.nan
    
    # Balanced Accuracy
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    
    # Matthews Correlation Coefficient
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    
    # Cohen's Kappa
    metrics["cohens_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    
    # Brier Score (Kalibrierung)
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    
    # Log Loss
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
    except Exception:
        metrics["log_loss"] = np.nan
    
    # Youden's J Statistic
    metrics["youdens_j"] = metrics["sensitivity"] + metrics["specificity"] - 1
    
    # Diagnostic Odds Ratio
    if fp > 0 and fn > 0:
        metrics["diagnostic_or"] = float((tp * tn) / (fp * fn))
    else:
        metrics["diagnostic_or"] = np.nan
    
    if logger:
        logger.info("Erweiterte Metriken:")
        logger.info(f"  Precision:    {metrics['precision']:.4f}")
        logger.info(f"  Recall/Sens:  {metrics['recall']:.4f}")
        logger.info(f"  Specificity:  {metrics['specificity']:.4f}")
        logger.info(f"  Bal. Accuracy:{metrics['balanced_accuracy']:.4f}")
        logger.info(f"  MCC:          {metrics['mcc']:.4f}")
        logger.info(f"  Brier Score:  {metrics['brier_score']:.4f}")
        logger.info(f"  Youden's J:   {metrics['youdens_j']:.4f}")
    
    return metrics


# ==============================================================================
# KONFIDENZINTERVALLE (BOOTSTRAPPING)
# ==============================================================================

def compute_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    logger=None
) -> Dict:
    """
    Berechnet Konfidenzintervalle für die Hauptmetriken via Bootstrapping.
    """
    if logger:
        logger.info(f"Bootstrap Konfidenzintervalle: {n_bootstrap} Iterationen, "
                     f"{ci_level*100:.0f}% CI")
    
    rng = np.random.RandomState(RANDOM_SEED)
    n_samples = len(y_true)
    
    alpha = (1 - ci_level) / 2
    
    bootstrap_metrics = {
        "auroc": [],
        "auprc": [],
        "f1": [],
        "balanced_accuracy": [],
    }
    
    for i in range(n_bootstrap):
        # Bootstrap-Sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        # Stelle sicher dass beide Klassen vorhanden sind
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        y_pred_boot = (y_prob_boot >= 0.5).astype(int)
        
        try:
            bootstrap_metrics["auroc"].append(
                roc_auc_score(y_true_boot, y_prob_boot)
            )
        except Exception:
            pass
        
        try:
            bootstrap_metrics["auprc"].append(
                average_precision_score(y_true_boot, y_prob_boot)
            )
        except Exception:
            pass
        
        try:
            bootstrap_metrics["f1"].append(
                f1_score(y_true_boot, y_pred_boot, zero_division=0)
            )
        except Exception:
            pass
        
        try:
            bootstrap_metrics["balanced_accuracy"].append(
                balanced_accuracy_score(y_true_boot, y_pred_boot)
            )
        except Exception:
            pass
    
    # Konfidenzintervalle berechnen
    ci_results = {}
    for metric_name, values in bootstrap_metrics.items():
        if len(values) > 10:
            values = np.array(values)
            ci_results[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_lower": float(np.percentile(values, alpha * 100)),
                "ci_upper": float(np.percentile(values, (1 - alpha) * 100)),
                "ci_level": ci_level,
                "n_bootstrap": len(values),
            }
            
            if logger:
                ci = ci_results[metric_name]
                logger.info(
                    f"  {metric_name}: {ci['mean']:.4f} "
                    f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
                    f"({ci_level*100:.0f}% CI)"
                )
    
    return ci_results


# ==============================================================================
# SCHWELLENWERT-ANALYSE
# ==============================================================================

def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    logger=None
) -> Dict:
    """
    Analysiert verschiedene Klassifikations-Schwellenwerte.
    Findet den optimalen Schwellenwert für verschiedene Kriterien.
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    
    threshold_results = []
    
    for thresh in thresholds:
        y_pred_t = (y_prob >= thresh).astype(int)
        
        if len(np.unique(y_pred_t)) < 2:
            continue
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        ba = balanced_accuracy_score(y_true, y_pred_t)
        youdens_j = sens + spec - 1
        
        threshold_results.append({
            "threshold": float(thresh),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "precision": float(prec),
            "f1": float(f1),
            "balanced_accuracy": float(ba),
            "youdens_j": float(youdens_j),
            "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn),
        })
    
    threshold_df = pd.DataFrame(threshold_results)
    
    # Optimale Schwellenwerte
    optimal = {}
    
    # Youden's J (maximiert Sensitivität + Spezifität)
    if len(threshold_df) > 0:
        best_j_idx = threshold_df["youdens_j"].idxmax()
        optimal["youdens_j"] = {
            "threshold": float(threshold_df.loc[best_j_idx, "threshold"]),
            "value": float(threshold_df.loc[best_j_idx, "youdens_j"]),
            "sensitivity": float(threshold_df.loc[best_j_idx, "sensitivity"]),
            "specificity": float(threshold_df.loc[best_j_idx, "specificity"]),
        }
        
        # Maximaler F1
        best_f1_idx = threshold_df["f1"].idxmax()
        optimal["max_f1"] = {
            "threshold": float(threshold_df.loc[best_f1_idx, "threshold"]),
            "value": float(threshold_df.loc[best_f1_idx, "f1"]),
        }
        
        # Maximale Balanced Accuracy
        best_ba_idx = threshold_df["balanced_accuracy"].idxmax()
        optimal["max_balanced_accuracy"] = {
            "threshold": float(threshold_df.loc[best_ba_idx, "threshold"]),
            "value": float(threshold_df.loc[best_ba_idx, "balanced_accuracy"]),
        }
        
        # Sensitivität >= 0.90
        high_sens = threshold_df[threshold_df["sensitivity"] >= 0.90]
        if len(high_sens) > 0:
            best_spec_idx = high_sens["specificity"].idxmax()
            optimal["sens_90"] = {
                "threshold": float(high_sens.loc[best_spec_idx, "threshold"]),
                "sensitivity": float(high_sens.loc[best_spec_idx, "sensitivity"]),
                "specificity": float(high_sens.loc[best_spec_idx, "specificity"]),
            }
    
    if logger:
        logger.info("Schwellenwert-Analyse:")
        for criterion, values in optimal.items():
            logger.info(f"  {criterion}: Threshold={values['threshold']:.2f}")
            for k, v in values.items():
                if k != "threshold":
                    logger.info(f"    {k}: {v:.4f}")
    
    return {
        "threshold_table": threshold_df.to_dict(orient="records"),
        "optimal_thresholds": optimal,
    }


# ==============================================================================
# KALIBRIERUNG
# ==============================================================================

def assess_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    logger=None
) -> Dict:
    """
    Bewertet die Kalibrierung des Modells.
    Ein gut kalibriertes Modell: P(y=1|prob=0.7) ≈ 0.7
    """
    results = {}
    
    # Brier Score
    results["brier_score"] = float(brier_score_loss(y_true, y_prob))
    
    # Kalibrierungskurve
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        results["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        
        # Expected Calibration Error (ECE)
        bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
        total = len(y_prob)
        
        ece = 0.0
        for i in range(len(prob_true)):
            if i < len(bin_counts) and bin_counts[i] > 0:
                ece += (bin_counts[i] / total) * abs(prob_true[i] - prob_pred[i])
        
        results["ece"] = float(ece)
        
        # Maximum Calibration Error (MCE)
        if len(prob_true) > 0:
            results["mce"] = float(np.max(np.abs(prob_true - prob_pred)))
        else:
            results["mce"] = np.nan
    
    except Exception as e:
        results["ece"] = np.nan
        results["mce"] = np.nan
        if logger:
            logger.warning(f"Kalibrierungsberechnung fehlgeschlagen: {e}")
    
    # Wahrscheinlichkeits-Verteilung
    results["prob_distribution"] = {
        "mean": float(np.mean(y_prob)),
        "std": float(np.std(y_prob)),
        "median": float(np.median(y_prob)),
        "min": float(np.min(y_prob)),
        "max": float(np.max(y_prob)),
        "q25": float(np.percentile(y_prob, 25)),
        "q75": float(np.percentile(y_prob, 75)),
    }
    
    if logger:
        logger.info("Kalibrierung:")
        logger.info(f"  Brier Score: {results['brier_score']:.4f}")
        logger.info(f"  ECE:         {results.get('ece', 'N/A')}")
        logger.info(f"  MCE:         {results.get('mce', 'N/A')}")
        logger.info(f"  Prob. Mean:  {results['prob_distribution']['mean']:.4f}")
    
    return results


# ==============================================================================
# FEHLERANALYSE
# ==============================================================================

def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: np.ndarray,
    feature_table: Optional[pd.DataFrame] = None,
    logger=None
) -> Dict:
    """
    Analysiert falsch klassifizierte Patienten.
    """
    results = {}
    
    # Identifiziere Fehler
    fp_mask = (y_pred == 1) & (y_true == 0)  # False Positives
    fn_mask = (y_pred == 0) & (y_true == 1)  # False Negatives
    tp_mask = (y_pred == 1) & (y_true == 1)  # True Positives
    tn_mask = (y_pred == 0) & (y_true == 0)  # True Negatives
    
    results["false_positives"] = {
        "patient_ids": patient_ids[fp_mask].tolist(),
        "probabilities": y_prob[fp_mask].tolist(),
        "count": int(fp_mask.sum()),
        "mean_probability": float(np.mean(y_prob[fp_mask])) if fp_mask.any() else np.nan,
    }
    
    results["false_negatives"] = {
        "patient_ids": patient_ids[fn_mask].tolist(),
        "probabilities": y_prob[fn_mask].tolist(),
        "count": int(fn_mask.sum()),
        "mean_probability": float(np.mean(y_prob[fn_mask])) if fn_mask.any() else np.nan,
    }
    
    # Konfidenz-Analyse: Fehler bei hoher vs. niedriger Konfidenz
    high_conf_mask = (y_prob >= 0.7) | (y_prob <= 0.3)
    low_conf_mask = (y_prob > 0.3) & (y_prob < 0.7)
    
    errors = (y_pred != y_true)
    
    results["confidence_analysis"] = {
        "high_confidence_error_rate": float(
            errors[high_conf_mask].mean()
        ) if high_conf_mask.any() else np.nan,
        "low_confidence_error_rate": float(
            errors[low_conf_mask].mean()
        ) if low_conf_mask.any() else np.nan,
        "n_high_confidence": int(high_conf_mask.sum()),
        "n_low_confidence": int(low_conf_mask.sum()),
    }
    
    # Demographische Analyse der Fehler
    if feature_table is not None:
        demo_error_analysis = _analyze_errors_by_demographics(
            y_true, y_pred, y_prob, patient_ids, feature_table, logger
        )
        results["demographic_errors"] = demo_error_analysis
    
    if logger:
        logger.info("Fehleranalyse:")
        logger.info(f"  False Positives: {results['false_positives']['count']}")
        logger.info(f"  False Negatives: {results['false_negatives']['count']}")
        logger.info(f"  FP mittlere Prob: "
                     f"{results['false_positives']['mean_probability']:.4f}")
        logger.info(f"  FN mittlere Prob: "
                     f"{results['false_negatives']['mean_probability']:.4f}")
        ca = results["confidence_analysis"]
        logger.info(f"  Fehlerrate hohe Konfidenz: "
                     f"{ca['high_confidence_error_rate']:.4f}")
        logger.info(f"  Fehlerrate niedrige Konfidenz: "
                     f"{ca['low_confidence_error_rate']:.4f}")
    
    return results


def _analyze_errors_by_demographics(
    y_true, y_pred, y_prob, patient_ids, feature_table, logger=None
) -> Dict:
    """Analysiert Fehler aufgeschlüsselt nach demographischen Gruppen."""
    results = {}
    
    demo_cols = {
        "age_group": ["demo_age_group"],
        "sex": ["demo_sex"],
        "bmi_category": ["demo_bmi_category"],
    }
    
    for group_name, col_candidates in demo_cols.items():
        col = None
        for candidate in col_candidates:
            if candidate in feature_table.columns:
                col = candidate
                break
        
        if col is None:
            continue
        
        group_results = {}
        
        for group_val in feature_table[col].dropna().unique():
            mask = feature_table[col].values == group_val
            if mask.sum() < 5:
                continue
            
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            group_errors = (group_y_pred != group_y_true)
            
            group_results[str(group_val)] = {
                "n": int(mask.sum()),
                "error_rate": float(group_errors.mean()),
                "accuracy": float(1 - group_errors.mean()),
                "n_errors": int(group_errors.sum()),
            }
        
        results[group_name] = group_results
    
    return results


# ==============================================================================
# FAIRNESS-ANALYSE
# ==============================================================================

def analyze_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_table: pd.DataFrame,
    logger=None
) -> Dict:
    """
    Analysiert die Fairness des Modells über demographische Gruppen.
    
    Metriken:
    - Equalized Odds: Gleiche TPR und FPR über Gruppen
    - Demographic Parity: Gleiche Vorhersagerate über Gruppen
    - Calibration Fairness: Gleiche Kalibrierung über Gruppen
    """
    results = {}
    
    # Geschlecht
    sex_col = None
    for candidate in ["demo_sex", "Sex"]:
        if candidate in feature_table.columns:
            sex_col = candidate
            break
    
    if sex_col is not None:
        results["sex"] = _compute_fairness_metrics(
            y_true, y_pred, y_prob,
            feature_table[sex_col].values,
            group_name="sex",
            logger=logger
        )
    
    # Altersgruppen
    age_col = None
    for candidate in ["demo_age_group"]:
        if candidate in feature_table.columns:
            age_col = candidate
            break
    
    if age_col is not None:
        results["age_group"] = _compute_fairness_metrics(
            y_true, y_pred, y_prob,
            feature_table[age_col].values,
            group_name="age_group",
            logger=logger
        )
    
    # BMI-Kategorien
    bmi_col = None
    for candidate in ["demo_bmi_category"]:
        if candidate in feature_table.columns:
            bmi_col = candidate
            break
    
    if bmi_col is not None:
        results["bmi_category"] = _compute_fairness_metrics(
            y_true, y_pred, y_prob,
            feature_table[bmi_col].values,
            group_name="bmi_category",
            logger=logger
        )
    
    # Fairness-Zusammenfassung
    if logger:
        logger.info("Fairness-Analyse:")
        for attr, attr_results in results.items():
            if "disparity" in attr_results:
                logger.info(f"  {attr}:")
                for metric, value in attr_results["disparity"].items():
                    logger.info(f"    {metric}: {value:.4f}")
    
    return results


def _compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray,
    group_name: str = "group",
    logger=None
) -> Dict:
    """
    Berechnet Fairness-Metriken für eine gegebene Gruppierung.
    
    Metriken pro Gruppe:
    - TPR (True Positive Rate / Sensitivity)
    - FPR (False Positive Rate)
    - Positive Prediction Rate (Demographic Parity)
    - AUROC
    
    Disparitäts-Metriken:
    - Max Disparity (max Differenz zwischen Gruppen)
    - Equalized Odds Difference
    - Demographic Parity Difference
    """
    results = {"group_name": group_name, "groups": {}, "disparity": {}}
    
    # Entferne NaN-Gruppen
    valid_mask = ~pd.isna(group_labels)
    if valid_mask.sum() < len(y_true) * 0.5:
        return results
    
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]
    y_prob_v = y_prob[valid_mask]
    groups_v = group_labels[valid_mask]
    
    unique_groups = np.unique(groups_v)
    
    if len(unique_groups) < 2:
        return results
    
    tprs = {}
    fprs = {}
    positive_rates = {}
    aurocs = {}
    accuracies = {}
    
    for group in unique_groups:
        group_mask = groups_v == group
        n_group = group_mask.sum()
        
        if n_group < 5:
            continue
        
        g_true = y_true_v[group_mask]
        g_pred = y_pred_v[group_mask]
        g_prob = y_prob_v[group_mask]
        
        # Confusion Matrix Elemente
        if len(np.unique(g_true)) < 2:
            # Nur eine Klasse in dieser Gruppe
            results["groups"][str(group)] = {
                "n": int(n_group),
                "n_positive": int(g_true.sum()),
                "accuracy": float(accuracy_score(g_true, g_pred)),
                "positive_rate": float(g_pred.mean()),
                "note": "single_class",
            }
            continue
        
        tn, fp, fn, tp = confusion_matrix(g_true, g_pred).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        pos_rate = g_pred.mean()
        
        try:
            group_auroc = roc_auc_score(g_true, g_prob)
        except Exception:
            group_auroc = np.nan
        
        group_acc = accuracy_score(g_true, g_pred)
        
        tprs[str(group)] = tpr
        fprs[str(group)] = fpr
        positive_rates[str(group)] = pos_rate
        aurocs[str(group)] = group_auroc
        accuracies[str(group)] = group_acc
        
        results["groups"][str(group)] = {
            "n": int(n_group),
            "n_positive": int(g_true.sum()),
            "prevalence": float(g_true.mean()),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "positive_rate": float(pos_rate),
            "auroc": float(group_auroc) if not np.isnan(group_auroc) else None,
            "accuracy": float(group_acc),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            "f1": float(f1_score(g_true, g_pred, zero_division=0)),
        }
    
    # Disparitäts-Metriken
    if len(tprs) >= 2:
        tpr_values = list(tprs.values())
        fpr_values = list(fprs.values())
        pr_values = list(positive_rates.values())
        auroc_values = [v for v in aurocs.values() if not np.isnan(v)]
        acc_values = list(accuracies.values())
        
        # Equalized Odds Difference (max of TPR diff and FPR diff)
        tpr_diff = max(tpr_values) - min(tpr_values)
        fpr_diff = max(fpr_values) - min(fpr_values)
        results["disparity"]["equalized_odds_diff"] = float(
            max(tpr_diff, fpr_diff)
        )
        
        # Equal Opportunity Difference (TPR diff only)
        results["disparity"]["equal_opportunity_diff"] = float(tpr_diff)
        
        # Demographic Parity Difference
        results["disparity"]["demographic_parity_diff"] = float(
            max(pr_values) - min(pr_values)
        )
        
        # AUROC Disparity
        if len(auroc_values) >= 2:
            results["disparity"]["auroc_disparity"] = float(
                max(auroc_values) - min(auroc_values)
            )
        
        # Accuracy Disparity
        results["disparity"]["accuracy_disparity"] = float(
            max(acc_values) - min(acc_values)
        )
        
        # Fairness-Bewertung (Schwellenwerte nach gängiger Praxis)
        # Disparity < 0.1 gilt als fair
        is_fair = all(
            v < 0.1 for k, v in results["disparity"].items()
            if not np.isnan(v)
        )
        results["is_fair"] = is_fair
    
    return results


# ==============================================================================
# VISUALISIERUNGEN
# ==============================================================================

def generate_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    eval_results: Dict,
    model_result: Optional[Dict] = None,
    output_dir: Path = None,
    logger=None
) -> Dict[str, Path]:
    """
    Erstellt alle Evaluations-Visualisierungen.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "evaluation" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # Stil setzen
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})
    
    # --- 1. ROC-Kurve ---
    try:
        path = _plot_roc_curve(y_true, y_prob, eval_results, output_dir)
        plot_paths["roc_curve"] = path
    except Exception as e:
        if logger:
            logger.warning(f"ROC-Plot fehlgeschlagen: {e}")
    
    # --- 2. Precision-Recall-Kurve ---
    try:
        path = _plot_pr_curve(y_true, y_prob, eval_results, output_dir)
        plot_paths["pr_curve"] = path
    except Exception as e:
        if logger:
            logger.warning(f"PR-Plot fehlgeschlagen: {e}")
    
    # --- 3. Confusion Matrix ---
    try:
        path = _plot_confusion_matrix(y_true, y_pred, output_dir)
        plot_paths["confusion_matrix"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Confusion Matrix Plot fehlgeschlagen: {e}")
    
    # --- 4. Wahrscheinlichkeits-Verteilung ---
    try:
        path = _plot_probability_distribution(y_true, y_prob, output_dir)
        plot_paths["probability_distribution"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Probability Distribution Plot fehlgeschlagen: {e}")
    
    # --- 5. Kalibrierungskurve ---
    try:
        path = _plot_calibration_curve(y_true, y_prob, output_dir)
        plot_paths["calibration_curve"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Calibration Plot fehlgeschlagen: {e}")
    
    # --- 6. Feature Importance ---
    if model_result is not None and model_result.get("feature_importance") is not None:
        try:
            path = _plot_feature_importance(
                model_result["feature_importance"], output_dir
            )
            plot_paths["feature_importance"] = path
        except Exception as e:
            if logger:
                logger.warning(f"Feature Importance Plot fehlgeschlagen: {e}")
    
    # --- 7. Schwellenwert-Analyse ---
    try:
        path = _plot_threshold_analysis(
            eval_results.get("threshold_analysis", {}), output_dir
        )
        plot_paths["threshold_analysis"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Threshold Plot fehlgeschlagen: {e}")
    
    # --- 8. Zusammenfassungs-Dashboard ---
    try:
        path = _plot_evaluation_dashboard(
            y_true, y_pred, y_prob, eval_results, output_dir
        )
        plot_paths["dashboard"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Dashboard Plot fehlgeschlagen: {e}")
    
    if logger:
        logger.info(f"Plots erstellt: {len(plot_paths)} Visualisierungen")
        for name, path in plot_paths.items():
            logger.debug(f"  {name}: {path}")
    
    plt.close('all')
    
    return plot_paths


def _plot_roc_curve(y_true, y_prob, eval_results, output_dir) -> Path:
    """ROC-Kurve mit AUROC und Konfidenzintervall."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auroc = eval_results["challenge_metrics"]["auroc"]
    
    ax.plot(fpr, tpr, color='#2196F3', lw=2.5,
            label=f'ROC (AUROC = {auroc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
            label='Random (AUROC = 0.5)')
    
    # Konfidenzintervall
    ci = eval_results.get("confidence_intervals", {}).get("auroc", {})
    if ci:
        ax.fill_between(
            fpr, tpr * 0.95, np.minimum(tpr * 1.05, 1.0),
            alpha=0.1, color='#2196F3',
            label=f'{ci.get("ci_level", 0.95)*100:.0f}% CI: '
                  f'[{ci.get("ci_lower", 0):.4f}, {ci.get("ci_upper", 0):.4f}]'
        )
    
    # Optimaler Punkt (Youden's J)
    optimal = eval_results.get("threshold_analysis", {}).get("optimal_thresholds", {})
    if "youdens_j" in optimal:
        opt_thresh = optimal["youdens_j"]["threshold"]
        opt_idx = np.argmin(np.abs(thresholds - opt_thresh))
        ax.plot(fpr[opt_idx], tpr[opt_idx], 'ro', markersize=10,
                label=f'Optimal (J={optimal["youdens_j"]["value"]:.3f}, '
                      f'thresh={opt_thresh:.2f})')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title('ROC Curve – Cognitive Impairment Prediction', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    
    path = output_dir / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_pr_curve(y_true, y_prob, eval_results, output_dir) -> Path:
    """Precision-Recall-Kurve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auprc = eval_results["challenge_metrics"]["auprc"]
    baseline = y_true.mean()
    
    ax.plot(recall, precision, color='#4CAF50', lw=2.5,
            label=f'PR Curve (AUPRC = {auprc:.4f})')
    ax.axhline(y=baseline, color='gray', lw=1, linestyle='--',
               label=f'Baseline (Prevalence = {baseline:.3f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=13)
    ax.set_ylabel('Precision (PPV)', fontsize=13)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    path = output_dir / "pr_curve.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_confusion_matrix(y_true, y_pred, output_dir) -> Path:
    """Confusion Matrix Heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Absolute Werte
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No CI', 'CI'],
                yticklabels=['No CI', 'CI'])
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_title('Confusion Matrix (Absolute)', fontsize=13)
    
    # Normalisierte Werte
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
                xticklabels=['No CI', 'CI'],
                yticklabels=['No CI', 'CI'])
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13)
    
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_probability_distribution(y_true, y_prob, output_dir) -> Path:
    """Verteilung der vorhergesagten Wahrscheinlichkeiten."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Getrennt nach Klassen
    prob_neg = y_prob[y_true == 0]
    prob_pos = y_prob[y_true == 1]
    
    bins = np.linspace(0, 1, 30)
    
    ax.hist(prob_neg, bins=bins, alpha=0.6, color='#2196F3',
            label=f'No CI (n={len(prob_neg)})', density=True)
    ax.hist(prob_pos, bins=bins, alpha=0.6, color='#F44336',
            label=f'CI (n={len(prob_pos)})', density=True)
    
    ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5,
               label='Threshold = 0.5')
    
    ax.set_xlabel('Predicted Probability', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Distribution of Predicted Probabilities', fontsize=14)
    ax.legend(fontsize=11)
    
    path = output_dir / "probability_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_calibration_curve(y_true, y_prob, output_dir) -> Path:
    """Kalibrierungskurve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Kalibrierungskurve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        axes[0].plot(prob_pred, prob_true, 's-', color='#4CAF50',
                     lw=2, markersize=8, label='Model')
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
        axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
        axes[0].set_ylabel('Fraction of Positives', fontsize=12)
        axes[0].set_title('Calibration Curve', fontsize=13)
        axes[0].legend(fontsize=11)
        axes[0].set_xlim([-0.02, 1.02])
        axes[0].set_ylim([-0.02, 1.02])
    except Exception:
        axes[0].text(0.5, 0.5, 'Calibration curve\nnot available',
                     ha='center', va='center', fontsize=14)
    
    # Histogramm der Wahrscheinlichkeiten
    axes[1].hist(y_prob, bins=20, color='#9E9E9E', edgecolor='white')
    axes[1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Prediction Histogram', fontsize=13)
    
    plt.tight_layout()
    path = output_dir / "calibration_curve.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_feature_importance(importance_df, output_dir, top_n=30) -> Path:
    """Feature Importance Barplot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, max(8, top_n * 0.35)))
    
    top_features = importance_df.head(top_n).copy()
    top_features = top_features.sort_values("importance_normalized", ascending=True)
    
    colors = plt.cm.viridis(
        np.linspace(0.3, 0.9, len(top_features))
    )
    
    ax.barh(
        range(len(top_features)),
        top_features["importance_normalized"].values,
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values, fontsize=9)
    ax.set_xlabel('Normalized Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
    
    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_threshold_analysis(threshold_data, output_dir) -> Path:
    """Schwellenwert-Analyse Plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if "threshold_table" not in threshold_data or not threshold_data["threshold_table"]:
        for ax in axes:
            ax.text(0.5, 0.5, 'No threshold data', ha='center', va='center')
        path = output_dir / "threshold_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
    
    df = pd.DataFrame(threshold_data["threshold_table"])
    
    # Sensitivität und Spezifität vs. Schwellenwert
    axes[0].plot(df["threshold"], df["sensitivity"], 'b-', lw=2, label='Sensitivity')
    axes[0].plot(df["threshold"], df["specificity"], 'r-', lw=2, label='Specificity')
    axes[0].plot(df["threshold"], df["balanced_accuracy"], 'g--', lw=1.5,
                 label='Balanced Accuracy')
    
    # Optimaler Punkt
    optimal = threshold_data.get("optimal_thresholds", {})
    if "youdens_j" in optimal:
        opt_t = optimal["youdens_j"]["threshold"]
        axes[0].axvline(x=opt_t, color='gray', linestyle=':', lw=1.5,
                        label=f'Optimal (J): {opt_t:.2f}')
    
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Sensitivity/Specificity vs. Threshold', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    
    # F1 und Youden's J vs. Schwellenwert
    axes[1].plot(df["threshold"], df["f1"], 'b-', lw=2, label='F1 Score')
    axes[1].plot(df["threshold"], df["youdens_j"], 'r-', lw=2, label="Youden's J")
    axes[1].plot(df["threshold"], df["precision"], 'g--', lw=1.5, label='Precision')
    
    if "max_f1" in optimal:
        opt_t = optimal["max_f1"]["threshold"]
        axes[1].axvline(x=opt_t, color='blue', linestyle=':', lw=1.5, alpha=0.5,
                        label=f'Best F1: {opt_t:.2f}')
    
    axes[1].set_xlabel('Threshold', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('F1 / Youden\'s J vs. Threshold', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    path = output_dir / "threshold_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_evaluation_dashboard(y_true, y_pred, y_prob, eval_results, output_dir) -> Path:
    """Zusammenfassungs-Dashboard mit allen wichtigen Metriken."""
    fig = plt.figure(figsize=(20, 16))
    
    # Layout: 3x3 Grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # --- 1. ROC-Kurve (oben links) ---
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = eval_results["challenge_metrics"]["auroc"]
    ax1.plot(fpr, tpr, 'b-', lw=2)
    ax1.plot([0, 1], [0, 1], 'k--', lw=0.5)
    ax1.set_title(f'ROC (AUROC={auroc:.3f})', fontsize=11)
    ax1.set_xlabel('FPR', fontsize=9)
    ax1.set_ylabel('TPR', fontsize=9)
    
    # --- 2. PR-Kurve (oben mitte) ---
    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = eval_results["challenge_metrics"]["auprc"]
    ax2.plot(rec, prec, 'g-', lw=2)
    ax2.axhline(y=y_true.mean(), color='gray', linestyle='--', lw=0.5)
    ax2.set_title(f'PR Curve (AUPRC={auprc:.3f})', fontsize=11)
    ax2.set_xlabel('Recall', fontsize=9)
    ax2.set_ylabel('Precision', fontsize=9)
    
    # --- 3. Confusion Matrix (oben rechts) ---
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['No CI', 'CI'], yticklabels=['No CI', 'CI'])
    ax3.set_title('Confusion Matrix', fontsize=11)
    
    # --- 4. Probability Distribution (mitte links) ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(y_prob[y_true == 0], bins=20, alpha=0.6, color='blue',
             label='No CI', density=True)
    ax4.hist(y_prob[y_true == 1], bins=20, alpha=0.6, color='red',
             label='CI', density=True)
    ax4.axvline(x=0.5, color='black', linestyle='--', lw=1)
    ax4.set_title('Probability Distribution', fontsize=11)
    ax4.legend(fontsize=8)
    
    # --- 5. Kalibrierung (mitte mitte) ---
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8)
        ax5.plot(prob_pred, prob_true, 's-', color='green', lw=2)
        ax5.plot([0, 1], [0, 1], 'k--', lw=0.5)
    except Exception:
        ax5.text(0.5, 0.5, 'N/A', ha='center', va='center')
    ax5.set_title('Calibration', fontsize=11)
    
    # --- 6. Metriken-Tabelle (mitte rechts) ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    cm_metrics = eval_results.get("challenge_metrics", {})
    ext_metrics = eval_results.get("extended_metrics", {})
    
    table_data = [
        ['AUROC', f'{cm_metrics.get("auroc", 0):.4f}'],
        ['AUPRC', f'{cm_metrics.get("auprc", 0):.4f}'],
        ['Accuracy', f'{cm_metrics.get("accuracy", 0):.4f}'],
        ['F-Measure', f'{cm_metrics.get("f_measure", 0):.4f}'],
        ['', ''],
        ['Sensitivity', f'{ext_metrics.get("sensitivity", 0):.4f}'],
        ['Specificity', f'{ext_metrics.get("specificity", 0):.4f}'],
        ['Precision', f'{ext_metrics.get("precision", 0):.4f}'],
        ['MCC', f'{ext_metrics.get("mcc", 0):.4f}'],
        ['Brier Score', f'{ext_metrics.get("brier_score", 0):.4f}'],
    ]
    
    table = ax6.table(
        cellText=table_data,
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax6.set_title('Key Metrics', fontsize=11, pad=20)
    
    # --- 7-9. Untere Reihe: Feature Importance (falls vorhanden) ---
    ax7 = fig.add_subplot(gs[2, :])
    
    fi = eval_results.get("feature_importance")
    if fi is not None and isinstance(fi, pd.DataFrame) and len(fi) > 0:
        top_n = min(15, len(fi))
        top_fi = fi.head(top_n).copy()
        top_fi = top_fi.sort_values("importance_normalized", ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_fi)))
        ax7.barh(
            range(len(top_fi)),
            top_fi["importance_normalized"].values,
            color=colors,
            edgecolor='white',
            linewidth=0.5
        )
        ax7.set_yticks(range(len(top_fi)))
        ax7.set_yticklabels(top_fi["feature"].values, fontsize=8)
        ax7.set_xlabel('Normalized Importance', fontsize=10)
        ax7.set_title(f'Top {top_n} Feature Importance', fontsize=11)
    else:
        ax7.text(0.5, 0.5, 'Feature Importance not available',
                 ha='center', va='center', fontsize=14, color='gray')
        ax7.set_title('Feature Importance', fontsize=11)
    
    # Gesamttitel
    fig.suptitle(
        'Cognitive Impairment Prediction – Evaluation Dashboard',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    path = output_dir / "evaluation_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ==============================================================================
# BERICHT SPEICHERN
# ==============================================================================

def _save_evaluation_report(
    results: Dict,
    output_dir: Path,
    timestamp: str,
    logger=None
):
    """Speichert den vollständigen Evaluationsbericht."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- JSON-Bericht (maschinenlesbar) ---
    json_path = output_dir / f"evaluation_report_{timestamp}.json"
    
    # Serialisierbare Version erstellen
    serializable = _make_serializable(results)
    
    try:
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        if logger:
            logger.info(f"JSON-Bericht gespeichert: {json_path}")
    except Exception as e:
        if logger:
            logger.warning(f"JSON-Bericht konnte nicht gespeichert werden: {e}")
    
    # --- Text-Bericht (menschenlesbar) ---
    txt_path = output_dir / f"evaluation_report_{timestamp}.txt"
    
    try:
        with open(txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("EVALUATION REPORT – Cognitive Impairment Prediction\n")
            f.write(f"PhysioNet Challenge 2026\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 70 + "\n\n")
            
            # Datensatz-Info
            f.write("DATASET\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Samples:  {results.get('n_samples', 'N/A')}\n")
            f.write(f"Positive (CI):  {results.get('n_positive', 'N/A')}\n")
            f.write(f"Negative:       {results.get('n_negative', 'N/A')}\n")
            f.write(f"Prevalence:     {results.get('prevalence', 0):.4f}\n\n")
            
            # Challenge-Metriken
            cm = results.get("challenge_metrics", {})
            f.write("CHALLENGE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"AUROC:          {cm.get('auroc', 'N/A')}\n")
            f.write(f"AUPRC:          {cm.get('auprc', 'N/A')}\n")
            f.write(f"Accuracy:       {cm.get('accuracy', 'N/A')}\n")
            f.write(f"F-Measure:      {cm.get('f_measure', 'N/A')}\n")
            f.write(f"Challenge Score:{cm.get('challenge_score', 'N/A')}\n\n")
            
            # Erweiterte Metriken
            em = results.get("extended_metrics", {})
            f.write("EXTENDED METRICS\n")
            f.write("-" * 40 + "\n")
            for key, value in em.items():
                if isinstance(value, float):
                    f.write(f"{key:<25} {value:.4f}\n")
                else:
                    f.write(f"{key:<25} {value}\n")
            f.write("\n")
            
            # Konfidenzintervalle
            ci = results.get("confidence_intervals", {})
            if ci:
                f.write("CONFIDENCE INTERVALS (95%)\n")
                f.write("-" * 40 + "\n")
                for metric, values in ci.items():
                    if isinstance(values, dict):
                        f.write(
                            f"{metric:<25} {values.get('mean', 0):.4f} "
                            f"[{values.get('ci_lower', 0):.4f}, "
                            f"{values.get('ci_upper', 0):.4f}]\n"
                        )
                f.write("\n")
            
            # Optimale Schwellenwerte
            ta = results.get("threshold_analysis", {})
            optimal = ta.get("optimal_thresholds", {})
            if optimal:
                f.write("OPTIMAL THRESHOLDS\n")
                f.write("-" * 40 + "\n")
                for criterion, values in optimal.items():
                    f.write(f"\n{criterion}:\n")
                    for k, v in values.items():
                        if isinstance(v, float):
                            f.write(f"  {k}: {v:.4f}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                f.write("\n")
            
            # Kalibrierung
            cal = results.get("calibration", {})
            if cal:
                f.write("CALIBRATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Brier Score:    {cal.get('brier_score', 'N/A')}\n")
                f.write(f"ECE:            {cal.get('ece', 'N/A')}\n")
                f.write(f"MCE:            {cal.get('mce', 'N/A')}\n\n")
            
            # Fehleranalyse
            ea = results.get("error_analysis", {})
            if ea:
                f.write("ERROR ANALYSIS\n")
                f.write("-" * 40 + "\n")
                fp = ea.get("false_positives", {})
                fn = ea.get("false_negatives", {})
                f.write(f"False Positives:  {fp.get('count', 0)}\n")
                f.write(f"  Mean Prob:      {fp.get('mean_probability', 'N/A')}\n")
                f.write(f"False Negatives:  {fn.get('count', 0)}\n")
                f.write(f"  Mean Prob:      {fn.get('mean_probability', 'N/A')}\n\n")
            
            # Fairness
            fairness = results.get("fairness", {})
            if fairness:
                f.write("FAIRNESS ANALYSIS\n")
                f.write("-" * 40 + "\n")
                for attr, attr_results in fairness.items():
                    if isinstance(attr_results, dict) and "disparity" in attr_results:
                        f.write(f"\n{attr}:\n")
                        for metric, value in attr_results["disparity"].items():
                            if isinstance(value, float):
                                f.write(f"  {metric}: {value:.4f}\n")
                        if "is_fair" in attr_results:
                            f.write(f"  Fair (all < 0.1): {attr_results['is_fair']}\n")
                f.write("\n")
            
            # Feature Importance Top 20
            fi = results.get("feature_importance")
            if fi is not None and isinstance(fi, pd.DataFrame) and len(fi) > 0:
                f.write("TOP 20 FEATURE IMPORTANCE\n")
                f.write("-" * 40 + "\n")
                for _, row in fi.head(20).iterrows():
                    f.write(
                        f"  {row['rank']:3d}. {row['feature']:<45} "
                        f"{row['importance_normalized']:.4f} "
                        f"(cum: {row['importance_cumulative']:.4f})\n"
                    )
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")
        
        if logger:
            logger.info(f"Text-Bericht gespeichert: {txt_path}")
    
    except Exception as e:
        if logger:
            logger.warning(f"Text-Bericht konnte nicht gespeichert werden: {e}")
    
    # --- Symlink zum neuesten Bericht ---
    try:
        latest_json = output_dir / "latest_report.json"
        if latest_json.exists() or latest_json.is_symlink():
            latest_json.unlink()
        latest_json.symlink_to(json_path.name)
    except OSError:
        # Symlinks nicht unterstützt
        try:
            with open(output_dir / "latest_report.txt", "w") as f:
                f.write(str(json_path))
        except Exception:
            pass


def _make_serializable(obj: Any) -> Any:
    """Konvertiert ein Objekt in eine JSON-serialisierbare Form."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, float):
        return obj if not np.isnan(obj) else None
    elif isinstance(obj, (int, str, bool)):
        return obj
    else:
        return str(obj)


# ==============================================================================
# ZUSAMMENFASSUNG LOGGEN
# ==============================================================================

def _log_evaluation_summary(results: Dict, logger):
    """Loggt eine kompakte Zusammenfassung der Evaluation."""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATIONS-ZUSAMMENFASSUNG")
    logger.info("=" * 60)
    
    # Challenge-Metriken
    cm = results.get("challenge_metrics", {})
    logger.info(f"\n  Challenge-Metriken:")
    logger.info(f"    AUROC:          {cm.get('auroc', 'N/A'):.4f}" 
                if isinstance(cm.get('auroc'), float) else f"    AUROC: N/A")
    logger.info(f"    AUPRC:          {cm.get('auprc', 'N/A'):.4f}"
                if isinstance(cm.get('auprc'), float) else f"    AUPRC: N/A")
    logger.info(f"    Accuracy:       {cm.get('accuracy', 'N/A'):.4f}"
                if isinstance(cm.get('accuracy'), float) else f"    Accuracy: N/A")
    logger.info(f"    F-Measure:      {cm.get('f_measure', 'N/A'):.4f}"
                if isinstance(cm.get('f_measure'), float) else f"    F-Measure: N/A")
    
    score = cm.get('challenge_score')
    if score is not None and not np.isnan(score):
        logger.info(f"    Challenge Score: {score:.4f}")
    
    # Konfidenzintervalle
    ci = results.get("confidence_intervals", {})
    if "auroc" in ci:
        auroc_ci = ci["auroc"]
        logger.info(f"\n  AUROC 95% CI: [{auroc_ci['ci_lower']:.4f}, "
                     f"{auroc_ci['ci_upper']:.4f}]")
    
    # Kalibrierung
    cal = results.get("calibration", {})
    brier = cal.get("brier_score")
    if brier is not None and not np.isnan(brier):
        logger.info(f"\n  Kalibrierung:")
        logger.info(f"    Brier Score: {brier:.4f}")
        ece = cal.get("ece")
        if ece is not None and not np.isnan(ece):
            logger.info(f"    ECE:         {ece:.4f}")
    
    # Fehleranalyse
    ea = results.get("error_analysis", {})
    if ea:
        fp_count = ea.get("false_positives", {}).get("count", 0)
        fn_count = ea.get("false_negatives", {}).get("count", 0)
        logger.info(f"\n  Fehler:")
        logger.info(f"    False Positives: {fp_count}")
        logger.info(f"    False Negatives: {fn_count}")
    
    # Fairness
    fairness = results.get("fairness", {})
    if fairness:
        all_fair = True
        for attr, attr_results in fairness.items():
            if isinstance(attr_results, dict) and "is_fair" in attr_results:
                if not attr_results["is_fair"]:
                    all_fair = False
                    break
        
        if all_fair:
            logger.info(f"\n  Fairness: ✓ Alle Gruppen fair (Disparity < 0.1)")
        else:
            logger.info(f"\n  Fairness: ⚠ Disparitäten erkannt")
            for attr, attr_results in fairness.items():
                if isinstance(attr_results, dict) and "disparity" in attr_results:
                    for metric, value in attr_results["disparity"].items():
                        if isinstance(value, float) and value >= 0.1:
                            logger.warning(f"    {attr}/{metric}: {value:.4f}")
    
    logger.info("\n" + "=" * 60)


# ==============================================================================
# CROSS-VALIDATION EVALUATION
# ==============================================================================

def evaluate_with_cross_validation(
    feature_table: pd.DataFrame,
    model_result: Dict,
    n_folds: int = CV_FOLDS,
    output_dir: Optional[Path] = None,
    logger=None
) -> Dict:
    """
    Führt eine vollständige Evaluation mit Cross-Validation durch.
    Jeder Fold wird separat evaluiert und die Ergebnisse aggregiert.
    
    Parameters
    ----------
    feature_table : pd.DataFrame
        Patient-Level Feature-Tabelle mit Target.
    model_result : Dict
        Trainiertes Modell mit allen Artefakten.
    n_folds : int
        Anzahl CV-Folds.
    output_dir : Path, optional
        Ausgabeverzeichnis.
    
    Returns
    -------
    Dict mit aggregierten CV-Evaluationsergebnissen.
    """
    from sklearn.base import clone
    from classification.train_model import prepare_training_data

    
    if output_dir is None:
        output_dir = OUTPUT_DIR / "evaluation" / "cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info(f"CV-Evaluation: {n_folds}-Fold")
    
    # Daten vorbereiten
    X, y, feature_names, _ = prepare_training_data(feature_table, logger)
    if X is None:
        return {}
    
    # Imputation + Feature Selection + Skalierung
    imputer = model_result["imputer"]
    scaler = model_result["scaler"]
    selector = model_result.get("feature_selector")
    
    X_imp = imputer.transform(X)
    if selector is not None:
        X_sel = selector.transform(X_imp)
    else:
        X_sel = X_imp
    X_scaled = scaler.transform(X_sel)
    
    # Patient IDs
    patient_ids = None
    if "patient_id" in feature_table.columns:
        # Filtere auf Patienten mit Target
        target_col = None
        for candidate in ["target", "Cognitive_Impairment"]:
            if candidate in feature_table.columns:
                target_col = candidate
                break
        if target_col:
            valid_mask = feature_table[target_col].notna()
            patient_ids = feature_table.loc[valid_mask, "patient_id"].values
    
    # Cross-Validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    all_patient_ids = []
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Modell klonen und trainieren
        fold_model = clone(model_result["model"])
        fold_model.fit(X_train, y_train)
        
        # Vorhersagen
        y_pred = fold_model.predict(X_val)
        y_prob = fold_model.predict_proba(X_val)[:, 1]
        
        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)
        all_y_pred.extend(y_pred)
        
        if patient_ids is not None:
            all_patient_ids.extend(patient_ids[val_idx])
        
        # Fold-Metriken
        fold_metrics = compute_challenge_metrics(y_val, y_pred, y_prob)
        fold_metrics["fold"] = fold_idx + 1
        fold_results.append(fold_metrics)
        
        if logger:
            logger.info(
                f"  Fold {fold_idx+1}/{n_folds}: "
                f"AUROC={fold_metrics['auroc']:.4f}, "
                f"F1={fold_metrics['f_measure']:.4f}, "
                f"Acc={fold_metrics['accuracy']:.4f}"
            )
    
    # Aggregierte Evaluation auf allen OOF-Vorhersagen
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)
    all_patient_ids = np.array(all_patient_ids) if all_patient_ids else None
    
    # Vollständige Evaluation
    aggregated_results = evaluate_model(
        y_true=all_y_true,
        y_pred=all_y_pred,
        y_prob=all_y_prob,
        patient_ids=all_patient_ids,
        feature_table=feature_table if all_patient_ids is not None else None,
        model_result=model_result,
        output_dir=output_dir,
        generate_plots=True,
        logger=logger,
    )
    
    # Fold-Ergebnisse hinzufügen
    aggregated_results["fold_results"] = fold_results
    
    # Fold-Variabilität
    fold_df = pd.DataFrame(fold_results)
    aggregated_results["fold_summary"] = {
        "auroc_mean": float(fold_df["auroc"].mean()),
        "auroc_std": float(fold_df["auroc"].std()),
        "f_measure_mean": float(fold_df["f_measure"].mean()),
        "f_measure_std": float(fold_df["f_measure"].std()),
        "accuracy_mean": float(fold_df["accuracy"].mean()),
        "accuracy_std": float(fold_df["accuracy"].std()),
        "auprc_mean": float(fold_df["auprc"].mean()),
        "auprc_std": float(fold_df["auprc"].std()),
    }
    
    if logger:
        fs = aggregated_results["fold_summary"]
        logger.info(f"\nCV-Zusammenfassung ({n_folds} Folds):")
        logger.info(f"  AUROC:     {fs['auroc_mean']:.4f} ± {fs['auroc_std']:.4f}")
        logger.info(f"  AUPRC:     {fs['auprc_mean']:.4f} ± {fs['auprc_std']:.4f}")
        logger.info(f"  F-Measure: {fs['f_measure_mean']:.4f} ± {fs['f_measure_std']:.4f}")
        logger.info(f"  Accuracy:  {fs['accuracy_mean']:.4f} ± {fs['accuracy_std']:.4f}")
    
    return aggregated_results
