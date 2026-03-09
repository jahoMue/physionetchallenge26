"""
04_feature_table/build_feature_table.py
==========================================
Zusammenführung aller Features in eine finale Parametertabelle.

Dieses Modul:
- Kombiniert ECG-, EEG- und Annotation-Features pro Segment
- Fügt Demographics/Patient-Level-Daten hinzu
- Aggregiert Segment-Features zu Patient-Level-Features
- Erstellt sowohl Segment-Level als auch Patient-Level Tabellen
- Führt Feature-Bereinigung durch (NaN-Handling, Konstantenentfernung)
- Speichert die finale Tabelle als Parquet/CSV

Zwei Ausgabe-Modi:
1. Segment-Level: Eine Zeile pro Segment (für zeitaufgelöste Modelle)
2. Patient-Level: Eine Zeile pro Patient (für klassische ML-Modelle)
   -> Aggregation über alle Segmente (Mean, Std, Min, Max, Percentile)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    FEATURE_DIR, DEMOGRAPHICS_FILE,
    TARGET_COLUMN, TIME_TO_EVENT_COLUMN,
    SEGMENT_LENGTH_SEC, EEG_HOMOLOG_PAIRS,
    SLEEP_EVENTS_OF_INTEREST
)


# ==============================================================================
# HAUPT-FUNKTION: FEATURE-TABELLE FÜR EINEN PATIENTEN
# ==============================================================================

def build_patient_feature_table(
    patient_id: str,
    ecg_features: Optional[pd.DataFrame],
    eeg_features: Optional[pd.DataFrame],
    annotation_features: Optional[pd.DataFrame],
    segment_metadata: Optional[pd.DataFrame],
    sleep_summary: Optional[Dict] = None,
    logger=None
) -> pd.DataFrame:
    """
    Erstellt die Segment-Level Feature-Tabelle für einen Patienten.
    
    Parameters
    ----------
    patient_id : str
        Patienten-ID.
    ecg_features : pd.DataFrame or None
        ECG-Features pro Segment (aus features_ecg.py).
    eeg_features : pd.DataFrame or None
        EEG-Features pro Segment (aus features_eeg.py).
    annotation_features : pd.DataFrame or None
        Annotation-Features pro Segment (aus features_annotations.py).
    segment_metadata : pd.DataFrame or None
        Segment-Metadaten (aus segment_signals.py).
    sleep_summary : Dict or None
        Schlafarchitektur-Zusammenfassung.
    logger : loguru.Logger, optional
    
    Returns
    -------
    pd.DataFrame
        Segment-Level Feature-Tabelle.
    """
    if logger:
        logger.info(f"Erstelle Feature-Tabelle für Patient {patient_id}")
    
    # --- Bestimme Anzahl Segmente ---
    n_segments = _determine_n_segments(
        ecg_features, eeg_features, annotation_features, segment_metadata
    )
    
    if n_segments == 0:
        if logger:
            logger.warning(f"Patient {patient_id}: Keine Segmente gefunden.")
        return pd.DataFrame()
    
    # --- Basis-DataFrame ---
    feature_table = pd.DataFrame({
        "patient_id": patient_id,
        "segment_idx": range(n_segments),
    })
    
    # --- Segment-Metadaten hinzufügen ---
    if segment_metadata is not None and len(segment_metadata) > 0:
        meta_cols = [
            "start_sec", "end_sec", "duration_sec", "is_complete",
            "ecg_sqi", "ecg_quality_ok", "ecg_n_rpeaks",
            "any_signal_quality_ok"
        ]
        # EEG-Qualitätsspalten dynamisch hinzufügen
        for col in segment_metadata.columns:
            if col.startswith("eeg_") and (col.endswith("_sqi") or col.endswith("_quality_ok")):
                meta_cols.append(col)
        
        for col in meta_cols:
            if col in segment_metadata.columns:
                values = segment_metadata[col].values
                if len(values) >= n_segments:
                    feature_table[f"meta_{col}"] = values[:n_segments]
                else:
                    padded = np.full(n_segments, np.nan)
                    padded[:len(values)] = values
                    feature_table[f"meta_{col}"] = padded
    
    # --- ECG-Features hinzufügen ---
    if ecg_features is not None and len(ecg_features) > 0:
        ecg_cols = [c for c in ecg_features.columns if c != "segment_idx"]
        for col in ecg_cols:
            values = ecg_features[col].values
            if len(values) >= n_segments:
                feature_table[col] = values[:n_segments]
            else:
                padded = np.full(n_segments, np.nan)
                padded[:len(values)] = values
                feature_table[col] = padded
        
        if logger:
            logger.info(f"  ECG-Features: {len(ecg_cols)} Spalten")
    
    # --- EEG-Features hinzufügen ---
    if eeg_features is not None and len(eeg_features) > 0:
        eeg_cols = [c for c in eeg_features.columns if c != "segment_idx"]
        for col in eeg_cols:
            values = eeg_features[col].values
            if len(values) >= n_segments:
                feature_table[col] = values[:n_segments]
            else:
                padded = np.full(n_segments, np.nan)
                padded[:len(values)] = values
                feature_table[col] = padded
        
        if logger:
            logger.info(f"  EEG-Features: {len(eeg_cols)} Spalten")
    
    # --- Annotation-Features hinzufügen ---
    if annotation_features is not None and len(annotation_features) > 0:
        ann_cols = [c for c in annotation_features.columns if c != "segment_idx"]
        for col in ann_cols:
            values = annotation_features[col].values
            if len(values) >= n_segments:
                feature_table[col] = values[:n_segments]
            else:
                padded = np.full(n_segments, np.nan)
                padded[:len(values)] = values
                feature_table[col] = padded
        
        if logger:
            logger.info(f"  Annotation-Features: {len(ann_cols)} Spalten")
    
    # --- Feature-Bereinigung ---
    feature_table = _clean_feature_table(feature_table, logger)
    
    if logger:
        n_features = len([c for c in feature_table.columns
                          if c not in ["patient_id", "segment_idx"]])
        logger.info(f"  Feature-Tabelle: {n_segments} Segmente × "
                     f"{n_features} Features")
    
    return feature_table


# ==============================================================================
# PATIENT-LEVEL AGGREGATION
# ==============================================================================

def aggregate_to_patient_level(
    segment_features: pd.DataFrame,
    sleep_summary: Optional[Dict] = None,
    logger=None
) -> pd.DataFrame:
    """
    Aggregiert Segment-Level Features zu Patient-Level Features.
    
    Für jedes numerische Feature werden berechnet:
    - Mean, Std, Min, Max, Median
    - 25. und 75. Perzentil
    - Skewness (Verteilungsform)
    - Anteil gültiger (nicht-NaN) Segmente
    
    Zusätzlich werden schlafphasen-spezifische Aggregationen berechnet:
    - Features nur während NREM
    - Features nur während REM
    - Features nur während N3 (Deep Sleep)
    
    Parameters
    ----------
    segment_features : pd.DataFrame
        Segment-Level Feature-Tabelle.
    sleep_summary : Dict or None
        Schlafarchitektur-Zusammenfassung.
    logger : loguru.Logger, optional
    
    Returns
    -------
    pd.DataFrame
        Patient-Level Feature-Tabelle (eine Zeile).
    """
    if segment_features is None or len(segment_features) == 0:
        if logger:
            logger.warning("Keine Segment-Features für Aggregation vorhanden.")
        return pd.DataFrame()
    
    patient_id = segment_features["patient_id"].iloc[0]
    
    if logger:
        logger.info(f"Aggregiere Features für Patient {patient_id}: "
                     f"{len(segment_features)} Segmente")
    
    patient_features = {"patient_id": patient_id}
    
    # --- Identifiziere numerische Feature-Spalten ---
    exclude_cols = ["patient_id", "segment_idx"]
    # Auch String/kategorische Spalten ausschließen
    numeric_cols = []
    for col in segment_features.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(segment_features[col]):
            numeric_cols.append(col)
    
    # --- Globale Aggregation (alle Segmente) ---
    patient_features.update(
        _aggregate_features(segment_features, numeric_cols, prefix="all")
    )
    
    # --- Schlafphasen-spezifische Aggregation ---
    stage_col = None
    for candidate in ["ann_dominant_stage", "dominant_stage"]:
        if candidate in segment_features.columns:
            stage_col = candidate
            break
    
    if stage_col is not None:
        # NREM (N1 + N2 + N3)
        nrem_mask = segment_features[stage_col].isin(["N1", "N2", "N3"])
        if nrem_mask.sum() > 0:
            nrem_data = segment_features[nrem_mask]
            patient_features.update(
                _aggregate_features(nrem_data, numeric_cols, prefix="nrem")
            )
            patient_features["n_nrem_segments"] = int(nrem_mask.sum())
        else:
            patient_features["n_nrem_segments"] = 0
        
        # REM
        rem_mask = segment_features[stage_col] == "REM"
        if rem_mask.sum() > 0:
            rem_data = segment_features[rem_mask]
            patient_features.update(
                _aggregate_features(rem_data, numeric_cols, prefix="rem")
            )
            patient_features["n_rem_segments"] = int(rem_mask.sum())
        else:
            patient_features["n_rem_segments"] = 0
        
        # N3 (Deep Sleep – besonders relevant für Cognitive Impairment)
        n3_mask = segment_features[stage_col] == "N3"
        if n3_mask.sum() > 0:
            n3_data = segment_features[n3_mask]
            patient_features.update(
                _aggregate_features(n3_data, numeric_cols, prefix="n3")
            )
            patient_features["n_n3_segments"] = int(n3_mask.sum())
        else:
            patient_features["n_n3_segments"] = 0
        
        # N2 (für Schlafspindel-Analyse)
        n2_mask = segment_features[stage_col] == "N2"
        if n2_mask.sum() > 0:
            n2_data = segment_features[n2_mask]
            patient_features.update(
                _aggregate_features(n2_data, numeric_cols, prefix="n2")
            )
            patient_features["n_n2_segments"] = int(n2_mask.sum())
        else:
            patient_features["n_n2_segments"] = 0
        
        # Wake
        wake_mask = segment_features[stage_col] == "W"
        patient_features["n_wake_segments"] = int(wake_mask.sum())
    
    # --- Schlafarchitektur-Features (bereits global) ---
    if sleep_summary is not None:
        for key, value in sleep_summary.items():
            patient_features[f"sleep_{key}"] = value
    
    # --- Nacht-Drittel-Vergleiche ---
    patient_features.update(
        _compute_night_third_features(segment_features, numeric_cols)
    )
    
    # --- Schlafzyklus-Vergleiche ---
    patient_features.update(
        _compute_cycle_features(segment_features, numeric_cols)
    )
    
    # --- Meta-Features ---
    patient_features["n_total_segments"] = len(segment_features)
    patient_features["n_features_per_segment"] = len(numeric_cols)
    
    # Qualitäts-Zusammenfassung
    for quality_col in ["ecg_quality_ok", "meta_ecg_quality_ok",
                         "meta_any_signal_quality_ok"]:
        if quality_col in segment_features.columns:
            patient_features[f"pct_{quality_col}"] = float(
                segment_features[quality_col].mean() * 100
            )
    
    # Erstelle DataFrame (eine Zeile)
    patient_df = pd.DataFrame([patient_features])
    
    if logger:
        n_features = len(patient_features) - 1  # Minus patient_id
        logger.info(f"  Patient-Level Features: {n_features}")
    
    return patient_df


def _aggregate_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    prefix: str
) -> Dict:
    """
    Aggregiert numerische Features mit verschiedenen Statistiken.
    """
    aggregated = {}
    
    # Wähle nur Features die sinnvoll aggregiert werden können
    # Überspringe Meta-Spalten und bereits globale Features
    skip_patterns = [
        "meta_", "ann_global_", "patient_id", "segment_idx",
        "_ctx_", "ann_time_sec", "ann_time_min", "ann_time_hours",
        "ann_time_normalized", "ann_night_third", "ann_night_quartile",
        "ann_cycle_number", "ann_cycle_position_pct",
    ]
    
    for col in numeric_cols:
        # Überspringe nicht-aggregierbare Spalten
        if any(pattern in col for pattern in skip_patterns):
            continue
        
        values = df[col].dropna().values
        
        if len(values) == 0:
            continue
        
        short_col = col  # Verwende den originalen Spaltennamen
        
        # Basis-Statistiken
        aggregated[f"{prefix}_{short_col}_mean"] = float(np.mean(values))
        aggregated[f"{prefix}_{short_col}_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        aggregated[f"{prefix}_{short_col}_median"] = float(np.median(values))
        aggregated[f"{prefix}_{short_col}_min"] = float(np.min(values))
        aggregated[f"{prefix}_{short_col}_max"] = float(np.max(values))
        
        # Perzentile
        if len(values) >= 4:
            aggregated[f"{prefix}_{short_col}_p25"] = float(np.percentile(values, 25))
            aggregated[f"{prefix}_{short_col}_p75"] = float(np.percentile(values, 75))
            aggregated[f"{prefix}_{short_col}_iqr"] = (
                aggregated[f"{prefix}_{short_col}_p75"] -
                aggregated[f"{prefix}_{short_col}_p25"]
            )
        
        # Anteil gültiger Werte
        total_in_df = len(df[col])
        n_valid = len(values)
        aggregated[f"{prefix}_{short_col}_valid_pct"] = (
            float(n_valid / total_in_df * 100) if total_in_df > 0 else 0
        )
    
    return aggregated


# ==============================================================================
# NACHT-DRITTEL-VERGLEICHE
# ==============================================================================

def _compute_night_third_features(
    segment_features: pd.DataFrame,
    numeric_cols: List[str]
) -> Dict:
    """
    Vergleicht Features zwischen den Dritteln der Nacht.
    
    Besonders relevant für Cognitive Impairment:
    - SWA nimmt normalerweise über die Nacht ab
    - REM nimmt normalerweise zu
    - Veränderungen in diesen Mustern können auf CI hinweisen
    """
    features = {}
    
    third_col = None
    for candidate in ["ann_night_third"]:
        if candidate in segment_features.columns:
            third_col = candidate
            break
    
    if third_col is None:
        return features
    
    # Wähle Schlüssel-Features für Drittel-Vergleich
    key_features = _select_key_features_for_comparison(
        segment_features, numeric_cols
    )
    
    for col in key_features:
        values_by_third = {}
        for third in [1, 2, 3]:
            mask = segment_features[third_col] == third
            if mask.sum() > 0:
                vals = segment_features.loc[mask, col].dropna().values
                if len(vals) > 0:
                    values_by_third[third] = np.mean(vals)
        
        if len(values_by_third) < 2:
            continue
        
        short_col = col
        
        # Mittelwerte pro Drittel
        for third, mean_val in values_by_third.items():
            features[f"third{third}_{short_col}_mean"] = mean_val
        
        # Differenzen zwischen Dritteln
        if 1 in values_by_third and 3 in values_by_third:
            features[f"third_diff_1_3_{short_col}"] = (
                values_by_third[3] - values_by_third[1]
            )
            # Relative Änderung
            if values_by_third[1] != 0:
                features[f"third_rel_change_1_3_{short_col}"] = (
                    (values_by_third[3] - values_by_third[1]) /
                    abs(values_by_third[1])
                )
        
        if 1 in values_by_third and 2 in values_by_third:
            features[f"third_diff_1_2_{short_col}"] = (
                values_by_third[2] - values_by_third[1]
            )
    
    return features


def _select_key_features_for_comparison(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> List[str]:
    """
    Wählt Schlüssel-Features für Nacht-Drittel und Zyklus-Vergleiche.
    Fokus auf schlafmedizinisch relevante Features.
    """
    key_patterns = [
        # HRV
        "hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_hf_power",
        "hrv_lf_hf_ratio", "hrv_sd1", "hrv_sample_entropy",
        "hrv_dfa_alpha1",
        # RSA
        "rsa_p2t_mean", "rsa_coupling_strength",
        # EEG Bandpower
        "delta_power_rel", "theta_power_rel", "alpha_power_rel",
        "sigma_power_rel", "beta_power_rel",
        # EEG Schlafspezifisch
        "swa_power", "spindle_power_total", "slowing_ratio",
        "theta_alpha_ratio", "dar",
        # EEG Komplexität
        "spectral_entropy", "sample_entropy", "permutation_entropy",
        "hjorth_complexity",
        # Annotation
        "ann_sleep_depth", "ann_total_event_count",
        "ann_arousal_count", "ann_respiratory_event_count",
    ]
    
    selected = []
    for col in numeric_cols:
        for pattern in key_patterns:
            if pattern in col and col not in selected:
                selected.append(col)
                break
    
    return selected


# ==============================================================================
# SCHLAFZYKLUS-VERGLEICHE
# ==============================================================================

def _compute_cycle_features(
    segment_features: pd.DataFrame,
    numeric_cols: List[str]
) -> Dict:
    """
    Vergleicht Features zwischen Schlafzyklen.
    
    Besonders relevant:
    - SWA-Abnahme über Zyklen (normalerweise exponentiell)
    - Spindel-Aktivität über Zyklen
    - HRV-Veränderungen über Zyklen
    """
    features = {}
    
    cycle_col = None
    for candidate in ["ann_cycle_number"]:
        if candidate in segment_features.columns:
            cycle_col = candidate
            break
    
    if cycle_col is None:
        return features
    
    # Anzahl Zyklen
    max_cycle = int(segment_features[cycle_col].max())
    features["n_sleep_cycles"] = max_cycle
    
    if max_cycle < 2:
        return features
    
    # Schlüssel-Features für Zyklus-Vergleich
    key_features = _select_key_features_for_comparison(
        segment_features, numeric_cols
    )
    
    for col in key_features:
        cycle_means = {}
        for cycle_num in range(1, max_cycle + 1):
            mask = segment_features[cycle_col] == cycle_num
            if mask.sum() > 0:
                vals = segment_features.loc[mask, col].dropna().values
                if len(vals) > 0:
                    cycle_means[cycle_num] = np.mean(vals)
        
        if len(cycle_means) < 2:
            continue
        
        short_col = col
        
        # Mittelwerte pro Zyklus (erste 4 Zyklen)
        for cycle_num in range(1, min(5, max_cycle + 1)):
            if cycle_num in cycle_means:
                features[f"cycle{cycle_num}_{short_col}_mean"] = cycle_means[cycle_num]
        
        # Trend über Zyklen (lineare Steigung)
        if len(cycle_means) >= 3:
            x = np.array(list(cycle_means.keys()))
            y = np.array(list(cycle_means.values()))
            try:
                coeffs = np.polyfit(x, y, 1)
                features[f"cycle_trend_{short_col}"] = float(coeffs[0])
            except Exception:
                pass
        
        # Differenz Zyklus 1 vs. letzter Zyklus
        if 1 in cycle_means and max_cycle in cycle_means:
            features[f"cycle_diff_first_last_{short_col}"] = (
                cycle_means[max_cycle] - cycle_means[1]
            )
            if cycle_means[1] != 0:
                features[f"cycle_rel_change_{short_col}"] = (
                    (cycle_means[max_cycle] - cycle_means[1]) /
                    abs(cycle_means[1])
                )
    
    return features


# ==============================================================================
# FEATURE-BEREINIGUNG
# ==============================================================================

def _clean_feature_table(
    df: pd.DataFrame,
    logger=None
) -> pd.DataFrame:
    """
    Bereinigt die Feature-Tabelle:
    - Entfernt konstante Spalten
    - Entfernt Spalten mit zu vielen NaN-Werten
    - Ersetzt Inf-Werte
    - Konvertiert Boolean zu Int
    """
    original_cols = len(df.columns)
    
    # --- Inf-Werte ersetzen ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # --- Boolean zu Int ---
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # --- Konstante Spalten entfernen ---
    # (Spalten wo alle Werte gleich sind – kein Informationsgehalt)
    cols_to_drop = []
    for col in df.columns:
        if col in ["patient_id", "segment_idx"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) <= 1:
                cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        if logger:
            logger.debug(f"  Entfernt: {len(cols_to_drop)} konstante Spalten")
    
    # --- Spalten mit >90% NaN entfernen ---
    nan_threshold = 0.9
    high_nan_cols = []
    for col in df.columns:
        if col in ["patient_id", "segment_idx"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            nan_ratio = df[col].isna().mean()
            if nan_ratio > nan_threshold:
                high_nan_cols.append(col)
    
    if high_nan_cols:
        df = df.drop(columns=high_nan_cols)
        if logger:
            logger.debug(f"  Entfernt: {len(high_nan_cols)} Spalten mit "
                         f">{nan_threshold*100:.0f}% NaN")
    
    if logger:
        final_cols = len(df.columns)
        removed = original_cols - final_cols
        logger.info(f"  Feature-Bereinigung: {original_cols} -> {final_cols} "
                     f"Spalten ({removed} entfernt)")
    
    return df


# ==============================================================================
# DEMOGRAPHICS HINZUFÜGEN
# ==============================================================================

def add_demographics(
    feature_table: pd.DataFrame,
    demographics_path: Path = DEMOGRAPHICS_FILE,
    logger=None
) -> pd.DataFrame:
    """
    Fügt Demographics-Daten zur Feature-Tabelle hinzu.
    
    Parameters
    ----------
    feature_table : pd.DataFrame
        Feature-Tabelle (Segment- oder Patient-Level).
    demographics_path : Path
        Pfad zur Demographics-CSV.
    
    Returns
    -------
    pd.DataFrame
        Feature-Tabelle mit Demographics.
    """
    if not demographics_path.exists():
        if logger:
            logger.warning(f"Demographics-Datei nicht gefunden: {demographics_path}")
        return feature_table
    
    try:
        demographics = pd.read_csv(demographics_path)
        
        if logger:
            logger.info(f"Demographics geladen: {len(demographics)} Patienten, "
                        f"Spalten: {list(demographics.columns)}")
        
        # Identifiziere die Join-Spalte
        join_col = None
        for candidate in ["BDSPPatientID", "PatientID", "patient_id", "SubjectID"]:
            if candidate in demographics.columns:
                join_col = candidate
                break
        
        if join_col is None:
            if logger:
                logger.warning("Keine passende Join-Spalte in Demographics gefunden.")
            return feature_table
        
        # Merge
        # Stelle sicher dass patient_id in beiden DataFrames gleichen Typ hat
        feature_table["patient_id"] = feature_table["patient_id"].astype(str)
        demographics[join_col] = demographics[join_col].astype(str)
        
        merged = feature_table.merge(
            demographics,
            left_on="patient_id",
            right_on=join_col,
            how="left"
        )
        
        # Entferne doppelte Join-Spalte falls nötig
        if join_col != "patient_id" and join_col in merged.columns:
            merged = merged.drop(columns=[join_col])
        
        # --- Demographics-Features aufbereiten ---
        merged = _process_demographics(merged, logger)
        
        if logger:
            n_matched = merged[TARGET_COLUMN].notna().sum() if TARGET_COLUMN in merged.columns else 0
            logger.info(f"Demographics gemerged: {n_matched}/{len(merged)} "
                        f"mit Target-Variable")
        
        return merged
    
    except Exception as e:
        if logger:
            logger.error(f"Fehler beim Laden der Demographics: {e}")
        return feature_table


def _process_demographics(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Verarbeitet Demographics-Spalten für ML.
    """
    # --- Age ---
    if "Age" in df.columns:
        df["demo_age"] = pd.to_numeric(df["Age"], errors="coerce")
        # Age-Gruppen
        df["demo_age_group"] = pd.cut(
            df["demo_age"],
            bins=[0, 50, 60, 70, 80, 120],
            labels=[0, 1, 2, 3, 4],
            right=False
        ).astype(float)
    
    # --- Sex ---
    if "Sex" in df.columns:
        sex_map = {"male": 0, "m": 0, "female": 1, "f": 1}
        df["demo_sex"] = df["Sex"].str.lower().map(sex_map)
    
    # --- BMI ---
    if "BMI" in df.columns:
        df["demo_bmi"] = pd.to_numeric(df["BMI"], errors="coerce")
        # BMI-Kategorien
        df["demo_bmi_category"] = pd.cut(
            df["demo_bmi"],
            bins=[0, 18.5, 25, 30, 35, 100],
            labels=[0, 1, 2, 3, 4],
            right=False
        ).astype(float)
    
    # --- Race ---
    if "Race" in df.columns:
        # One-Hot Encoding
        race_dummies = pd.get_dummies(df["Race"], prefix="demo_race", dummy_na=False)
        # Konvertiere zu numerisch
        for col in race_dummies.columns:
            race_dummies[col] = race_dummies[col].astype(int)
        df = pd.concat([df, race_dummies], axis=1)
    

    # --- Ethnicity ---
    if "Ethnicity" in df.columns:
        ethnicity_dummies = pd.get_dummies(
            df["Ethnicity"], prefix="demo_ethnicity", dummy_na=False
        )
        for col in ethnicity_dummies.columns:
            ethnicity_dummies[col] = ethnicity_dummies[col].astype(int)
        df = pd.concat([df, ethnicity_dummies], axis=1)
    
    # --- Time to Event ---
    if TIME_TO_EVENT_COLUMN in df.columns:
        df["demo_time_to_event"] = pd.to_numeric(
            df[TIME_TO_EVENT_COLUMN], errors="coerce"
        )
    
    # --- Target Variable ---
    if TARGET_COLUMN in df.columns:
        df["target"] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    
    # --- SiteID (als kategorisch) ---
    if "SiteID" in df.columns:
        df["demo_site_id"] = pd.Categorical(df["SiteID"]).codes
    
    return df


# ==============================================================================
# MULTI-PATIENT FEATURE-TABELLE
# ==============================================================================

def build_cohort_feature_table(
    patient_segment_tables: Dict[str, pd.DataFrame],
    patient_sleep_summaries: Optional[Dict[str, Dict]] = None,
    demographics_path: Path = DEMOGRAPHICS_FILE,
    output_dir: Path = FEATURE_DIR,
    save: bool = True,
    logger=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Erstellt die Feature-Tabellen für die gesamte Kohorte.
    
    Gibt sowohl Segment-Level als auch Patient-Level Tabellen zurück.
    
    Parameters
    ----------
    patient_segment_tables : Dict[str, pd.DataFrame]
        Dictionary: patient_id -> Segment-Level Feature-Tabelle.
    patient_sleep_summaries : Dict[str, Dict], optional
        Dictionary: patient_id -> Schlafarchitektur-Zusammenfassung.
    demographics_path : Path
        Pfad zur Demographics-CSV.
    output_dir : Path
        Ausgabeverzeichnis.
    save : bool
        Ob die Tabellen gespeichert werden sollen.
    logger : loguru.Logger, optional
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (segment_level_table, patient_level_table)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("Erstelle Kohorten-Feature-Tabellen")
        logger.info(f"Anzahl Patienten: {len(patient_segment_tables)}")
        logger.info("=" * 60)
    
    # --- Segment-Level Tabelle ---
    segment_tables = []
    for patient_id, seg_table in patient_segment_tables.items():
        if seg_table is not None and len(seg_table) > 0:
            segment_tables.append(seg_table)
    
    if not segment_tables:
        if logger:
            logger.error("Keine Segment-Tabellen vorhanden!")
        return pd.DataFrame(), pd.DataFrame()
    
    segment_level = pd.concat(segment_tables, ignore_index=True)
    
    if logger:
        n_patients = segment_level["patient_id"].nunique()
        n_segments = len(segment_level)
        n_features = len([c for c in segment_level.columns
                          if c not in ["patient_id", "segment_idx"]])
        logger.info(f"Segment-Level: {n_patients} Patienten, "
                     f"{n_segments} Segmente, {n_features} Features")
    
    # --- Patient-Level Tabelle ---
    patient_tables = []
    for patient_id, seg_table in patient_segment_tables.items():
        if seg_table is None or len(seg_table) == 0:
            continue
        
        sleep_summary = None
        if patient_sleep_summaries and patient_id in patient_sleep_summaries:
            sleep_summary = patient_sleep_summaries[patient_id]
        
        try:
            patient_df = aggregate_to_patient_level(
                seg_table, sleep_summary, logger
            )
            if len(patient_df) > 0:
                patient_tables.append(patient_df)
        except Exception as e:
            if logger:
                logger.error(f"Aggregation fehlgeschlagen für {patient_id}: {e}")
    
    if not patient_tables:
        if logger:
            logger.error("Keine Patient-Level Tabellen erstellt!")
        return segment_level, pd.DataFrame()
    
    patient_level = pd.concat(patient_tables, ignore_index=True)
    
    if logger:
        n_patients = len(patient_level)
        n_features = len([c for c in patient_level.columns
                          if c not in ["patient_id"]])
        logger.info(f"Patient-Level: {n_patients} Patienten, "
                     f"{n_features} Features")
    
    # --- Demographics hinzufügen ---
    segment_level = add_demographics(segment_level, demographics_path, logger)
    patient_level = add_demographics(patient_level, demographics_path, logger)
    
    # --- Finale Bereinigung der Patient-Level Tabelle ---
    patient_level = _final_cleanup_patient_level(patient_level, logger)
    
    # --- Speichern ---
    if save:
        _save_feature_tables(segment_level, patient_level, output_dir, logger)
    
    # --- Zusammenfassung ---
    if logger:
        _log_feature_table_summary(segment_level, patient_level, logger)
    
    return segment_level, patient_level


def _final_cleanup_patient_level(
    df: pd.DataFrame,
    logger=None
) -> pd.DataFrame:
    """
    Finale Bereinigung der Patient-Level Feature-Tabelle.
    """
    if len(df) == 0:
        return df
    
    original_cols = len(df.columns)
    
    # --- Inf ersetzen ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # --- Konstante Spalten entfernen ---
    cols_to_drop = []
    for col in df.columns:
        if col in ["patient_id", "target", "demo_time_to_event"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) <= 1:
                cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # --- Spalten mit >80% NaN entfernen ---
    nan_threshold = 0.8
    high_nan_cols = []
    for col in df.columns:
        if col in ["patient_id", "target", "demo_time_to_event"]:
            continue
        if df[col].isna().mean() > nan_threshold:
            high_nan_cols.append(col)
    
    if high_nan_cols:
        df = df.drop(columns=high_nan_cols)
    
    # --- Duplikate Spalten entfernen ---
    df = df.loc[:, ~df.columns.duplicated()]
    
    if logger:
        final_cols = len(df.columns)
        removed = original_cols - final_cols
        logger.info(f"Finale Bereinigung: {original_cols} -> {final_cols} "
                     f"Spalten ({removed} entfernt)")
    
    return df


# ==============================================================================
# SPEICHERN
# ==============================================================================

def _save_feature_tables(
    segment_level: pd.DataFrame,
    patient_level: pd.DataFrame,
    output_dir: Path,
    logger=None
):
    """Speichert die Feature-Tabellen."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Segment-Level
    seg_parquet = output_dir / "features_segment_level.parquet"
    seg_csv = output_dir / "features_segment_level.csv"
    
    try:
        segment_level.to_parquet(seg_parquet, index=False)
        if logger:
            logger.info(f"Segment-Level gespeichert: {seg_parquet}")
    except Exception as e:
        if logger:
            logger.warning(f"Parquet-Speicherung fehlgeschlagen: {e}, "
                           f"versuche CSV...")
        segment_level.to_csv(seg_csv, index=False)
        if logger:
            logger.info(f"Segment-Level gespeichert: {seg_csv}")
    
    # Patient-Level
    pat_parquet = output_dir / "features_patient_level.parquet"
    pat_csv = output_dir / "features_patient_level.csv"
    
    try:
        patient_level.to_parquet(pat_parquet, index=False)
        if logger:
            logger.info(f"Patient-Level gespeichert: {pat_parquet}")
    except Exception as e:
        if logger:
            logger.warning(f"Parquet-Speicherung fehlgeschlagen: {e}, "
                           f"versuche CSV...")
        patient_level.to_csv(pat_csv, index=False)
        if logger:
            logger.info(f"Patient-Level gespeichert: {pat_csv}")
    
    # Feature-Liste speichern (für Reproduzierbarkeit)
    feature_list_path = output_dir / "feature_list.txt"
    with open(feature_list_path, "w") as f:
        f.write("=== SEGMENT-LEVEL FEATURES ===\n")
        for col in sorted(segment_level.columns):
            dtype = segment_level[col].dtype
            n_nan = segment_level[col].isna().sum()
            f.write(f"{col}\t{dtype}\tNaN: {n_nan}/{len(segment_level)}\n")
        
        f.write(f"\n=== PATIENT-LEVEL FEATURES ===\n")
        for col in sorted(patient_level.columns):
            dtype = patient_level[col].dtype
            n_nan = patient_level[col].isna().sum()
            f.write(f"{col}\t{dtype}\tNaN: {n_nan}/{len(patient_level)}\n")
    
    if logger:
        logger.info(f"Feature-Liste gespeichert: {feature_list_path}")


def _log_feature_table_summary(
    segment_level: pd.DataFrame,
    patient_level: pd.DataFrame,
    logger
):
    """Loggt eine Zusammenfassung der Feature-Tabellen."""
    logger.info("=" * 60)
    logger.info("FEATURE-TABELLEN ZUSAMMENFASSUNG")
    logger.info("=" * 60)
    
    # Segment-Level
    logger.info(f"\nSegment-Level:")
    logger.info(f"  Patienten: {segment_level['patient_id'].nunique()}")
    logger.info(f"  Segmente: {len(segment_level)}")
    logger.info(f"  Features: {len(segment_level.columns)}")
    
    # NaN-Statistik
    numeric_cols = segment_level.select_dtypes(include=[np.number]).columns
    mean_nan_pct = segment_level[numeric_cols].isna().mean().mean() * 100
    logger.info(f"  Mittlerer NaN-Anteil: {mean_nan_pct:.1f}%")
    
    # Patient-Level
    logger.info(f"\nPatient-Level:")
    logger.info(f"  Patienten: {len(patient_level)}")
    logger.info(f"  Features: {len(patient_level.columns)}")
    
    numeric_cols_pat = patient_level.select_dtypes(include=[np.number]).columns
    mean_nan_pct_pat = patient_level[numeric_cols_pat].isna().mean().mean() * 100
    logger.info(f"  Mittlerer NaN-Anteil: {mean_nan_pct_pat:.1f}%")
    
    # Target-Verteilung
    if "target" in patient_level.columns:
        target_dist = patient_level["target"].value_counts(dropna=False).to_dict()
        logger.info(f"\nTarget-Verteilung:")
        for val, count in sorted(target_dist.items(), key=lambda x: str(x[0])):
            pct = count / len(patient_level) * 100
            logger.info(f"  {val}: {count} ({pct:.1f}%)")
    
    # Feature-Gruppen
    logger.info(f"\nFeature-Gruppen (Patient-Level):")
    
    group_prefixes = {
        "ECG/HRV": ["hr_", "hrv_", "rr_", "rsa_", "ecg_"],
        "EEG": ["eeg_"],
        "Annotation": ["ann_"],
        "Demographics": ["demo_"],
        "Schlafarchitektur": ["sleep_"],
        "Nacht-Drittel": ["third"],
        "Schlafzyklus": ["cycle"],
        "Meta": ["meta_", "n_", "pct_"],
    }
    
    for group_name, prefixes in group_prefixes.items():
        count = sum(
            1 for col in patient_level.columns
            if any(col.startswith(p) or f"_{p}" in col for p in prefixes)
        )
        if count > 0:
            logger.info(f"  {group_name}: {count} Features")
    
    logger.info("=" * 60)


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def _determine_n_segments(
    ecg_features: Optional[pd.DataFrame],
    eeg_features: Optional[pd.DataFrame],
    annotation_features: Optional[pd.DataFrame],
    segment_metadata: Optional[pd.DataFrame]
) -> int:
    """Bestimmt die Anzahl der Segmente aus den verfügbaren DataFrames."""
    candidates = []
    
    for df in [ecg_features, eeg_features, annotation_features, segment_metadata]:
        if df is not None and len(df) > 0:
            candidates.append(len(df))
    
    if not candidates:
        return 0
    
    # Nehme das Maximum (alle sollten gleich sein, aber Sicherheit)
    return max(candidates)


def load_and_merge_saved_features(
    output_dir: Path = FEATURE_DIR,
    level: str = "patient",
    logger=None
) -> Optional[pd.DataFrame]:
    """
    Lädt gespeicherte Feature-Tabellen.
    
    Parameters
    ----------
    output_dir : Path
        Verzeichnis mit den gespeicherten Features.
    level : str
        "segment" oder "patient".
    
    Returns
    -------
    pd.DataFrame or None
    """
    if level == "segment":
        filename = "features_segment_level"
    elif level == "patient":
        filename = "features_patient_level"
    else:
        raise ValueError(f"Unbekanntes Level: {level}")
    
    # Versuche Parquet
    parquet_path = output_dir / f"{filename}.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            if logger:
                logger.info(f"Features geladen: {parquet_path} "
                             f"({len(df)} Zeilen, {len(df.columns)} Spalten)")
            return df
        except Exception as e:
            if logger:
                logger.warning(f"Parquet-Laden fehlgeschlagen: {e}")
    
    # Fallback: CSV
    csv_path = output_dir / f"{filename}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if logger:
                logger.info(f"Features geladen: {csv_path} "
                             f"({len(df)} Zeilen, {len(df.columns)} Spalten)")
            return df
        except Exception as e:
            if logger:
                logger.error(f"CSV-Laden fehlgeschlagen: {e}")
    
    if logger:
        logger.warning(f"Keine Feature-Datei gefunden in {output_dir}")
    
    return None


def get_feature_importance_groups() -> Dict[str, List[str]]:
    """
    Definiert Feature-Gruppen für Feature-Importance-Analyse.
    Nützlich für die Interpretation der ML-Ergebnisse.
    
    Returns
    -------
    Dict[str, List[str]]
        Feature-Gruppen mit Beschreibungen.
    """
    return {
        "hrv_time_domain": [
            "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_pnn20",
            "hrv_sdsd", "hrv_mean_nn", "hrv_cv_nn", "hrv_hti"
        ],
        "hrv_frequency_domain": [
            "hrv_vlf_power", "hrv_lf_power", "hrv_hf_power",
            "hrv_lf_hf_ratio", "hrv_lf_norm", "hrv_hf_norm",
            "hrv_total_power"
        ],
        "hrv_nonlinear": [
            "hrv_sd1", "hrv_sd2", "hrv_sd1_sd2_ratio",
            "hrv_sample_entropy", "hrv_dfa_alpha1",
            "hrv_csi", "hrv_cvi"
        ],
        "cardiac_autonomic": [
            "rsa_p2t_mean", "rsa_coupling_strength",
            "rsa_resp_rate", "hr_mean", "hr_cv"
        ],
        "eeg_spectral": [
            "delta_power", "theta_power", "alpha_power",
            "sigma_power", "beta_power", "spectral_entropy"
        ],
        "eeg_sleep_markers": [
            "swa_power", "spindle_power_total", "spindle_prominence",
            "slow_spindle_power", "fast_spindle_power"
        ],
        "eeg_slowing": [
            "theta_alpha_ratio", "delta_alpha_ratio",
            "slowing_ratio", "dar", "dtabr"
        ],
        "eeg_complexity": [
            "sample_entropy", "permutation_entropy",
            "higuchi_fd", "dfa_alpha", "lzc",
            "hjorth_complexity"
        ],
        "sleep_architecture": [
            "sleep_efficiency", "tst_min", "waso_min",
            "n3_pct", "rem_pct", "sol_min", "rem_latency_min"
        ],
        "sleep_fragmentation": [
            "arousal_index", "transition_index",
            "fragmentation_index", "stage_stability"
        ],
        "respiratory_events": [
            "ahi", "central_apnea_index", "obstructive_apnea_index",
            "hypopnea_index", "apnea_burden"
        ],
        "demographics": [
            "age", "sex", "bmi"
        ],
        "temporal_dynamics": [
            "third_diff", "cycle_trend", "ctx_trend"
        ],
    }
