"""
03_feature_extraction/features_annotations.py
================================================
Annotation-basierte Feature-Extraktion pro Segment:

Schlafstadien-Features:
- Dominantes Stadium und Stabilität
- Stadien-Fraktionen und Übergänge
- Zeitlicher Verlauf der Schlafarchitektur
- Schlafzyklus-Zuordnung (NREM-REM Zyklen)

Event-Features:
- Event-Präsenz, Anzahl, Dauer
- Bidirektionale Distanzen zum nächsten/vorherigen Event
- Event-Dichte (pro Typ und aggregiert)
- Event-Muster und Clustering

Kontextuelle Features:
- Features der Nachbar-Segmente (±N Segmente)
- Gleitende Mittelwerte und Trends
- Zeitliche Position in der Nacht

Schlafzyklen-Features:
- NREM-REM Zyklus-Nummer
- Position innerhalb des Zyklus
- Zyklusdauer und -zusammensetzung
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter

from config import (
    SEGMENT_LENGTH_SEC, SLEEP_EPOCH_SEC,
    SLEEP_EVENTS_OF_INTEREST, EVENT_DISTANCE_MAX_SEC,
    CONTEXT_WINDOW_SEGMENTS, EEG_FREQUENCY_BANDS
)


# ==============================================================================
# HAUPT-FUNKTION
# ==============================================================================

def extract_annotation_features(
    stages_per_segment: Optional[pd.DataFrame],
    events_per_segment: Optional[pd.DataFrame],
    sleep_summary: Optional[Dict],
    total_duration_sec: float,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    context_window: int = CONTEXT_WINDOW_SEGMENTS,
    logger=None
) -> pd.DataFrame:
    """
    Extrahiert alle annotation-basierten Features pro Segment.
    
    Parameters
    ----------
    stages_per_segment : pd.DataFrame or None
        Schlafstadien pro Segment (aus preprocess_annotations.py).
    events_per_segment : pd.DataFrame or None
        Events pro Segment (aus preprocess_annotations.py).
    sleep_summary : Dict or None
        Schlafarchitektur-Zusammenfassung.
    total_duration_sec : float
        Gesamtdauer der Aufnahme in Sekunden.
    segment_length_sec : float
        Segmentlänge in Sekunden.
    context_window : int
        Anzahl Nachbar-Segmente für kontextuelle Features (±N).
    logger : loguru.Logger, optional
    
    Returns
    -------
    pd.DataFrame
        Feature-Tabelle mit einer Zeile pro Segment.
    """
    n_segments = int(np.ceil(total_duration_sec / segment_length_sec))
    
    if logger:
        logger.info(f"Annotation Feature-Extraktion: {n_segments} Segmente")
    
    # --- Schlafstadien-Features ---
    stage_features = _extract_stage_features(
        stages_per_segment, n_segments, logger
    )
    
    # --- Event-Features ---
    event_features = _extract_event_features(
        events_per_segment, n_segments, logger
    )
    
    # --- Schlafzyklus-Features ---
    cycle_features = _extract_sleep_cycle_features(
        stages_per_segment, n_segments, segment_length_sec, logger
    )
    
    # --- Zeitliche Position ---
    temporal_features = _extract_temporal_position_features(
        n_segments, total_duration_sec, segment_length_sec,
        stages_per_segment, logger
    )
    
    # --- Globale Schlafarchitektur-Features (Patient-Level, repliziert) ---
    global_features = _extract_global_sleep_features(
        sleep_summary, n_segments, logger
    )
    
    # --- Zusammenführen ---
    all_features = stage_features.copy()
    
    for df in [event_features, cycle_features, temporal_features, global_features]:
        if df is not None and len(df) > 0:
            # Merge über segment_idx
            merge_cols = [c for c in df.columns if c != "segment_idx"]
            for col in merge_cols:
                if col not in all_features.columns:
                    all_features[col] = df[col].values[:len(all_features)]
    
    # --- Kontextuelle Features (nach Zusammenführung) ---
    all_features = _add_contextual_features(
        all_features, context_window, logger
    )
    
    if logger:
        n_features = len([c for c in all_features.columns if c != "segment_idx"])
        logger.info(f"Annotation Features: {n_features} Features pro Segment")
    
    return all_features


# ==============================================================================
# SCHLAFSTADIEN-FEATURES
# ==============================================================================

def _extract_stage_features(
    stages_per_segment: Optional[pd.DataFrame],
    n_segments: int,
    logger=None
) -> pd.DataFrame:
    """
    Extrahiert erweiterte Schlafstadien-Features pro Segment.
    """
    features = pd.DataFrame({"segment_idx": range(n_segments)})
    
    if stages_per_segment is None or len(stages_per_segment) == 0:
        if logger:
            logger.warning("Keine Schlafstadien verfügbar für Feature-Extraktion.")
        features = _add_empty_stage_features(features)
        return features
    
    stages = stages_per_segment.copy()
    
    # --- Basis-Features (bereits in preprocess_annotations berechnet) ---
    basis_cols = [
        "dominant_stage", "dominant_stage_numeric", "stage_stability",
        "n_stage_transitions", "n1_fraction", "n2_fraction",
        "n3_fraction", "rem_fraction", "w_fraction"
    ]
    
    for col in basis_cols:
        if col in stages.columns:
            features[f"ann_{col}"] = stages[col].values[:n_segments]
        else:
            features[f"ann_{col}"] = np.nan
    
    # --- One-Hot Encoding des dominanten Stadiums ---
    if "dominant_stage" in stages.columns:
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            features[f"ann_is_{stage.lower()}"] = (
                stages["dominant_stage"].values[:n_segments] == stage
            ).astype(int)
    
    # --- Schlaftiefe-Score ---
    # W=0, N1=1, N2=2, N3=3, REM=2 (REM ähnlich N2 in Tiefe)
    depth_map = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 2, "Unknown": 0}
    if "dominant_stage" in stages.columns:
        features["ann_sleep_depth"] = stages["dominant_stage"].map(depth_map).values[:n_segments]
    else:
        features["ann_sleep_depth"] = 0
    
    # --- Ist Schlaf (nicht Wake) ---
    if "dominant_stage" in stages.columns:
        features["ann_is_sleep"] = (
            stages["dominant_stage"].values[:n_segments] != "W"
        ).astype(int)
    else:
        features["ann_is_sleep"] = 0
    
    # --- Ist NREM ---
    if "dominant_stage" in stages.columns:
        features["ann_is_nrem"] = stages["dominant_stage"].isin(
            ["N1", "N2", "N3"]
        ).values[:n_segments].astype(int)
    else:
        features["ann_is_nrem"] = 0
    
    # --- Stadien-Wechsel Features ---
    if "dominant_stage" in stages.columns:
        stage_sequence = stages["dominant_stage"].values[:n_segments]
        
        # Wechsel zum vorherigen Segment
        prev_stages = np.roll(stage_sequence, 1)
        prev_stages[0] = stage_sequence[0]
        features["ann_stage_changed"] = (stage_sequence != prev_stages).astype(int)
        
        # Wechsel zum nächsten Segment
        next_stages = np.roll(stage_sequence, -1)
        next_stages[-1] = stage_sequence[-1]
        features["ann_stage_will_change"] = (stage_sequence != next_stages).astype(int)
        
        # Dauer im aktuellen Stadium (Anzahl aufeinanderfolgender Segmente)
        features["ann_stage_duration_segments"] = _compute_run_lengths(stage_sequence)
        
        # Dauer in Sekunden
        features["ann_stage_duration_sec"] = (
            features["ann_stage_duration_segments"] * SEGMENT_LENGTH_SEC
        )
        
        # Zeit seit letztem Stadien-Wechsel
        features["ann_time_since_transition_sec"] = _time_since_event(
            features["ann_stage_changed"].values, SEGMENT_LENGTH_SEC
        )
        
        # Zeit bis zum nächsten Stadien-Wechsel
        features["ann_time_to_transition_sec"] = _time_to_event(
            features["ann_stage_will_change"].values, SEGMENT_LENGTH_SEC
        )
    
    if logger:
        if "ann_dominant_stage" in features.columns:
            stage_dist = features["ann_dominant_stage"].value_counts().to_dict()
            logger.info(f"Schlafstadien-Verteilung: {stage_dist}")
    
    return features


def _add_empty_stage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt leere Schlafstadien-Features hinzu."""
    empty_cols = {
        "ann_dominant_stage": "Unknown",
        "ann_dominant_stage_numeric": 0,
        "ann_stage_stability": 0.0,
        "ann_n_stage_transitions": 0,
        "ann_n1_fraction": 0.0,
        "ann_n2_fraction": 0.0,
        "ann_n3_fraction": 0.0,
        "ann_rem_fraction": 0.0,
        "ann_w_fraction": 0.0,
        "ann_is_w": 0, "ann_is_n1": 0, "ann_is_n2": 0,
        "ann_is_n3": 0, "ann_is_rem": 0,
        "ann_sleep_depth": 0,
        "ann_is_sleep": 0,
        "ann_is_nrem": 0,
        "ann_stage_changed": 0,
        "ann_stage_will_change": 0,
        "ann_stage_duration_segments": 0,
        "ann_stage_duration_sec": 0.0,
        "ann_time_since_transition_sec": 0.0,
        "ann_time_to_transition_sec": 0.0,
    }
    for col, default in empty_cols.items():
        df[col] = default
    return df


# ==============================================================================
# EVENT-FEATURES
# ==============================================================================

def _extract_event_features(
    events_per_segment: Optional[pd.DataFrame],
    n_segments: int,
    logger=None
) -> pd.DataFrame:
    """
    Extrahiert erweiterte Event-Features pro Segment.
    Baut auf den Basis-Features aus preprocess_annotations auf.
    """
    features = pd.DataFrame({"segment_idx": range(n_segments)})
    
    if events_per_segment is None or len(events_per_segment) == 0:
        if logger:
            logger.warning("Keine Event-Daten verfügbar.")
        features = _add_empty_event_features(features)
        return features
    
    events = events_per_segment.copy()
    
    # --- Basis-Event-Features übernehmen ---
    event_cols = [c for c in events.columns if c != "segment_idx"]
    for col in event_cols:
        if col not in features.columns:
            values = events[col].values
            if len(values) >= n_segments:
                features[f"ann_{col}"] = values[:n_segments]
            else:
                # Padding mit Defaults
                padded = np.full(n_segments, np.nan)
                padded[:len(values)] = values
                features[f"ann_{col}"] = padded
    
    # --- Erweiterte Event-Features ---
    
    # Event-Burden: Gesamtdauer aller Events relativ zur Segmentlänge
    for event_type in SLEEP_EVENTS_OF_INTEREST:
        dur_col = f"ann_{event_type}_total_duration_sec"
        if dur_col in features.columns:
            features[f"ann_{event_type}_burden"] = (
                features[dur_col] / SEGMENT_LENGTH_SEC
            )
    
    # Apnea-Burden (alle Apnea-Typen zusammen)
    apnea_dur_col = "ann_total_apnea_duration_sec"
    if apnea_dur_col in features.columns:
        features["ann_apnea_burden"] = (
            features[apnea_dur_col] / SEGMENT_LENGTH_SEC
        )
    
    # --- Event-Clustering: Sind Events gehäuft? ---
    for event_type in SLEEP_EVENTS_OF_INTEREST:
        count_col = f"ann_{event_type}_count"
        if count_col in features.columns:
            counts = features[count_col].values
            
            # Gleitender Mittelwert der Event-Anzahl (±2 Segmente)
            features[f"ann_{event_type}_count_rolling_mean"] = _rolling_mean(
                counts, window=5
            )
            
            # Ist die aktuelle Event-Anzahl über dem Durchschnitt?
            mean_count = np.nanmean(counts)
            if mean_count > 0:
                features[f"ann_{event_type}_above_average"] = (
                    counts > mean_count
                ).astype(int)
            else:
                features[f"ann_{event_type}_above_average"] = 0
    
    # --- Kombinierte Event-Scores ---
    # Respiratory Disturbance Score
    apnea_types = ["central_apnea", "obstructive_apnea", "mixed_apnea", "hypopnea"]
    resp_count = np.zeros(n_segments)
    for at in apnea_types:
        col = f"ann_{at}_count"
        if col in features.columns:
            resp_count += features[col].fillna(0).values
    features["ann_respiratory_event_count"] = resp_count
    features["ann_respiratory_event_index"] = (
        resp_count / (SEGMENT_LENGTH_SEC / 3600)  # Events pro Stunde
    )
    
    # Arousal + Respiratory combined
    arousal_col = "ann_arousal_count"
    if arousal_col in features.columns:
        features["ann_combined_event_count"] = (
            resp_count + features[arousal_col].fillna(0).values
        )
    
    if logger:
        for event_type in SLEEP_EVENTS_OF_INTEREST:
            col = f"ann_{event_type}_present"
            if col in features.columns:
                n_present = features[col].sum()
                logger.info(f"Event '{event_type}': {n_present}/{n_segments} "
                            f"Segmente betroffen")
    
    return features


def _add_empty_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt leere Event-Features hinzu."""
    for event_type in SLEEP_EVENTS_OF_INTEREST:
        prefix = f"ann_{event_type}"
        df[f"{prefix}_present"] = False
        df[f"{prefix}_count"] = 0
        df[f"{prefix}_total_duration_sec"] = 0.0
        df[f"{prefix}_dist_to_next_sec"] = EVENT_DISTANCE_MAX_SEC
        df[f"{prefix}_dist_to_prev_sec"] = EVENT_DISTANCE_MAX_SEC
        df[f"{prefix}_density"] = 0.0
        df[f"{prefix}_burden"] = 0.0
        df[f"{prefix}_count_rolling_mean"] = 0.0
        df[f"{prefix}_above_average"] = 0
    
    df["ann_any_event_present"] = False
    df["ann_total_event_count"] = 0
    df["ann_min_dist_to_next_event_sec"] = EVENT_DISTANCE_MAX_SEC
    df["ann_min_dist_to_prev_event_sec"] = EVENT_DISTANCE_MAX_SEC
    df["ann_any_apnea_present"] = False
    df["ann_total_apnea_count"] = 0
    df["ann_total_apnea_duration_sec"] = 0.0
    df["ann_apnea_burden"] = 0.0
    df["ann_respiratory_event_count"] = 0
    df["ann_respiratory_event_index"] = 0.0
    df["ann_combined_event_count"] = 0
    
    return df


# ==============================================================================
# SCHLAFZYKLUS-FEATURES
# ==============================================================================

def _extract_sleep_cycle_features(
    stages_per_segment: Optional[pd.DataFrame],
    n_segments: int,
    segment_length_sec: float,
    logger=None
) -> pd.DataFrame:
    """
    Identifiziert NREM-REM Schlafzyklen und berechnet Zyklus-Features.
    
    Ein Schlafzyklus besteht typischerweise aus:
    NREM (N1 -> N2 -> N3) -> REM
    Dauer: 90-120 Minuten
    """
    features = pd.DataFrame({"segment_idx": range(n_segments)})
    
    if stages_per_segment is None or "dominant_stage" not in stages_per_segment.columns:
        features["ann_cycle_number"] = 0
        features["ann_cycle_position_pct"] = np.nan
        features["ann_cycle_phase"] = "unknown"
        features["ann_is_first_cycle"] = 0
        features["ann_is_last_cycle"] = 0
        features["ann_n_total_cycles"] = 0
        features["ann_cycle_nrem_duration_min"] = np.nan
        features["ann_cycle_rem_duration_min"] = np.nan
        return features
    
    stages = stages_per_segment["dominant_stage"].values[:n_segments]
    
    # --- Zyklen identifizieren ---
    cycles = _identify_sleep_cycles(stages, segment_length_sec)
    
    # --- Features zuweisen ---
    cycle_numbers = np.zeros(n_segments, dtype=int)
    cycle_positions = np.full(n_segments, np.nan)
    cycle_phases = ["unknown"] * n_segments
    cycle_nrem_dur = np.full(n_segments, np.nan)
    cycle_rem_dur = np.full(n_segments, np.nan)
    
    for cycle in cycles:
        cycle_num = cycle["cycle_number"]
        start_idx = cycle["start_segment"]
        end_idx = cycle["end_segment"]
        nrem_end = cycle.get("nrem_end_segment", start_idx)
        cycle_len = end_idx - start_idx + 1
        
        for seg_idx in range(start_idx, min(end_idx + 1, n_segments)):
            cycle_numbers[seg_idx] = cycle_num
            
            # Position innerhalb des Zyklus (0-1)
            if cycle_len > 0:
                cycle_positions[seg_idx] = (seg_idx - start_idx) / cycle_len
            
            # Phase: NREM oder REM
            if seg_idx <= nrem_end:
                cycle_phases[seg_idx] = "nrem"
            else:
                cycle_phases[seg_idx] = "rem"
            
            # Zyklus-Dauern
            cycle_nrem_dur[seg_idx] = cycle.get("nrem_duration_min", np.nan)
            cycle_rem_dur[seg_idx] = cycle.get("rem_duration_min", np.nan)
    
    n_total_cycles = len(cycles)
    
    features["ann_cycle_number"] = cycle_numbers
    features["ann_cycle_position_pct"] = cycle_positions
    features["ann_cycle_phase"] = cycle_phases
    features["ann_is_first_cycle"] = (cycle_numbers == 1).astype(int)
    features["ann_is_last_cycle"] = (
        (cycle_numbers == n_total_cycles) & (cycle_numbers > 0)
    ).astype(int)
    features["ann_n_total_cycles"] = n_total_cycles
    features["ann_cycle_nrem_duration_min"] = cycle_nrem_dur
    features["ann_cycle_rem_duration_min"] = cycle_rem_dur
    
    # One-Hot für Zyklus-Phase
    features["ann_cycle_is_nrem"] = (
        np.array(cycle_phases) == "nrem"
    ).astype(int)
    features["ann_cycle_is_rem"] = (
        np.array(cycle_phases) == "rem"
    ).astype(int)
    
    if logger:
        logger.info(f"Schlafzyklen identifiziert: {n_total_cycles}")
        for cycle in cycles:
            logger.debug(
                f"  Zyklus {cycle['cycle_number']}: "
                f"Seg {cycle['start_segment']}-{cycle['end_segment']}, "
                f"NREM={cycle.get('nrem_duration_min', 0):.1f}min, "
                f"REM={cycle.get('rem_duration_min', 0):.1f}min"
            )
    
    return features


def _identify_sleep_cycles(
    stages: np.ndarray,
    segment_length_sec: float
) -> List[Dict]:
    """
    Identifiziert NREM-REM Schlafzyklen.
    
    Algorithmus:
    1. Finde zusammenhängende NREM-Blöcke (N1/N2/N3)
    2. Finde darauffolgende REM-Blöcke
    3. Ein Zyklus = NREM-Block + REM-Block
    4. Mindestdauer: NREM >= 15min, REM >= 5min (außer letzter Zyklus)
    """
    cycles = []
    n = len(stages)
    
    if n == 0:
        return cycles
    
    # Klassifiziere Segmente als NREM, REM, oder Wake
    seg_type = []
    for s in stages:
        if s in ["N1", "N2", "N3"]:
            seg_type.append("NREM")
        elif s == "REM":
            seg_type.append("REM")
        else:
            seg_type.append("WAKE")
    
    seg_type = np.array(seg_type)
    
    # Finde Schlafbeginn (erstes NREM oder REM)
    sleep_start = None
    for i, st in enumerate(seg_type):
        if st in ["NREM", "REM"]:
            sleep_start = i
            break
    
    if sleep_start is None:
        return cycles
    
    # Durchlaufe die Aufnahme und identifiziere Zyklen
    cycle_num = 0
    i = sleep_start
    
    while i < n:
        # Suche NREM-Block
        nrem_start = None
        for j in range(i, n):
            if seg_type[j] == "NREM":
                nrem_start = j
                break
        
        if nrem_start is None:
            break
        
        # Finde Ende des NREM-Blocks
        # Erlaube kurze Wake-Unterbrechungen (≤ 2 Segmente)
        nrem_end = nrem_start
        wake_count = 0
        for j in range(nrem_start, n):
            if seg_type[j] == "NREM":
                nrem_end = j
                wake_count = 0
            elif seg_type[j] == "WAKE":
                wake_count += 1
                if wake_count > 2:
                    break
                nrem_end = j
            elif seg_type[j] == "REM":
                break
        
        nrem_duration_min = (nrem_end - nrem_start + 1) * segment_length_sec / 60.0
        
        # Suche REM-Block nach NREM
        rem_start = None
        rem_end = None
        for j in range(nrem_end + 1, min(nrem_end + 10, n)):  # Suche in den nächsten 10 Segmenten
            if seg_type[j] == "REM":
                if rem_start is None:
                    rem_start = j
                rem_end = j
            elif seg_type[j] == "NREM" and rem_start is not None:
                break
            elif seg_type[j] == "WAKE" and rem_start is not None:
                # Erlaube kurze Wake-Unterbrechungen
                wake_ahead = sum(
                    1 for k in range(j, min(j + 3, n))
                    if seg_type[k] == "WAKE"
                )
                if wake_ahead > 2:
                    break
        
        rem_duration_min = 0
        if rem_start is not None and rem_end is not None:
            rem_duration_min = (rem_end - rem_start + 1) * segment_length_sec / 60.0
        
        # Mindestdauer-Kriterien
        min_nrem_min = 15.0
        min_rem_min = 0.0  # REM kann im ersten Zyklus fehlen
        
        if nrem_duration_min >= min_nrem_min:
            cycle_num += 1
            
            cycle_end = rem_end if rem_end is not None else nrem_end
            
            cycles.append({
                "cycle_number": cycle_num,
                "start_segment": nrem_start,
                "end_segment": cycle_end,
                "nrem_start_segment": nrem_start,
                "nrem_end_segment": nrem_end,
                "rem_start_segment": rem_start,
                "rem_end_segment": rem_end,
                "nrem_duration_min": nrem_duration_min,
                "rem_duration_min": rem_duration_min,
                "total_duration_min": (cycle_end - nrem_start + 1) * segment_length_sec / 60.0,
                "has_rem": rem_start is not None,
            })
            
            i = cycle_end + 1
        else:
            i = nrem_end + 1
    
    return cycles


# ==============================================================================
# ZEITLICHE POSITION
# ==============================================================================

def _extract_temporal_position_features(
    n_segments: int,
    total_duration_sec: float,
    segment_length_sec: float,
    stages_per_segment: Optional[pd.DataFrame],
    logger=None
) -> pd.DataFrame:
    """
    Berechnet Features zur zeitlichen Position in der Nacht.
    """
    features = pd.DataFrame({"segment_idx": range(n_segments)})
    
    # --- Absolute und relative Position ---
    segment_centers = np.array([
        (i * segment_length_sec + segment_length_sec / 2)
        for i in range(n_segments)
    ])
    
    features["ann_time_sec"] = segment_centers
    features["ann_time_min"] = segment_centers / 60.0
    features["ann_time_hours"] = segment_centers / 3600.0
    
    # Normalisierte Position (0 = Anfang, 1 = Ende)
    features["ann_time_normalized"] = segment_centers / total_duration_sec
    
    # Drittel der Nacht (1, 2, 3)
    features["ann_night_third"] = np.clip(
        np.floor(features["ann_time_normalized"] * 3) + 1, 1, 3
    ).astype(int)
    
    # Quartil der Nacht
    features["ann_night_quartile"] = np.clip(
        np.floor(features["ann_time_normalized"] * 4) + 1, 1, 4
    ).astype(int)
    
    # --- Zeit relativ zum Schlafbeginn ---
    if stages_per_segment is not None and "dominant_stage" in stages_per_segment.columns:
        stages = stages_per_segment["dominant_stage"].values[:n_segments]
        
        # Schlafbeginn: Erstes Nicht-Wake-Segment
        sleep_onset_idx = None
        for i, s in enumerate(stages):
            if s != "W" and s != "Unknown":
                sleep_onset_idx = i
                break
        
        if sleep_onset_idx is not None:
            sleep_onset_sec = sleep_onset_idx * segment_length_sec
            features["ann_time_since_sleep_onset_min"] = (
                (segment_centers - sleep_onset_sec) / 60.0
            )
            features["ann_time_since_sleep_onset_min"] = features[
                "ann_time_since_sleep_onset_min"
            ].clip(lower=0)
        else:
            features["ann_time_since_sleep_onset_min"] = np.nan
        
        # Schlafende: Letztes Nicht-Wake-Segment
        sleep_end_idx = None
        for i in range(len(stages) - 1, -1, -1):
            if stages[i] != "W" and stages[i] != "Unknown":
                sleep_end_idx = i
                break
        
        if sleep_end_idx is not None and sleep_onset_idx is not None:
            sleep_duration_sec = (sleep_end_idx - sleep_onset_idx) * segment_length_sec
            if sleep_duration_sec > 0:
                features["ann_sleep_time_normalized"] = (
                    (segment_centers - sleep_onset_idx * segment_length_sec) /
                    sleep_duration_sec
                ).clip(0, 1)
            else:
                features["ann_sleep_time_normalized"] = np.nan
        else:
            features["ann_sleep_time_normalized"] = np.nan
    else:
        features["ann_time_since_sleep_onset_min"] = np.nan
        features["ann_sleep_time_normalized"] = np.nan
    
    return features


# ==============================================================================
# GLOBALE SCHLAFARCHITEKTUR-FEATURES
# ==============================================================================

def _extract_global_sleep_features(
    sleep_summary: Optional[Dict],
    n_segments: int,
    logger=None
) -> pd.DataFrame:
    """
    Extrahiert globale Schlafarchitektur-Features (Patient-Level).
    Diese werden für jedes Segment repliziert, da sie den gesamten
    Patienten charakterisieren.
    
    Besonders relevant für Cognitive Impairment:
    - Schlafeffizienz
    - SWS-Anteil (reduziert bei CI)
    - REM-Anteil und -Latenz (verändert bei CI)
    - Arousal-Index
    - AHI
    - WASO
    - Schlafstadien-Fragmentierung
    
    Referenz: Schlafarchitektur-Marker für kognitive Beeinträchtigung
    basieren auf etablierten Zusammenhängen zwischen Schlafqualität
    und kognitiver Funktion [[10]].
    """
    features = pd.DataFrame({"segment_idx": range(n_segments)})
    
    if sleep_summary is None or len(sleep_summary) == 0:
        if logger:
            logger.warning("Keine Schlafarchitektur-Zusammenfassung verfügbar.")
        features = _add_empty_global_features(features, n_segments)
        return features
    
    # --- Schlafarchitektur-Metriken (repliziert für jedes Segment) ---
    global_feature_keys = {
        # Schlafzeiten
        "total_recording_time_min": "ann_global_trt_min",
        "total_scored_time_min": "ann_global_scored_time_min",
        "total_sleep_time_min": "ann_global_tst_min",
        "sleep_efficiency": "ann_global_sleep_efficiency",
        "waso_min": "ann_global_waso_min",
        "sleep_onset_latency_min": "ann_global_sol_min",
        "rem_latency_min": "ann_global_rem_latency_min",
        
        # Stadien-Anteile
        "w_pct_tst": "ann_global_w_pct",
        "n1_pct_tst": "ann_global_n1_pct",
        "n2_pct_tst": "ann_global_n2_pct",
        "n3_pct_tst": "ann_global_n3_pct",
        "rem_pct_tst": "ann_global_rem_pct",
        
        # Stadien-Zeiten
        "w_time_min": "ann_global_w_time_min",
        "n1_time_min": "ann_global_n1_time_min",
        "n2_time_min": "ann_global_n2_time_min",
        "n3_time_min": "ann_global_n3_time_min",
        "rem_time_min": "ann_global_rem_time_min",
        
        # Fragmentierung
        "n_stage_transitions": "ann_global_n_transitions",
        "stage_transition_index": "ann_global_transition_index",
        
        # Event-Indizes
        "ahi": "ann_global_ahi",
        "arousal_index": "ann_global_arousal_index",
        
        # Event-Counts
        "arousal_count": "ann_global_arousal_count",
        "central_apnea_count": "ann_global_central_apnea_count",
        "obstructive_apnea_count": "ann_global_obstructive_apnea_count",
        "mixed_apnea_count": "ann_global_mixed_apnea_count",
        "hypopnea_count": "ann_global_hypopnea_count",
        
        # Event-Indizes pro Typ
        "arousal_index": "ann_global_arousal_index",
        "central_apnea_index": "ann_global_central_apnea_index",
        "obstructive_apnea_index": "ann_global_obstructive_apnea_index",
        "hypopnea_index": "ann_global_hypopnea_index",
        
        # Event-Dauern
        "arousal_mean_duration_sec": "ann_global_arousal_mean_dur",
        "central_apnea_mean_duration_sec": "ann_global_ca_mean_dur",
        "obstructive_apnea_mean_duration_sec": "ann_global_oa_mean_dur",
        "hypopnea_mean_duration_sec": "ann_global_hyp_mean_dur",
    }
    
    for summary_key, feature_name in global_feature_keys.items():
        value = sleep_summary.get(summary_key, np.nan)
        features[feature_name] = value
    
    # --- Abgeleitete globale Features ---
    
    # NREM-Anteil (N1+N2+N3)
    n1_pct = sleep_summary.get("n1_pct_tst", 0) or 0
    n2_pct = sleep_summary.get("n2_pct_tst", 0) or 0
    n3_pct = sleep_summary.get("n3_pct_tst", 0) or 0
    features["ann_global_nrem_pct"] = n1_pct + n2_pct + n3_pct
    
    # Deep Sleep Anteil (N3 allein – besonders relevant für CI)
    features["ann_global_deep_sleep_pct"] = n3_pct
    
    # Light Sleep Anteil (N1+N2)
    features["ann_global_light_sleep_pct"] = n1_pct + n2_pct
    
    # N3/REM Ratio (verändert bei Neurodegeneration)
    rem_pct = sleep_summary.get("rem_pct_tst", 0) or 0
    features["ann_global_n3_rem_ratio"] = (
        n3_pct / rem_pct if rem_pct > 0 else np.nan
    )
    
    # Schlaffragmentierung: Übergänge pro Stunde Schlaf
    tst_hours = (sleep_summary.get("total_sleep_time_min", 0) or 0) / 60.0
    n_transitions = sleep_summary.get("n_stage_transitions", 0) or 0
    features["ann_global_fragmentation_index"] = (
        n_transitions / tst_hours if tst_hours > 0 else np.nan
    )
    
    # AHI-Kategorien
    ahi = sleep_summary.get("ahi", 0) or 0
    features["ann_global_ahi_category"] = _categorize_ahi(ahi)
    
    # Arousal-Index-Kategorien
    ai = sleep_summary.get("arousal_index", 0) or 0
    features["ann_global_ai_category"] = _categorize_arousal_index(ai)
    
    if logger:
        logger.info("Globale Schlafarchitektur-Features:")
        logger.info(f"  TST: {sleep_summary.get('total_sleep_time_min', 'N/A')} min")
        logger.info(f"  SE: {sleep_summary.get('sleep_efficiency', 'N/A')}")
        logger.info(f"  AHI: {ahi:.1f}")
        logger.info(f"  Arousal Index: {ai:.1f}")
        logger.info(f"  N3%: {n3_pct:.1f}%, REM%: {rem_pct:.1f}%")
    
    return features


def _add_empty_global_features(df: pd.DataFrame, n_segments: int) -> pd.DataFrame:
    """Fügt leere globale Features hinzu."""
    global_cols = [
        "ann_global_trt_min", "ann_global_scored_time_min",
        "ann_global_tst_min", "ann_global_sleep_efficiency",
        "ann_global_waso_min", "ann_global_sol_min",
        "ann_global_rem_latency_min",
        "ann_global_w_pct", "ann_global_n1_pct", "ann_global_n2_pct",
        "ann_global_n3_pct", "ann_global_rem_pct",
        "ann_global_w_time_min", "ann_global_n1_time_min",
        "ann_global_n2_time_min", "ann_global_n3_time_min",
        "ann_global_rem_time_min",
        "ann_global_n_transitions", "ann_global_transition_index",
        "ann_global_ahi", "ann_global_arousal_index",
        "ann_global_arousal_count", "ann_global_central_apnea_count",
        "ann_global_obstructive_apnea_count", "ann_global_mixed_apnea_count",
        "ann_global_hypopnea_count",
        "ann_global_central_apnea_index", "ann_global_obstructive_apnea_index",
        "ann_global_hypopnea_index",
        "ann_global_arousal_mean_dur", "ann_global_ca_mean_dur",
        "ann_global_oa_mean_dur", "ann_global_hyp_mean_dur",
        "ann_global_nrem_pct", "ann_global_deep_sleep_pct",
        "ann_global_light_sleep_pct", "ann_global_n3_rem_ratio",
        "ann_global_fragmentation_index",
    ]
    for col in global_cols:
        df[col] = np.nan
    
    df["ann_global_ahi_category"] = 0
    df["ann_global_ai_category"] = 0
    
    return df


def _categorize_ahi(ahi: float) -> int:
    """
    Kategorisiert den AHI.
    0: Normal (<5)
    1: Mild (5-15)
    2: Moderate (15-30)
    3: Severe (>30)
    """
    if ahi < 5:
        return 0
    elif ahi < 15:
        return 1
    elif ahi < 30:
        return 2
    else:
        return 3


def _categorize_arousal_index(ai: float) -> int:
    """
    Kategorisiert den Arousal-Index.
    0: Normal (<10)
    1: Mild (10-25)
    2: Moderate (25-50)
    3: Severe (>50)
    """
    if ai < 10:
        return 0
    elif ai < 25:
        return 1
    elif ai < 50:
        return 2
    else:
        return 3


# ==============================================================================
# KONTEXTUELLE FEATURES
# ==============================================================================

def _add_contextual_features(
    features_df: pd.DataFrame,
    context_window: int = CONTEXT_WINDOW_SEGMENTS,
    logger=None
) -> pd.DataFrame:
    """
    Fügt kontextuelle Features hinzu: Informationen aus Nachbar-Segmenten.
    
    Für jedes numerische Feature werden berechnet:
    - Gleitender Mittelwert (±N Segmente)
    - Differenz zum vorherigen Segment (Trend)
    - Differenz zum gleitenden Mittelwert (Abweichung)
    
    Nur für ausgewählte, wichtige Features um Feature-Explosion zu vermeiden.
    """
    if logger:
        logger.info(f"Kontextuelle Features: Fenster ±{context_window} Segmente")
    
    # Wähle Features für Kontext-Berechnung
    context_features = _select_features_for_context(features_df)
    
    if not context_features:
        if logger:
            logger.warning("Keine Features für Kontext-Berechnung gefunden.")
        return features_df
    
    window_size = 2 * context_window + 1  # Gesamtfenstergröße
    
    for feat_name in context_features:
        if feat_name not in features_df.columns:
            continue
        
        values = features_df[feat_name].values.astype(float)
        
        # --- Gleitender Mittelwert ---
        rolling_mean = _rolling_mean(values, window=window_size)
        features_df[f"{feat_name}_ctx_mean"] = rolling_mean
        
        # --- Gleitende Standardabweichung ---
        rolling_std = _rolling_std(values, window=window_size)
        features_df[f"{feat_name}_ctx_std"] = rolling_std
        
        # --- Differenz zum vorherigen Segment (Trend) ---
        diff = np.full(len(values), np.nan)
        diff[1:] = values[1:] - values[:-1]
        features_df[f"{feat_name}_ctx_diff"] = diff
        
        # --- Abweichung vom gleitenden Mittelwert ---
        deviation = values - rolling_mean
        features_df[f"{feat_name}_ctx_deviation"] = deviation
        
        # --- Trend über Kontextfenster (lineare Steigung) ---
        trend = _rolling_trend(values, window=window_size)
        features_df[f"{feat_name}_ctx_trend"] = trend
    
    if logger:
        n_ctx_features = len(context_features) * 5  # 5 kontextuelle Features pro Basis-Feature
        logger.info(f"Kontextuelle Features hinzugefügt: {n_ctx_features} "
                     f"(basierend auf {len(context_features)} Basis-Features)")
    
    return features_df


def _select_features_for_context(df: pd.DataFrame) -> List[str]:
    """
    Wählt die wichtigsten Features für die Kontext-Berechnung aus.
    Vermeidet Feature-Explosion durch gezielte Auswahl.
    """
    selected = []
    
    # Schlafstadien-Features
    stage_features = [
        "ann_sleep_depth",
        "ann_stage_stability",
        "ann_n1_fraction", "ann_n2_fraction",
        "ann_n3_fraction", "ann_rem_fraction",
        "ann_w_fraction",
    ]
    
    # Event-Features
    event_features = [
        "ann_total_event_count",
        "ann_arousal_count",
        "ann_total_apnea_count",
        "ann_respiratory_event_count",
        "ann_min_dist_to_next_event_sec",
        "ann_min_dist_to_prev_event_sec",
    ]
    
    # Alle Kandidaten
    candidates = stage_features + event_features
    
    # Nur Features die tatsächlich im DataFrame existieren
    for feat in candidates:
        if feat in df.columns:
            # Prüfe ob numerisch
            if pd.api.types.is_numeric_dtype(df[feat]):
                selected.append(feat)
    
    return selected


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Berechnet den gleitenden Mittelwert."""
    result = np.full(len(values), np.nan)
    half_w = window // 2
    
    for i in range(len(values)):
        start = max(0, i - half_w)
        end = min(len(values), i + half_w + 1)
        window_vals = values[start:end]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            result[i] = np.mean(valid)
    
    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Berechnet die gleitende Standardabweichung."""
    result = np.full(len(values), np.nan)
    half_w = window // 2
    
    for i in range(len(values)):
        start = max(0, i - half_w)
        end = min(len(values), i + half_w + 1)
        window_vals = values[start:end]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 1:
            result[i] = np.std(valid, ddof=1)
    
    return result


def _rolling_trend(values: np.ndarray, window: int) -> np.ndarray:
    """
    Berechnet den linearen Trend (Steigung) im gleitenden Fenster.
    Positive Werte = ansteigend, negative = abfallend.
    """
    result = np.full(len(values), np.nan)
    half_w = window // 2
    
    for i in range(len(values)):
        start = max(0, i - half_w)
        end = min(len(values), i + half_w + 1)
        window_vals = values[start:end]
        
        valid_mask = ~np.isnan(window_vals)
        if valid_mask.sum() >= 3:
            x = np.arange(valid_mask.sum())
            y = window_vals[valid_mask]
            try:
                coeffs = np.polyfit(x, y, 1)
                result[i] = coeffs[0]  # Steigung
            except Exception:
                pass
    
    return result


def _compute_run_lengths(sequence: np.ndarray) -> np.ndarray:
    """
    Berechnet für jedes Element die Länge des aktuellen Runs
    (Anzahl aufeinanderfolgender gleicher Werte bis zum aktuellen Punkt).
    """
    n = len(sequence)
    run_lengths = np.ones(n, dtype=int)
    
    for i in range(1, n):
        if sequence[i] == sequence[i - 1]:
            run_lengths[i] = run_lengths[i - 1] + 1
        else:
            run_lengths[i] = 1
    
    return run_lengths


def _time_since_event(event_flags: np.ndarray, segment_length_sec: float) -> np.ndarray:
    """
    Berechnet die Zeit seit dem letzten Event (in Sekunden).
    event_flags: 1 = Event, 0 = kein Event
    """
    n = len(event_flags)
    result = np.full(n, np.nan)
    
    last_event_idx = None
    for i in range(n):
        if event_flags[i] == 1:
            last_event_idx = i
            result[i] = 0.0
        elif last_event_idx is not None:
            result[i] = (i - last_event_idx) * segment_length_sec
    
    return result


def _time_to_event(event_flags: np.ndarray, segment_length_sec: float) -> np.ndarray:
    """
    Berechnet die Zeit bis zum nächsten Event (in Sekunden).
    event_flags: 1 = Event, 0 = kein Event
    """
    n = len(event_flags)
    result = np.full(n, np.nan)
    
    next_event_idx = None
    for i in range(n - 1, -1, -1):
        if event_flags[i] == 1:
            next_event_idx = i
            result[i] = 0.0
        elif next_event_idx is not None:
            result[i] = (next_event_idx - i) * segment_length_sec
    
    return result

