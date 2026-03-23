"""
01_preprocessing/preprocess_annotations.py
============================================
Verarbeitung der Schlaf-Annotationen:
- Schlafstadien-Extraktion (30s Epochen)
- Event-Extraktion (Arousal, Central Apnea, Obstructive Apnea, etc.)
- Mapping auf Segmente
- Berechnung von Event-Distanzen und Event-Dichte

Supports the PhysioNet Challenge 2026 numeric annotation encoding:
  Sleep stages (stage): 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable
  Arousals (arousal):   0=No event, 1=Arousal
  Respiratory (resp):   0=No Event, 1=Obstructive Apnea, 2=Central Apnea,
                         3=Mixed Apnea, 4=Obstructive Hypopnea,
                         5=Central Hypopnea, 6=Mixed Hypopnea, 7=RERA,
                         8=Apnea (unspecified), 9=Hypopnea (unspecified)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter

from config import (
    SLEEP_STAGE_ENCODING, SLEEP_EVENTS_OF_INTEREST,
    SEGMENT_LENGTH_SEC, SLEEP_EPOCH_SEC,
    EVENT_DISTANCE_MAX_SEC, EVENT_DENSITY_WINDOW_SEC
)


# ==============================================================================
# NUMERIC ENCODING MAPS (PhysioNet Challenge 2026)
# ==============================================================================

# Sleep stage numeric encoding -> label
NUMERIC_STAGE_MAP = {
    1: "N3",
    2: "N2",
    3: "N1",
    4: "REM",
    5: "W",
    9: None,  # Unavailable
}

# Respiratory event numeric encoding -> standardized event type
NUMERIC_RESP_EVENT_MAP = {
    0: None,                    # No Event
    1: "obstructive_apnea",     # Obstructive Apnea
    2: "central_apnea",         # Central Apnea
    3: "mixed_apnea",           # Mixed Apnea
    4: "obstructive_hypopnea",  # Obstructive Hypopnea
    5: "central_hypopnea",      # Central Hypopnea
    6: "mixed_hypopnea",        # Mixed Hypopnea
    7: "rera",                  # RERA
    8: "apnea_unspecified",     # Apnea (unspecified)
    9: "hypopnea_unspecified",  # Hypopnea (unspecified)
}

# Arousal event numeric encoding -> standardized event type
NUMERIC_AROUSAL_EVENT_MAP = {
    0: None,       # No event
    1: "arousal",  # Arousal
}

# Mapping from fine-grained respiratory types to the categories tracked
# in SLEEP_EVENTS_OF_INTEREST
RESP_TO_INTEREST_MAP = {
    "obstructive_apnea": "obstructive_apnea",
    "central_apnea": "central_apnea",
    "mixed_apnea": "mixed_apnea",
    "obstructive_hypopnea": "hypopnea",
    "central_hypopnea": "hypopnea",
    "mixed_hypopnea": "hypopnea",
    "rera": "rera",
    "apnea_unspecified": "obstructive_apnea",  # conservative default
    "hypopnea_unspecified": "hypopnea",
    "arousal": "arousal",
}


# ==============================================================================
# SCHLAFSTADIEN
# ==============================================================================

def parse_sleep_stages(
    annotations: Dict,
    total_duration_sec: float,
    logger=None
) -> Optional[pd.DataFrame]:
    """
    Extrahiert Schlafstadien aus den Annotationen.
    
    Parameters
    ----------
    annotations : Dict
        Rohe Annotationen (aus io_utils.load_annotations).
    total_duration_sec : float
        Gesamtdauer der Aufnahme in Sekunden.
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame mit Spalten: epoch_idx, start_sec, end_sec, stage, stage_label
    """
    if annotations is None:
        if logger:
            logger.warning("Keine Annotationen vorhanden.")
        return None
    
    stages_list = []
    
    # Versuche verschiedene Annotationsformate zu parsen
    for key, data in annotations.items():
        if isinstance(data, pd.DataFrame):
            stages_df = _parse_stages_from_dataframe(data, logger)
            if stages_df is not None and len(stages_df) > 0:
                stages_list.append(stages_df)
        else:
            # WFDB Annotation Objekt
            stages_df = _parse_stages_from_wfdb(data, logger)
            if stages_df is not None and len(stages_df) > 0:
                stages_list.append(stages_df)
    
    if not stages_list:
        if logger:
            logger.warning("Keine Schlafstadien in den Annotationen gefunden.")
        return None
    
    # Kombiniere alle gefundenen Stadien und entferne Duplikate
    combined = pd.concat(stages_list, ignore_index=True)
    combined = combined.drop_duplicates(subset=["epoch_idx"]).sort_values("epoch_idx")
    combined = combined.reset_index(drop=True)
    
    if logger:
        stage_counts = combined["stage_label"].value_counts().to_dict()
        logger.info(f"Schlafstadien extrahiert: {len(combined)} Epochen, "
                     f"Verteilung: {stage_counts}")
    
    return combined


def _parse_stages_from_dataframe(
    df: pd.DataFrame,
    logger=None
) -> Optional[pd.DataFrame]:
    """
    Parst Schlafstadien aus einem DataFrame (TSV/CSV).
    Unterstützt sowohl numerische Kodierung als auch Text-Annotationen.
    """
    # Spalten normalisieren
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Mögliche Spaltennamen für Onset/Start
    onset_col = None
    for col_name in ["onset", "start", "start_sec", "start_time", "time"]:
        if col_name in df.columns:
            onset_col = col_name
            break
    
    # Mögliche Spaltennamen für Duration
    duration_col = None
    for col_name in ["duration", "duration_sec", "dur", "length"]:
        if col_name in df.columns:
            duration_col = col_name
            break
    
    # Mögliche Spaltennamen für Stage/Annotation
    stage_col = None
    for col_name in ["annotation", "description", "stage", "sleep_stage",
                      "label", "event", "type", "value"]:
        if col_name in df.columns:
            stage_col = col_name
            break
    
    if onset_col is None or stage_col is None:
        if logger:
            logger.debug(f"DataFrame hat keine erkennbaren Schlafstadien-Spalten: "
                         f"{list(df.columns)}")
        return None
    
    stages_rows = []
    for _, row in df.iterrows():
        raw_value = row[stage_col]
        annotation = str(raw_value).strip()
        
        # Try numeric encoding first
        stage_label = _map_numeric_to_stage(raw_value)
        
        # Fall back to text-based mapping
        if stage_label is None:
            stage_label = _map_annotation_to_stage(annotation.lower())
        
        if stage_label is not None:
            onset = float(row[onset_col])
            duration = float(row[duration_col]) if duration_col and pd.notna(row.get(duration_col)) else SLEEP_EPOCH_SEC
            epoch_idx = int(onset / SLEEP_EPOCH_SEC)
            
            stages_rows.append({
                "epoch_idx": epoch_idx,
                "start_sec": onset,
                "end_sec": onset + duration,
                "duration_sec": duration,
                "stage_raw": annotation,
                "stage_label": stage_label,
                "stage_numeric": _stage_to_numeric(stage_label),
            })
    
    if not stages_rows:
        return None
    
    return pd.DataFrame(stages_rows)


def _parse_stages_from_wfdb(
    ann_obj,
    logger=None
) -> Optional[pd.DataFrame]:
    """
    Parst Schlafstadien aus einem WFDB Annotation-Objekt.
    """
    try:
        samples = ann_obj.sample
        symbols = ann_obj.symbol if hasattr(ann_obj, 'symbol') else []
        aux_notes = ann_obj.aux_note if hasattr(ann_obj, 'aux_note') else []
        fs = ann_obj.fs if hasattr(ann_obj, 'fs') else None
        
        if fs is None:
            if logger:
                logger.warning("WFDB Annotation hat keine Sampling-Rate.")
            return None
        
        stages_rows = []
        for i, sample in enumerate(samples):
            onset_sec = sample / fs
            
            # Versuche Stage aus Symbol oder aux_note zu lesen
            annotation_text = ""
            if i < len(aux_notes) and aux_notes[i]:
                annotation_text = str(aux_notes[i]).strip()
            elif i < len(symbols) and symbols[i]:
                annotation_text = str(symbols[i]).strip()
            
            # Try numeric encoding first
            stage_label = _map_numeric_to_stage(annotation_text)
            
            # Fall back to text-based mapping
            if stage_label is None:
                stage_label = _map_annotation_to_stage(annotation_text.lower())
            
            if stage_label is not None:
                epoch_idx = int(onset_sec / SLEEP_EPOCH_SEC)
                stages_rows.append({
                    "epoch_idx": epoch_idx,
                    "start_sec": onset_sec,
                    "end_sec": onset_sec + SLEEP_EPOCH_SEC,
                    "duration_sec": SLEEP_EPOCH_SEC,
                    "stage_raw": annotation_text,
                    "stage_label": stage_label,
                    "stage_numeric": _stage_to_numeric(stage_label),
                })
        
        if not stages_rows:
            return None
        
        return pd.DataFrame(stages_rows)
    
    except Exception as e:
        if logger:
            logger.error(f"Fehler beim Parsen der WFDB-Annotation: {e}")
        return None


def _map_numeric_to_stage(value) -> Optional[str]:
    """
    Maps a numeric stage code to a standardized sleep stage label.
    
    PhysioNet Challenge 2026 encoding:
        1 = N3, 2 = N2, 3 = N1, 4 = REM, 5 = Wake, 9 = Unavailable
    
    Parameters
    ----------
    value : int, float, or str
        The raw value from the annotation.
    
    Returns
    -------
    str or None
        "W", "N1", "N2", "N3", "REM", or None if not recognized.
    """
    try:
        numeric_val = int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None
    
    return NUMERIC_STAGE_MAP.get(numeric_val, None)


def _map_annotation_to_stage(annotation: str) -> Optional[str]:
    """
    Mappt eine Text-Annotation auf ein standardisiertes Schlafstadium.
    
    Returns
    -------
    str or None
        "W", "N1", "N2", "N3", "REM", oder None wenn kein Stadium erkannt.
    """
    ann = annotation.lower().strip().replace(" ", "")
    
    # First try numeric encoding (handles cases where numeric values
    # are passed as strings)
    numeric_result = _map_numeric_to_stage(ann)
    if numeric_result is not None:
        return numeric_result
    
    # Direkte Mappings
    direct_map = {
        "w": "W", "wake": "W", "wakefulness": "W",
        "sleepstagew": "W", "stage-w": "W", "stagew": "W",
        "n1": "N1", "nrem1": "N1", "stage1": "N1",
        "sleepstagen1": "N1", "stage-n1": "N1", "stagen1": "N1",
        "s1": "N1", "sleepstage1": "N1",
        "n2": "N2", "nrem2": "N2", "stage2": "N2",
        "sleepstagen2": "N2", "stage-n2": "N2", "stagen2": "N2",
        "s2": "N2", "sleepstage2": "N2",
        "n3": "N3", "nrem3": "N3", "stage3": "N3", "stage4": "N3",
        "sleepstagen3": "N3", "stage-n3": "N3", "stagen3": "N3",
        "s3": "N3", "s4": "N3", "sws": "N3",
        "sleepstage3": "N3", "sleepstage4": "N3",
        "rem": "REM", "r": "REM", "stager": "REM",
        "sleepstagerem": "REM", "stage-rem": "REM", "stagerem": "REM",
        "rapideyemovement": "REM",
    }
    
    # Versuche numerisches Encoding (aus config SLEEP_STAGE_ENCODING)
    for numeric_val, label in SLEEP_STAGE_ENCODING.items():
        if ann == str(numeric_val):
            return label if label != "Unknown" else None
    
    # Direkte Suche
    if ann in direct_map:
        return direct_map[ann]
    
    # Teilstring-Suche
    for key, value in direct_map.items():
        if key in ann:
            return value
    
    return None


def _stage_to_numeric(stage_label: str) -> int:
    """Konvertiert Schlafstadium-Label in numerischen Wert."""
    mapping = {"W": 5, "N1": 3, "N2": 2, "N3": 1, "REM": 4, "Unknown": 0}
    return mapping.get(stage_label, 0)


# ==============================================================================
# SCHLAF-EVENTS (Arousal, Apnea, etc.)
# ==============================================================================

def parse_sleep_events(
    annotations: Dict,
    total_duration_sec: float,
    logger=None
) -> Optional[pd.DataFrame]:
    """
    Extrahiert Schlaf-Events (Arousal, Apnea, Hypopnea, etc.).
    
    Supports both numeric encoding (PhysioNet Challenge 2026) and
    text-based annotations.
    
    Parameters
    ----------
    annotations : Dict
        Rohe Annotationen.
    total_duration_sec : float
        Gesamtdauer der Aufnahme in Sekunden.
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame mit Spalten: event_type, start_sec, end_sec, duration_sec
    """
    if annotations is None:
        if logger:
            logger.warning("Keine Annotationen für Event-Extraktion vorhanden.")
        return None
    
    events_list = []
    
    for key, data in annotations.items():
        # Determine annotation channel type from key name
        ann_channel = _detect_annotation_channel(key)
        
        if isinstance(data, pd.DataFrame):
            events_df = _parse_events_from_dataframe(data, ann_channel, logger)
            if events_df is not None and len(events_df) > 0:
                events_list.append(events_df)
        else:
            events_df = _parse_events_from_wfdb(data, ann_channel, logger)
            if events_df is not None and len(events_df) > 0:
                events_list.append(events_df)
    
    if not events_list:
        if logger:
            logger.warning("Keine Schlaf-Events in den Annotationen gefunden.")
        return None
    
    combined = pd.concat(events_list, ignore_index=True)
    combined = combined.sort_values("start_sec").reset_index(drop=True)
    
    # Entferne Duplikate (gleicher Typ, gleiche Startzeit)
    combined = combined.drop_duplicates(
        subset=["event_type", "start_sec"], keep="first"
    ).reset_index(drop=True)
    
    if logger:
        event_counts = combined["event_type"].value_counts().to_dict()
        logger.info(f"Schlaf-Events extrahiert: {len(combined)} Events, "
                     f"Verteilung: {event_counts}")
    
    return combined


def _detect_annotation_channel(key: str) -> Optional[str]:
    """
    Detect the annotation channel type from the key/filename.
    
    Returns
    -------
    str or None
        "resp", "arousal", "stage", or None if unknown.
    """
    key_lower = key.lower()
    if "resp" in key_lower:
        return "resp"
    elif "arousal" in key_lower or "arsl" in key_lower:
        return "arousal"
    elif "stage" in key_lower or "sleep" in key_lower or "hypno" in key_lower:
        return "stage"
    return None


def _parse_events_from_dataframe(
    df: pd.DataFrame,
    ann_channel: Optional[str] = None,
    logger=None
) -> Optional[pd.DataFrame]:
    """
    Parst Events aus einem DataFrame.
    Supports numeric encoding for resp and arousal channels.
    """
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Finde relevante Spalten
    onset_col = None
    for col_name in ["onset", "start", "start_sec", "start_time", "time"]:
        if col_name in df.columns:
            onset_col = col_name
            break
    
    duration_col = None
    for col_name in ["duration", "duration_sec", "dur", "length"]:
        if col_name in df.columns:
            duration_col = col_name
            break
    
    event_col = None
    for col_name in ["annotation", "description", "event", "type",
                      "label", "event_type", "value"]:
        if col_name in df.columns:
            event_col = col_name
            break
    
    if onset_col is None or event_col is None:
        return None
    
    events_rows = []
    for _, row in df.iterrows():
        raw_value = row[event_col]
        annotation = str(raw_value).strip()
        
        # Try numeric encoding first based on channel type
        event_type = _map_numeric_to_event(raw_value, ann_channel)
        
        # Fall back to text-based mapping
        if event_type is None:
            event_type = _map_annotation_to_event(annotation.lower())
        
        if event_type is not None:
            onset = float(row[onset_col])
            duration = float(row[duration_col]) if duration_col and pd.notna(row.get(duration_col)) else 0.0
            
            events_rows.append({
                "event_type": event_type,
                "start_sec": onset,
                "end_sec": onset + duration,
                "duration_sec": duration,
                "raw_annotation": annotation,
            })
    
    if not events_rows:
        return None
    
    return pd.DataFrame(events_rows)


def _parse_events_from_wfdb(
    ann_obj,
    ann_channel: Optional[str] = None,
    logger=None
) -> Optional[pd.DataFrame]:
    """Parst Events aus einem WFDB Annotation-Objekt."""
    try:
        samples = ann_obj.sample
        aux_notes = ann_obj.aux_note if hasattr(ann_obj, 'aux_note') else []
        symbols = ann_obj.symbol if hasattr(ann_obj, 'symbol') else []
        fs = ann_obj.fs if hasattr(ann_obj, 'fs') else None
        
        if fs is None:
            return None
        
        events_rows = []
        i = 0
        while i < len(samples):
            onset_sec = samples[i] / fs
            
            annotation_text = ""
            if i < len(aux_notes) and aux_notes[i]:
                annotation_text = str(aux_notes[i]).strip()
            elif i < len(symbols) and symbols[i]:
                annotation_text = str(symbols[i]).strip()
            
            # Try numeric encoding first
            event_type = _map_numeric_to_event(annotation_text, ann_channel)
            
            # Fall back to text-based mapping
            if event_type is None:
                event_type = _map_annotation_to_event(annotation_text.lower())
            
            if event_type is not None:
                # Versuche Dauer zu bestimmen
                duration = _estimate_event_duration(
                    event_type, i, samples, aux_notes, fs
                )
                
                events_rows.append({
                    "event_type": event_type,
                    "start_sec": onset_sec,
                    "end_sec": onset_sec + duration,
                    "duration_sec": duration,
                    "raw_annotation": annotation_text,
                })
            
            i += 1
        
        if not events_rows:
            return None
        
        return pd.DataFrame(events_rows)
    
    except Exception as e:
        if logger:
            logger.error(f"Fehler beim Parsen der WFDB-Events: {e}")
        return None


def _map_numeric_to_event(value, ann_channel: Optional[str] = None) -> Optional[str]:
    """
    Maps a numeric event code to a standardized event type string.
    
    Uses the annotation channel type to determine which encoding map to use.
    
    PhysioNet Challenge 2026 encoding:
        Respiratory (resp):
            0=No Event, 1=Obstructive Apnea, 2=Central Apnea,
            3=Mixed Apnea, 4=Obstructive Hypopnea, 5=Central Hypopnea,
            6=Mixed Hypopnea, 7=RERA, 8=Apnea (unspecified),
            9=Hypopnea (unspecified)
        Arousal (arousal):
            0=No event, 1=Arousal
    
    Parameters
    ----------
    value : int, float, or str
        The raw annotation value.
    ann_channel : str or None
        The annotation channel type: "resp", "arousal", or None.
    
    Returns
    -------
    str or None
        Standardized event type, mapped through RESP_TO_INTEREST_MAP,
        or None if no event / not recognized.
    """
    try:
        numeric_val = int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None
    
    raw_event_type = None
    
    if ann_channel == "resp":
        raw_event_type = NUMERIC_RESP_EVENT_MAP.get(numeric_val, None)
    elif ann_channel == "arousal":
        raw_event_type = NUMERIC_AROUSAL_EVENT_MAP.get(numeric_val, None)
    else:
        # Unknown channel: try resp first (more codes), then arousal
        raw_event_type = NUMERIC_RESP_EVENT_MAP.get(numeric_val, None)
        if raw_event_type is None:
            raw_event_type = NUMERIC_AROUSAL_EVENT_MAP.get(numeric_val, None)
    
    if raw_event_type is None:
        return None
    
    # Map fine-grained type to the categories in SLEEP_EVENTS_OF_INTEREST
    mapped = RESP_TO_INTEREST_MAP.get(raw_event_type, raw_event_type)
    
    return mapped


def _map_annotation_to_event(annotation: str) -> Optional[str]:
    """
    Mappt eine Annotation auf einen standardisierten Event-Typ.
    
    Returns
    -------
    str or None
        "arousal", "central_apnea", "obstructive_apnea", "mixed_apnea",
        "hypopnea", oder None.
    """
    ann = annotation.lower().strip()
    
    # Skip empty or "no event" annotations
    if not ann or ann in ("0", "0.0", "no event", "none", ""):
        return None
    
    # Try numeric encoding (handles string-encoded numbers without channel info)
    try:
        numeric_val = int(float(ann))
        # Try resp map first (it has more distinct codes)
        raw_type = NUMERIC_RESP_EVENT_MAP.get(numeric_val, None)
        if raw_type is not None:
            return RESP_TO_INTEREST_MAP.get(raw_type, raw_type)
        # Try arousal map
        raw_type = NUMERIC_AROUSAL_EVENT_MAP.get(numeric_val, None)
        if raw_type is not None:
            return RESP_TO_INTEREST_MAP.get(raw_type, raw_type)
    except (ValueError, TypeError):
        pass
    
    # --- Text-based matching (legacy / other datasets) ---
    
    # Arousal
    arousal_keywords = [
        "arousal", "arsl", "ar ", "spontaneous arousal",
        "respiratory arousal", "limb movement arousal"
    ]
    for kw in arousal_keywords:
        if kw in ann:
            return "arousal"
    
    # Central Apnea
    central_keywords = [
        "central apnea", "central_apnea", "centralapnea",
        "ca ", "central ap", "csa"
    ]
    for kw in central_keywords:
        if kw in ann:
            return "central_apnea"
    
    # Obstructive Apnea
    obstructive_keywords = [
        "obstructive apnea", "obstructive_apnea", "obstructiveapnea",
        "oa ", "obst ap", "osa"
    ]
    for kw in obstructive_keywords:
        if kw in ann:
            return "obstructive_apnea"
    
    # Mixed Apnea
    mixed_keywords = [
        "mixed apnea", "mixed_apnea", "mixedapnea", "ma "
    ]
    for kw in mixed_keywords:
        if kw in ann:
            return "mixed_apnea"
    
    # Hypopnea
    hypopnea_keywords = [
        "hypopnea", "hyp ", "hypop", "hypo"
    ]
    for kw in hypopnea_keywords:
        if kw in ann:
            return "hypopnea"
    
    return None


def _estimate_event_duration(
    event_type: str,
    current_idx: int,
    samples: np.ndarray,
    aux_notes: list,
    fs: float
) -> float:
    """
    Schätzt die Dauer eines Events.
    Standardwerte basierend auf typischen Dauern.
    """
    # Typische Dauern als Fallback
    default_durations = {
        "arousal": 3.0,           # 3-15 Sekunden
        "central_apnea": 20.0,    # ≥10 Sekunden
        "obstructive_apnea": 20.0,
        "mixed_apnea": 20.0,
        "hypopnea": 15.0,         # ≥10 Sekunden
    }
    
    # Versuche End-Marker zu finden
    if current_idx + 1 < len(samples):
        next_onset = samples[current_idx + 1] / fs
        current_onset = samples[current_idx] / fs
        gap = next_onset - current_onset
        
        # Wenn der nächste Eintrag ein End-Marker sein könnte
        if current_idx + 1 < len(aux_notes):
            next_note = str(aux_notes[current_idx + 1]).lower().strip()
            if "end" in next_note or ")" in next_note:
                return gap
        
        # For numeric annotations: if the next sample has value 0 (no event),
        # the gap is the event duration
        if current_idx + 1 < len(aux_notes):
            try:
                next_val = int(float(str(aux_notes[current_idx + 1]).strip()))
                if next_val == 0:
                    return gap
            except (ValueError, TypeError):
                pass
    
    return default_durations.get(event_type, 10.0)


# ==============================================================================
# MAPPING AUF SEGMENTE
# ==============================================================================

def map_stages_to_segments(
    stages_df: Optional[pd.DataFrame],
    total_duration_sec: float,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    logger=None
) -> pd.DataFrame:
    """
    Mappt Schlafstadien auf Segmente und bestimmt das dominante Stadium.
    
    Parameters
    ----------
    stages_df : pd.DataFrame or None
        Schlafstadien-DataFrame.
    total_duration_sec : float
        Gesamtdauer in Sekunden.
    segment_length_sec : float
        Segmentlänge in Sekunden.
    
    Returns
    -------
        pd.DataFrame
        DataFrame mit Spalten: segment_idx, start_sec, end_sec,
        dominant_stage, dominant_stage_numeric, stage_stability,
        n1_fraction, n2_fraction, n3_fraction, rem_fraction, w_fraction
    """
    n_segments = int(np.ceil(total_duration_sec / segment_length_sec))
    
    segment_records = []
    
    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_length_sec
        seg_end = min(seg_start + segment_length_sec, total_duration_sec)
        
        record = {
            "segment_idx": seg_idx,
            "start_sec": seg_start,
            "end_sec": seg_end,
        }
        
        if stages_df is None or len(stages_df) == 0:
            record.update({
                "dominant_stage": "Unknown",
                "dominant_stage_numeric": 0,
                "stage_stability": 0.0,
                "n1_fraction": 0.0,
                "n2_fraction": 0.0,
                "n3_fraction": 0.0,
                "rem_fraction": 0.0,
                "w_fraction": 0.0,
                "unknown_fraction": 1.0,
                "n_stage_transitions": 0,
            })
            segment_records.append(record)
            continue
        
        # Finde alle Epochen die in dieses Segment fallen
        overlapping = stages_df[
            (stages_df["start_sec"] < seg_end) &
            (stages_df["end_sec"] > seg_start)
        ]
        
        if len(overlapping) == 0:
            record.update({
                "dominant_stage": "Unknown",
                "dominant_stage_numeric": 0,
                "stage_stability": 0.0,
                "n1_fraction": 0.0,
                "n2_fraction": 0.0,
                "n3_fraction": 0.0,
                "rem_fraction": 0.0,
                "w_fraction": 0.0,
                "unknown_fraction": 1.0,
                "n_stage_transitions": 0,
            })
            segment_records.append(record)
            continue
        
        # Berechne gewichtete Anteile (nach Überlappungsdauer)
        stage_durations = Counter()
        stages_sequence = []
        
        for _, epoch in overlapping.iterrows():
            overlap_start = max(epoch["start_sec"], seg_start)
            overlap_end = min(epoch["end_sec"], seg_end)
            overlap_duration = overlap_end - overlap_start
            
            if overlap_duration > 0:
                stage_durations[epoch["stage_label"]] += overlap_duration
                stages_sequence.append(epoch["stage_label"])
        
        total_annotated = sum(stage_durations.values())
        segment_duration = seg_end - seg_start
        
        # Fraktionen berechnen
        fractions = {}
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            frac = stage_durations.get(stage, 0) / segment_duration if segment_duration > 0 else 0
            fractions[f"{stage.lower()}_fraction"] = frac
        
        fractions["unknown_fraction"] = max(0, 1.0 - sum(fractions.values()))
        
        # Dominantes Stadium
        if stage_durations:
            dominant_stage = max(stage_durations, key=stage_durations.get)
        else:
            dominant_stage = "Unknown"
        
        # Stabilität: Anteil des dominanten Stadiums
        stability = (
            stage_durations.get(dominant_stage, 0) / total_annotated
            if total_annotated > 0 else 0
        )
        
        # Anzahl Stadien-Übergänge
        n_transitions = sum(
            1 for i in range(1, len(stages_sequence))
            if stages_sequence[i] != stages_sequence[i - 1]
        )
        
        record.update({
            "dominant_stage": dominant_stage,
            "dominant_stage_numeric": _stage_to_numeric(dominant_stage),
            "stage_stability": stability,
            "n_stage_transitions": n_transitions,
            **fractions,
        })
        
        segment_records.append(record)
    
    return pd.DataFrame(segment_records)


def map_events_to_segments(
    events_df: Optional[pd.DataFrame],
    total_duration_sec: float,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    logger=None
) -> pd.DataFrame:
    """
    Mappt Schlaf-Events auf Segmente.
    Berechnet pro Segment:
    - Ob ein Event stattgefunden hat (Flag)
    - Anzahl Events pro Typ
    - Distanz zum nächsten und vorherigen Event (bidirektional)
    - Event-Dichte
    
    Parameters
    ----------
    events_df : pd.DataFrame or None
        Events-DataFrame.
    total_duration_sec : float
        Gesamtdauer in Sekunden.
    segment_length_sec : float
        Segmentlänge in Sekunden.
    
    Returns
    -------
    pd.DataFrame
        Feature-DataFrame pro Segment.
    """
    n_segments = int(np.ceil(total_duration_sec / segment_length_sec))
    
    # Initialisiere leere Records
    segment_records = []
    
    # Event-Typen die wir tracken
    event_types = SLEEP_EVENTS_OF_INTEREST
    
    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_length_sec
        seg_end = min(seg_start + segment_length_sec, total_duration_sec)
        seg_center = (seg_start + seg_end) / 2.0
        
        record = {
            "segment_idx": seg_idx,
            "start_sec": seg_start,
            "end_sec": seg_end,
        }
        
        for event_type in event_types:
            prefix = event_type
            
            if events_df is None or len(events_df) == 0:
                record[f"{prefix}_present"] = False
                record[f"{prefix}_count"] = 0
                record[f"{prefix}_total_duration_sec"] = 0.0
                record[f"{prefix}_dist_to_next_sec"] = EVENT_DISTANCE_MAX_SEC
                record[f"{prefix}_dist_to_prev_sec"] = EVENT_DISTANCE_MAX_SEC
                record[f"{prefix}_density"] = 0.0
                continue
            
            # Events dieses Typs
            type_events = events_df[events_df["event_type"] == event_type]
            
            if len(type_events) == 0:
                record[f"{prefix}_present"] = False
                record[f"{prefix}_count"] = 0
                record[f"{prefix}_total_duration_sec"] = 0.0
                record[f"{prefix}_dist_to_next_sec"] = EVENT_DISTANCE_MAX_SEC
                record[f"{prefix}_dist_to_prev_sec"] = EVENT_DISTANCE_MAX_SEC
                record[f"{prefix}_density"] = 0.0
                continue
            
            # Events die in dieses Segment fallen
            overlapping = type_events[
                (type_events["start_sec"] < seg_end) &
                (type_events["end_sec"] > seg_start)
            ]
            
            record[f"{prefix}_present"] = len(overlapping) > 0
            record[f"{prefix}_count"] = len(overlapping)
            
            # Gesamtdauer der Events im Segment
            total_event_dur = 0.0
            for _, ev in overlapping.iterrows():
                ov_start = max(ev["start_sec"], seg_start)
                ov_end = min(ev["end_sec"], seg_end)
                total_event_dur += max(0, ov_end - ov_start)
            record[f"{prefix}_total_duration_sec"] = total_event_dur
            
            # --- Distanz zum nächsten Event (vorwärts) ---
            future_events = type_events[type_events["start_sec"] >= seg_end]
            if len(future_events) > 0:
                next_event_start = future_events["start_sec"].min()
                dist_next = next_event_start - seg_end
            elif len(overlapping) > 0:
                dist_next = 0.0  # Event ist im Segment
            else:
                dist_next = EVENT_DISTANCE_MAX_SEC
            record[f"{prefix}_dist_to_next_sec"] = min(dist_next, EVENT_DISTANCE_MAX_SEC)
            
            # --- Distanz zum vorherigen Event (rückwärts) ---
            past_events = type_events[type_events["end_sec"] <= seg_start]
            if len(past_events) > 0:
                prev_event_end = past_events["end_sec"].max()
                dist_prev = seg_start - prev_event_end
            elif len(overlapping) > 0:
                dist_prev = 0.0
            else:
                dist_prev = EVENT_DISTANCE_MAX_SEC
            record[f"{prefix}_dist_to_prev_sec"] = min(dist_prev, EVENT_DISTANCE_MAX_SEC)
            
            # --- Event-Dichte (Events pro EVENT_DENSITY_WINDOW_SEC) ---
            density_start = max(0, seg_center - EVENT_DENSITY_WINDOW_SEC / 2)
            density_end = min(total_duration_sec, seg_center + EVENT_DENSITY_WINDOW_SEC / 2)
            density_events = type_events[
                (type_events["start_sec"] < density_end) &
                (type_events["end_sec"] > density_start)
            ]
            density_window_actual = density_end - density_start
            record[f"{prefix}_density"] = (
                len(density_events) / (density_window_actual / 60.0)
                if density_window_actual > 0 else 0.0
            )  # Events pro Minute
        
        # --- Aggregierte Event-Features (über alle Typen) ---
        any_event_present = any(
            record.get(f"{et}_present", False) for et in event_types
        )
        total_event_count = sum(
            record.get(f"{et}_count", 0) for et in event_types
        )
        
        # Minimale Distanz zum nächsten Event (über alle Typen)
        min_dist_next = min(
            record.get(f"{et}_dist_to_next_sec", EVENT_DISTANCE_MAX_SEC)
            for et in event_types
        )
        min_dist_prev = min(
            record.get(f"{et}_dist_to_prev_sec", EVENT_DISTANCE_MAX_SEC)
            for et in event_types
        )
        
        # Aggregierte Apnea-Features (alle Apnea-Typen zusammen)
        apnea_types = ["central_apnea", "obstructive_apnea", "mixed_apnea"]
        any_apnea_present = any(
            record.get(f"{at}_present", False) for at in apnea_types
        )
        total_apnea_count = sum(
            record.get(f"{at}_count", 0) for at in apnea_types
        )
        total_apnea_duration = sum(
            record.get(f"{at}_total_duration_sec", 0.0) for at in apnea_types
        )
        
        record["any_event_present"] = any_event_present
        record["total_event_count"] = total_event_count
        record["min_dist_to_next_event_sec"] = min_dist_next
        record["min_dist_to_prev_event_sec"] = min_dist_prev
        record["any_apnea_present"] = any_apnea_present
        record["total_apnea_count"] = total_apnea_count
        record["total_apnea_duration_sec"] = total_apnea_duration
        
        segment_records.append(record)
    
    result_df = pd.DataFrame(segment_records)
    
    if logger:
        for et in event_types:
            n_segments_with = result_df[f"{et}_present"].sum()
            logger.info(f"Event '{et}': {n_segments_with}/{n_segments} Segmente betroffen")
        logger.info(f"Gesamt: {result_df['any_event_present'].sum()}/{n_segments} "
                     f"Segmente mit mindestens einem Event")
    
    return result_df


# ==============================================================================
# ZUSAMMENFASSUNG: ALLE ANNOTATIONEN FÜR EINEN PATIENTEN
# ==============================================================================

def process_all_annotations(
    annotations: Dict,
    total_duration_sec: float,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    logger=None
) -> Dict:
    """
    Verarbeitet alle Annotationen für einen Patienten.
    Kombiniert Schlafstadien und Events in segment-basierte Features.
    
    Parameters
    ----------
    annotations : Dict
        Rohe Annotationen.
    total_duration_sec : float
        Gesamtdauer der Aufnahme in Sekunden.
    segment_length_sec : float
        Segmentlänge in Sekunden.
    
    Returns
    -------
    Dict mit:
        - stages_raw: Rohe Schlafstadien (30s Epochen)
        - events_raw: Rohe Events
        - stages_per_segment: Schlafstadien pro Segment
        - events_per_segment: Events pro Segment
        - sleep_summary: Zusammenfassung der Schlafarchitektur
    """
    result = {
        "stages_raw": None,
        "events_raw": None,
        "stages_per_segment": None,
        "events_per_segment": None,
        "sleep_summary": {},
    }
    
    if logger:
        logger.info("=" * 50)
        logger.info("Starte Annotation-Verarbeitung")
        logger.info("=" * 50)
    
    # --- Schlafstadien ---
    stages_df = parse_sleep_stages(annotations, total_duration_sec, logger)
    result["stages_raw"] = stages_df
    
    stages_per_segment = map_stages_to_segments(
        stages_df, total_duration_sec, segment_length_sec, logger
    )
    result["stages_per_segment"] = stages_per_segment
    
    # --- Events ---
    events_df = parse_sleep_events(annotations, total_duration_sec, logger)
    result["events_raw"] = events_df
    
    events_per_segment = map_events_to_segments(
        events_df, total_duration_sec, segment_length_sec, logger
    )
    result["events_per_segment"] = events_per_segment
    
    # --- Schlafarchitektur-Zusammenfassung ---
    result["sleep_summary"] = compute_sleep_summary(
        stages_df, events_df, total_duration_sec, logger
    )
    
    return result


def compute_sleep_summary(
    stages_df: Optional[pd.DataFrame],
    events_df: Optional[pd.DataFrame],
    total_duration_sec: float,
    logger=None
) -> Dict:
    """
    Berechnet eine Zusammenfassung der Schlafarchitektur.
    Diese globalen Features können als Patient-Level-Features verwendet werden.
    
    Returns
    -------
    Dict mit Schlafarchitektur-Metriken.
    """
    summary = {}
    
    # --- Schlafstadien-Metriken ---
    if stages_df is not None and len(stages_df) > 0:
        total_scored_sec = stages_df["duration_sec"].sum()
        
        # Total Sleep Time (TST): Alles außer Wake
        sleep_stages = stages_df[stages_df["stage_label"] != "W"]
        tst_sec = sleep_stages["duration_sec"].sum() if len(sleep_stages) > 0 else 0
        
        summary["total_recording_time_min"] = total_duration_sec / 60.0
        summary["total_scored_time_min"] = total_scored_sec / 60.0
        summary["total_sleep_time_min"] = tst_sec / 60.0
        summary["sleep_efficiency"] = tst_sec / total_scored_sec if total_scored_sec > 0 else 0
        
        # Anteile der Schlafstadien an TST
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            stage_data = stages_df[stages_df["stage_label"] == stage]
            stage_sec = stage_data["duration_sec"].sum() if len(stage_data) > 0 else 0
            summary[f"{stage.lower()}_time_min"] = stage_sec / 60.0
            summary[f"{stage.lower()}_pct_tst"] = (
                stage_sec / tst_sec * 100 if tst_sec > 0 else 0
            )
        
        # Schlaflatenz: Zeit bis zum ersten Schlafstadium
        first_sleep = stages_df[stages_df["stage_label"] != "W"]
        if len(first_sleep) > 0:
            summary["sleep_onset_latency_min"] = first_sleep.iloc[0]["start_sec"] / 60.0
        else:
            summary["sleep_onset_latency_min"] = np.nan
        
        # REM-Latenz: Zeit vom Schlafbeginn bis zum ersten REM
        first_rem = stages_df[stages_df["stage_label"] == "REM"]
        if len(first_rem) > 0 and len(first_sleep) > 0:
            sleep_onset = first_sleep.iloc[0]["start_sec"]
            rem_onset = first_rem.iloc[0]["start_sec"]
            summary["rem_latency_min"] = (rem_onset - sleep_onset) / 60.0
        else:
            summary["rem_latency_min"] = np.nan
        
        # Anzahl Stadien-Wechsel
        stage_sequence = stages_df.sort_values("start_sec")["stage_label"].values
        n_transitions = sum(
            1 for i in range(1, len(stage_sequence))
            if stage_sequence[i] != stage_sequence[i - 1]
        )
        summary["n_stage_transitions"] = n_transitions
        summary["stage_transition_index"] = (
            n_transitions / (total_scored_sec / 3600)
            if total_scored_sec > 0 else 0
        )  # Übergänge pro Stunde
        
        # Wake After Sleep Onset (WASO)
        if len(first_sleep) > 0:
            sleep_onset_sec = first_sleep.iloc[0]["start_sec"]
            wake_after_onset = stages_df[
                (stages_df["stage_label"] == "W") &
                (stages_df["start_sec"] > sleep_onset_sec)
            ]
            waso_sec = wake_after_onset["duration_sec"].sum() if len(wake_after_onset) > 0 else 0
            summary["waso_min"] = waso_sec / 60.0
        else:
            summary["waso_min"] = np.nan
    
    else:
        summary["total_recording_time_min"] = total_duration_sec / 60.0
        summary["total_sleep_time_min"] = np.nan
        summary["sleep_efficiency"] = np.nan
    
    # --- Event-Metriken ---
    if events_df is not None and len(events_df) > 0:
        tst_hours = summary.get("total_sleep_time_min", 0) / 60.0
        
        for event_type in SLEEP_EVENTS_OF_INTEREST:
            type_events = events_df[events_df["event_type"] == event_type]
            count = len(type_events)
            summary[f"{event_type}_count"] = count
            summary[f"{event_type}_index"] = (
                count / tst_hours if tst_hours > 0 else 0
            )  # Events pro Stunde Schlaf
            
            if count > 0:
                summary[f"{event_type}_mean_duration_sec"] = type_events["duration_sec"].mean()
                summary[f"{event_type}_max_duration_sec"] = type_events["duration_sec"].max()
            else:
                summary[f"{event_type}_mean_duration_sec"] = 0.0
                summary[f"{event_type}_max_duration_sec"] = 0.0
        
        # AHI (Apnea-Hypopnea Index)
        apnea_types = ["central_apnea", "obstructive_apnea", "mixed_apnea", "hypopnea"]
        total_respiratory_events = sum(
            len(events_df[events_df["event_type"] == at]) for at in apnea_types
        )
        summary["ahi"] = (
            total_respiratory_events / tst_hours if tst_hours > 0 else 0
        )
        
        # Arousal Index
        arousal_count = len(events_df[events_df["event_type"] == "arousal"])
        summary["arousal_index"] = (
            arousal_count / tst_hours if tst_hours > 0 else 0
        )
    
    else:
        for event_type in SLEEP_EVENTS_OF_INTEREST:
            summary[f"{event_type}_count"] = 0
            summary[f"{event_type}_index"] = 0.0
        summary["ahi"] = np.nan
        summary["arousal_index"] = np.nan
    
    if logger:
        logger.info("Schlafarchitektur-Zusammenfassung:")
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
    
    return summary

