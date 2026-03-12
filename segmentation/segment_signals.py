"""
02_segmentation/segment_signals.py
====================================
Segmentierung aller Signale und Annotationen in einheitliche Zeitfenster.

Dieses Modul:
- Segmentiert ECG, EEG und Respirationssignale in konfigurierbare Zeitfenster
- Synchronisiert die Segmentierung über alle Signaltypen
- Verwaltet Segment-Metadaten (Zeitstempel, Qualität, Schlafphase)
- Unterstützt nachträgliche Änderung der Segmentlänge (Re-Segmentierung)

Die Segmentlänge wird zentral in config.py definiert und kann jederzeit
geändert werden, ohne die Rohdaten erneut verarbeiten zu müssen.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import (
    SEGMENT_LENGTH_SEC, SEGMENT_OVERLAP_SEC,
    SLEEP_EPOCH_SEC, ECG_SQI_THRESHOLD, EEG_SQI_THRESHOLD
)


# ==============================================================================
# DATENKLASSEN
# ==============================================================================

@dataclass
class SignalSegment:
    """
    Repräsentiert ein einzelnes Segment eines Signals.
    """
    segment_idx: int
    start_sec: float
    end_sec: float
    start_sample: int
    end_sample: int
    data: np.ndarray
    fs: float
    channel_name: str
    signal_type: str  # "ecg", "eeg", "resp"
    sqi: float = 0.0
    quality_ok: bool = False
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
    
    @property
    def n_samples(self) -> int:
        return len(self.data)


@dataclass
class SegmentedRecording:
    """
    Container für alle segmentierten Signale eines Patienten.
    Hält ECG, EEG, Respirations-Segmente und Annotationen synchron.
    """
    patient_id: str
    total_duration_sec: float
    segment_length_sec: float
    segment_overlap_sec: float
    n_segments: int
    
    # Segmentierte Signale
    ecg_segments: List[Optional[SignalSegment]] = field(default_factory=list)
    eeg_segments: Dict[str, List[Optional[SignalSegment]]] = field(default_factory=dict)
    resp_segments: Dict[str, List[Optional[SignalSegment]]] = field(default_factory=dict)
    
    # Annotationen pro Segment
    stages_per_segment: Optional[pd.DataFrame] = None
    events_per_segment: Optional[pd.DataFrame] = None
    
    # Segment-Metadaten
    segment_metadata: Optional[pd.DataFrame] = None


# ==============================================================================
# KERN-SEGMENTIERUNG
# ==============================================================================

def compute_segment_boundaries(
    total_duration_sec: float,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    overlap_sec: float = SEGMENT_OVERLAP_SEC,
) -> pd.DataFrame:
    """
    Berechnet die Segment-Grenzen für eine gegebene Aufnahmedauer.
    
    Parameters
    ----------
    total_duration_sec : float
        Gesamtdauer der Aufnahme in Sekunden.
    segment_length_sec : float
        Segmentlänge in Sekunden.
    overlap_sec : float
        Überlappung zwischen Segmenten in Sekunden.
    
    Returns
    -------
    pd.DataFrame
        DataFrame mit Spalten: segment_idx, start_sec, end_sec, duration_sec
    """
    step_sec = segment_length_sec - overlap_sec
    if step_sec <= 0:
        raise ValueError(
            f"Segment-Schritt muss positiv sein. "
            f"segment_length={segment_length_sec}, overlap={overlap_sec}"
        )
    
    boundaries = []
    seg_idx = 0
    start = 0.0
    
    while start < total_duration_sec:
        end = min(start + segment_length_sec, total_duration_sec)
        
        # Mindestlänge: 50% der Segmentlänge (letztes Segment)
        if (end - start) < segment_length_sec * 0.5 and seg_idx > 0:
            break
        
        boundaries.append({
            "segment_idx": seg_idx,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "is_complete": (end - start) >= segment_length_sec * 0.99,
        })
        
        start += step_sec
        seg_idx += 1
    
    return pd.DataFrame(boundaries)


def segment_signal(
    signal: np.ndarray,
    fs: float,
    boundaries: pd.DataFrame,
    channel_name: str = "unknown",
    signal_type: str = "unknown",
    sqi_values: Optional[np.ndarray] = None,
    sqi_threshold: float = 0.5,
    logger=None
) -> List[Optional[SignalSegment]]:
    """
    Segmentiert ein einzelnes Signal anhand vorgegebener Grenzen.
    
    Parameters
    ----------
    signal : np.ndarray
        Das zu segmentierende Signal.
    fs : float
        Sampling-Rate in Hz.
    boundaries : pd.DataFrame
        Segment-Grenzen (aus compute_segment_boundaries).
    channel_name : str
        Name des Kanals.
    signal_type : str
        Signaltyp ("ecg", "eeg", "resp").
    sqi_values : np.ndarray, optional
        SQI-Werte pro Segment (gleiche Länge wie boundaries).
    sqi_threshold : float
        SQI-Schwellenwert für Qualitäts-Flag.
    logger : loguru.Logger, optional
    
    Returns
    -------
    List[Optional[SignalSegment]]
        Liste von SignalSegment-Objekten. None für ungültige Segmente.
    """
    segments = []
    total_samples = len(signal)
    
    for _, row in boundaries.iterrows():
        seg_idx = int(row["segment_idx"])
        start_sample = int(row["start_sec"] * fs)
        end_sample = int(row["end_sec"] * fs)
        
        # Sicherheitscheck: Grenzen nicht überschreiten
        start_sample = max(0, start_sample)
        end_sample = min(total_samples, end_sample)
        
        if start_sample >= end_sample:
            segments.append(None)
            if logger:
                logger.warning(
                    f"Segment {seg_idx} [{channel_name}]: Ungültige Grenzen "
                    f"({start_sample} >= {end_sample})"
                )
            continue
        
        # Signal-Daten extrahieren
        seg_data = signal[start_sample:end_sample].copy()
        
        # NaN/Inf Check
        if np.any(np.isnan(seg_data)) or np.any(np.isinf(seg_data)):
            n_invalid = np.sum(np.isnan(seg_data)) + np.sum(np.isinf(seg_data))
            invalid_ratio = n_invalid / len(seg_data)
            
            if invalid_ratio > 0.1:
                # Mehr als 10% ungültig -> Segment verwerfen
                segments.append(None)
                if logger:
                    logger.warning(
                        f"Segment {seg_idx} [{channel_name}]: "
                        f"{invalid_ratio*100:.1f}% NaN/Inf -> verworfen"
                    )
                continue
            else:
                # Wenige ungültige Werte -> Interpolieren
                seg_data = _interpolate_invalid(seg_data)
                if logger:
                    logger.debug(
                        f"Segment {seg_idx} [{channel_name}]: "
                        f"{n_invalid} NaN/Inf Werte interpoliert"
                    )
        
        # SQI zuweisen
        sqi = 0.0
        if sqi_values is not None and seg_idx < len(sqi_values):
            sqi = float(sqi_values[seg_idx])
        
        quality_ok = sqi >= sqi_threshold
        
        segment = SignalSegment(
            segment_idx=seg_idx,
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            start_sample=start_sample,
            end_sample=end_sample,
            data=seg_data,
            fs=fs,
            channel_name=channel_name,
            signal_type=signal_type,
            sqi=sqi,
            quality_ok=quality_ok,
        )
        
        segments.append(segment)
    
    if logger:
        n_valid = sum(1 for s in segments if s is not None)
        n_quality = sum(1 for s in segments if s is not None and s.quality_ok)
        logger.info(
            f"Segmentierung [{channel_name}]: {n_valid}/{len(segments)} gültig, "
            f"{n_quality}/{len(segments)} mit ausreichender Qualität"
        )
    
    return segments


# ==============================================================================
# VOLLSTÄNDIGE SEGMENTIERUNG EINES PATIENTEN
# ==============================================================================

def segment_patient_recording(
    patient_id: str,
    ecg_preprocessed: Optional[Dict],
    eeg_strategies: Optional[Dict],
    resp_signals: Optional[Dict],
    annotation_data: Optional[Dict],
    total_duration_sec: float,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    overlap_sec: float = SEGMENT_OVERLAP_SEC,
    logger=None
) -> SegmentedRecording:
    """
    Segmentiert alle Signale und Annotationen eines Patienten.
    
    Parameters
    ----------
    patient_id : str
        Patienten-ID.
    ecg_preprocessed : Dict or None
        Ergebnis von preprocess_ecg_signal().
    eeg_strategies : Dict or None
        Ergebnis von decide_eeg_channel_strategy().
        Keys: Region-Namen ("frontal", "central", "occipital")
    resp_signals : Dict or None
        Dictionary mit Respirationssignalen.
        Keys: Kanalnamen, Values: {"signal": np.ndarray, "fs": float}
    annotation_data : Dict or None
        Ergebnis von process_all_annotations().
    total_duration_sec : float
        Gesamtdauer der Aufnahme in Sekunden.
    segment_length_sec : float
        Segmentlänge in Sekunden.
    overlap_sec : float
        Überlappung in Sekunden.
    logger : loguru.Logger, optional
    
    Returns
    -------
    SegmentedRecording
        Container mit allen segmentierten Daten.
    """
    if logger:
        logger.info("=" * 60)
        logger.info(f"Starte Segmentierung für Patient {patient_id}")
        logger.info(f"Segmentlänge: {segment_length_sec}s, "
                     f"Überlappung: {overlap_sec}s")
        logger.info("=" * 60)
    
    # --- Segment-Grenzen berechnen ---
    boundaries = compute_segment_boundaries(
        total_duration_sec, segment_length_sec, overlap_sec
    )
    n_segments = len(boundaries)
    
    if logger:
        logger.info(f"Anzahl Segmente: {n_segments} "
                     f"(Aufnahmedauer: {total_duration_sec/60:.1f} min)")
    
    # --- Initialisiere SegmentedRecording ---
    recording = SegmentedRecording(
        patient_id=patient_id,
        total_duration_sec=total_duration_sec,
        segment_length_sec=segment_length_sec,
        segment_overlap_sec=overlap_sec,
        n_segments=n_segments,
    )
    
    # --- ECG segmentieren ---
    if ecg_preprocessed is not None and ecg_preprocessed.get("ecg_cleaned") is not None:
        ecg_sqi = None
        if ecg_preprocessed.get("sqi_per_segment") is not None:
            ecg_sqi = ecg_preprocessed["sqi_per_segment"]["sqi"].values
        
        recording.ecg_segments = segment_signal(
            signal=ecg_preprocessed["ecg_cleaned"],
            fs=ecg_preprocessed.get("fs", 
                len(ecg_preprocessed["ecg_cleaned"]) / total_duration_sec),
            boundaries=boundaries,
            channel_name="ECG",
            signal_type="ecg",
            sqi_values=ecg_sqi,
            sqi_threshold=ECG_SQI_THRESHOLD,
            logger=logger,
        )
        
        # R-Peaks pro Segment zuweisen
        if ecg_preprocessed.get("rpeaks") is not None:
            _assign_rpeaks_to_segments(
                recording.ecg_segments,
                ecg_preprocessed["rpeaks"],
                ecg_preprocessed.get("rr_intervals"),
                ecg_preprocessed.get("rr_valid_mask"),
            )
    else:
        recording.ecg_segments = [None] * n_segments
        if logger:
            logger.warning("Kein ECG-Signal verfügbar für Segmentierung.")
    
    # --- EEG segmentieren (pro Region) ---
    if eeg_strategies is not None:
        for region, strategy_data in eeg_strategies.items():
            if strategy_data["strategy"] == "none" or strategy_data["signal"] is None:
                recording.eeg_segments[region] = [None] * n_segments
                if logger:
                    logger.warning(f"EEG [{region}]: Kein Signal -> übersprungen")
                continue
            
            eeg_sqi = None
            if strategy_data.get("sqi_per_segment") is not None:
                eeg_sqi = strategy_data["sqi_per_segment"]["sqi"].values
            
            # Sampling-Rate aus Signal-Länge und Dauer ableiten
            eeg_signal = strategy_data["signal"]
            eeg_fs = len(eeg_signal) / total_duration_sec
            
            recording.eeg_segments[region] = segment_signal(
                signal=eeg_signal,
                fs=eeg_fs,
                boundaries=boundaries,
                channel_name=f"EEG_{region}_{strategy_data['strategy']}",
                signal_type="eeg",
                sqi_values=eeg_sqi,
                sqi_threshold=EEG_SQI_THRESHOLD,
                logger=logger,
            )
    else:
        if logger:
            logger.warning("Keine EEG-Strategien verfügbar.")
    
    # --- Respirationssignale segmentieren ---
    if resp_signals is not None:
        for resp_name, resp_data in resp_signals.items():
            if resp_data.get("signal") is None:
                continue
            
            recording.resp_segments[resp_name] = segment_signal(
                signal=resp_data["signal"],
                fs=resp_data["fs"],
                boundaries=boundaries,
                channel_name=f"RESP_{resp_name}",
                signal_type="resp",
                sqi_values=None,  # Kein SQI für Resp (vorerst)
                sqi_threshold=0.0,
                logger=logger,
            )
    
    # --- Annotationen zuweisen ---
    if annotation_data is not None:
        recording.stages_per_segment = annotation_data.get("stages_per_segment")
        recording.events_per_segment = annotation_data.get("events_per_segment")
    
    # --- Segment-Metadaten erstellen ---
    recording.segment_metadata = _build_segment_metadata(
        recording, boundaries, logger
    )
    
    if logger:
        logger.info(f"Segmentierung abgeschlossen: {n_segments} Segmente erstellt.")
    
    return recording


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def _assign_rpeaks_to_segments(
    ecg_segments: List[Optional[SignalSegment]],
    rpeaks: np.ndarray,
    rr_intervals: Optional[np.ndarray] = None,
    rr_valid_mask: Optional[np.ndarray] = None,
):
    """
    Weist R-Peaks und RR-Intervalle den ECG-Segmenten zu.
    Speichert die Daten als zusätzliche Attribute im SignalSegment.
    """
    for seg in ecg_segments:
        if seg is None:
            continue
        
        # R-Peaks in diesem Segment (in Sample-Indizes relativ zum Segment)
        mask = (rpeaks >= seg.start_sample) & (rpeaks < seg.end_sample)
        seg_rpeaks_global = rpeaks[mask]
        seg_rpeaks_local = seg_rpeaks_global - seg.start_sample
        
        # Als zusätzliche Attribute speichern
        seg.rpeaks_local = seg_rpeaks_local
        seg.rpeaks_global = seg_rpeaks_global
        seg.n_rpeaks = len(seg_rpeaks_local)
        
        # RR-Intervalle für dieses Segment
        if rr_intervals is not None:
            # RR-Intervalle sind zwischen aufeinanderfolgenden R-Peaks
            # Finde die Indizes der R-Peaks in diesem Segment
            rpeak_indices = np.where(mask)[0]
            
            # RR-Intervalle: Index i entspricht dem Intervall zwischen
            # rpeak[i] und rpeak[i+1], also brauchen wir rpeak_indices[:-1]
            # aber die RR-Intervalle sind um 1 verschoben (diff)
            rr_mask = np.zeros(len(rr_intervals), dtype=bool)
            for idx in rpeak_indices:
                if idx < len(rr_intervals):
                    rr_mask[idx] = True
            # Letzter R-Peak hat kein zugehöriges RR-Intervall
            if len(rpeak_indices) > 0 and rpeak_indices[-1] < len(rr_intervals):
                pass  # Bereits gesetzt
            
            seg.rr_intervals = rr_intervals[rr_mask]
            
            if rr_valid_mask is not None:
                seg.rr_valid_mask = rr_valid_mask[rr_mask]
            else:
                seg.rr_valid_mask = np.ones(len(seg.rr_intervals), dtype=bool)
        else:
            seg.rr_intervals = np.array([])
            seg.rr_valid_mask = np.array([], dtype=bool)


def _build_segment_metadata(
    recording: SegmentedRecording,
    boundaries: pd.DataFrame,
    logger=None
) -> pd.DataFrame:
    """
    Erstellt eine Metadaten-Tabelle für alle Segmente.
    Fasst Qualitätsinformationen aller Signaltypen zusammen.
    """
    metadata = boundaries.copy()
    metadata["patient_id"] = recording.patient_id
    
    # --- ECG Metadaten ---
    ecg_sqi = []
    ecg_quality = []
    ecg_n_rpeaks = []
    
    for seg in recording.ecg_segments:
        if seg is not None:
            ecg_sqi.append(seg.sqi)
            ecg_quality.append(seg.quality_ok)
            ecg_n_rpeaks.append(getattr(seg, 'n_rpeaks', 0))
        else:
            ecg_sqi.append(0.0)
            ecg_quality.append(False)
            ecg_n_rpeaks.append(0)
    
    metadata["ecg_sqi"] = ecg_sqi
    metadata["ecg_quality_ok"] = ecg_quality
    metadata["ecg_n_rpeaks"] = ecg_n_rpeaks
    
    # --- EEG Metadaten (pro Region) ---
    for region, segments in recording.eeg_segments.items():
        region_sqi = []
        region_quality = []
        
        for seg in segments:
            if seg is not None:
                region_sqi.append(seg.sqi)
                region_quality.append(seg.quality_ok)
            else:
                region_sqi.append(0.0)
                region_quality.append(False)
        
        metadata[f"eeg_{region}_sqi"] = region_sqi
        metadata[f"eeg_{region}_quality_ok"] = region_quality
    
    # --- Kombinierte Qualitäts-Flags ---
    # Mindestens ECG ODER mindestens ein EEG-Kanal muss gut sein
    metadata["any_signal_quality_ok"] = metadata["ecg_quality_ok"]
    for region in recording.eeg_segments:
        col = f"eeg_{region}_quality_ok"
        if col in metadata.columns:
            metadata["any_signal_quality_ok"] = (
                metadata["any_signal_quality_ok"] | metadata[col]
            )
    
    # --- Schlafstadien-Info ---
    if recording.stages_per_segment is not None:
        stages = recording.stages_per_segment
        if len(stages) == len(metadata):
            metadata["dominant_stage"] = stages["dominant_stage"].values
            metadata["stage_stability"] = stages["stage_stability"].values
        else:
            # Längen stimmen nicht überein -> Merge über segment_idx
            metadata = metadata.merge(
                stages[["segment_idx", "dominant_stage", "stage_stability"]],
                on="segment_idx",
                how="left"
            )
    
    # --- Event-Info (Zusammenfassung) ---
    if recording.events_per_segment is not None:
        events = recording.events_per_segment
        if len(events) == len(metadata):
            metadata["any_event_present"] = events["any_event_present"].values
            metadata["total_event_count"] = events["total_event_count"].values
        else:
            metadata = metadata.merge(
                events[["segment_idx", "any_event_present", "total_event_count"]],
                on="segment_idx",
                how="left"
            )
    
    if logger:
        n_total = len(metadata)
        n_ecg_ok = metadata["ecg_quality_ok"].sum()
        n_any_ok = metadata["any_signal_quality_ok"].sum()
        logger.info(f"Segment-Metadaten: {n_total} Segmente")
        logger.info(f"  ECG Qualität OK: {n_ecg_ok}/{n_total} "
                     f"({n_ecg_ok/n_total*100:.1f}%)")
        logger.info(f"  Mindestens ein Signal OK: {n_any_ok}/{n_total} "
                     f"({n_any_ok/n_total*100:.1f}%)")
        
        for region in recording.eeg_segments:
            col = f"eeg_{region}_quality_ok"
            if col in metadata.columns:
                n_ok = metadata[col].sum()
                logger.info(f"  EEG [{region}] Qualität OK: {n_ok}/{n_total} "
                             f"({n_ok/n_total*100:.1f}%)")
    
    return metadata


def _interpolate_invalid(data: np.ndarray) -> np.ndarray:
    """
    Ersetzt NaN und Inf Werte durch lineare Interpolation.
    """
    result = data.copy()
    invalid_mask = np.isnan(result) | np.isinf(result)
    
    if not invalid_mask.any():
        return result
    
    valid_indices = np.where(~invalid_mask)[0]
    invalid_indices = np.where(invalid_mask)[0]
    
    if len(valid_indices) == 0:
        # Alles ungültig -> Nullen
        return np.zeros_like(result)
    
    if len(valid_indices) == len(result):
        return result
    
    # Lineare Interpolation
    result[invalid_indices] = np.interp(
        invalid_indices,
        valid_indices,
        result[valid_indices]
    )
    
    return result


# ==============================================================================
# RE-SEGMENTIERUNG (Segmentlänge nachträglich ändern)
# ==============================================================================

def resegment_recording(
    patient_id: str,
    ecg_preprocessed: Optional[Dict],
    eeg_strategies: Optional[Dict],
    resp_signals: Optional[Dict],
    annotation_data: Optional[Dict],
    total_duration_sec: float,
    new_segment_length_sec: float,
    new_overlap_sec: float = 0.0,
    logger=None
) -> SegmentedRecording:
    """
    Re-segmentiert eine Aufnahme mit neuer Segmentlänge.
    
    Dies ermöglicht es, die Segmentlänge nachträglich zu ändern,
    ohne die Vorverarbeitung erneut durchführen zu müssen.
    
    Parameters
    ----------
    patient_id : str
        Patienten-ID.
    ecg_preprocessed : Dict or None
        Vorverarbeitete ECG-Daten.
    eeg_strategies : Dict or None
        EEG-Kanal-Strategien.
    resp_signals : Dict or None
        Respirationssignale.
    annotation_data : Dict or None
        Annotationsdaten (müssen für neue Segmentlänge neu gemappt werden).
    total_duration_sec : float
        Gesamtdauer.
    new_segment_length_sec : float
        Neue Segmentlänge in Sekunden.
    new_overlap_sec : float
        Neue Überlappung in Sekunden.
    logger : loguru.Logger, optional
    
    Returns
    -------
    SegmentedRecording
        Neu segmentierte Aufnahme.
    """
    if logger:
        logger.info(f"Re-Segmentierung: {new_segment_length_sec}s "
                     f"(Überlappung: {new_overlap_sec}s)")
    
    # Annotationen müssen für die neue Segmentlänge neu gemappt werden
    if annotation_data is not None:
        from preprocessing.preprocess_annotations import (
            map_stages_to_segments, map_events_to_segments
        )
        
        # Neu-Mapping der Schlafstadien
        if annotation_data.get("stages_raw") is not None:
            annotation_data["stages_per_segment"] = map_stages_to_segments(
                annotation_data["stages_raw"],
                total_duration_sec,
                new_segment_length_sec,
                logger
            )
        
        # Neu-Mapping der Events
        if annotation_data.get("events_raw") is not None:
            annotation_data["events_per_segment"] = map_events_to_segments(
                annotation_data["events_raw"],
                total_duration_sec,
                new_segment_length_sec,
                logger
            )
        
        # SQI muss ebenfalls neu berechnet werden
        # (wird in segment_patient_recording über die Rohwerte gemacht)
    
    # Nutze die Standard-Segmentierungsfunktion mit neuen Parametern
    return segment_patient_recording(
        patient_id=patient_id,
        ecg_preprocessed=ecg_preprocessed,
        eeg_strategies=eeg_strategies,
        resp_signals=resp_signals,
        annotation_data=annotation_data,
        total_duration_sec=total_duration_sec,
        segment_length_sec=new_segment_length_sec,
        overlap_sec=new_overlap_sec,
        logger=logger,
    )


# ==============================================================================
# UTILITY: SEGMENT-STATISTIKEN
# ==============================================================================

def get_segment_statistics(recording: SegmentedRecording) -> pd.DataFrame:
    """
    Erstellt eine Übersichtstabelle mit Statistiken pro Segment.
    Nützlich für Debugging und Qualitätskontrolle.
    
    Returns
    -------
    pd.DataFrame
        Statistiken pro Segment.
    """
    if recording.segment_metadata is None:
        return pd.DataFrame()
    
    stats = recording.segment_metadata.copy()
    
    # ECG-Statistiken pro Segment
    for i, seg in enumerate(recording.ecg_segments):
        if seg is not None and i < len(stats):
            stats.loc[i, "ecg_mean"] = np.mean(seg.data)
            stats.loc[i, "ecg_std"] = np.std(seg.data)
            stats.loc[i, "ecg_n_rpeaks"] = getattr(seg, 'n_rpeaks', 0)
            
            rr = getattr(seg, 'rr_intervals', np.array([]))
            if len(rr) > 0:
                valid = getattr(seg, 'rr_valid_mask', np.ones(len(rr), dtype=bool))
                valid_rr = rr[valid] if valid.any() else rr
                stats.loc[i, "ecg_mean_hr"] = 60000 / np.mean(valid_rr) if len(valid_rr) > 0 else np.nan
                stats.loc[i, "ecg_rr_artifact_pct"] = (1 - valid.mean()) * 100 if len(valid) > 0 else 100
    
    # EEG-Statistiken pro Segment und Region
    for region, segments in recording.eeg_segments.items():
        for i, seg in enumerate(segments):
            if seg is not None and i < len(stats):
                stats.loc[i, f"eeg_{region}_mean"] = np.mean(seg.data)
                stats.loc[i, f"eeg_{region}_std"] = np.std(seg.data)
                stats.loc[i, f"eeg_{region}_max_abs"] = np.max(np.abs(seg.data))
    
    return stats


def print_segmentation_summary(recording: SegmentedRecording,
    logger=None):
    """
    Gibt eine lesbare Zusammenfassung der Segmentierung aus.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Segmentierung: Patient {recording.patient_id}")
    logger.info(f"{'='*60}")
    logger.info(f"Aufnahmedauer:    {recording.total_duration_sec/60:.1f} min")
    logger.info(f"Segmentlänge:    {recording.segment_length_sec}s")
    logger.info(f"Überlappung:     {recording.segment_overlap_sec}s")
    logger.info(f"Anzahl Segmente: {recording.n_segments}")
    logger.info(f"{'-'*60}")
    
    # --- ECG ---
    n_ecg_valid = sum(1 for s in recording.ecg_segments if s is not None)
    n_ecg_quality = sum(
        1 for s in recording.ecg_segments 
        if s is not None and s.quality_ok
    )
    logger.info(f"\nECG:")
    logger.info(f"  Gültige Segmente:     {n_ecg_valid}/{recording.n_segments}")
    logger.info(f"  Qualität OK:          {n_ecg_quality}/{recording.n_segments} "
          f"({n_ecg_quality/recording.n_segments*100:.1f}%)" 
          if recording.n_segments > 0 else "  Qualität OK: N/A")
    
    # Mittlere Herzfrequenz über alle gültigen Segmente
    hrs = []
    for seg in recording.ecg_segments:
        if seg is not None:
            rr = getattr(seg, 'rr_intervals', np.array([]))
            valid_mask = getattr(seg, 'rr_valid_mask', np.ones(len(rr), dtype=bool))
            valid_rr = rr[valid_mask] if len(rr) > 0 and valid_mask.any() else np.array([])
            if len(valid_rr) > 0:
                hrs.append(60000 / np.mean(valid_rr))
    if hrs:
        logger.info(f"  Mittlere HR:          {np.mean(hrs):.1f} bpm "
              f"(Range: {np.min(hrs):.1f}-{np.max(hrs):.1f})")
    
    # --- EEG ---
    logger.info(f"\nEEG:")
    if recording.eeg_segments:
        for region, segments in recording.eeg_segments.items():
            n_valid = sum(1 for s in segments if s is not None)
            n_quality = sum(
                1 for s in segments 
                if s is not None and s.quality_ok
            )
            # Strategie aus Kanalnamen extrahieren
            strategy = "unknown"
            for s in segments:
                if s is not None:
                    strategy = s.channel_name.split("_")[-1] if "_" in s.channel_name else "unknown"
                    break
            
            logger.info(f"  [{region}] Strategie: {strategy}")
            logger.info(f"    Gültige Segmente:   {n_valid}/{recording.n_segments}")
            logger.info(f"    Qualität OK:        {n_quality}/{recording.n_segments} "
                  f"({n_quality/recording.n_segments*100:.1f}%)"
                  if recording.n_segments > 0 else "    Qualität OK: N/A")
    else:
        logger.info("  Keine EEG-Kanäle verfügbar.")
    
    # --- Respiration ---
    logger.info(f"\nRespiration:")
    if recording.resp_segments:
        for resp_name, segments in recording.resp_segments.items():
            n_valid = sum(1 for s in segments if s is not None)
            logger.info(f"  [{resp_name}] Gültige Segmente: {n_valid}/{recording.n_segments}")
    else:
        logger.info("  Keine Respirationssignale verfügbar.")
    
    # --- Annotationen ---
    logger.info(f"\nAnnotationen:")
    if recording.stages_per_segment is not None:
        stages = recording.stages_per_segment
        stage_counts = stages["dominant_stage"].value_counts().to_dict()
        logger.info(f"  Schlafstadien vorhanden: Ja")
        logger.info(f"  Stadien-Verteilung (dominant pro Segment):")
        for stage, count in sorted(stage_counts.items()):
            pct = count / len(stages) * 100
            logger.info(f"    {stage}: {count} Segmente ({pct:.1f}%)")
        
        # Mittlere Stabilität
        mean_stability = stages["stage_stability"].mean()
        logger.info(f"  Mittlere Stadien-Stabilität: {mean_stability:.2f}")
    else:
        logger.info("  Schlafstadien: Nicht verfügbar")
    
    if recording.events_per_segment is not None:
        events = recording.events_per_segment
        logger.info(f"  Events vorhanden: Ja")
        
        if "any_event_present" in events.columns:
            n_with_events = events["any_event_present"].sum()
            logger.info(f"  Segmente mit Events: {n_with_events}/{len(events)} "
                  f"({n_with_events/len(events)*100:.1f}%)")
        
        if "any_apnea_present" in events.columns:
            n_with_apnea = events["any_apnea_present"].sum()
            logger.info(f"  Segmente mit Apnoe:  {n_with_apnea}/{len(events)} "
                  f"({n_with_apnea/len(events)*100:.1f}%)")
        
        if "arousal_present" in events.columns:
            n_with_arousal = events["arousal_present"].sum()
            logger.info(f"  Segmente mit Arousal: {n_with_arousal}/{len(events)} "
                  f"({n_with_arousal/len(events)*100:.1f}%)")
    else:
        logger.info("  Events: Nicht verfügbar")
    
    # --- Segment-Metadaten Zusammenfassung ---
    if recording.segment_metadata is not None:
        meta = recording.segment_metadata
        logger.info(f"\nGesamtqualität:")
        if "any_signal_quality_ok" in meta.columns:
            n_any_ok = meta["any_signal_quality_ok"].sum()
            logger.info(f"  Segmente mit mind. 1 gutem Signal: "
                  f"{n_any_ok}/{len(meta)} ({n_any_ok/len(meta)*100:.1f}%)")
        
        # Nutzbare Segmente für Feature-Extraktion
        n_usable = 0
        for i in range(len(meta)):
            ecg_ok = meta.loc[i, "ecg_quality_ok"] if "ecg_quality_ok" in meta.columns else False
            eeg_ok = any(
                meta.loc[i, f"eeg_{region}_quality_ok"]
                for region in recording.eeg_segments
                if f"eeg_{region}_quality_ok" in meta.columns
            ) if recording.eeg_segments else False
            if ecg_ok or eeg_ok:
                n_usable += 1
        
        logger.info(f"  Nutzbare Segmente (ECG oder EEG OK): "
              f"{n_usable}/{len(meta)} ({n_usable/len(meta)*100:.1f}%)")
    
    logger.info(f"\n{'='*60}\n")


# ==============================================================================
# BATCH-VERARBEITUNG
# ==============================================================================

def segment_multiple_patients(
    patient_data: Dict[str, Dict],
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    overlap_sec: float = SEGMENT_OVERLAP_SEC,
    logger=None
) -> Dict[str, SegmentedRecording]:
    """
    Segmentiert Aufnahmen für mehrere Patienten.
    
    Parameters
    ----------
    patient_data : Dict[str, Dict]
        Dictionary mit Patienten-IDs als Keys und Dictionaries mit
        vorverarbeiteten Daten als Values. Jedes Dictionary sollte enthalten:
        - ecg_preprocessed
        - eeg_strategies
        - resp_signals
        - annotation_data
        - total_duration_sec
    segment_length_sec : float
        Segmentlänge in Sekunden.
    overlap_sec : float
        Überlappung in Sekunden.
    logger : loguru.Logger, optional
    
    Returns
    -------
    Dict[str, SegmentedRecording]
        Dictionary mit Patienten-IDs als Keys und SegmentedRecording als Values.
    """
    results = {}
    n_patients = len(patient_data)
    
    for i, (patient_id, data) in enumerate(patient_data.items()):
        if logger:
            logger.info(f"Segmentierung Patient {i+1}/{n_patients}: {patient_id}")
        
        try:
            recording = segment_patient_recording(
                patient_id=patient_id,
                ecg_preprocessed=data.get("ecg_preprocessed"),
                eeg_strategies=data.get("eeg_strategies"),
                resp_signals=data.get("resp_signals"),
                annotation_data=data.get("annotation_data"),
                total_duration_sec=data.get("total_duration_sec", 0),
                segment_length_sec=segment_length_sec,
                overlap_sec=overlap_sec,
                logger=logger,
            )
            results[patient_id] = recording
            
            if logger:
                logger.info(f"Patient {patient_id}: Segmentierung erfolgreich "
                            f"({recording.n_segments} Segmente)")
        
        except Exception as e:
            if logger:
                logger.error(f"Patient {patient_id}: Segmentierung fehlgeschlagen: {e}")
            results[patient_id] = None
    
    if logger:
        n_success = sum(1 for v in results.values() if v is not None)
        logger.info(f"Batch-Segmentierung abgeschlossen: "
                     f"{n_success}/{n_patients} erfolgreich")
    
    return results
