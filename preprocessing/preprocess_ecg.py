"""
preprocessing/preprocess_ecg.py
===================================
ECG-Vorverarbeitung mit NeuroKit2:
- Bandpassfilterung (nk.ecg_clean in Chunks)
- R-Peak-Detektion (nk.ecg_peaks in Chunks)
- Signal Quality Index (SQI) Berechnung

Lange Aufnahmen (>10 min) werden in Chunks verarbeitet,
um Speicherprobleme und C-Level Crashes zu vermeiden.
NeuroKit2 wird für alle Kernfunktionen verwendet [[1]] [[3]].
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt, find_peaks
from typing import Dict, Optional

from config import (
    ECG_FILTER, ECG_SQI_THRESHOLD, ECG_SQI_WINDOW_SEC,
    HRV_RR_MIN_MS, HRV_RR_MAX_MS, SEGMENT_LENGTH_SEC
)

# Maximale Chunk-Größe für NeuroKit2 (in Sekunden)
# 10 Minuten ist ein sicherer Wert für nk.ecg_clean()
NK_CHUNK_DURATION_SEC = 600  # 10 Minuten
NK_CHUNK_OVERLAP_SEC = 10    # 10s Überlappung für Filterartefakte

# R-Peak Chunk-Größe (kann kleiner sein)
RPEAK_CHUNK_DURATION_SEC = 300  # 5 Minuten
RPEAK_CHUNK_OVERLAP_SEC = 2     # 2s Überlappung


def preprocess_ecg_signal(
    ecg_raw: np.ndarray,
    fs: float,
    logger=None
) -> Dict:
    """
    Vollständige ECG-Vorverarbeitung mit NeuroKit2.
    Verarbeitet lange Aufnahmen in Chunks.
    
    Parameters
    ----------
    ecg_raw : np.ndarray
        Rohes ECG-Signal.
    fs : float
        Sampling-Rate in Hz.
    logger : loguru.Logger, optional
    
    Returns
    -------
    Dict mit:
        - ecg_cleaned: Gefiltertes Signal
        - rpeaks: R-Peak Indizes (global)
        - rr_intervals: RR-Intervalle in ms
        - rr_times: Zeitpunkte der RR-Intervalle in Sekunden
        - rr_valid_mask: Boolean-Maske für gültige RR-Intervalle
        - sqi_per_segment: SQI pro Segment (DataFrame)
        - quality_mask: Boolean-Maske pro Segment
        - info: Zusätzliche Informationen
    """
    result = {
        "ecg_cleaned": None,
        "rpeaks": None,
        "rr_intervals": None,
        "rr_times": None,
        "rr_valid_mask": None,
        "sqi_per_segment": None,
        "quality_mask": None,
        "info": {}
    }

    if logger:
        logger.info(f"ECG Preprocessing: {len(ecg_raw)} samples, fs={fs} Hz, "
                     f"duration={len(ecg_raw)/fs:.1f}s")

    # ==================================================================
    # 1. FILTERUNG (nk.ecg_clean in Chunks)
    # ==================================================================
    try:
        ecg_cleaned = _clean_ecg_chunked(ecg_raw, fs, logger)
        result["ecg_cleaned"] = ecg_cleaned

        if logger:
            logger.info("ECG Filterung erfolgreich (NeuroKit2, chunk-weise).")
    except Exception as e:
        if logger:
            logger.error(f"ECG Filterung fehlgeschlagen: {e}")
        # Fallback: scipy Bandpassfilter
        try:
            ecg_cleaned = _bandpass_filter_scipy(ecg_raw, fs)
            result["ecg_cleaned"] = ecg_cleaned
            if logger:
                logger.info("ECG Filterung: Fallback auf scipy Bandpass.")
        except Exception as e2:
            if logger:
                logger.error(f"ECG Filterung komplett fehlgeschlagen: {e2}")
            return result

    # ==================================================================
    # 2. R-PEAK-DETEKTION (nk.ecg_peaks in Chunks)
    # ==================================================================
    try:
        if logger:
            logger.info("R-Peak-Detektion (NeuroKit2, chunk-weise)...")

        rpeaks = _detect_rpeaks_chunked(ecg_cleaned, fs, logger)

        if logger:
            logger.info(f"R-Peaks detektiert: {len(rpeaks)}")

        if len(rpeaks) < 2:
            if logger:
                logger.warning("Zu wenige R-Peaks detektiert.")
            result["rpeaks"] = rpeaks
            return result

        result["rpeaks"] = rpeaks

    except Exception as e:
        if logger:
            logger.error(f"R-Peak-Detektion fehlgeschlagen: {e}")
        return result

    # ==================================================================
    # 3. RR-INTERVALLE BERECHNEN UND BEREINIGEN
    # ==================================================================
    rr_intervals_samples = np.diff(rpeaks)
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000
    rr_times = rpeaks[1:] / fs

    # Physiologische Grenzen
    valid_rr_mask = (
        (rr_intervals_ms >= HRV_RR_MIN_MS) &
        (rr_intervals_ms <= HRV_RR_MAX_MS)
    )

    # Median-basierte Artefakterkennung
    rr_median = np.median(rr_intervals_ms[valid_rr_mask]) if valid_rr_mask.any() else 800
    deviation = np.abs(rr_intervals_ms - rr_median) / rr_median
    valid_rr_mask &= (deviation < 0.20)

    n_artifacts = (~valid_rr_mask).sum()
    if logger:
        logger.info(f"RR-Intervalle: {len(rr_intervals_ms)} total, "
                     f"{n_artifacts} Artefakte "
                     f"({n_artifacts/len(rr_intervals_ms)*100:.1f}%)")

    result["rr_intervals"] = rr_intervals_ms
    result["rr_times"] = rr_times
    result["rr_valid_mask"] = valid_rr_mask
    result["info"]["n_rpeaks"] = len(rpeaks)
    result["info"]["n_rr_artifacts"] = int(n_artifacts)
    result["info"]["mean_hr"] = 60000 / rr_median if rr_median > 0 else 0

    # ==================================================================
    # 4. SIGNAL QUALITY INDEX (SQI) PRO SEGMENT
    # ==================================================================
    if logger:
        logger.info("SQI-Berechnung...")

    try:
        sqi_segments = compute_ecg_sqi_segments(
            ecg_cleaned, rpeaks, fs,
            segment_length_sec=SEGMENT_LENGTH_SEC,
            logger=logger
        )
        result["sqi_per_segment"] = sqi_segments

        quality_mask = np.array([
            sqi >= ECG_SQI_THRESHOLD for sqi in sqi_segments["sqi"]
        ])
        result["quality_mask"] = quality_mask

        if logger:
            n_good = quality_mask.sum()
            n_total = len(quality_mask)
            logger.info(f"ECG Qualität: {n_good}/{n_total} Segmente über "
                         f"SQI-Schwelle ({ECG_SQI_THRESHOLD}): "
                         f"{n_good/n_total*100:.1f}%")
    except Exception as e:
        if logger:
            logger.warning(f"SQI-Berechnung fehlgeschlagen: {e}")

    if logger:
        logger.info("ECG Preprocessing abgeschlossen.")

    return result


# ==============================================================================
# CHUNK-WEISE ECG CLEANING MIT NEUROKIT2
# ==============================================================================

def _clean_ecg_chunked(
    ecg_raw: np.ndarray,
    fs: float,
    logger=None,
    chunk_duration_sec: float = NK_CHUNK_DURATION_SEC,
    overlap_sec: float = NK_CHUNK_OVERLAP_SEC,
) -> np.ndarray:
    """
    Filtert das ECG-Signal mit nk.ecg_clean() in Chunks.
    
    Für kurze Signale (<= 2 Chunks) wird direkt verarbeitet.
    Für lange Signale wird in überlappende Chunks aufgeteilt,
    jeder Chunk einzeln mit NeuroKit2 gefiltert, und die
    Ergebnisse nahtlos zusammengefügt.
    """
    chunk_samples = int(chunk_duration_sec * fs)
    overlap_samples = int(overlap_sec * fs)

    # Kurzes Signal: Direkt verarbeiten
    if len(ecg_raw) <= chunk_samples * 2:
        return nk.ecg_clean(ecg_raw, sampling_rate=int(fs), method="neurokit")

    # Langes Signal: Chunk-weise
    ecg_cleaned = np.zeros_like(ecg_raw)
    n_chunks = int(np.ceil(len(ecg_raw) / chunk_samples))

    if logger:
        logger.info(f"ECG Filterung in {n_chunks} Chunks à "
                     f"{chunk_duration_sec}s...")

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_samples

        # Erweitere Chunk um Überlappung für Filterartefakte an den Rändern
        chunk_start = max(0, start - overlap_samples)
        chunk_end = min(len(ecg_raw), start + chunk_samples + overlap_samples)

        chunk = ecg_raw[chunk_start:chunk_end]

        # Mindestlänge für NeuroKit2
        if len(chunk) < int(fs * 3):
            ecg_cleaned[start:min(start + chunk_samples, len(ecg_raw))] = \
                chunk[start - chunk_start:start - chunk_start + min(chunk_samples, len(ecg_raw) - start)]
            continue

        try:
            chunk_cleaned = nk.ecg_clean(
                chunk, sampling_rate=int(fs), method="neurokit"
            )
        except Exception:
            # Fallback für diesen Chunk: scipy Bandpass
            try:
                chunk_cleaned = _bandpass_filter_scipy(chunk, fs)
            except Exception:
                chunk_cleaned = chunk.copy()

        # Kopiere nur den nicht-überlappenden Teil
        local_start = start - chunk_start
        dest_end = min(start + chunk_samples, len(ecg_raw))
        local_end = local_start + (dest_end - start)

        ecg_cleaned[start:dest_end] = chunk_cleaned[local_start:local_end]

        if logger and (chunk_idx + 1) % 10 == 0:
            logger.info(f"  ECG Filterung: {chunk_idx+1}/{n_chunks} Chunks")

    return ecg_cleaned


# ==============================================================================
# CHUNK-WEISE R-PEAK-DETEKTION MIT NEUROKIT2
# ==============================================================================

def _detect_rpeaks_chunked(
    ecg_cleaned: np.ndarray,
    fs: float,
    logger=None,
    chunk_duration_sec: float = RPEAK_CHUNK_DURATION_SEC,
    overlap_sec: float = RPEAK_CHUNK_OVERLAP_SEC,
) -> np.ndarray:
    """
    Detektiert R-Peaks mit nk.ecg_peaks() in Chunks.
    
    Jeder Chunk wird einzeln verarbeitet, Peaks werden in
    globale Indizes konvertiert, und Duplikate an den
    Chunk-Grenzen werden entfernt.
    """
    chunk_samples = int(chunk_duration_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    n_chunks = int(np.ceil(len(ecg_cleaned) / chunk_samples))

    all_rpeaks = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_samples
        chunk_start = max(0, start - overlap_samples)
        chunk_end = min(len(ecg_cleaned), start + chunk_samples + overlap_samples)

        chunk = ecg_cleaned[chunk_start:chunk_end]

        if len(chunk) < int(fs * 3):  # Mindestens 3 Sekunden
            continue

        # Versuche NeuroKit2
        chunk_peaks = _detect_rpeaks_single_chunk_nk(chunk, fs)

        # Fallback: scipy
        if chunk_peaks is None or len(chunk_peaks) == 0:
            chunk_peaks = _detect_rpeaks_scipy(chunk, fs)

        if chunk_peaks is not None and len(chunk_peaks) > 0:
            # Konvertiere zu globalen Indizes
            global_peaks = chunk_peaks + chunk_start

            # Behalte nur Peaks im nicht-überlappenden Bereich
            valid_start = 0 if chunk_idx == 0 else start
            valid_end = len(ecg_cleaned) if chunk_idx == n_chunks - 1 else start + chunk_samples

            valid_peaks = global_peaks[
                (global_peaks >= valid_start) & (global_peaks < valid_end)
            ]
            all_rpeaks.extend(valid_peaks.tolist())

        if logger and (chunk_idx + 1) % 20 == 0:
            logger.info(f"  R-Peak-Detektion: {chunk_idx+1}/{n_chunks} Chunks")

    rpeaks = np.array(sorted(set(all_rpeaks)))

    # Entferne Duplikate die zu nah beieinander sind (< 200ms)
    if len(rpeaks) > 1:
        min_distance = int(0.2 * fs)  # 200ms
        cleaned_peaks = [rpeaks[0]]
        for peak in rpeaks[1:]:
            if peak - cleaned_peaks[-1] >= min_distance:
                cleaned_peaks.append(peak)
        rpeaks = np.array(cleaned_peaks)

    return rpeaks


def _detect_rpeaks_single_chunk_nk(
    chunk: np.ndarray,
    fs: float
) -> np.ndarray:
    """
    Detektiert R-Peaks in einem einzelnen Chunk mit NeuroKit2.
    """
    try:
        _, peaks_info = nk.ecg_peaks(
            chunk, sampling_rate=int(fs), method="neurokit"
        )
        peaks = np.array(peaks_info["ECG_R_Peaks"])
        if len(peaks) > 0:
            return peaks
    except Exception:
        pass
    return None


# ==============================================================================
# SCIPY FALLBACK FUNKTIONEN
# ==============================================================================

def _bandpass_filter_scipy(
    ecg_raw: np.ndarray,
    fs: float,
) -> np.ndarray:
    """
    Bandpassfilterung mit scipy als Fallback.
    Wird nur verwendet wenn nk.ecg_clean() fehlschlägt.
    """
    nyq = 0.5 * fs
    low = max(ECG_FILTER["lowcut"] / nyq, 0.001)
    high = min(ECG_FILTER["highcut"] / nyq, 0.999)
    b, a = butter(ECG_FILTER["order"], [low, high], btype='band')
    return filtfilt(b, a, ecg_raw)


def _detect_rpeaks_scipy(
    ecg_signal: np.ndarray,
    fs: float
) -> np.ndarray:
    """
    R-Peak-Detektion mit scipy als Fallback.
    Wird nur verwendet wenn nk.ecg_peaks() fehlschlägt.
    """
    try:
        # Quadriere das Signal um R-Peaks hervorzuheben
        ecg_squared = ecg_signal ** 2

        # Gleitender Mittelwert
        window_size = max(1, int(0.12 * fs))  # 120ms Fenster
        kernel = np.ones(window_size) / window_size
        ecg_smooth = np.convolve(ecg_squared, kernel, mode='same')

        # Schwellenwert
        threshold = np.mean(ecg_smooth) + np.std(ecg_smooth)

        # Mindestabstand: 300ms (200 bpm max)
        min_distance = int(0.3 * fs)

        peaks, _ = find_peaks(
            ecg_smooth,
            height=threshold,
            distance=min_distance
        )

        # Verfeinere Peak-Positionen im Originalsignal
        refined_peaks = []
        half_window = int(0.05 * fs)

        for peak in peaks:
            start = max(0, peak - half_window)
            end = min(len(ecg_signal), peak + half_window + 1)
            local_max = start + np.argmax(np.abs(ecg_signal[start:end]))
            refined_peaks.append(local_max)

        return np.array(refined_peaks)
    except Exception:
        return np.array([])


# ==============================================================================
# SQI BERECHNUNG
# ==============================================================================

def compute_ecg_sqi_segments(
    ecg_cleaned: np.ndarray,
    rpeaks: np.ndarray,
    fs: float,
    segment_length_sec: float = 300,
    logger=None
) -> pd.DataFrame:
    """
    Berechnet den Signal Quality Index (SQI) pro Segment.
    
    Kombiniert drei SQI-Metriken:
    1. Korrelations-SQI: Ähnlichkeit der QRS-Komplexe zum Template
    2. Peak-Regularitäts-SQI: Regelmäßigkeit der RR-Intervalle
    3. SNR-SQI: Signal-Rausch-Verhältnis
    """
    segment_length_samples = int(segment_length_sec * fs)
    n_segments = int(np.ceil(len(ecg_cleaned) / segment_length_samples))

    sqi_records = []

    for seg_idx in range(n_segments):
        start_sample = seg_idx * segment_length_samples
        end_sample = min(start_sample + segment_length_samples, len(ecg_cleaned))
        start_sec = start_sample / fs
        end_sec = end_sample / fs

        seg_signal = ecg_cleaned[start_sample:end_sample]
        seg_rpeaks = rpeaks[(rpeaks >= start_sample) & (rpeaks < end_sample)]
        seg_rpeaks_local = seg_rpeaks - start_sample
        n_peaks = len(seg_rpeaks_local)

        csqi = _compute_correlation_sqi(seg_signal, seg_rpeaks_local, fs)
        psqi = _compute_peak_regularity_sqi(seg_rpeaks_local, fs)
        snr_sqi = _compute_snr_sqi(seg_signal, seg_rpeaks_local, fs)

        combined_sqi = 0.4 * csqi + 0.3 * psqi + 0.3 * snr_sqi
        combined_sqi = np.clip(combined_sqi, 0.0, 1.0)

        sqi_records.append({
            "segment_idx": seg_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "sqi": combined_sqi,
            "csqi": csqi,
            "psqi": psqi,
            "snr_sqi": snr_sqi,
            "n_rpeaks": n_peaks,
        })

    return pd.DataFrame(sqi_records)


def _compute_correlation_sqi(signal, rpeaks_local, fs):
    """Korrelation der QRS-Komplexe mit dem Median-Template."""
    if len(rpeaks_local) < 3:
        return 0.0
    half_window = int(0.1 * fs)
    qrs_complexes = []
    for peak in rpeaks_local:
        start = int(peak) - half_window
        end = int(peak) + half_window
        if start >= 0 and end < len(signal):
            qrs = signal[start:end]
            qrs_std = np.std(qrs)
            if qrs_std > 0:
                qrs_norm = (qrs - np.mean(qrs)) / qrs_std
                qrs_complexes.append(qrs_norm)
    if len(qrs_complexes) < 3:
        return 0.0
    qrs_array = np.array(qrs_complexes)
    template = np.median(qrs_array, axis=0)
    template_std = np.std(template)
    if template_std == 0:
        return 0.0
    template_norm = (template - np.mean(template)) / template_std
    correlations = []
    for qrs in qrs_array:
        corr = np.corrcoef(qrs, template_norm)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    if not correlations:
        return 0.0
    mean_corr = np.mean(correlations)
    return np.clip((mean_corr + 1) / 2, 0.0, 1.0)


def _compute_peak_regularity_sqi(rpeaks_local, fs):
    """Regelmäßigkeit der RR-Intervalle."""
    if len(rpeaks_local) < 3:
        return 0.0
    rr = np.diff(rpeaks_local) / fs * 1000  # ms
    if len(rr) < 2:
        return 0.0
    rr_mean = np.mean(rr)
    if rr_mean == 0:
        return 0.0
    rr_cv = np.std(rr) / rr_mean
    rr_diff = np.abs(np.diff(rr))
    mean_successive_diff = np.mean(rr_diff) if len(rr_diff) > 0 else 0
    cv_sqi = np.clip(1.0 - (rr_cv / 0.5), 0.0, 1.0)
    sd_sqi = np.clip(1.0 - (mean_successive_diff / 200), 0.0, 1.0)
    return 0.5 * cv_sqi + 0.5 * sd_sqi


def _compute_snr_sqi(signal, rpeaks_local, fs):
    """Signal-Rausch-Verhältnis basierend auf QRS vs. Baseline."""
    if len(rpeaks_local) < 3 or len(signal) == 0:
        return 0.0
    half_window = int(0.05 * fs)
    qrs_mask = np.zeros(len(signal), dtype=bool)
    for peak in rpeaks_local:
        start = max(0, int(peak) - half_window)
        end = min(len(signal), int(peak) + half_window)
        qrs_mask[start:end] = True
    signal_power = np.mean(signal[qrs_mask] ** 2) if qrs_mask.any() else 0
    noise_power = np.mean(signal[~qrs_mask] ** 2) if (~qrs_mask).any() else 1
    if noise_power == 0:
        return 1.0
    snr = signal_power / noise_power
    snr_sqi = np.clip((snr - 1) / 4, 0.0, 1.0)
    return snr_sqi
