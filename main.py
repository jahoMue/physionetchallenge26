"""
main.py
========
Pipeline-Orchestrierung für die PhysioNet Challenge 2026:
Screening for Cognitive Impairment During Sleep Studies.

Dieses Skript steuert die gesamte Pipeline:
1. Daten laden (WFDB Records, Annotationen, Demographics)
2. Preprocessing (ECG, EEG, Annotationen)
3. Segmentierung
4. Feature-Extraktion
5. Feature-Tabelle erstellen
6. Modell trainieren
7. Evaluation

Verwendung:
    python main.py                          # Vollständige Pipeline
    python main.py --step preprocess        # Nur Preprocessing
    python main.py --step features          # Nur Feature-Extraktion
    python main.py --step train             # Nur Training
    python main.py --step evaluate          # Nur Evaluation
    python main.py --patients 10            # Nur erste 10 Patienten
    python main.py --segment-length 300     # Segmentlänge ändern
"""
import argparse
import sys
import time
import traceback
import numpy as np
import pandas as pd
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# Konfiguration – EINZIGER Import-Block (Zeile ~37)
from config import (
    # Pfade
    DATA_DIR, OUTPUT_DIR, FEATURE_DIR, MODEL_DIR, LOG_DIR,
    # Signalverarbeitung
    SEGMENT_LENGTH_SEC, SEGMENT_OVERLAP_SEC,
    # Daten
    DEMOGRAPHICS_FILE, TARGET_COLUMN,
    # Qualität
    ECG_SQI_THRESHOLD, EEG_SQI_THRESHOLD,
    # Allgemein
    RANDOM_SEED, RSA_ENABLED,
    # Visualisierung
    PLOT_DIR, PLOT_ENABLED, PLOT_PER_PATIENT,
    PLOT_COHORT, PLOT_MAX_PATIENTS, PLOT_FORMAT, PLOT_DPI,
    NUM_WORKERS,
)


# I/O
from utils.io_utils import (
    load_demographics, get_patient_list, load_record,
    load_annotations, find_ecg_channel, find_eeg_channels,
    find_resp_channels, extract_signal, rereference_eeg_to_bipolar
)

# Preprocessing
from preprocessing.preprocess_ecg import preprocess_ecg_signal
from preprocessing.preprocess_eeg import (
    preprocess_eeg_signal, decide_eeg_channel_strategy
)
from preprocessing.preprocess_annotations import process_all_annotations

# Segmentierung
from segmentation.segment_signals import (
    segment_patient_recording, print_segmentation_summary
)

# Feature-Extraktion
from feature_extraction.features_ecg import extract_ecg_features_all_segments
from feature_extraction.features_eeg import extract_eeg_features_all_segments
from feature_extraction.features_annotations import extract_annotation_features

# CAP-Feature-Extraktion
from CAP_classification.signal_eeg import SignalEEG
from CAP_classification.signal_features import get_features_paper
from CAP_classification.cap_classification import cap_classification


# Feature-Tabelle
from feature_table.build_feature_table import (
    build_patient_feature_table, aggregate_to_patient_level,
    build_cohort_feature_table, add_demographics,
    load_and_merge_saved_features
)

# Klassifikation
from classification.train_model import (
    train_model, train_multiple_models, load_model, predict
)
from classification.evaluate_model import (
    evaluate_model, evaluate_with_cross_validation
)

# Visualisierung
from utils.visualization import (
    plot_ecg_preprocessing,
    plot_eeg_preprocessing,
    plot_hypnogram,
    plot_patient_dashboard,
    generate_all_patient_plots,
    generate_all_cohort_plots,
)

# Logging - configured at module level for multiprocessing safety
from utils.logger import setup_logger, get_patient_logger, PipelineStats
logger = setup_logger(module_name="main")




# ==============================================================================
# PIPELINE-SCHRITTE
# ==============================================================================
# [CAP-Integration] CAP-Feature-Extraktion pro Patient
def extract_cap_features_for_patient(
    eeg_signal, fs, event, duration, eventtime, patient_id,
    lstm_predict_fn=None, flags=None, logger=None
):
    cap_feature_keys = [
        "ID", "SLDUR", "NRAPH", "APHDUR", "AVGAPHDUR", "NRAPHPH", "RAPHSL",
        "NRA1PH", "NRA2PH", "NRA3PH", "A1PHDUR", "A2PHDUR", "A3PHDUR",
        "AVGA1PHDUR", "AVGA2PHDUR", "AVGA3PHDUR",
        "A1RAPHPH", "A2RAPHPH", "A3RAPHPH",
        "A1SL", "A2SL", "A3SL",
        "A1PHH", "A2PHH", "A3PHH",
        "NCAPSEQ", "CAPDUR", "AVGCAPDUR", "CAPR", "AVGCYCLEDUR", "AVGBPHDUR"
    ]
    cap_feature_keys_prefixed = [f"cap_{k}" for k in cap_feature_keys]
    cap_features = {k: np.nan for k in cap_feature_keys_prefixed}
    cap_features["cap_ID"] = patient_id
    try:
        eeg = SignalEEG(eeg_signal, fs, event, duration, eventtime, patient_id)
        eeg.eeg_features = get_features_paper(eeg.eeg, eeg.fs, eeg.event, eeg.duration, eeg.eventtime)
        input_list = eeg.create_multi_class_input()
        stats = cap_classification(
            input_list, eeg, flags or {"Scoring": "CAP"},
        )
    except Exception as e:
        if logger:
            logger.warning(f"[CAP-Integration] CAP-Feature-Extraktion fehlgeschlagen für {patient_id}: {e}")
    return stats


def process_single_patient(
    patient_id: str,
    patient_dir: Path,
    record_name: str = None,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    overlap_sec: float = SEGMENT_OVERLAP_SEC,
    feature_output_dir: Path = FEATURE_DIR,  # NEW parameter
) -> Optional[Dict]:
    """
    Verarbeitet einen einzelnen Patienten durch die gesamte Pipeline:
    Laden -> Preprocessing -> Segmentierung -> Feature-Extraktion.
    
    CHANGED: Features werden auf Disk gespeichert statt im Result-Dict
    zurückgegeben. Das Result enthält nur leichtgewichtige Metadaten.
    """
    patient_logger = get_patient_logger(logger, "pipeline", patient_id)
    stats = PipelineStats(patient_id)
    
    # CHANGED: result dict is now lightweight – no DataFrames
    result = {
        "patient_id": patient_id,
        "seg_features_path": None,       # NEW: path to saved file
        "pat_features_path": None,       # NEW: path to saved file
        "sleep_summary": None,
        "stats": None,
        "success": False,
    }
    
    start_time = time.time()
    
    # Initialize variables for cleanup
    ecg_preprocessed = None
    eeg_preprocessed = {}
    eeg_strategies = None
    annotation_data = None
    segment_features = None
    sleep_summary = None
    recording = None
    record = None                        # NEW: track for explicit cleanup
    eeg_filtered_signals = {}            # NEW: track for explicit cleanup
    
    try:
        patient_logger.info(f"{'='*50}")
        patient_logger.info(f"Starte Verarbeitung: Patient {patient_id}")
        patient_logger.info(f"{'='*50}")
        
        # ==============================================================
        # SCHRITT 1: DATEN LADEN (unchanged)
        # ==============================================================
        patient_logger.info("Schritt 1: Daten laden")
        
        record, metadata = load_record(patient_dir, record_name=record_name)
        
        if record is None:
            patient_logger.error(f"Record konnte nicht geladen werden: {metadata}")
            stats.update("errors", f"Record laden fehlgeschlagen")
            result["stats"] = stats.get_summary()
            return result
        
        total_duration_sec = metadata["duration_sec"]
        sig_names = metadata["sig_name"]
        fs = metadata["fs"]
        
        patient_logger.info(f"Record geladen: {metadata['n_sig']} Kanäle, "
                            f"fs={fs} Hz, Dauer={total_duration_sec/60:.1f} min")
        patient_logger.info(f"Kanäle: {sig_names}")
        
        annotations = load_annotations(
            patient_dir, record_name=record_name, patient_id=patient_id
        )
        
        if annotations:
            patient_logger.info(f"Annotationen geladen: {list(annotations.keys())}")
        else:
            patient_logger.warning("Keine Annotationen gefunden.")
        
        # ==============================================================
        # SCHRITT 2: KANÄLE IDENTIFIZIEREN (unchanged)
        # ==============================================================
        patient_logger.info("Schritt 2: Kanäle identifizieren")
        
        ecg_idx = find_ecg_channel(sig_names)
        if ecg_idx is not None:
            patient_logger.info(f"ECG-Kanal gefunden: {sig_names[ecg_idx]} (Index {ecg_idx})")
        else:
            patient_logger.warning("Kein ECG-Kanal gefunden.")
        
        eeg_channels = find_eeg_channels(sig_names)
        if eeg_channels:
            patient_logger.info(f"EEG-Kanäle gefunden: {eeg_channels}")
            stats.update("eeg_channels_used", list(eeg_channels.keys()))
        else:
            if patient_logger:
                patient_logger.warning("Keine EEG-Kanäle gefunden.")
                
        # Re-Referencing
        record_reref, rereferenced = rereference_eeg_to_bipolar(record)

        if rereferenced:
            record = record_reref
            sig_names = record_reref.ch_names
            eeg_channels = []
            eeg_channels = find_eeg_channels(sig_names)


        # Respiration
        resp_channels = find_resp_channels(sig_names)
        if resp_channels and RSA_ENABLED:
            patient_logger.info(f"Respirations-Kanäle gefunden: {resp_channels}")
        
        # ==============================================================
        # SCHRITT 3: PREPROCESSING (unchanged)
        # ==============================================================
        patient_logger.info("Schritt 3: Preprocessing")

        if ecg_idx is not None:
            ecg_signal, ecg_fs = extract_signal(record, ecg_idx)
            ecg_preprocessed = preprocess_ecg_signal(
                ecg_signal, ecg_fs, logger=patient_logger
            )
            if ecg_preprocessed:
                ecg_preprocessed["fs"] = ecg_fs

        eeg_fs_value = None
        for channel_name, channel_idx in eeg_channels.items():
            eeg_signal, eeg_fs = extract_signal(record, channel_idx)
            eeg_fs_value = eeg_fs
            try:
                from preprocessing.preprocess_eeg import _bandpass_filter, _notch_filter
                from config import EEG_FILTER
                
                filtered = _bandpass_filter(
                    eeg_signal, eeg_fs,
                    lowcut=EEG_FILTER["lowcut"],
                    highcut=EEG_FILTER["highcut"],
                    order=EEG_FILTER["order"]
                )
                if EEG_FILTER.get("notch"):
                    filtered = _notch_filter(filtered, eeg_fs, freq=EEG_FILTER["notch"])
                
                eeg_filtered_signals[channel_name] = filtered
            except Exception as e:
                patient_logger.error(f"EEG [{channel_name}] Filterung fehlgeschlagen: {e}")

        from preprocessing.preprocess_eeg import detect_bad_channels, _detect_eeg_unit_scale
        scale_factor = _detect_eeg_unit_scale(eeg_filtered_signals, logger=patient_logger)
        globally_bad_channels = detect_bad_channels(
            raw_signals=eeg_filtered_signals,
            fs=eeg_fs_value if eeg_fs_value else 200.0,
            logger=patient_logger
        )

        for channel_name, channel_idx in eeg_channels.items():
            eeg_signal, eeg_fs = extract_signal(record, channel_idx)
            is_bad = channel_name in globally_bad_channels
            eeg_result = preprocess_eeg_signal(
                eeg_signal, eeg_fs,
                channel_name=channel_name,
                is_globally_bad=is_bad,
                scale_to_uv=scale_factor,
                logger=patient_logger
            )
            eeg_preprocessed[channel_name] = eeg_result

        if eeg_preprocessed:
            eeg_strategies = decide_eeg_channel_strategy(
                eeg_preprocessed, logger=patient_logger
            )
            for region, strategy in eeg_strategies.items():
                stats.update("eeg_strategy", f"{region}:{strategy['strategy']}")

        resp_signals = None
        if resp_channels and RSA_ENABLED:
            resp_signals = {}
            for resp_name, resp_idx in resp_channels.items():
                resp_sig, resp_fs = extract_signal(record, resp_idx)
                resp_signals[resp_name] = {
                    "signal": resp_sig,
                    "fs": resp_fs,
                }
        
        if annotations:
            annotation_data = process_all_annotations(
                annotations, total_duration_sec,
                segment_length_sec=segment_length_sec,
                logger=patient_logger
            )
            if annotation_data and annotation_data.get("sleep_summary"):
                summary = annotation_data["sleep_summary"]
                stats.update("total_sleep_duration_min",
                             summary.get("total_sleep_time_min", 0))
                stats.update("total_arousals",
                             summary.get("arousal_count", 0))
                stats.update("total_central_apneas",
                             summary.get("central_apnea_count", 0))
                stats.update("total_obstructive_apneas",
                             summary.get("obstructive_apnea_count", 0))
        
        # ==============================================================
        # SCHRITT 4: SEGMENTIERUNG (unchanged)
        # ==============================================================
        patient_logger.info("Schritt 4: Segmentierung")
        
        recording = segment_patient_recording(
            patient_id=patient_id,
            ecg_preprocessed=ecg_preprocessed,
            eeg_strategies=eeg_strategies,
            resp_signals=resp_signals,
            annotation_data=annotation_data,
            total_duration_sec=total_duration_sec,
            segment_length_sec=segment_length_sec,
            overlap_sec=overlap_sec,
            logger=patient_logger,
        )
        
        stats.update("total_segments", recording.n_segments)
        
        n_ecg_ok = sum(
            1 for s in recording.ecg_segments
            if s is not None and s.quality_ok
        )
        stats.update("valid_ecg_segments", n_ecg_ok)
        stats.update("rejected_ecg_segments",
                      recording.n_segments - n_ecg_ok)
        
        for region, segments in recording.eeg_segments.items():
            n_eeg_ok = sum(
                1 for s in segments
                if s is not None and s.quality_ok
            )
            stats.update("valid_eeg_segments", n_eeg_ok)
        
        print_segmentation_summary(recording, logger=patient_logger)
        
        # ==============================================================
        # NEW: FREE FULL-LENGTH SIGNALS AFTER SEGMENTATION
        # The segments now hold their own data slices.
        # We no longer need the full preprocessed signals.
        # ==============================================================
        # Free the raw record (largest single object)
        del record
        record = None
        
        # Free filtered EEG signals (no longer needed)
        del eeg_filtered_signals
        eeg_filtered_signals = {}
        
        # Free full-length cleaned signals from preprocessing results
        if ecg_preprocessed:
            ecg_preprocessed.pop("ecg_cleaned", None)
            ecg_preprocessed.pop("rpeaks", None)
        
        for ch_name in list(eeg_preprocessed.keys()):
            if eeg_preprocessed[ch_name]:
                eeg_preprocessed[ch_name].pop("eeg_cleaned", None)
        
        gc.collect()
        patient_logger.info("Speicher freigegeben: Vollständige Signale nach Segmentierung entfernt.")
        
        # ==============================================================
        # SCHRITT 5: FEATURE-EXTRAKTION (unchanged)
        # ==============================================================
        patient_logger.info("Schritt 5: Feature-Extraktion")
        
        ecg_features = extract_ecg_features_all_segments(
            ecg_segments=recording.ecg_segments,
            resp_segments=recording.resp_segments if RSA_ENABLED else None,
            logger=patient_logger,
        )
        
        eeg_features = extract_eeg_features_all_segments(
            eeg_segments=recording.eeg_segments,
            eeg_strategies=eeg_strategies,
            logger=patient_logger,
        )
        
        sleep_summary = (
            annotation_data.get("sleep_summary")
            if annotation_data else None
        )
        
        annotation_features = extract_annotation_features(
            stages_per_segment=recording.stages_per_segment,
            events_per_segment=recording.events_per_segment,
            sleep_summary=sleep_summary,
            total_duration_sec=total_duration_sec,
            segment_length_sec=segment_length_sec,
            logger=patient_logger,
        )

        # --- CAP Features ---
        cap_features = {}

        try:
            cap_eeg_channel = None
            if eeg_preprocessed:
                cap_eeg_channel = next(iter(eeg_preprocessed.keys()))
            if cap_eeg_channel:
                cap_eeg_signal = eeg_preprocessed[cap_eeg_channel]["eeg_cleaned"]
                #cap_fs = eeg_preprocessed[cap_eeg_channel]["fs"]
                cap_fs = eeg_fs
                cap_event = annotation_data["stages_raw"].stage_numeric if annotation_data and "stages_raw" in annotation_data else None
                cap_duration = annotation_data["stages_raw"].duration_sec if annotation_data and "stages_raw" in annotation_data else None
                cap_eventtime = annotation_data["stages_raw"].start_sec if annotation_data and "stages_raw" in annotation_data else None
                if cap_event is not None and cap_duration is not None and cap_eventtime is not None:
                    cap_features = extract_cap_features_for_patient(
                        cap_eeg_signal, cap_fs, cap_event, cap_duration, cap_eventtime, patient_id,
                        logger=patient_logger
                    )
        except Exception as e:
            if patient_logger:
                patient_logger.warning(f"[CAP-Integration] CAP-Feature-Extraktion übersprungen: {e}")


        
        # ==============================================================
        # SCHRITT 6: FEATURE-TABELLE ERSTELLEN (unchanged logic)
        # ==============================================================
        patient_logger.info("Schritt 6: Feature-Tabelle erstellen")
        
        segment_features = build_patient_feature_table(
            patient_id=patient_id,
            ecg_features=ecg_features,
            eeg_features=eeg_features,
            annotation_features=annotation_features,
            segment_metadata=recording.segment_metadata,
            sleep_summary=sleep_summary,
            logger=patient_logger,
        )
        
        patient_features = aggregate_to_patient_level(
            segment_features, sleep_summary, logger=patient_logger
        )
        result["segment_features"] = segment_features
        result["patient_features"] = patient_features
        
        # ==============================================================
        # NEW: SAVE FEATURES TO DISK INSTEAD OF RETURNING THEM
        # ==============================================================
        safe_id = patient_id.replace("/", "_").replace("\\", "_")
        seg_path = feature_output_dir / f"{safe_id}_segment_features.parquet"
        pat_path = feature_output_dir / f"{safe_id}_patient_features.parquet"
        
        if segment_features is not None and len(segment_features) > 0:
            segment_features.to_parquet(seg_path, engine="pyarrow", index=False)
            result["seg_features_path"] = str(seg_path)
            patient_logger.info(f"Segment-Features gespeichert: {seg_path} "
                                f"({segment_features.shape})")
        
        if patient_features is not None and len(patient_features) > 0:
            patient_features.to_parquet(pat_path, engine="pyarrow", index=False)
            result["pat_features_path"] = str(pat_path)
            patient_logger.info(f"Patient-Features gespeichert: {pat_path} "
                                f"({patient_features.shape})")
        
        # Store only the lightweight sleep_summary dict (small)
        result["sleep_summary"] = sleep_summary
        result["success"] = True
        
        # Zeitstatistik
        elapsed = time.time() - start_time
        
        patient_logger.info(f"Patient {patient_id} abgeschlossen in {elapsed:.1f}s")
        patient_logger.info(f"  Segmente: {recording.n_segments}")
        patient_logger.info(f"  Segment-Features: {segment_features.shape if segment_features is not None else 'N/A'}")
        patient_logger.info(f"  Patient-Features: {patient_features.shape if patient_features is not None else 'N/A'}")
        
        stats.log_summary(patient_logger)
        result["stats"] = stats.get_summary()
        
        # ==============================================================
        # SCHRITT 7: VISUALISIERUNG (optional)
        # NOTE: We still have ecg_preprocessed (without ecg_cleaned)
        # and eeg_preprocessed (without eeg_cleaned). If you need
        # the signals for plotting, you must do it BEFORE the cleanup
        # block above. For now, we skip signal-based plots in workers.
        # ==============================================================
        if PLOT_ENABLED and PLOT_PER_PATIENT:
            patient_logger.info("Schritt 7: Visualisierung")
            try:
                sqi_ecg = None
                if ecg_preprocessed and ecg_preprocessed.get("sqi_per_segment") is not None:
                    sqi_ecg = ecg_preprocessed["sqi_per_segment"]
                
                sqi_eeg = {}
                if eeg_preprocessed:
                    for ch_name, ch_data in eeg_preprocessed.items():
                        if ch_data and ch_data.get("sqi_per_segment") is not None:
                            sqi_eeg[ch_name] = ch_data["sqi_per_segment"]
                
                # NOTE: eeg_signals_for_plot is empty because we freed
                # eeg_cleaned above. Signal-based plots are skipped.
                eeg_signals_for_plot = {}
                
                stages_raw = None
                events_raw = None
                if annotation_data:
                    stages_raw = annotation_data.get("stages_raw")
                    events_raw = annotation_data.get("events_raw")
                
                safe_patient_id = patient_id.replace("/", "_")
                patient_plot_dir = PLOT_DIR / safe_patient_id
                plot_paths = generate_all_patient_plots(
                    patient_id=safe_patient_id,
                    segment_features=segment_features,
                    ecg_preprocessed=ecg_preprocessed,
                    eeg_signals=eeg_signals_for_plot if eeg_signals_for_plot else None,
                    eeg_strategies=eeg_strategies,
                    stages_df=stages_raw,
                    events_df=events_raw,
                    sleep_summary=sleep_summary,
                    sqi_ecg=sqi_ecg,
                    sqi_eeg=sqi_eeg if sqi_eeg else None,
                    output_dir=patient_plot_dir,
                    logger=patient_logger,
                )
                result["plot_paths"] = plot_paths
                patient_logger.info(f"Visualisierung: {len(plot_paths)} Plots erstellt")
            except Exception as e:
                patient_logger.warning(f"Visualisierung fehlgeschlagen: {e}")
        
        # ==============================================================
        # NEW: EXPLICIT CLEANUP BEFORE RETURNING
        # ==============================================================
        del segment_features, patient_features
        del ecg_features, eeg_features, annotation_features
        del recording, ecg_preprocessed, eeg_preprocessed
        del annotation_data, eeg_strategies, resp_signals
        gc.collect()
        
    except Exception as e:
        patient_logger.error(f"Fehler bei Patient {patient_id}: {e}")
        patient_logger.error(traceback.format_exc())
        stats.update("errors", str(e))
        result["stats"] = stats.get_summary()
    
    return result




def run_preprocessing_pipeline(
    patient_list=None, max_patients=None,
    segment_length_sec=SEGMENT_LENGTH_SEC,
    overlap_sec=SEGMENT_OVERLAP_SEC
):
    """
    Führt die Preprocessing- und Feature-Extraktions-Pipeline
    für alle Patienten durch.
    
    CHANGED: Workers save features to disk. Main process loads them
    after all workers complete, controlling peak memory usage.
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    if patient_list is None:
        patient_list = get_patient_list(DATA_DIR)
    
    if max_patients is not None:
        patient_list = patient_list[:max_patients]
    
    logger.info(f"Patienten zu verarbeiten: {len(patient_list)}")
    
    # CHANGED: collect paths and summaries instead of DataFrames
    successful_results = {}   # patient_id -> {seg_path, pat_path, sleep_summary}
    patient_stats = []
    
    total_start = time.time()
    
    max_workers = min(NUM_WORKERS, len(patient_list)) if len(patient_list) > 0 else 1
    logger.info(f"Starte Verarbeitung mit {max_workers} parallelen Workern.")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, patient_id in enumerate(patient_list):
            if "/" in patient_id:
                site_id, record_name = patient_id.split("/", 1)
                patient_dir = DATA_DIR / site_id
            else:
                patient_dir = DATA_DIR / patient_id
                record_name = None
            
            if not patient_dir.exists():
                logger.warning(f"Verzeichnis nicht gefunden: {patient_dir}")
                continue
            
            future = executor.submit(
                process_single_patient,
                patient_id=patient_id,
                patient_dir=patient_dir,
                record_name=record_name,
                segment_length_sec=segment_length_sec,
                overlap_sec=overlap_sec,
                feature_output_dir=FEATURE_DIR,  # NEW parameter
            )
            futures[future] = patient_id

        for i, future in enumerate(as_completed(futures)):
            patient_id = futures[future]
            try:
                result = future.result()
                
                # CHANGED: collect paths, not DataFrames
                if result and result["success"]:
                    successful_results[patient_id] = {
                        "seg_features_path": result.get("seg_features_path"),
                        "pat_features_path": result.get("pat_features_path"),
                        "sleep_summary": result.get("sleep_summary"),
                    }
                
                if result and result.get("stats"):
                    patient_stats.append(result["stats"])
            
            except Exception as e:
                logger.error(f"Fehler bei Patient {patient_id}: {e}")
                logger.error(traceback.format_exc())
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - total_start
                rate = (i + 1) / elapsed
                remaining = (len(patient_list) - i - 1) / rate if rate > 0 else 0
                logger.info(f"Fortschritt: {i+1}/{len(patient_list)} "
                            f"({elapsed:.0f}s, ~{remaining:.0f}s verbleibend)")
    
    total_elapsed = time.time() - total_start
    
    n_success = len(successful_results)
    logger.info(f"\nPreprocessing abgeschlossen:")
    logger.info(f"  Erfolgreich: {n_success}/{len(patient_list)}")
    if len(patient_list) > 0:
        logger.info(f"  Gesamtzeit: {total_elapsed:.1f}s "
                     f"({total_elapsed/len(patient_list):.1f}s/Patient)")
    else:
        logger.info(f"  Gesamtzeit: {total_elapsed:.1f}s")
        logger.warning("Keine Patienten gefunden! Prüfe das Datenverzeichnis.")
    
    # Pipeline-Statistiken speichern
    if patient_stats:
        stats_df = pd.DataFrame(patient_stats)
        stats_path = OUTPUT_DIR / "pipeline_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Pipeline-Statistiken gespeichert: {stats_path}")
    
    # ==================================================================
    # NEW: LOAD FEATURES FROM DISK (sequential, memory-controlled)
    # ==================================================================
    logger.info("Lade gespeicherte Features von Disk...")
    
    patient_segment_tables = {}
    patient_sleep_summaries = {}
    
    for patient_id, paths in successful_results.items():
        # Load segment features
        seg_path = paths.get("seg_features_path")
        if seg_path and Path(seg_path).exists():
            try:
                patient_segment_tables[patient_id] = pd.read_parquet(seg_path)
            except Exception as e:
                logger.error(f"Fehler beim Laden von {seg_path}: {e}")
        
        # Collect sleep summaries
        if paths.get("sleep_summary"):
            patient_sleep_summaries[patient_id] = paths["sleep_summary"]
    
    logger.info(f"Features geladen: {len(patient_segment_tables)} Patienten")
    
    # Free the paths dict (no longer needed)
    del successful_results
    gc.collect()
    
    # ==================================================================
    # REST: Build cohort table (unchanged logic)
    # ==================================================================
    segment_level, patient_level = pd.DataFrame(), pd.DataFrame()

    if patient_segment_tables:
        segment_level, patient_level = build_cohort_feature_table(
            patient_segment_tables=patient_segment_tables,
            patient_sleep_summaries=patient_sleep_summaries,
            demographics_path=DEMOGRAPHICS_FILE,
            output_dir=FEATURE_DIR,
            save=True,
            logger=logger
        )
    
    # Kohorten-Visualisierungen (unchanged)
    if PLOT_ENABLED and PLOT_COHORT and len(patient_level) > 0:
        logger.info("Erstelle Kohorten-Visualisierungen...")
        try:
            cohort_plot_paths = generate_all_cohort_plots(
                patient_level_features=patient_level,
                segment_level_features=segment_level,
                target_col="target",
                output_dir=PLOT_DIR / "cohort",
                logger=logger
            )
            if logger:
                logger.info(f"Kohorten-Plots erstellt: {len(cohort_plot_paths)}")
        except Exception as e:
            if logger:
                logger.warning(f"Kohorten-Visualisierung fehlgeschlagen: {e}")
    
    return segment_level, patient_level



def run_training_pipeline(
    patient_level_features: Optional[pd.DataFrame] = None,
    model_types: Optional[List[str]] = None,
    tune: bool = False
) -> Dict:
    """
    Führt die Trainings-Pipeline durch.
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Features laden falls nicht übergeben
    if patient_level_features is None:
        patient_level_features = load_and_merge_saved_features(
            FEATURE_DIR, level="patient", logger=logger
        )
    
    if patient_level_features is None or len(patient_level_features) == 0:
        logger.error("Keine Patient-Level Features verfügbar!")
        return {}
    
    logger.info(f"Feature-Tabelle: {patient_level_features.shape}")
    
    # Multi-Model Training
    if model_types is None:
        model_types = ["xgboost", "lightgbm", "random_forest", "ensemble"]
    
    results = train_multiple_models(
        feature_table=patient_level_features,
        model_types=model_types,
        handle_imbalance="class_weight",
        feature_selection=True,
        tune=tune,
        output_dir=MODEL_DIR,
        logger=logger,
    )
    
    return results


def run_evaluation_pipeline(
    patient_level_features: Optional[pd.DataFrame] = None,
    model_result: Optional[Dict] = None,
) -> Dict:
    """
    Führt die Evaluations-Pipeline durch.
    """
    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE")
    logger.info("=" * 60)
    
    # Features laden
    if patient_level_features is None:
        patient_level_features = load_and_merge_saved_features(
            FEATURE_DIR, level="patient", logger=logger
        )
    
    if patient_level_features is None or len(patient_level_features) == 0:
        logger.error("Keine Features für Evaluation verfügbar!")
        return {}
    
    # Modell laden
    if model_result is None:
        model_result = load_model(logger=logger)
    
    if model_result is None:
        logger.error("Kein Modell für Evaluation verfügbar!")
        return {}
    
    # CV-Evaluation
    eval_results = evaluate_with_cross_validation(
        feature_table=patient_level_features,
        model_result=model_result,
        output_dir=OUTPUT_DIR / "evaluation",
        logger=logger,
    )
    
    # Zusätzliche Visualisierungen nach Evaluation
    if PLOT_ENABLED and eval_results:
        try:
            from utils.visualization import plot_feature_comparison_ci
            
            if patient_level_features is not None:
                plot_feature_comparison_ci(
                    patient_level_features,
                    target_col="target",
                    output_dir=OUTPUT_DIR / "evaluation" / "plots",
                )
        except Exception:
            pass
    
    return eval_results


# ==============================================================================
# VOLLSTÄNDIGE PIPELINE
# ==============================================================================

def run_full_pipeline(
    max_patients: Optional[int] = None,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    model_types: Optional[List[str]] = None,
    tune: bool = False
) -> Dict:
    """
    Führt die vollständige Pipeline durch:
    Preprocessing -> Feature-Extraktion -> Training -> Evaluation.
    """
    pipeline_start = time.time()
    
    logger.info("=" * 70)
    logger.info("PHYSIONET CHALLENGE 2026 – VOLLSTÄNDIGE PIPELINE")
    logger.info("Screening for Cognitive Impairment During Sleep Studies")
    logger.info(f"Zeitstempel: {datetime.now().isoformat()}")
    logger.info("=" * 70)
    
    results = {
        "preprocessing": None,
        "training": None,
        "evaluation": None,
        "total_time_sec": 0,
    }
    
    # --- Schritt 1: Preprocessing & Feature-Extraktion ---
    logger.info("\n" + "▶" * 30)
    logger.info("PHASE 1: PREPROCESSING & FEATURE-EXTRAKTION")
    logger.info("▶" * 30)
    
    segment_level, patient_level = run_preprocessing_pipeline(
        max_patients=max_patients,
        segment_length_sec=segment_length_sec,
    )
    
    results["preprocessing"] = {
        "n_patients": patient_level["patient_id"].nunique() if len(patient_level) > 0 else 0,
        "n_segments": len(segment_level),
        "n_features_segment": len(segment_level.columns) if len(segment_level) > 0 else 0,
        "n_features_patient": len(patient_level.columns) if len(patient_level) > 0 else 0,
    }
    
    if len(patient_level) == 0:
        logger.error("Preprocessing hat keine Ergebnisse geliefert. Abbruch.")
        return results
    
    # --- Schritt 2: Training ---
    logger.info("\n" + "▶" * 30)
    logger.info("PHASE 2: MODELL-TRAINING")
    logger.info("▶" * 30)
    
    training_results = run_training_pipeline(
        patient_level_features=patient_level,
        model_types=model_types,
        tune=tune,
    )
    results["training"] = {
        model_type: {
            "auroc": r.get("cv_results", {}).get("auroc_mean", None),
            "f1": r.get("cv_results", {}).get("f1_mean", None),
        }
        for model_type, r in training_results.items()
        if isinstance(r, dict) and "cv_results" in r
    }
    
    # --- Schritt 3: Evaluation ---
    logger.info("\n" + "▶" * 30)
    logger.info("PHASE 3: EVALUATION")
    logger.info("▶" * 30)

    # --- Schritt 4: Finale Visualisierungen ---
    if PLOT_ENABLED:
        logger.info("\n" + "▶" * 30)
        logger.info("PHASE 4: FINALE VISUALISIERUNGEN")
        logger.info("▶" * 30)
        
        try:
            # Kohorten-Plots (falls nicht schon in Preprocessing erstellt)
            cohort_plot_dir = PLOT_DIR / "cohort"
            if not (cohort_plot_dir / "cohort_overview.png").exists():
                generate_all_cohort_plots(
                    patient_level_features=patient_level,
                    segment_level_features=segment_level,
                    target_col="target",
                    output_dir=cohort_plot_dir,
                    logger=logger,
                )
            
            # Evaluation-Plots sind bereits in evaluate_model.py integriert
            
            # Zähle alle erstellten Plots
            n_plots = sum(
                1 for p in PLOT_DIR.rglob(f"*.{PLOT_FORMAT}")
            )
            logger.info(f"Gesamtanzahl Plots: {n_plots}")
        
        except Exception as e:
            logger.warning(f"Finale Visualisierung fehlgeschlagen: {e}")

    
    # Bestes Modell für Evaluation auswählen
    best_model_type = None
    best_auroc = 0
    for model_type, r in training_results.items():
        if isinstance(r, dict) and "cv_results" in r:
            auroc = r["cv_results"].get("auroc_mean", 0)
            if auroc > best_auroc:
                best_auroc = auroc
                best_model_type = model_type
    
    if best_model_type and best_model_type in training_results:
        eval_results = run_evaluation_pipeline(
            patient_level_features=patient_level,
            model_result=training_results[best_model_type],
        )
        results["evaluation"] = eval_results
    
    # --- Zusammenfassung ---
    total_time = time.time() - pipeline_start
    results["total_time_sec"] = total_time
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE ABGESCHLOSSEN")
    logger.info("=" * 70)
    logger.info(f"Gesamtzeit: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Patienten: {results['preprocessing']['n_patients']}")
    logger.info(f"Bestes Modell: {best_model_type} "
                 f"(AUROC={best_auroc:.4f})")
    
    if results.get("evaluation"):
        cm = results["evaluation"].get("challenge_metrics", {})
        logger.info(f"\nFinale Challenge-Metriken:")
        logger.info(f"  AUROC:     {cm.get('auroc', 'N/A')}")
        logger.info(f"  AUPRC:     {cm.get('auprc', 'N/A')}")
        logger.info(f"  Accuracy:  {cm.get('accuracy', 'N/A')}")
        logger.info(f"  F-Measure: {cm.get('f_measure', 'N/A')}")
    
    logger.info("=" * 70)
    
    return results


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

def parse_arguments():
    """Parst Kommandozeilen-Argumente."""
    parser = argparse.ArgumentParser(
        description="PhysioNet Challenge 2026 – "
                    "Cognitive Impairment Screening Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py                              # Vollständige Pipeline
  python main.py --step preprocess            # Nur Preprocessing
  python main.py --step train                 # Nur Training
  python main.py --step evaluate              # Nur Evaluation
  python main.py --patients 10                # Erste 10 Patienten
  python main.py --segment-length 300         # 5-Minuten Segmente
  python main.py --segment-length 30          # 30-Sekunden Segmente
  python main.py --models xgboost lightgbm    # Bestimmte Modelle
  python main.py --tune                       # Mit Hyperparameter-Tuning
        """
    )
    
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "preprocess", "features", "train", "evaluate"],
        default="all",
        help="Pipeline-Schritt (default: all)"
    )
    
    parser.add_argument(
        "--patients",
        type=int,
        default=None,
        help="Maximale Anzahl Patienten (default: alle)"
    )
    
    parser.add_argument(
        "--patient-ids",
        type=str,
        nargs="+",
        default=None,
        help="Spezifische Patienten-IDs"
    )
    
    parser.add_argument(
        "--segment-length",
        type=float,
        default=SEGMENT_LENGTH_SEC,
        help=f"Segmentlänge in Sekunden (default: {SEGMENT_LENGTH_SEC})"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=SEGMENT_OVERLAP_SEC,
        help=f"Segment-Überlappung in Sekunden (default: {SEGMENT_OVERLAP_SEC})"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Modelltypen für Training "
             "(default: xgboost lightgbm random_forest ensemble)"
    )
    
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Hyperparameter-Tuning aktivieren"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Datenverzeichnis (default: {DATA_DIR})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Ausgabeverzeichnis (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Keine Visualisierungen erstellen"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ausführliches Logging (DEBUG-Level)"
    )

    
    parser.add_argument(
        "--plot-format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Plot-Format (default: png)"
    )
    
    parser.add_argument(
        "--plot-patients",
        type=int,
        default=None,
        help=f"Max. Patienten für Einzelplots (default: {PLOT_MAX_PATIENTS})"
    )

    
    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Hauptfunktion – Einstiegspunkt der Pipeline."""
    global logger
    
    # --- Argumente parsen ---
    args = parse_arguments()
    
    # --- Konfiguration überschreiben falls nötig ---
    # --- Konfiguration überschreiben, BEVOR der Logger eingerichtet wird ---
    if args.data_dir:
        import config
        config.DATA_DIR = Path(args.data_dir)
    
    if args.output_dir:
        import config
        config.OUTPUT_DIR = Path(args.output_dir)
        config.FEATURE_DIR = config.OUTPUT_DIR / "features"
        config.MODEL_DIR = config.OUTPUT_DIR / "models"
        config.LOG_DIR = config.OUTPUT_DIR / "logs"
        for d in [config.OUTPUT_DIR, config.FEATURE_DIR,
                   config.MODEL_DIR, config.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        import config
        config.LOG_LEVEL = "DEBUG"

        # --- Visualisierungs-Konfiguration ---
    if args.no_plots:
        import config
        config.PLOT_ENABLED = False
    
    if args.plot_format:
        import config
        config.PLOT_FORMAT = args.plot_format
    
    if args.plot_patients is not None:
        import config
        config.PLOT_MAX_PATIENTS = args.plot_patients

    
    # --- Logger einrichten und binden ---
    # Muss nach dem Parsen der Argumente erfolgen, um LOG_LEVEL zu berücksichtigen.
    logger = setup_logger(module_name="main")
    logger = logger.bind(module="main", patient_id="GLOBAL")
    
    # --- Jetzt kann geloggt werden ---
    if args.no_plots:
        logger.info("Visualisierungen deaktiviert (--no-plots)")

    logger.info("=" * 70)
    logger.info("PhysioNet Challenge 2026")
    logger.info("Screening for Cognitive Impairment During Sleep Studies")
    logger.info(f"Zeitstempel: {datetime.now().isoformat()}")
    logger.info("=" * 70)
    logger.info(f"Konfiguration:")
    logger.info(f"  Schritt:          {args.step}")
    logger.info(f"  Datenverzeichnis: {DATA_DIR}")
    logger.info(f"  Ausgabe:          {OUTPUT_DIR}")
    logger.info(f"  Segmentlänge:     {args.segment_length}s")
    logger.info(f"  Überlappung:      {args.overlap}s")
    logger.info(f"  Max Patienten:    {args.patients or 'alle'}")
    logger.info(f"  Modelle:          {args.models or 'Standard'}")
    logger.info(f"  Tuning:           {args.tune}")
    logger.info(f"  Verbose:          {args.verbose}")
    logger.info("=" * 70)
    
    # --- Patienten-Liste ---
    patient_list = None
    if args.patient_ids:
        patient_list = args.patient_ids
        logger.info(f"Spezifische Patienten: {patient_list}")
    
    # --- Pipeline ausführen ---
    try:
        if args.step == "all":
            # Vollständige Pipeline
            results = run_full_pipeline(
                max_patients=args.patients,
                segment_length_sec=args.segment_length,
                model_types=args.models,
                tune=args.tune
            )
            _print_final_summary(results, logger)
        
        elif args.step == "preprocess":
            # Nur Preprocessing & Feature-Extraktion
            segment_level, patient_level = run_preprocessing_pipeline(
                patient_list=patient_list,
                max_patients=args.patients,
                segment_length_sec=args.segment_length,
                overlap_sec=args.overlap
            )
            
            logger.info(f"\nPreprocessing abgeschlossen:")
            if len(segment_level) > 0:
                logger.info(f"  Segment-Level: {segment_level.shape}")
            if len(patient_level) > 0:
                logger.info(f"  Patient-Level: {patient_level.shape}")
        
        elif args.step == "features":
            # Feature-Tabelle aus gespeicherten Daten laden und anzeigen
            segment_level = load_and_merge_saved_features(
                FEATURE_DIR, level="segment", logger=logger
            )
            patient_level = load_and_merge_saved_features(
                FEATURE_DIR, level="patient", logger=logger
            )
            
            if segment_level is not None:
                logger.info(f"Segment-Level Features: {segment_level.shape}")
                logger.info(f"  Patienten: {segment_level['patient_id'].nunique()}")
                logger.info(f"  Spalten: {list(segment_level.columns[:20])}...")
            
            if patient_level is not None:
                logger.info(f"Patient-Level Features: {patient_level.shape}")
                
                # Target-Verteilung
                for target_col in ["target", TARGET_COLUMN, "Cognitive_Impairment"]:
                    if target_col in patient_level.columns:
                        dist = patient_level[target_col].value_counts(dropna=False)
                        logger.info(f"  Target-Verteilung ({target_col}):")
                        for val, count in dist.items():
                            logger.info(f"    {val}: {count}")
                        break
        
        elif args.step == "train":
            # Nur Training
            results = run_training_pipeline(
                model_types=args.models,
                tune=args.tune
            )
            
            if results:
                logger.info("\nTraining abgeschlossen:")
                for model_type, r in results.items():
                    if isinstance(r, dict) and "cv_results" in r:
                        cv = r["cv_results"]
                        logger.info(
                            f"  {model_type}: "
                            f"AUROC={cv.get('auroc_mean', 0):.4f} ± "
                            f"{cv.get('auroc_std', 0):.4f}"
                        )
        
        elif args.step == "evaluate":
            # Nur Evaluation
            results = run_evaluation_pipeline()
            
            if results:
                cm = results.get("challenge_metrics", {})
                logger.info("\nEvaluation abgeschlossen:")
                logger.info(f"  AUROC:     {cm.get('auroc', 'N/A')}")
                logger.info(f"  AUPRC:     {cm.get('auprc', 'N/A')}")
                logger.info(f"  Accuracy:  {cm.get('accuracy', 'N/A')}")
                logger.info(f"  F-Measure: {cm.get('f_measure', 'N/A')}")
    
    except KeyboardInterrupt:
        logger.warning("\nPipeline durch Benutzer abgebrochen (Ctrl+C)")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\nFataler Fehler in der Pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("\nPipeline beendet.")
    


def _print_final_summary(results: Dict, logger):
    """Gibt eine finale Zusammenfassung der gesamten Pipeline aus."""
    logger.info("\n" + "█" * 70)
    logger.info("█" + " " * 68 + "█")
    logger.info("█" + "  FINALE ZUSAMMENFASSUNG".center(68) + "█")
    logger.info("█" + " " * 68 + "█")
    logger.info("█" * 70)
    
    # Preprocessing
    prep = results.get("preprocessing", {})
    if prep:
        logger.info(f"\n  📊 DATEN:")
        logger.info(f"     Patienten:           {prep.get('n_patients', 'N/A')}")
        logger.info(f"     Segmente:            {prep.get('n_segments', 'N/A')}")
        logger.info(f"     Features (Segment):  {prep.get('n_features_segment', 'N/A')}")
        logger.info(f"     Features (Patient):  {prep.get('n_features_patient', 'N/A')}")
    
    # Training
    training = results.get("training", {})
    if training:
        logger.info(f"\n  🤖 MODELLE:")
        best_model = None
        best_auroc = 0
        for model_type, metrics in training.items():
            if isinstance(metrics, dict):
                auroc = metrics.get("auroc", 0) or 0
                f1 = metrics.get("f1", 0) or 0
                logger.info(f"     {model_type:<20} AUROC={auroc:.4f}  F1={f1:.4f}")
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_model = model_type
        
        if best_model:
            logger.info(f"\n     🏆 Bestes Modell: {best_model} (AUROC={best_auroc:.4f})")
    
    # Evaluation
    evaluation = results.get("evaluation", {})
    if evaluation:
        cm = evaluation.get("challenge_metrics", {})
        logger.info(f"\n  📈 CHALLENGE-METRIKEN:")
        logger.info(f"     AUROC:               {cm.get('auroc', 'N/A')}")
        logger.info(f"     AUPRC:               {cm.get('auprc', 'N/A')}")
        logger.info(f"     Accuracy:            {cm.get('accuracy', 'N/A')}")
        logger.info(f"     F-Measure:           {cm.get('f_measure', 'N/A')}")
        
        score = cm.get("challenge_score")
        if score is not None:
            logger.info(f"     Challenge Score:     {score:.4f}")
        
        # Konfidenzintervalle
        ci = evaluation.get("confidence_intervals", {})
        if "auroc" in ci:
            auroc_ci = ci["auroc"]
            logger.info(f"\n     AUROC 95% CI:        "
                        f"[{auroc_ci['ci_lower']:.4f}, "
                        f"{auroc_ci['ci_upper']:.4f}]")
    
    # Zeitstatistik
    total_time = results.get("total_time_sec", 0)
    logger.info(f"\n  ⏱️  Gesamtzeit:          {total_time:.1f}s "
                f"({total_time/60:.1f} min)")
    
    # Ausgabeverzeichnisse
    logger.info(f"\n  📁 AUSGABE:")
    logger.info(f"     Features:  {FEATURE_DIR}")
    logger.info(f"     Modelle:   {MODEL_DIR}")
    logger.info(f"     Logs:      {LOG_DIR}")
    
    logger.info("\n" + "█" * 70)

    # Plots
    if PLOT_ENABLED:
        n_plots = sum(1 for _ in PLOT_DIR.rglob(f"*.png")) if PLOT_DIR.exists() else 0
        logger.info(f"     Plots:     {PLOT_DIR} ({n_plots} Dateien)")



# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
