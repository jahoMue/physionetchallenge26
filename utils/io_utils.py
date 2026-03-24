"""
utils/io_utils.py
=================
Utility-Funktionen zum Laden und Speichern von Daten.
Unterstützt EDF-Dateien und die PhysioNet Challenge 2026 Datenstruktur.

Datenstruktur:
training_set/
├── physiological_data/
│   ├── S0001/
│   │   ├── sub-S0001xxx_ses-1.edf
│   │   └── ...
│   ├── I0002/
│   └── I0006/
├── algorithmic_annotations/
│   ├── S0001/
│   │   ├── sub-S0001xxx_ses-1_caisr_annotations.edf
│   │   └── ...
│   ├── I0002/
│   └── I0006/
├── human_annotations/
│   ├── S0001/
│   │   ├── sub-S0001xxx_ses-1_expert_annotations.edf
│   │   ├── sub-S0001xxx_ses-1_expert_annotations.annot
│   │   └── ...
│   ├── I0002/
│   └── I0006/
└── demographics.csv
"""
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import mne

from config import (
    DATA_DIR, DEMOGRAPHICS_FILE,
    ALGORITHMIC_ANNOTATIONS_DIR, HUMAN_ANNOTATIONS_DIR,
    ECG_CHANNEL_NAMES, EEG_CHANNEL_MAPPING,
    RESP_CHANNEL_NAMES, SLEEP_STAGE_ENCODING,
    SIGNAL_DTYPE,
)


# ==============================================================================
# DEMOGRAPHICS
# ==============================================================================

def load_demographics() -> pd.DataFrame:
    """Lädt die Demographics-Datei."""
    df = pd.read_csv(DEMOGRAPHICS_FILE)
    return df


# ==============================================================================
# PATIENT / RECORD LIST
# ==============================================================================

def get_patient_list(data_dir: Path = DATA_DIR) -> List[str]:
    """
    Ermittelt alle verfügbaren Records.

    Returns
    -------
    List[str]
        Liste von "site_id/record_name" Strings.
        record_name ist der Dateiname OHNE .edf Extension.
    """
    record_list = []

    for site_dir in sorted(data_dir.iterdir()):
        if not site_dir.is_dir() or site_dir.name.startswith('.'):
            continue

        edf_files = sorted(site_dir.glob("*.edf"))

        for edf_file in edf_files:
            record_name = edf_file.stem
            record_id = f"{site_dir.name}/{record_name}"
            record_list.append(record_id)

    return record_list


# ==============================================================================
# RECORD LADEN (EDF)
# ==============================================================================

def load_record(
    patient_dir: Path,
    record_name: str = None
) -> Tuple[Optional[mne.io.Raw], Optional[dict]]:
    """
    Lädt einen EDF-Record.

    Parameters
    ----------
    patient_dir : Path
        Pfad zum Site-Verzeichnis.
    record_name : str, optional
        Name des Records (ohne .edf Extension).
        Falls None, wird die erste .edf-Datei verwendet.

    Returns
    -------
    Tuple[mne.io.Raw, dict]
        MNE Raw-Objekt und Metadaten-Dictionary.
    """
    if record_name is None:
        edf_files = sorted(patient_dir.glob("*.edf"))
        if not edf_files:
            return None, {"error": f"No EDF files found in {patient_dir}"}
        record_name = edf_files[0].stem

    edf_path = patient_dir / f"{record_name}.edf"

    if not edf_path.exists():
        return None, {"error": f"EDF file not found: {edf_path}"}

    try:
        raw = mne.io.read_raw_edf(
            str(edf_path),
            preload=True,
            verbose=False
        )

        # ==============================================================
        # NEW: Convert MNE's internal data buffer to float32
        # MNE stores data as float64 internally. For PSG signals with
        # 12-16 bit ADC resolution, float32 (24-bit mantissa) is more
        # than sufficient. This halves the memory of the Raw object.
        # ==============================================================
        raw._data = raw._data.astype(SIGNAL_DTYPE)

        metadata = {
            "record_name": record_name,
            "n_sig": len(raw.ch_names),
            "fs": raw.info['sfreq'],
            "sig_len": raw.n_times,
            "sig_name": [s.lower().strip() for s in raw.ch_names],
            "sig_name_original": list(raw.ch_names),
            "units": [
                raw.info['chs'][i].get('unit_mul', 0)
                for i in range(len(raw.ch_names))
            ],
            "duration_sec": raw.n_times / raw.info['sfreq'],
            "file_path": str(edf_path),
        }
        return raw, metadata

    except Exception as e:
        return None, {"error": str(e)}



# ==============================================================================
# ANNOTATIONEN LADEN
# ==============================================================================

def load_annotations(
    patient_dir: Path,
    record_name: str = None,
    patient_id: str = None
) -> Optional[Dict]:
    """
    Lädt Schlaf-Annotationen aus den separaten Annotations-Verzeichnissen.
    """
    annotations = {}

    # Bestimme site_id und record_name
    if patient_id is not None and "/" in patient_id:
        site_id, rec_name = patient_id.split("/", 1)
    else:
        site_id = patient_dir.name
        rec_name = record_name

    if rec_name is None:
        edf_files = sorted(patient_dir.glob("*.edf"))
        if edf_files:
            rec_name = edf_files[0].stem
        else:
            return None

    # --- Human Annotations (EDF + .annot) ---
    human_dir = HUMAN_ANNOTATIONS_DIR / site_id
    if human_dir.exists():
        for edf_file in sorted(human_dir.glob(f"{rec_name}*.edf")):
            ann_key = f"human_{edf_file.stem}"
            try:
                ann_df = _load_annotations_from_edf(edf_file)
                if ann_df is not None and len(ann_df) > 0:
                    annotations[ann_key] = ann_df
            except Exception as e:
                print(f"[DEBUG] Failed to load human EDF annotation {edf_file.name}: {e}")

        for annot_file in sorted(human_dir.glob(f"{rec_name}*.annot")):
            ann_key = f"human_{annot_file.stem}"
            try:
                ann_df = _parse_annot_file(annot_file)
                if ann_df is not None and len(ann_df) > 0:
                    annotations[ann_key] = ann_df
            except Exception as e:
                print(f"[DEBUG] Failed to load .annot file {annot_file.name}: {e}")

    # --- Algorithmic Annotations (EDF only) ---
    algo_dir = ALGORITHMIC_ANNOTATIONS_DIR / site_id
    if algo_dir.exists():
        for edf_file in sorted(algo_dir.glob(f"{rec_name}*.edf")):
            ann_key = f"algo_{edf_file.stem}"
            try:
                ann_df = _load_annotations_from_edf(edf_file)
                if ann_df is not None and len(ann_df) > 0:
                    annotations[ann_key] = ann_df
            except Exception as e:
                print(f"[DEBUG] Failed to load algo EDF annotation {edf_file.name}: {e}")

    # --- Fallback: Embedded annotations in the physiological EDF ---
    phys_edf = patient_dir / f"{rec_name}.edf"
    if phys_edf.exists():
        try:
            ann_df = _load_annotations_from_edf(phys_edf)
            if ann_df is not None and len(ann_df) > 0:
                annotations["embedded_annotations"] = ann_df
        except Exception:
            pass

    # --- DEBUG: Log result ---
    #print(f"[DEBUG] load_annotations result: {len(annotations)} annotation sets loaded")
    #for key, df in annotations.items():
    #    print(f"  {key}: {len(df)} entries")
    #    if len(df) > 0:
    #        print(f"    Columns: {list(df.columns)}")
    #        print(f"    First descriptions: {df['description'].head(5).tolist()}")

    return annotations if annotations else None


def _load_annotations_from_edf(edf_path: Path) -> Optional[pd.DataFrame]:
    """
    Lädt Annotationen aus einer EDF/EDF+ Datei.
    
    Versucht mehrere Methoden:
    1. MNE read_annotations (direkt)
    2. MNE read_raw_edf -> raw.annotations
    3. pyedflib readAnnotations
    4. EDF signal channels (Annotation als Signalwerte)
    """
    ann_df = None
    
    # --- Methode 1: MNE read_annotations (direkt, ohne Raw) ---
    try:
        mne_annotations = mne.read_annotations(str(edf_path))
        #print(f"[DEBUG] Method 1 (mne.read_annotations) for {edf_path.name}: "
        #      f"{len(mne_annotations)} annotations found")
        if len(mne_annotations) > 0:
            ann_df = _mne_annotations_to_dataframe(mne_annotations)
            #print(f"[DEBUG]   DataFrame: {len(ann_df)} rows")
            #if len(ann_df) > 0:
                #print(f"[DEBUG]   Descriptions: {ann_df['description'].unique()[:10].tolist()}")
            if ann_df is not None and len(ann_df) > 0:
                # Filtere leere/triviale Annotationen
                ann_df = ann_df[ann_df['description'].str.strip() != '']
                ann_df = ann_df[~ann_df['description'].str.lower().isin(
                    ['', 'sleep onset', 'recording start', 'lights off',
                     'lights on', 'recording end']
                )]
                #print(f"[DEBUG]   After filtering: {len(ann_df)} rows")
                if len(ann_df) > 0:
                    return ann_df
    except Exception as e:
        print(f"[DEBUG] Method 1 failed for {edf_path.name}: {e}")
    
    # --- Methode 2: MNE read_raw_edf -> raw.annotations ---
    try:
        raw_ann = mne.io.read_raw_edf(
            str(edf_path), preload=False, verbose=False
        )
        mne_annotations = raw_ann.annotations
        #print(f"[DEBUG] Method 2 (read_raw_edf) for {edf_path.name}: "
        #      f"{len(mne_annotations)} annotations, "
        #      f"{len(raw_ann.ch_names)} channels: {raw_ann.ch_names[:5]}")
        if len(mne_annotations) > 0:
            ann_df = _mne_annotations_to_dataframe(mne_annotations)
            if ann_df is not None and len(ann_df) > 0:
                ann_df = ann_df[ann_df['description'].str.strip() != '']
                #print(f"[DEBUG]   After filtering: {len(ann_df)} rows")
                if len(ann_df) > 0:
                    raw_ann.close()
                    return ann_df
        
        # --- Methode 2b: Annotations stored as signal channels ---
        # Some annotation EDFs store sleep stages as signal values
        if len(raw_ann.ch_names) > 0:
            # print(f"[DEBUG] Method 2b: Trying to read annotations from signal channels")
            ann_df = _extract_annotations_from_signal_channels(raw_ann)
            if ann_df is not None and len(ann_df) > 0:
                #print(f"[DEBUG]   Signal channel annotations: {len(ann_df)} rows")
                raw_ann.close()
                return ann_df
        
        raw_ann.close()
    except Exception as e:
        print(f"[DEBUG] Method 2 failed for {edf_path.name}: {e}")
    
    # --- Methode 3: pyedflib als Fallback ---
    try:
        import pyedflib
        f = pyedflib.EdfReader(str(edf_path))
        
        n_signals = f.signals_in_file
        #print(f"[DEBUG] Method 3 (pyedflib) for {edf_path.name}: "
        #      f"{n_signals} signals")
        
        # Print signal labels
        signal_labels = [f.getLabel(i) for i in range(n_signals)]
        #print(f"[DEBUG]   Signal labels: {signal_labels}")
        
        annotations = f.readAnnotations()
        
        if annotations is not None:
            #print(f"[DEBUG]   Annotations tuple length: {len(annotations)}")
            if len(annotations) >= 3:
                onsets = annotations[0]
                durations = annotations[1]
                descriptions = annotations[2]
                #print(f"[DEBUG]   Onsets: {len(onsets)}, "
                #      f"Durations: {len(durations)}, "
                #      f"Descriptions: {len(descriptions)}")
                #if len(descriptions) > 0:
                    #print(f"[DEBUG]   First 5 descriptions: "
                    #      f"{[str(d) for d in descriptions[:5]]}")
                
                if len(onsets) > 0:
                    records = []
                    for i in range(len(onsets)):
                        desc = str(descriptions[i]).strip() if i < len(descriptions) else ""
                        if desc == '':
                            continue
                        records.append({
                            "onset": float(onsets[i]),
                            "duration": float(durations[i]) if i < len(durations) else 0.0,
                            "description": desc,
                        })
                    
                    if records:
                        ann_df = pd.DataFrame(records)
                        ann_df = ann_df[~ann_df['description'].str.lower().isin(
                            ['', 'recording start']
                        )]
                        #print(f"[DEBUG]   pyedflib result: {len(ann_df)} rows")
                        if len(ann_df) > 0:
                            f.close()
                            return ann_df
        #else:
            #print(f"[DEBUG]   readAnnotations returned None")
        
        # --- Methode 3b: Read signal data as annotations ---
        if n_signals > 0:
            #print(f"[DEBUG] Method 3b: Trying to read signal data as annotations")
            ann_df = _extract_annotations_from_pyedflib(f, signal_labels)
            if ann_df is not None and len(ann_df) > 0:
                #print(f"[DEBUG]   Signal-based annotations: {len(ann_df)} rows")
                f.close()
                return ann_df
        
        f.close()
    except Exception as e:
        print(f"[DEBUG] Method 3 failed for {edf_path.name}: {e}")
    
    #print(f"[DEBUG] All methods failed for {edf_path.name}")
    return None


def _extract_annotations_from_signal_channels(raw: mne.io.Raw) -> Optional[pd.DataFrame]:
    """
    Extracts sleep stages and events from EDF annotation signal channels.
    """
    try:
        data = raw.get_data().astype(SIGNAL_DTYPE)  # CHANGED: cast to float32
        ch_names = [ch.lower().strip() for ch in raw.ch_names]
        fs = raw.info['sfreq']
        
        all_annotations = []
        
        for ch_idx, ch_name in enumerate(ch_names):
            signal = data[ch_idx]
            
            # --- SLEEP STAGES ---
            if 'stage' in ch_name:
                stage_df = _parse_stage_channel(signal, fs, ch_name)
                if stage_df is not None and len(stage_df) > 0:
                    all_annotations.append(stage_df)
            
            # --- AROUSALS ---
            elif 'arousal' in ch_name:
                # Skip probability channels (e.g., caisr_prob_arous)
                if 'prob' in ch_name:
                    continue
                event_df = _parse_binary_event_channel(
                    signal, fs, ch_name, event_type="arousal"
                )
                if event_df is not None and len(event_df) > 0:
                    all_annotations.append(event_df)
            
                        # --- RESPIRATORY EVENTS ---
            elif 'resp' in ch_name:
                if 'prob' in ch_name:
                    continue
                # Respiratory channels use multi-class encoding (0-9),
                # NOT binary. Use the multiclass parser.
                event_df = _parse_multiclass_event_channel(
                    signal, fs, ch_name
                )
                if event_df is not None and len(event_df) > 0:
                    all_annotations.append(event_df)
            
            # --- LIMB MOVEMENTS ---
            elif 'limb' in ch_name:
                if 'prob' in ch_name:
                    continue
                event_df = _parse_binary_event_channel(
                    signal, fs, ch_name, event_type="limb_movement"
                )
                if event_df is not None and len(event_df) > 0:
                    all_annotations.append(event_df)
        
        if all_annotations:
            combined = pd.concat(all_annotations, ignore_index=True)
            return combined
        
        return None
    except Exception:
        return None


def _parse_stage_channel(
    signal: np.ndarray, 
    fs: float, 
    ch_name: str
) -> Optional[pd.DataFrame]:
    """
    Parses a sleep stage channel.
    
    Stage values may be continuous floats due to interpolation or 
    probability encoding. We round to nearest integer and map to 
    standard sleep stage labels.
    
    Common encodings:
    - 0=Wake, 1=N1, 2=N2, 3=N3, 5=REM (AASM)
    - 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    - 0=Wake, 1=S1, 2=S2, 3=S3, 4=S4, 5=REM (R&K)
    """
    epoch_sec = 30  # Standard sleep epoch
    epoch_samples = int(epoch_sec * fs)
    
    if epoch_samples == 0:
        return None
    
    n_epochs = len(signal) // epoch_samples
    
    if n_epochs == 0:
        return None
    
    records = []
    for epoch_idx in range(n_epochs):
        start_sample = epoch_idx * epoch_samples
        end_sample = start_sample + epoch_samples
        epoch_data = signal[start_sample:end_sample]
        
        # Get the dominant value in this epoch
        # Round to nearest integer since values may have float noise
        rounded = np.round(epoch_data).astype(int)
        values, counts = np.unique(rounded, return_counts=True)
        dominant_value = values[np.argmax(counts)]
        
        # Map to stage label
        stage_label = _map_numeric_to_stage(dominant_value)
        
        records.append({
            "onset": float(epoch_idx * epoch_sec),
            "duration": float(epoch_sec),
            "description": stage_label,
        })
    
    if records:
        df = pd.DataFrame(records)
        # Keep all stages including Wake and Unknown for proper hypnogram
        df = df[df['description'] != '']
        if len(df) > 0:
            return df
    
    return None


def _parse_binary_event_channel(
    signal: np.ndarray,
    fs: float,
    ch_name: str,
    event_type: str = "event"
) -> Optional[pd.DataFrame]:
    """
    Parses a binary event channel (0/1) into event annotations.
    
    Finds contiguous regions where signal == 1 and creates 
    event annotations with onset and duration.
    """
    # Threshold at 0.5 for binary classification
    binary = (signal >= 0.5).astype(int)
    
    # Find transitions (0->1 = start, 1->0 = end)
    diff = np.diff(binary, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        return None
    
    # Ensure starts and ends are paired
    if len(ends) < len(starts):
        ends = np.append(ends, len(signal))
    
    records = []
    for start_sample, end_sample in zip(starts, ends):
        onset_sec = start_sample / fs
        duration_sec = (end_sample - start_sample) / fs
        
        # Filter very short events (< 1 second likely noise)
        if duration_sec < 1.0:
            continue
        
        # Map event type to standard description
        if event_type == "arousal":
            description = "arousal"
        elif event_type == "respiratory_event":
            description = "respiratory_event"
        elif event_type == "limb_movement":
            description = "limb_movement"
        else:
            description = event_type
        
        records.append({
            "onset": float(onset_sec),
            "duration": float(duration_sec),
            "description": description,
        })
    
    if records:
        return pd.DataFrame(records)
    return None

def _parse_multiclass_event_channel(
    signal: np.ndarray,
    fs: float,
    ch_name: str,
    event_map: Dict[int, str] = None,
) -> Optional[pd.DataFrame]:
    """
    Parses a multi-class event channel (e.g., respiratory events with codes 0-9)
    into event annotations with onset, duration, and the specific event type.

    PhysioNet Challenge 2026 respiratory encoding:
        0 = No Event
        1 = Obstructive Apnea
        2 = Central Apnea
        3 = Mixed Apnea
        4 = Obstructive Hypopnea
        5 = Central Hypopnea
        6 = Mixed Hypopnea
        7 = RERA
        8 = Apnea (unspecified)
        9 = Hypopnea (unspecified)

    Parameters
    ----------
    signal : np.ndarray
        The raw signal values from the annotation channel.
    fs : float
        Sampling frequency of the annotation channel.
    ch_name : str
        Channel name (for logging/debugging).
    event_map : Dict[int, str], optional
        Mapping from integer code to event description string.
        If None, uses the default respiratory event map.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: onset, duration, description
    """
    if event_map is None:
        event_map = {
            1: "obstructive_apnea",
            2: "central_apnea",
            3: "mixed_apnea",
            4: "obstructive_hypopnea",
            5: "central_hypopnea",
            6: "mixed_hypopnea",
            7: "rera",
            8: "apnea_unspecified",
            9: "hypopnea_unspecified",
        }

    # Round to nearest integer (signal may have float noise)
    rounded = np.round(signal).astype(int)

    records = []
    n_samples = len(rounded)
    i = 0

    while i < n_samples:
        code = rounded[i]

        # Skip "no event" samples (code 0 or not in map)
        if code not in event_map:
            i += 1
            continue

        # Found the start of an event — find its end
        event_start_sample = i
        event_code = code

        while i < n_samples and rounded[i] == event_code:
            i += 1

        event_end_sample = i  # one past the last sample of this event

        onset_sec = event_start_sample / fs
        duration_sec = (event_end_sample - event_start_sample) / fs

        # Filter very short events (< 1 second likely noise)
        if duration_sec < 1.0:
            continue

        description = event_map[event_code]

        records.append({
            "onset": float(onset_sec),
            "duration": float(duration_sec),
            "description": description,
        })

    if records:
        return pd.DataFrame(records)
    return None

def _map_numeric_to_stage(value: int) -> str:
    """
    Maps numeric values to sleep stage labels.
    Handles multiple common encoding schemes.
    """
    # Try multiple encodings
    # Encoding 1: AASM with REM=5
    aasm_5 = {
        0: "Sleep stage W",
        1: "Sleep stage N1",
        2: "Sleep stage N2",
        3: "Sleep stage N3",
        5: "Sleep stage R",
    }
    
    # Encoding 2: AASM with REM=4
    aasm_4 = {
        0: "Sleep stage W",
        1: "Sleep stage N1",
        2: "Sleep stage N2",
        3: "Sleep stage N3",
        4: "Sleep stage R",
    }
    
    # Encoding 3: R&K
    rk = {
        0: "Sleep stage W",
        1: "Sleep stage N1",
        2: "Sleep stage N2",
        3: "Sleep stage N3",
        4: "Sleep stage N3",  # S4 -> N3
        5: "Sleep stage R",
    }
    
    # Try AASM with REM=5 first (most common in modern datasets)
    if value in aasm_5:
        return aasm_5[value]
    
    # Then AASM with REM=4
    if value in aasm_4:
        return aasm_4[value]
    
    # Movement/artifact
    if value in [6, 7, 8, 9]:
        return "Sleep stage W"  # Treat as wake
    
    return "Unknown"
def _extract_annotations_from_pyedflib(f, signal_labels: list) -> Optional[pd.DataFrame]:
    """
    Extracts annotations from signal data using pyedflib.
    Handles stage channels, arousal channels, and event channels.
    """
    import pyedflib
    
    try:
        n_signals = f.signals_in_file
        all_annotations = []
        
        for i in range(n_signals):
            label = signal_labels[i].lower().strip()
            signal = f.readSignal(i)
            fs = f.getSampleFrequency(i)
            
            if fs <= 0 or len(signal) == 0:
                continue
            
            # --- SLEEP STAGES ---
            if 'stage' in label:
                stage_df = _parse_stage_channel(signal, fs, label)
                if stage_df is not None and len(stage_df) > 0:
                    all_annotations.append(stage_df)
            
            # --- AROUSALS ---
            elif 'arousal' in label and 'prob' not in label:
                unique_vals = np.unique(signal)
                if len(unique_vals) <= 5:  # Binary or near-binary
                    event_df = _parse_binary_event_channel(
                        signal, fs, label, event_type="arousal"
                    )
                    if event_df is not None and len(event_df) > 0:
                        all_annotations.append(event_df)
            
                        # --- RESPIRATORY ---
            elif 'resp' in label and 'prob' not in label:
                # Respiratory channels use multi-class encoding (0-9)
                event_df = _parse_multiclass_event_channel(
                    signal, fs, label
                )
                if event_df is not None and len(event_df) > 0:
                    all_annotations.append(event_df)

            
            # --- LIMB ---
            elif 'limb' in label and 'prob' not in label:
                unique_vals = np.unique(signal)
                if len(unique_vals) <= 5:
                    event_df = _parse_binary_event_channel(
                        signal, fs, label, event_type="limb_movement"
                    )
                    if event_df is not None and len(event_df) > 0:
                        all_annotations.append(event_df)
        
        if all_annotations:
            combined = pd.concat(all_annotations, ignore_index=True)
            return combined
        
        return None
    except Exception:
        return None


def _map_numeric_to_stage(value: float) -> str:
    """
    Maps numeric values to sleep stage labels.
    Handles multiple common encoding schemes.
    """
    # Round to nearest integer
    val = int(round(value))
    
    # AASM standard encoding (most common)
    # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, 5=Unknown/Movement
    aasm_map = {
        0: "Sleep stage W",
        1: "Sleep stage N1",
        2: "Sleep stage N2",
        3: "Sleep stage N3",
        4: "Sleep stage R",
        5: "Sleep stage W",  # Movement/Unknown -> Wake
    }
    
    # R&K encoding
    # 0=Wake, 1=S1, 2=S2, 3=S3, 4=S4, 5=REM
    rk_map = {
        0: "Sleep stage W",
        1: "Sleep stage N1",
        2: "Sleep stage N2",
        3: "Sleep stage N3",
        4: "Sleep stage N3",  # S4 -> N3
        5: "Sleep stage R",
    }
    
    # Try AASM first (most common in modern datasets)
    if val in aasm_map:
        return aasm_map[val]
    
    return "Unknown"



# ==============================================================================
# ANNOTATION PARSING HELPERS
# ==============================================================================

def _mne_annotations_to_dataframe(mne_annotations) -> pd.DataFrame:
    """
    Konvertiert MNE Annotations zu einem standardisierten DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame mit Spalten: onset, duration, description
    """
    records = []
    for ann in mne_annotations:
        records.append({
            "onset": float(ann['onset']),
            "duration": float(ann['duration']),
            "description": str(ann['description']).strip(),
        })

    return pd.DataFrame(records)


def _parse_annot_file(annot_path: Path) -> Optional[pd.DataFrame]:
    """
    Parst eine .annot Datei.

    .annot Dateien können verschiedene Formate haben:
    - Tab-separiert mit Spalten onset, duration, annotation
    - XML-basiert
    - Zeilenbasiert
    """
    try:
        with open(annot_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Prüfe ob es XML ist
        if content.strip().startswith('<?xml') or content.strip().startswith('<'):
            return _parse_annot_xml(content)

        # Versuche als Tab/Komma-separierte Datei
        for sep in ['\t', ',', ';']:
            try:
                df = pd.read_csv(annot_path, sep=sep, comment='#',
                                  encoding='utf-8', on_bad_lines='skip')

                # Normalisiere Spaltennamen
                df.columns = [c.lower().strip() for c in df.columns]

                # Suche nach onset/duration/description Spalten
                onset_col = None
                for candidate in ['onset', 'start', 'time', 'start_sec']:
                    if candidate in df.columns:
                        onset_col = candidate
                        break

                dur_col = None
                for candidate in ['duration', 'dur', 'length', 'duration_sec']:
                    if candidate in df.columns:
                        dur_col = candidate
                        break

                desc_col = None
                for candidate in ['description', 'annotation', 'label',
                                   'event', 'type', 'stage']:
                    if candidate in df.columns:
                        desc_col = candidate
                        break

                if onset_col and desc_col:
                    result = pd.DataFrame({
                        'onset': pd.to_numeric(df[onset_col], errors='coerce'),
                        'duration': pd.to_numeric(
                            df[dur_col], errors='coerce'
                        ) if dur_col else 30.0,
                        'description': df[desc_col].astype(str),
                    })
                    result = result.dropna(subset=['onset'])
                    if len(result) > 0:
                        return result
            except Exception:
                continue

        # Versuche zeilenweises Parsen
        return _parse_annot_lines(content)

    except Exception:
        return None


def _parse_annot_xml(content: str) -> Optional[pd.DataFrame]:
    """Parst XML-basierte .annot Dateien."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)

        records = []
        for event in root.iter():
            onset = None
            duration = None
            description = None

            for child in event:
                tag = child.tag.lower()
                text = child.text.strip() if child.text else ""

                if tag in ['onset', 'start', 'time']:
                    try:
                        onset = float(text)
                    except ValueError:
                        pass
                elif tag in ['duration', 'dur']:
                    try:
                        duration = float(text)
                    except ValueError:
                        pass
                elif tag in ['annotation', 'description', 'label',
                              'event', 'type', 'stage']:
                    description = text

            if onset is not None and description:
                records.append({
                    'onset': onset,
                    'duration': duration or 30.0,
                    'description': description,
                })

        if records:
            return pd.DataFrame(records)
        return None
    except Exception:
        return None


def _parse_annot_lines(content: str) -> Optional[pd.DataFrame]:
    """Parst .annot Dateien zeilenweise als Fallback."""
    records = []
    lines = content.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('%'):
            continue

        # Versuche verschiedene Trennzeichen
        parts = re.split(r'[\t,;]+', line)

        if len(parts) >= 2:
            try:
                onset = float(parts[0])
                if len(parts) >= 3:
                    try:
                        duration = float(parts[1])
                        description = parts[2].strip()
                    except ValueError:
                        duration = 30.0
                        description = parts[1].strip()
                else:
                    duration = 30.0
                    description = parts[1].strip()

                records.append({
                    'onset': onset,
                    'duration': duration,
                    'description': description,
                })
            except ValueError:
                continue

    if records:
        return pd.DataFrame(records)
    return None


# ==============================================================================
# KANAL-IDENTIFIKATION
# ==============================================================================

def identify_channel(sig_names: List[str], target_names: List[str]) -> Optional[int]:
    """Identifiziert einen Kanal anhand möglicher Namens-Varianten."""
    # Erster Durchlauf: Exaktes Matching
    for i, name in enumerate(sig_names):
        name_clean = name.lower().strip()
        for target in target_names:
            if name_clean == target.lower().strip():
                return i
    # Zweiter Durchlauf: Teilstring-Matching
    # Vorläufig entfernt, da sonst doppelte Registrierung der Mastoid Channel
    """
    for i, name in enumerate(sig_names):
        name_clean = name.lower().strip()
        for target in target_names:
            if target.lower().strip() in name_clean:
                return i
    """
    return None
    


def find_ecg_channel(sig_names: List[str]) -> Optional[int]:
    """Findet den ECG-Kanal."""
    return identify_channel(sig_names, ECG_CHANNEL_NAMES)


def find_eeg_channels(sig_names: List[str]) -> Dict[str, Optional[int]]:
    """
    Findet alle verfügbaren EEG-Kanäle.

    Returns
    -------
    Dict[str, int]
        Mapping: Standardisierter Kanalname -> Index im Record.
    """
    found_channels = {}
    for standard_name, variants in EEG_CHANNEL_MAPPING.items():
        idx = identify_channel(sig_names, variants)
        if idx is not None:
            found_channels[standard_name] = idx
    return found_channels


def find_resp_channels(sig_names: List[str]) -> Dict[str, int]:
    """Findet Respirations-Kanäle."""
    found = {}
    for i, name in enumerate(sig_names):
        name_clean = name.lower().strip()
        for resp_name in RESP_CHANNEL_NAMES:
            if resp_name in name_clean:
                found[name_clean] = i
                break
    return found


# ==============================================================================
# SIGNAL-EXTRAKTION
# ==============================================================================

def extract_signal(raw: mne.io.Raw, channel_idx: int) -> Tuple[np.ndarray, float]:
    """
    Extrahiert ein einzelnes Signal aus dem MNE Raw-Objekt.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw-Objekt.
    channel_idx : int
        Index des Kanals.

    Returns
    -------
    Tuple[np.ndarray, float]
        Signal-Array (float32) und Sampling-Rate.
    """
    signal = raw.get_data(picks=[channel_idx])[0].astype(SIGNAL_DTYPE)  # CHANGED: cast to float32
    fs = raw.info['sfreq']
    return signal, fs


# ==============================================================================
# FEATURE SPEICHERN / LADEN
# ==============================================================================

def save_features(features_df: pd.DataFrame, patient_id: str,
                  output_dir: Path, suffix: str = ""):
    """Speichert Feature-DataFrame als Parquet-Datei."""
    safe_id = patient_id.replace("/", "_")
    filename = f"{safe_id}_features{suffix}.parquet"
    filepath = output_dir / filename
    features_df.to_parquet(filepath, index=False)
    return filepath


def load_features(patient_id: str, output_dir: Path,
                  suffix: str = "") -> Optional[pd.DataFrame]:
    """Lädt Feature-DataFrame aus Parquet-Datei."""
    safe_id = patient_id.replace("/", "_")
    filename = f"{safe_id}_features{suffix}.parquet"
    filepath = output_dir / filename
    if filepath.exists():
        return pd.read_parquet(filepath)
    return None

# ==============================================================================
# Re-referencing
# ==============================================================================

def rereference_eeg_to_bipolar(
    raw: mne.io.Raw,
) -> Tuple[mne.io.Raw, bool]:
    """
    Erzeugt bipolare EEG-Ableitungen (F3-M2, F4-M1, C3-M2, C4-M1) aus unipolaren Kanälen.

    Parameter
    ---------
    raw : mne.io.Raw
        Das geladene EEG-Rohdatenobjekt (muss preload=True haben).
    eeg_channel_map : dict
        Mapping von Standardnamen ('F3', 'F4', 'C3', 'C4', 'M1', 'M2') zu tatsächlichen Kanalnamen im Datensatz.

    Rückgabe
    --------
    raw_bipolar : mne.io.Raw
        Raw-Objekt mit den bipolaren Kanälen.
    ch_names : list[str]
        Liste der erzeugten (oder bereits vorhandenen) bipolaren Kanalnamen.
    ch_index_map : dict[str, int]
        Dictionary: Kanalname → Index im Raw-Objekt (raw_bipolar.ch_names).
    """
    # Definiere gewünschte Ableitungen
    bipolar_pairs = [
        ('F3', 'M2', 'F3-M2'),
        ('F4', 'M1', 'F4-M1'),
        ('C3', 'M2', 'C3-M2'),
        ('C4', 'M1', 'C4-M1'),
    ]

    rereferenced = False
    # Prüfe, ob bereits bipolare Kanäle vorhanden sind (z.B. "F3-M2")
    already_bipolar = any('-M' in ch for ch in raw.ch_names)
    if already_bipolar:
        #existing_bipolar_names = [ch for ch in raw.ch_names if '-' in ch]
        #ch_index_map = {name: raw.ch_names.index(name) for name in existing_bipolar_names}
        return raw, rereferenced

    # Erzeuge Listen für set_bipolar_reference
    anodes, cathodes, ch_names = [], [], []
    for anode, cathode, new_name in bipolar_pairs:
        #anode_actual   = eeg_channel_map.get(anode)
        #cathode_actual = eeg_channel_map.get(cathode)
        if anode in raw.ch_names and cathode in raw.ch_names:
            anodes.append(anode)
            cathodes.append(cathode)
            ch_names.append(new_name)

    if not anodes:
        #raise RuntimeError("Keine passenden Kanäle für bipolare Ableitungen gefunden.")
        raw_bipolar = []
        rereferenced = False
        return raw_bipolar, rereferenced

    # Bipolare Referenzierung durchführen
    raw_bipolar = mne.set_bipolar_reference(
        raw, anode=anodes, cathode=cathodes, ch_name=ch_names,
        drop_refs=True, copy=True
    )
    rereferenced = True
    # Index-Mapping: Kanalname → Index im Raw-Objekt
    #ch_index_map = {name: raw_bipolar.ch_names.index(name) for name in ch_names}

    return raw_bipolar, rereferenced
