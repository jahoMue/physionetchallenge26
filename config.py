"""
config.py
==========
Zentrale Konfiguration für die PhysioNet Challenge 2026 Pipeline.
"""

from pathlib import Path

# ==============================================================================
# PFADE – Alle Pfade als Path-Objekte!
# ==============================================================================

# Basis-Verzeichnisse
PROJECT_DIR = Path(__file__).parent.resolve()

# Training-Set Verzeichnis
TRAINING_SET_DIR = Path(r"E:\Physionet26Data\training_set")

# Unterverzeichnisse der Datenstruktur
PHYSIOLOGICAL_DATA_DIR = TRAINING_SET_DIR / "physiological_data"
ALGORITHMIC_ANNOTATIONS_DIR = TRAINING_SET_DIR / "algorithmic_annotations"
HUMAN_ANNOTATIONS_DIR = TRAINING_SET_DIR / "human_annotations"

# DATA_DIR zeigt auf physiological_data (für Abwärtskompatibilität)
DATA_DIR = PHYSIOLOGICAL_DATA_DIR

# Ausgabe-Verzeichnisse
OUTPUT_DIR = PROJECT_DIR / "output"
FEATURE_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"

# Demographics
DEMOGRAPHICS_FILE = TRAINING_SET_DIR / "demographics.csv"

# Site-IDs
SITE_IDS = ["S0001", "I0002", "I0006"]

# Verzeichnisse erstellen
for d in [OUTPUT_DIR, FEATURE_DIR, MODEL_DIR, LOG_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# SIGNALVERARBEITUNG
# ==============================================================================
SEGMENT_LENGTH_SEC = 300           # 5 Minuten
SEGMENT_OVERLAP_SEC = 0            # Keine Überlappung
SLEEP_EPOCH_SEC = 30               # Standard-Schlafepoche

# ==============================================================================
# QUALITÄT
# ==============================================================================
ECG_SQI_THRESHOLD = 0.5
EEG_SQI_THRESHOLD = 0.5

# ==============================================================================
# HRV
# ==============================================================================
HRV_MIN_RR_INTERVALS = 10
HRV_RR_MIN_MS = 300
HRV_RR_MAX_MS = 2000

# ==============================================================================
# EEG
# ==============================================================================
EEG_FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

EEG_HOMOLOG_PAIRS = {
    "frontal": ("F3", "F4"),
    "central": ("C3", "C4"),
}

# ==============================================================================
# SCHLAF
# ==============================================================================
SLEEP_STAGE_ENCODING = {
    0: "Unknown",
    1: "N3",
    2: "N2",
    3: "N1",
    4: "REM",
    5: "W",
}

SLEEP_EVENTS_OF_INTEREST = [
    "arousal",
    "central_apnea",
    "obstructive_apnea",
    "mixed_apnea",
    "hypopnea",
]

EVENT_DISTANCE_MAX_SEC = 3600
EVENT_DENSITY_WINDOW_SEC = 300

# ==============================================================================
# KONTEXTUELLE FEATURES
# ==============================================================================
CONTEXT_WINDOW_SEGMENTS = 3

# ==============================================================================
# KLASSIFIKATION
# ==============================================================================
TARGET_COLUMN = "Cognitive_Impairment"
TIME_TO_EVENT_COLUMN = "Time_to_Event"
RANDOM_SEED = 42
CV_FOLDS = 5
TEST_SIZE = 0.2

# ==============================================================================
# RSA
# ==============================================================================
RSA_ENABLED = True

# ==============================================================================
# LOGGING
# ==============================================================================
LOG_LEVEL = "INFO"                 # "DEBUG", "INFO", "WARNING", "ERROR"
LOG_TO_FILE = True                 # Log in Datei schreiben
LOG_TO_CONSOLE = True              # Log auf Konsole ausgeben
LOG_ROTATION = "10 MB"             # Log-Datei Rotation
LOG_RETENTION = "30 days"          # Log-Dateien aufbewahren

# ==============================================================================
# VISUALISIERUNG
# ==============================================================================
PLOT_ENABLED = True
PLOT_FORMAT = "png"
PLOT_DPI = 150
PLOT_PER_PATIENT = True
PLOT_COHORT = True
PLOT_MAX_PATIENTS = 20

# ==============================================================================
# KANALNAMEN (für io_utils.py)
# ==============================================================================

# ECG-Kanalnamen (mögliche Varianten in PSG-Daten)
ECG_CHANNEL_NAMES = [
    "ecg", "ekg", "ecg1", "ecg2", "ecg i", "ecg ii",
    "ekg1", "ekg2", "ecg-la", "ecg-ra",
]

# EEG-Kanal-Mapping: Standardname -> mögliche Varianten
EEG_CHANNEL_MAPPING = {
    "F3": ["f3", "f3-m2", "f3-a2", "eeg f3-m2", "eeg f3-a2", "eeg f3"],
    "F4": ["f4", "f4-m1", "f4-a1", "eeg f4-m1", "eeg f4-a1", "eeg f4"],
    "C3": ["c3", "c3-m2", "c3-a2", "eeg c3-m2", "eeg c3-a2", "eeg c3"],
    "C4": ["c4", "c4-m1", "c4-a1", "eeg c4-m1", "eeg c4-a1", "eeg c4"],
    "M1": ["m1", "eeg m1"],
    "M2": ["m2", "eeg m2"]
}

# Respirations-Kanalnamen
RESP_CHANNEL_NAMES = [
    "airflow", "nasal", "oral", "thorax", "abdomen",
    "chest", "abdo", "flow", "thermistor", "cannula",
    "resp", "respiratory", "effort",
]

# ==============================================================================
# ECG FILTER-EINSTELLUNGEN (für preprocess_ecg.py)
# ==============================================================================
ECG_FILTER = {
    "lowcut": 0.5,
    "highcut": 40.0,
    "order": 4,
}
ECG_SQI_WINDOW_SEC = 10  # SQI-Berechnungsfenster

# ==============================================================================
# EEG FILTER- UND QUALITÄTS-EINSTELLUNGEN (für preprocess_eeg.py)
# ==============================================================================
EEG_FILTER = {
    "lowcut": 0.3,
    "highcut": 35.0,
    "order": 4,
    "notch": 50.0,  # Netzfrequenz (50 Hz EU, 60 Hz US)
}
EEG_AMPLITUDE_MAX_UV = 200.0    # Maximale physiologische Amplitude (µV)
EEG_AMPLITUDE_MIN_UV = 0.5      # Minimale Amplitude (Flatliner-Erkennung)
EEG_CORRELATION_THRESHOLD = 0.7 # Mindestkorrelation für Kanal-Mittelung
