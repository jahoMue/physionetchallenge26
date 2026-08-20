import numpy as np
import onnxruntime as ort
from CAP_classification.post_processing import label_reconstruction, post_processing_multi_class, cap_sequences, get_output_statistics
from config import CAP_MODEL_DIR, CAP_CLASSIFICATION_ENABLED

_GLOBAL_ORT_SESSION = None

_CAP_PARAM_KEYS = [
    "cap_SLDUR",
    "cap_NRAPH", "cap_APHDUR", "cap_AVGAPHDUR", "cap_NRAPHPH", "cap_RAPHSL",
    "cap_NRA1", "cap_A1DUR", "cap_AVGA1DUR", "cap_RA1APH", "cap_RA1NRE", "cap_A1IND",
    "cap_NRA2", "cap_A2DUR", "cap_AVGA2DUR", "cap_RA2APH", "cap_RA2NRE", "cap_A2IND",
    "cap_NRA3", "cap_A3DUR", "cap_AVGA3DUR", "cap_RA3APH", "cap_RA3NRE", "cap_A3IND",
    "cap_NRCAP", "cap_CAPDUR", "cap_RCAPSL", "cap_AVGCAPDUR",
    "cap_AVGCYCLEDUR", "cap_AVGBPHADUR",
]

def _nan_cap_stats(name=''):
    """Gibt das CAP-Statistik-Dict mit allen Parametern auf NaN zurück."""
    stats = {'ID': name}
    for key in _CAP_PARAM_KEYS:
        stats[key] = np.nan
    return stats

def _get_ort_session():
    global _GLOBAL_ORT_SESSION
    if _GLOBAL_ORT_SESSION is None:
        so = ort.SessionOptions()
        # Verhindert, dass der Arena-Allocator sehr grosse zusammenhaengende
        # Puffer fuer lange Sequenzen spekulativ reserviert.
        so.enable_cpu_mem_arena = False
        _GLOBAL_ORT_SESSION = ort.InferenceSession(CAP_MODEL_DIR, sess_options=so)
    return _GLOBAL_ORT_SESSION

def cap_classification(input_list, eeg, flags, chunk_sec=3600, overlap_sec=120):
    """
    input_list: Liste von Feature-Fenstern (L x 30)
    eeg: SignalEEG-Objekt
    flags: Dict mit Optionen
    lstm_predict_fn: Funktion, die ein Input-Array (seq_len, features) nimmt und Vorhersagen zurückgibt
    """
    if not CAP_CLASSIFICATION_ENABLED:
        return _nan_cap_stats(getattr(eeg, 'name', ''))

    ort_session = _get_ort_session()
    input_name = ort_session.get_inputs()[0].name

    # Per-Zeitschritt-Normalisierung (median/iqr ueber die 14 Feature-Dims).
    # Vektorisiert statt Python-Liste aus T Kopien -> deutlich weniger Speicher.
    X = np.asarray(input_list, dtype=np.float32)
    med = np.median(X, axis=1, keepdims=True)
    q75, q25 = np.percentile(X, [75, 25], axis=1, keepdims=True)
    X_norm = ((X - med) / ((q75 - q25) + 1e-8)).astype(np.float32)

    T = X_norm.shape[0]

    def _run_chunk(arr2d):
        x_batch = np.expand_dims(arr2d, axis=0)
        outputs = ort_session.run(None, {input_name: x_batch})
        return np.atleast_1d(np.argmax(outputs[0], axis=-1)).reshape(-1)

    # Chunking entlang der Zeitachse deckelt den Aktivierungsspeicher
    # unabhaengig von der Nachtlaenge. Overlap dient als LSTM-"Warmup",
    # der verworfen wird (CAP-A-Phasen sind < 60 s, 120 s Overlap genuegt).
    if T <= chunk_sec:
        predictions = _run_chunk(X_norm)
    else:
        predictions = np.empty(T, dtype=np.int64)
        step = max(1, chunk_sec - overlap_sec)
        start = 0
        while start < T:
            end = min(start + chunk_sec, T)
            chunk_pred = _run_chunk(X_norm[start:end])
            keep_from = 0 if start == 0 else overlap_sec
            predictions[start + keep_from:end] = chunk_pred[keep_from:]
            if end >= T:
                break
            start += step

    del X_norm, X

    pred_rec, x_rec = label_reconstruction(eeg, predictions, input_list, 30)
    pred_rec = post_processing_multi_class(pred_rec, x_rec)
    events_vec = np.repeat(eeg.event, 30)
    if flags.get('Scoring', '') == 'CAP':
        pred_rec, CAP_start, CAP_stop = cap_sequences(pred_rec, events_vec)
    else:
        CAP_start, CAP_stop = [], []

    return get_output_statistics(predictions, pred_rec, CAP_start, CAP_stop, eeg.name)
