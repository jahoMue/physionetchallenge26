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
        _GLOBAL_ORT_SESSION = ort.InferenceSession(CAP_MODEL_DIR)
    return _GLOBAL_ORT_SESSION

def cap_classification(input_list, eeg, flags):
    """
    input_list: Liste von Feature-Fenstern (L x 30)
    eeg: SignalEEG-Objekt
    flags: Dict mit Optionen
    lstm_predict_fn: Funktion, die ein Input-Array (seq_len, features) nimmt und Vorhersagen zurückgibt
    """
    if not CAP_CLASSIFICATION_ENABLED:
        return _nan_cap_stats(getattr(eeg, 'name', ''))

    ort_session = _get_ort_session()

    # Normalisierung (medianiqr)
    def medianiqr_norm(x):
        med = np.median(x, axis=0, keepdims=True)
        iqr = np.subtract(*np.percentile(x, [75, 25], axis=0, keepdims=True))
        return (x - med) / (iqr + 1e-8)
    
    input_norm = [medianiqr_norm(x) for x in input_list]

    # LSTM Initialisieren
    def lstm_predict_fn(input_norm):
        # x_batch: (batch, seq_len, features)
        x = np.array(input_norm)
        x_batch = np.expand_dims(x.astype(np.float32), axis=0)
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: x_batch})
        pred = np.argmax(outputs[0], axis=-1)
        return np.atleast_1d(pred)  # Immer 1D-Array zurückgeben
    # LSTM-Vorhersage
    predictions = lstm_predict_fn(input_norm)
    # Mehrheitsvoting
    pred_arr = np.stack(predictions)
    pred_1 = np.sum(predictions == 1, axis=0)
    pred_2 = np.sum(predictions == 2, axis=0)
    pred_3 = np.sum(predictions == 3, axis=0)
    n_pred = pred_arr.shape[0]
    pred = (pred_1 > n_pred//2).astype(int) + 2*(pred_2 > n_pred//2) + 3*(pred_3 > n_pred//2)
    # Label-Rekonstruktion und Postprocessing
    x = input_list
    pred_rec, x_rec = label_reconstruction(eeg, predictions, x, 30)
    pred_rec = post_processing_multi_class(pred_rec, x_rec)
    # Event-Vektor für CAPsequences
    events_vec = np.repeat(eeg.event, 30)
    if flags.get('Scoring', '') == 'CAP':
        pred_rec, CAP_start, CAP_stop = cap_sequences(pred_rec, events_vec)
    else:
        CAP_start, CAP_stop = [], []
    stats = get_output_statistics(predictions, pred_rec, CAP_start, CAP_stop, eeg.name)
    return stats
