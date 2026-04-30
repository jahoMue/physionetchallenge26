import numpy as np
import onnxruntime as ort
from CAP_classification.post_processing import label_reconstruction, post_processing_multi_class, cap_sequences, get_output_statistics
from config import CAP_MODEL_DIR

def cap_classification(input_list, eeg, flags):
    """
    input_list: Liste von Feature-Fenstern (L x 30)
    eeg: SignalEEG-Objekt
    flags: Dict mit Optionen
    lstm_predict_fn: Funktion, die ein Input-Array (seq_len, features) nimmt und Vorhersagen zurückgibt
    """

    ort_session = ort.InferenceSession(CAP_MODEL_DIR)

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