import numpy as np

def label_reconstruction(eeg, labels, x, seq_length):
    # x_rec (2D) wird von KEINER nachgelagerten Funktion genutzt
    # (post_processing_multi_class ignoriert sein zweites Argument),
    # verursachte aber durch wiederholtes np.concatenate eine
    # O(T^2)-Speicherexplosion -> OOM. Wir rekonstruieren daher nur
    # den 1D-Label-Vektor und geben None statt x_rec zurueck.
    labels_rec = np.concatenate([np.zeros(seq_length - 1), labels])
    for i in range(len(eeg.event)):
        if eeg.event[i] == 0 or eeg.event[i] == 5 or eeg.event[i] > 8:
            dur = int(eeg.duration[i])
            if eeg.eventtime[i] == 0:
                labels_rec = np.concatenate([np.zeros(dur), labels_rec])
            elif eeg.eventtime[i] > len(labels_rec):
                labels_rec = np.concatenate([labels_rec, np.zeros(dur)])
            else:
                et = int(eeg.eventtime[i])
                labels_rec = np.concatenate([
                    labels_rec[:et],
                    np.zeros(dur),
                    labels_rec[et:],
                ])
    return labels_rec, None

def post_processing_multi_class(y, x):
    # Entspricht postProcessingMultiClass.m
    y_post = y.copy()
    for step in range(4):
        CAP_flag = False
        score_flag = False
        ind = 0
        CAP_start = []
        CAP_stop = []
        duration = []
        i = 0
        while i < len(y_post):
            if CAP_flag:
                if y_post[i] == 0:
                    CAP_flag = False
                    score_flag = True
                    CAP_stop.append(i)
                    duration.append(i - CAP_start[ind])
                    if duration[ind] == 1:
                        y_post[i-1] = 0
                    else:
                        # Dominante Klasse im Run
                        run = y_post[CAP_start[ind]:CAP_stop[ind]]
                        vals, counts = np.unique(run, return_counts=True)
                        dominant = vals[np.argmax(counts)]
                        y_post[CAP_start[ind]:CAP_stop[ind]] = dominant
                    ind += 1
                else:
                    pass  # Zählt Klassen, für Py: nicht nötig
            else:
                if y_post[i] > 0:
                    CAP_flag = True
                    CAP_start.append(i)
            i += 1
        if CAP_flag:
            CAP_stop.append(i)
            duration.append(i - CAP_start[ind])
            if duration[ind] == 1:
                y_post[-1] = 0
            else:
                run = y_post[CAP_start[ind]:]
                vals, counts = np.unique(run, return_counts=True)
                dominant = vals[np.argmax(counts)]
                y_post[CAP_start[ind]:] = dominant
        # Entferne isolierte Nullen innerhalb von A-Phasen
        # (Optional: weitere Nachbearbeitung wie in MATLAB)
    return y_post

def cap_sequences(pred, events):
    pred_bin = pred > 0
    pred_diff = np.diff(pred_bin.astype(int))
    A_start = np.where(pred_diff == 1)[0] + 1
    A_stop = np.where(pred_diff == -1)[0]
    CAP_start = []
    CAP_stop = []
    pred_out = pred.copy()
    if len(A_stop) > 0:
        if A_stop[0] < A_start[0]:
            A_start = np.insert(A_start, 0, 0)
        if A_start[-1] > A_stop[-1]:
            A_stop = np.append(A_stop, len(pred)-1)
        i = 0
        CAP_status = 0
        while i < len(A_start):
            if A_stop[i] - A_start[i] <= 60:
                if CAP_status > 0:
                    if (A_start[i] - A_stop[i-1] - 1 <= 60 and
                        not np.any((events[A_stop[i-1]:A_start[i]] == 0) | (events[A_stop[i-1]:A_start[i]] > 4))):
                        CAP_status += 1
                    elif CAP_status > 2:
                        CAP_start.append(A_start[i-CAP_status])
                        CAP_stop.append(A_start[i-1]-1)
                        pred_out[A_start[i-1]:A_stop[i-1]+1] = 0
                        CAP_status = 1
                    else:
                        for j in range(CAP_status):
                            pred_out[A_start[i-j]:A_stop[i-j]+1] = 0
                        CAP_status = 1
                else:
                    CAP_status = 1
            else:
                if CAP_status > 0:
                    if CAP_status > 2:
                        CAP_start.append(A_start[i-CAP_status])
                        CAP_stop.append(A_stop[i-1])
                    else:
                        for j in range(CAP_status):
                            pred_out[A_start[i-j]:A_stop[i-j]+1] = 0
                pred_out[A_start[i]:A_stop[i]+1] = 0
                CAP_status = 0
            i += 1
        if CAP_status > 2:
            CAP_start.append(A_start[i-CAP_status])
            CAP_stop.append(A_stop[i-1])
    return pred_out, CAP_start, CAP_stop

def get_output_statistics(predictions, rec_pred, CAP_start, CAP_stop, name):
    """
    Exakte Übersetzung von getOutputStatistics.m (MATLAB) nach Python.
    Gibt ein Dictionary mit allen 31 CAP-Parametern zurück.
    """
    stats = {}
    # 1. ID
    stats['ID'] = name
    # 2. SLDUR: Schlafdauer (len(predictions)+13)
    sldur = len(predictions) + 13
    stats['cap_SLDUR'] = sldur

    # 3. A-Phasen-Detektion
    pred_diff = np.diff((rec_pred > 0).astype(int))
    A_start = np.where(pred_diff == 1)[0] + 1
    A_stop = np.where(pred_diff == -1)[0]
    if len(A_stop) > 0:
        if A_stop[0] < A_start[0]:
            A_start = np.insert(A_start, 0, 0)
        if A_start[-1] > A_stop[-1]:
            A_stop = np.append(A_stop, len(rec_pred)-1)
        A_dur = A_stop - A_start + 1
        A_type = rec_pred[A_start]
    else:
        A_start = np.array([], dtype=int)
        A_stop = np.array([], dtype=int)
        A_dur = np.array([], dtype=int)
        A_type = np.array([], dtype=int)
    # 3. NRAPH: Anzahl A-Phasen
    nraph = len(A_start)
    stats['cap_NRAPH'] = nraph
    # 4. APHDUR: Gesamtdauer A-Phasen
    aphdur = int(np.sum(rec_pred > 0))
    stats['cap_APHDUR'] = aphdur
    # 5. AVGAPHDUR: Durchschnittliche A-Phasen-Dauer
    stats['cap_AVGAPHDUR'] = aphdur / nraph if nraph > 0 else 0
    # 6. NRAPHPH: A-Phasen pro Stunde
    stats['cap_NRAPHPH'] = nraph / sldur * 3600 if sldur > 0 else 0
    # 7. RAPHSL: Anteil A-Phasen an Schlafdauer
    stats['cap_RAPHSL'] = aphdur / sldur if sldur > 0 else 0

    # --- Typ-spezifische A-Phasen (A1, A2, A3) ---
    # A1
    pred_1 = (rec_pred == 1).astype(int)
    A1_start = np.where(np.diff(pred_1) == 1)[0]
    A1_stop = np.where(np.diff(pred_1) == -1)[0]
    if len(A1_stop) > 0:
        if A1_stop[0] < A1_start[0]:
            A1_start = np.insert(A1_start, 0, 0)
        if A1_start[-1] > A1_stop[-1]:
            A1_stop = np.append(A1_stop, len(rec_pred)-1)
        nra1 = len(A1_start)
        a1dur = int(np.sum(pred_1 > 0))
        stats['cap_NRA1'] = nra1
        stats['cap_A1DUR'] = a1dur
        stats['cap_AVGA1DUR'] = a1dur / nra1 if nra1 > 0 else 0
        stats['cap_RA1APH'] = nra1 / nraph if nraph > 0 else 0
        stats['cap_RA1NRE'] = a1dur / sldur if sldur > 0 else 0
        stats['cap_A1IND'] = nra1 / sldur * 3600 if sldur > 0 else 0
    else:
        stats['cap_NRA1'] = 0
        stats['cap_A1DUR'] = 0
        stats['cap_AVGA1DUR'] = 0
        stats['cap_RA1APH'] = 0
        stats['cap_RA1NRE'] = 0
        stats['cap_A1IND'] = 0

    # A2
    pred_2 = (rec_pred == 2).astype(int)
    A2_start = np.where(np.diff(pred_2) == 1)[0]
    A2_stop = np.where(np.diff(pred_2) == -1)[0]
    if len(A2_stop) > 0 and len(A2_start) > 0:
        if A2_stop[0] < A2_start[0]:
            A2_start = np.insert(A2_start, 0, 0)
        if A2_start[-1] > A2_stop[-1]:
            A2_stop = np.append(A2_stop, len(rec_pred)-1)
        nra2 = len(A2_start)
        a2dur = int(np.sum(pred_2 > 0))
        stats['cap_NRA2'] = nra2
        stats['cap_A2DUR'] = a2dur
        stats['cap_AVGA2DUR'] = a2dur / nra2 if nra2 > 0 else 0
        stats['cap_RA2APH'] = nra2 / nraph if nraph > 0 else 0
        stats['cap_RA2NRE'] = a2dur / sldur if sldur > 0 else 0
        stats['cap_A2IND'] = nra2 / sldur * 3600 if sldur > 0 else 0
    else:
        stats['cap_NRA2'] = 0
        stats['cap_A2DUR'] = 0
        stats['cap_AVGA2DUR'] = 0
        stats['cap_RA2APH'] = 0
        stats['cap_RA2NRE'] = 0
        stats['cap_A2IND'] = 0

    # A3
    pred_3 = (rec_pred == 3).astype(int)
    A3_start = np.where(np.diff(pred_3) == 1)[0]
    A3_stop = np.where(np.diff(pred_3) == -1)[0]
    if len(A3_stop) > 0:
        if A3_stop[0] < A3_start[0]:
            A3_start = np.insert(A3_start, 0, 0)
        if A3_start[-1] > A3_stop[-1]:
            A3_stop = np.append(A3_stop, len(rec_pred)-1)
        nra3 = len(A3_start)
        a3dur = int(np.sum(pred_3 > 0))
        stats['cap_NRA3'] = nra3
        stats['cap_A3DUR'] = a3dur
        stats['cap_AVGA3DUR'] = a3dur / nra3 if nra3 > 0 else 0
        stats['cap_RA3APH'] = nra3 / nraph if nraph > 0 else 0
        stats['cap_RA3NRE'] = a3dur / sldur if sldur > 0 else 0
        stats['cap_A3IND'] = nra3 / sldur * 3600 if sldur > 0 else 0
    else:
        stats['cap_NRA3'] = 0
        stats['cap_A3DUR'] = 0
        stats['cap_AVGA3DUR'] = 0
        stats['cap_RA3APH'] = 0
        stats['cap_RA3NRE'] = 0
        stats['cap_A3IND'] = 0

    # --- CAP-Statistiken ---
    if len(CAP_start) > 0:
        nrcap = len(CAP_start)
        cap_duration = 0
        b_phase_duration = []
        cap_cycle_duration = []
        for i in range(len(CAP_start)):
            cap_duration += CAP_stop[i] - CAP_start[i]
            pred_tmp = rec_pred[CAP_start[i]:CAP_stop[i]+1]  # inclusive
            cap_diff = np.diff((pred_tmp > 0).astype(int))
            cap_a_start = np.where(cap_diff == 1)[0] + 1
            cap_a_start1 = np.concatenate([[0], cap_a_start, [len(pred_tmp)-1]])
            cap_a_start2 = np.concatenate([cap_a_start, [len(pred_tmp)-1]])
            cap_a_stop = np.where(cap_diff == -1)[0]
            # --- Fix: Längen angleichen ---
            while len(cap_a_start2) > len(cap_a_stop):
                cap_a_start2 = cap_a_start2[:-1]
            while len(cap_a_stop) > len(cap_a_start2):
                cap_a_stop = cap_a_stop[:-1]
            cap_cycle_duration.extend(np.diff(cap_a_start1).tolist())
            b_phase_duration.extend((cap_a_start2 - cap_a_stop).tolist())
        stats['cap_NRCAP'] = nrcap
        stats['cap_CAPDUR'] = cap_duration
        stats['cap_RCAPSL'] = cap_duration / sldur * 100 if sldur > 0 else 0
        stats['cap_AVGCAPDUR'] = cap_duration / nrcap if nrcap > 0 else 0
        stats['cap_AVGCYCLEDUR'] = float(np.mean(cap_cycle_duration)) if cap_cycle_duration else 0
        stats['cap_AVGBPHADUR'] = float(np.mean(b_phase_duration)) if b_phase_duration else 0
    else:
        stats['cap_NRCAP'] = 0
        stats['cap_CAPDUR'] = 0
        stats['cap_RCAPSL'] = 0
        stats['cap_AVGCAPDUR'] = 0
        stats['cap_AVGCYCLEDUR'] = 0
        stats['cap_AVGBPHADUR'] = 0

    return stats