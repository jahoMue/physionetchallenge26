import numpy as np

class SignalEEG:
    def __init__(self, eeg, fs, event, duration, eventtime, name=''):
        self.eeg = eeg
        self.fs = fs
        self.event = event
        self.duration = duration
        self.eventtime = eventtime
        self.name = name
        self.eeg_features = None

    @staticmethod
    def cut_signal(signal, start, stop):
        return signal[start:stop]

    @staticmethod
    def align_signal(signal, factor):
        return signal * factor

    @staticmethod
    def get_cell(data, labels, channel):
        idx = labels.index(channel)
        return data[idx]

    def create_multi_class_input(self, seq_length=30):
        # Entspricht createMultiClassInput in MATLAB
        X = self.eeg_features
        #n_windows = X.shape[0] - seq_length + 1
        #input_list = [X[i:i+seq_length].T for i in range(n_windows)]
        return X