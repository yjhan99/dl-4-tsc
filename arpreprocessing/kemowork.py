import itertools as it
import pickle

import numpy as np
import scipy.stats

from arpreprocessing.helpers import filter_signal, get_empatica_sampling
from arpreprocessing.preprocessor import Preprocessor
from arpreprocessing.signal import Signal, NoSuchSignal
from arpreprocessing.subject import Subject


class KEmoWork(Preprocessor):
    SUBJECTS_IDS = [1,2,3,4,8,10,12,13,14,16,18,19,20,21,22,23,25,26,27]
    CHANNELS_NAMES = ['muse', 'e4_acc', 'e4_bvp', 'e4_eda', 'e4_temp', 'polar_ecg']

    def __init__(self, logger, path, label_type):
        Preprocessor.__init__(self, logger, path, label_type, "KEmoWork", [], None, subject_cls=KEmoWorkSubject)

    def get_subjects_ids(self):
        return self.SUBJECTS_IDS


def original_sampling(channel_name: str):
    if channel_name.startswith("EDA"):
        return 4
    if channel_name.startswith("EEG"):
        return 256
    if channel_name.startswith("TEMP"):
        return 4
    if channel_name.startswith("ACC"):
        return 32
    if channel_name.startswith("BVP"):
        return 64
    if channel_name.startswith("ECG"):
        return 130
    if channel_name == "label":
        return 10
    raise NoSuchSignal(channel_name)


def target_sampling(channel_name: str):
    if channel_name.startswith("EDA"):
        return 4
    if channel_name.startswith("EEG"):
        return 128
    if channel_name.startswith("TEMP"):
        return 4
    if channel_name.startswith("ACC"):
        return 8
    if channel_name.startswith("BVP"):
        return 64
    if channel_name.startswith("ECG"):
        return 65
    if channel_name == "label":
        return 10
    raise NoSuchSignal(channel_name)


class KEmoWorkSubject(Subject):
    def __init__(self, logger, path, label_type, subject_id, channels_names, get_sampling_fn):
        Subject.__init__(self, logger, path, label_type, subject_id, channels_names, get_sampling_fn)
        self._logger = logger
        self._path = path
        self._label_type = label_type
        self.id = subject_id

        data = self._load_subject_data_from_file()
        self._data = self._restructure_data(data)
        self._process_data()

    def _process_data(self):
        data = self._filter_all_signals(self._data)
        self._create_sliding_windows(data)

    def _load_subject_data_from_file(self):
        self._logger.info("Loading data for subject {}".format(self.id))
        data = self.load_subject_data_from_file(self._path, self.id)
        self._logger.info("Finished loading data for subject {}".format(self.id))

        return data

    @staticmethod
    def load_subject_data_from_file(path, id):
        with open("{0}/S{1}/S{1}.pkl".format(path, id), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    def _restructure_data(self, data):
        self._logger.info("Restructuring data for subject {}".format(self.id))
        signals = self.restructure_data(data, self._label_type)
        self._logger.info("Finished restructuring data for subject {}".format(self.id))

        return signals

    @staticmethod
    def restructure_data(data, label_type):
        # new_data = {'label': np.array(data['label'][label_type]), "signal": {}}
        new_data = {'label': np.array(data['label'][label_type].reshape(1,-1))[0], "signal": {}}
        for sensor in data['signal']:
            for i in range(len(data['signal'][sensor][0])):
                signal_name = '_'.join([sensor, str(i)])
                signal = np.array([x[i] for x in data['signal'][sensor]])
                new_data["signal"][signal_name] = signal
        return new_data

    def _filter_all_signals(self, data):
        self._logger.info("Filtering signals for subject {}".format(self.id))
        signals = data["signal"]
        for signal_name in signals:
            signals[signal_name] = filter_signal(signal_name, signals[signal_name], original_sampling, target_sampling)
        self._logger.info("Finished filtering signals for subject {}".format(self.id))
        return data

    def _create_sliding_windows(self, data):
        self._logger.info("Creating sliding windows for subject {}".format(self.id))

        self.x = [Signal(signal_name, target_sampling(signal_name), []) for signal_name in data["signal"]]

        # for i in range(0, len(data["signal"]["EDA_0"]) - 4*10, 4*1): #10sec*4Hz window and 1sec*4Hz sliding
        for i in range(0, len(data["signal"]["EDA_0"]) - 240, 120): # 60sec*4Hz window and 30sec*4Hz sliding
            first_index, last_index = self._indexes_for_signal(i, "label")
            personalized_threshold = np.mean(data["label"])
            label_window_mean = np.mean(data["label"][first_index:last_index])

            if label_window_mean not in range(0,20):
                continue

            channel_id = 0
            for signal in data["signal"]:
                first_index, last_index = self._indexes_for_signal(i, signal)
                self.x[channel_id].data.append(data["signal"][signal][first_index:last_index])
                channel_id += 1

            self.y.append(np.float64(1.0)) if label_window_mean > personalized_threshold else self.y.append(np.float64(0.0))

        self._logger.info("Finished creating sliding windows for subject {}".format(self.id))

    @staticmethod
    def _indexes_for_signal(i, signal):
        freq = target_sampling(signal)
        first_index = int((i * freq) // 4)
        window_size = int(10 * freq)
        return first_index, first_index + window_size
