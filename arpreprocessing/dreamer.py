import itertools as it
import pickle

import numpy as np
import scipy.stats

from arpreprocessing.helpers import filter_signal, get_empatica_sampling
from arpreprocessing.preprocessor import Preprocessor
from arpreprocessing.signal import Signal, NoSuchSignal
from arpreprocessing.subject import Subject
from arpreprocessing.DataAugmentation_TimeseriesData import DataAugmentation
    

class Dreamer(Preprocessor):
    SUBJECTS_IDS = tuple(range(0, 23))
    CHANNELS_NAMES = [f"eeg_channel_{i+1}" for i in range(14)] + [f"ecg_channel_{i+1}" for i in range(2)]

    def __init__(self, logger, path):
        Preprocessor.__init__(self, logger, path, "Dreamer", [], None, subject_cls=DreamerSubject)

    def get_subjects_ids(self):
        return self.SUBJECTS_IDS


def original_sampling(channel_name: str):
    if channel_name.startswith("eeg"):
        return 128
    if channel_name.startswith("ecg"):
        return 256
    return NoSuchSignal(channel_name)


def target_sampling(channel_name: str):
    if channel_name.startswith("eeg"):
        return 32
    if channel_name.startswith("ecg"):
        return 64
    # if channel_name == "label":
    #     return 10
    raise NoSuchSignal(channel_name)


class DreamerSubject(Subject):
    def __init__(self, logger, path, subject_id, channels_names, get_sampling_fn):
        Subject.__init__(self, logger, path, subject_id, channels_names, get_sampling_fn)
        self._logger = logger
        self._path = path
        self.id = subject_id

        for video_num in range(0,18):
            data = self._load_subject_data_from_file(video_num)
            self._data = self._restructure_data(data)
            self._process_data()

    def _process_data(self):
        data = self._filter_all_signals(self._data)
        self._create_sliding_windows(data)

    def _load_subject_data_from_file(self, video_num):
        self._logger.info("Loading data for subject {}".format(self.id))
        data = self.load_subject_data_from_file(self._path, self.id, video_num)
        self._logger.info("Finished loading data for subject {}".format(self.id))

        return data
    
    @staticmethod
    def load_subject_data_from_file(path, id, video_num):
        with open("{0}/S{1}/S{1}_V{2}.pkl".format(path, id, video_num), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    def _restructure_data(self, data):
        self._logger.info("Restructuring data for subject {}".format(self.id))
        signals = self.restructure_data(data)
        # signals = self.restructure_data_with_augmentation(data)
        self._logger.info("Finished restructuring data for subject {}".format(self.id))

        return signals
    
    @staticmethod
    def restructure_data(data):
        new_data = {'label': np.array(data['label']["arousal"]), "signal": {}}
        for type in data['signal']:
            print('type:', type)
            data['signal'][type] = data['signal'][type].reshape(-1,1)
            for i in range(len(data["signal"][type][0])):
                signal = np.array([x[i] for x in data['signal'][type]])
                new_data["signal"][type] = signal
        # for device in data['signal']:
        #     print('device:', device)
        #     for type in data['signal'][device]:
        #         print('type:', type)
        #         for i in range(len(data['signal'][device][type][0])):
        #             signal_name = '_'.join([device, type, str(i)])
        #             signal = np.array([x[i] for x in data['signal'][device][type]])
        #             new_data["signal"][signal_name] = signal
        return new_data
    
    @staticmethod
    def restructure_data_with_augmentation(data):
        duplicated_labels = np.tile(data['label'], 7)
        new_data = {'label': duplicated_labels, "signal": {}}
        for device in data['signal']:
            print('device:', device)
            for type in data['signal'][device]:
                print('type:', type)
                for i in range(len(data['signal'][device][type][0])):
                    signal_name = '_'.join([device, type, str(i)])
                    signal = np.array([x[i] for x in data['signal'][device][type]])
                    data_augmentor = DataAugmentation(signal)
                    signal_augmented = data_augmentor.apply_all_augmentations()
                    new_data["signal"][signal_name] = signal_augmented
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

        # self.x = [Signal(signal_name, target_sampling(signal_name), []) for signal_name in data["signal"]]
        if len(self.x) == 0:
            self.x = [Signal(signal_name, target_sampling(signal_name), []) for idx,signal_name in enumerate(data["signal"])]
        else:
            self.x = [Signal(signal_name, target_sampling(signal_name), self.x[idx].data) for idx,signal_name in enumerate(data["signal"])]

        for i in range(0, len(data["signal"]["eeg_channel_1"]) - 10*32, 5*32): # 10sec*4Hz window and 5sec*4Hz sliding
        # for i in range(0, len(data["signal"]["wrist_EDA_0"]) - 240, 120): # 60sec*4Hz window and 30sec*4Hz sliding
            # first_index, last_index = self._indexes_for_signal(i, "label")
            # label_id = scipy.stats.mstats.mode(data["label"][first_index:last_index])[0][0]
            label_id = data["label"]

            channel_id = 0
            for signal in data["signal"]:
                first_index, last_index = self._indexes_for_signal(i, signal)
                self.x[channel_id].data.append(data["signal"][signal][first_index:last_index])
                channel_id += 1

            self.y.append(label_id)

        self._logger.info("Finished creating sliding windows for subject {}".format(self.id))

    @staticmethod
    def _indexes_for_signal(i, signal):
        freq = target_sampling(signal)
        first_index = int((i * freq) // 32) # Due to eeg's sampling rate 4Hz
        window_size = int(10 * freq)
        # window_size = int(60 * freq)
        return first_index, first_index + window_size