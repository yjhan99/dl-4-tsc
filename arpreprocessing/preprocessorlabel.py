import gzip
import pickle
from abc import ABC, abstractmethod

from arpreprocessing.dataset import Dataset
from arpreprocessing.subjectlabel import SubjectLabel


class PreprocessorLabel(ABC):
    def __init__(self, logger, path, label_type, name, channels_names, get_sampling_fn, subject_cls=SubjectLabel):
        self._logger = logger
        self._path = path
        self._label_type = label_type
        self._name = name
        self.subjects = [subject_cls(self._logger, self._path, self._label_type, id_, channels_names, get_sampling_fn) for id_ in
                         self.get_subjects_ids()]

    def get_dataset(self):
        return Dataset(self._name, self, self._logger)

    @abstractmethod
    def get_subjects_ids(self):
        raise NotImplementedError()


def load_gzipped_pkl(path):
    with gzip.open(path + ".pickle.gz", 'rb') as f:
        load = pickle.load(f)
        return load
