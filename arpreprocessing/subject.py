from arpreprocessing.signal import Signal


class Subject:
    def __init__(self, logger, path, label_type, id_, channels_names, get_sampling_fn):
        self._logger = logger
        self._path = path
        self._label = label_type
        self.id = id_

        self.x = [Signal(signal_name, get_sampling_fn(signal_name), []) for signal_name in channels_names]
        self.y = []
