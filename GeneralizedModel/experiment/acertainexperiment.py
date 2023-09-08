from arpreprocessing.ascertain import Ascertain
from experiment.experiment import Experiment, n_fold_split, prepare_experimental_setups_n_iterations

SIGNALS_LEN = len(Ascertain.CHANNELS_NAMES)


class AscertainExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        folds = n_fold_split(Ascertain.SUBJECTS_IDS, n, seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]

        Experiment.__init__(self, "ASCERTAIN", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids)
