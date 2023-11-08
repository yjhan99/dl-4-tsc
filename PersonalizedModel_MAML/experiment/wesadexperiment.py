import sys
sys.path.append('/home/haerang/git/dl-4-tsc/arpreprocessing')
sys.path.append('/home/haerang/git/dl-4-tsc/multimodal_classifiers')

from arpreprocessing.wesad import Wesad
#from experiment.experiment import Experiment, prepare_experimental_setups_n_iterations, n_fold_split
from experiment.experiment import prepare_experimental_setups_n_iterations, n_fold_split
from experiment.experiment import Experiment
import random

SIGNALS_LEN = 14


class WesadExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        # Person specific
        folds = n_fold_split(Wesad.SUBJECTS_IDS, n, seed=seed)
        # Cluster specific (trait-based)
        # folds = n_fold_split_cluster_trait(Wesad.SUBJECTS_IDS, n, "WESAD", seed=seed)
        # Cluster specific (feature-based)
        # folds = n_fold_split_cluster_feature(Wesad.SUBJECTS_IDS, n, seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]

        Experiment.__init__(self, "WESAD", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids)

