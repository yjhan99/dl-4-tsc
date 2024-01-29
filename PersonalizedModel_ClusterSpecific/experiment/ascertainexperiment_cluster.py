from arpreprocessing.ascertain import ASCERTAIN
from clustering.ascertainclustering import n_fold_split_cluster_trait, n_fold_split_cluster_trait_experiment
from experiment.experiment_cluster import Experiment, prepare_experimental_setups_n_iterations

SIGNALS_LEN = 6

class ASCERTAINExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        # folds = n_fold_split_cluster_trait(ASCERTAIN.SUBJECTS_IDS, n, "ASCERTAIN", seed=seed)
        folds = n_fold_split_cluster_trait_experiment(ASCERTAIN.SUBJECTS_IDS, n, "ASCERTAIN", seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]

        Experiment.__init__(self, "ASCERTAIN", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids)
