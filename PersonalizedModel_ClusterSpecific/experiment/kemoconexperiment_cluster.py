from arpreprocessing.kemocon import KEmoCon
from clustering.kemoconclustering import n_fold_split_cluster_trait, n_fold_split_cluster_trait_experiment
from experiment.experiment_cluster import Experiment, prepare_experimental_setups_n_iterations

SIGNALS_LEN = 6


class KEmoConExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        folds = n_fold_split_cluster_trait(KEmoCon.SUBJECTS_IDS, n, seed=seed)
        # folds = n_fold_split_cluster_trait_experiment(KEmoCon.SUBJECTS_IDS, n, seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]

        Experiment.__init__(self, "KEmoCon", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids)
