from arpreprocessing.wesad import Wesad
from clustering.wesadclustering import n_fold_split_cluster_trait_mtl
from experiment.experiment import Experiment, prepare_experimental_setups_n_iterations, n_fold_split

SIGNALS_LEN = 14


class WesadExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        folds = n_fold_split(Wesad.SUBJECTS_IDS, n, seed=seed)
        clusters = n_fold_split_cluster_trait_mtl(Wesad.SUBJECTS_IDS, n, "WESAD", seed=seed)
        # clusters = n_fold_split_cluster_feature_mtl(Wesad.SUBJECTS_IDS, n, seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]
        self.cluste_ids = clusters[i]["cluster"]

        Experiment.__init__(self, "WESAD", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids, self.cluste_ids)

