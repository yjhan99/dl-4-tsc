from arpreprocessing.wesad import Wesad
from clustering.wesadclustering import n_fold_split_mtl_cluster_trait, n_fold_split_cluster_feature
from experiment.experiment_cluster import Experiment, prepare_experimental_setups_n_iterations

SIGNALS_LEN = 14


class WesadExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        # Cluster specific (trait-based)
        folds = n_fold_split_mtl_cluster_trait(Wesad.SUBJECTS_IDS, n, "WESAD", seed=seed)
        # Cluster specific (feature-based)
        # folds = n_fold_split_cluster_feature(Wesad.SUBJECTS_IDS, n, seed=seed)

        self.test_ids = folds[i]["test"]
        self.task_test = folds[i]["task_test"]
        self.task_rest = []
        for n in range(len(folds[0])-2):
            values = list(folds[i].values())
            self.task_rest.append(values[n])
            # setattr(self, f'task_{n+1}', folds[i].keys()[n])

        Experiment.__init__(self, "WESAD", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.task_rest, self.task_test, self.test_ids)

