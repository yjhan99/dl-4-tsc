from arpreprocessing.wesad import Wesad
from experiment.experiment import Experiment, prepare_experimental_setups_n_iterations, n_fold_split
import random

SIGNALS_LEN = 14

def n_fold_split_cluster_trait(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)
    subject_ids = list(subject_ids)
    subject_cluster = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    subject_cluster.append(words[1])

    test_sets = [subject_ids[i::n] for i in range(n)]

    # TODO: 같은 cluster만 이용해서 set 만들기
    for idx, test_set in enumerate(test_sets):
        
        result.append({"train": test_set, "val": test_set, "test": test_set})

    random.seed()
    return result


def n_fold_split_cluster_feature(subject_ids, n, seed=5):
    result = []

    random.seed(seed)
    subject_ids = list(subject_ids)

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_set in test_sets:
        result.append({"train": test_set, "val": test_set, "test": test_set})

    random.seed()
    return result

class WesadExperimentNFold(Experiment):
    def __init__(self, logger_obj, n, i, seed=5):
        # Person specific
        # folds = n_fold_split(Wesad.SUBJECTS_IDS, n, seed=seed)
        # Cluster specific (trait-based)
        folds = n_fold_split_cluster_trait(Wesad.SUBJECTS_IDS, n, "WESAD", seed=seed)
        # Cluster specific (feature-based)
        # folds = n_fold_split_cluster_feature(Wesad.SUBJECTS_IDS, n, seed=seed)

        self.test_ids = folds[i]["test"]
        self.val_ids = folds[i]["val"]
        self.train_ids = folds[i]["train"]

        Experiment.__init__(self, "WESAD", logger_obj, SIGNALS_LEN, dataset_name_suffix=f"_{n}fold_{i:02d}")

    def prepare_experimental_setups(self):
        prepare_experimental_setups_n_iterations(self, self.train_ids, self.val_ids, self.test_ids)

