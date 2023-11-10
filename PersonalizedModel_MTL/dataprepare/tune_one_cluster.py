import sys

from arpreprocessing.wesad import Wesad
from arpreprocessing.kemowork import KEmoWork
from utils.loggerwrapper import GLOBAL_LOGGER
from utils.utils import set_available_gpus
from clustering_mtl.wesadclustering import n_fold_split_cluster_trait_wesad, n_fold_split_cluster_feature_wesad
from clustering_mtl.kemoworkclustering import n_fold_split_cluster_trait_kemowork, n_fold_split_cluster_feature_kemowork


def prepare_dataset(name):
    if name.startswith("wesad"):
        # n_fold_split_cluster_feature_wesad(Wesad.SUBJECTS_IDS, 15, seed=5)
        n_fold_split_cluster_feature_wesad(Wesad.SUBJECTS_IDS, 15, seed=5)
        # n_fold_split_cluster_trait_wesad(Wesad.SUBJECTS_IDS, 15, "WESAD", seed=5)
        n_fold_split_cluster_trait_wesad(Wesad.SUBJECTS_IDS, 15, "WESAD", seed=5)
        # return WesadExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    if name.startswith("kemowork"):
        n_fold_split_cluster_feature_kemowork(KEmoWork.SUBJECTS_IDS, 19, seed=5)
        n_fold_split_cluster_trait_kemowork(KEmoWork.SUBJECTS_IDS, 19, "KEmoWork", seed=5)
        # return KEmoWorkExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))

if __name__ == '__main__':
    try:
        _, gpu_id, dataset_name  = sys.argv

        set_available_gpus(gpu_id)
        prepare_dataset(dataset_name)

    except Exception as e:
        GLOBAL_LOGGER.exception(e)
        raise e
