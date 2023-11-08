import sys

from experiment.hyperparametertuning import HyperparameterTuning
from experiment.wesadexperiment import WesadExperimentNFold
from experiment.kemoworkexperiment import KEmoWorkExperimentNFold
from utils.loggerwrapper import GLOBAL_LOGGER
from utils.utils import set_available_gpus


def get_dataset(name):
    if name.startswith("wesad_fold_"):
        return WesadExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    if name.startswith("kemowork_fold_"):
        return KEmoWorkExperimentNFold(GLOBAL_LOGGER, int(name[-5:-3]), int(name[-2:]))
    raise Exception(f"No such dataset/experiment as {name}")


if __name__ == '__main__':
    try:
        _, gpu_id, dataset_name, clas, max_eval, num_shot = sys.argv

        set_available_gpus(gpu_id)
        dataset = get_dataset(dataset_name)

        HyperparameterTuning(dataset, [gpu_id]).tune_one(clas, int(max_eval))

    except Exception as e:
        GLOBAL_LOGGER.exception(e)
        raise e
