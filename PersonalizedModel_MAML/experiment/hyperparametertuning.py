import os
import pickle
import shutil
import time
from multiprocessing import Queue

import pandas as pd
from filelock import FileLock, Timeout
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from experiment.experiment import Experiment, Hyperparameters
from utils.loggerwrapper import GLOBAL_LOGGER
from utils.utils import NoSuchClassifier


def get_search_space(no_channels, classifier_name):
    result = {}

    # optimizer_subspace = [hp.randint("lr", -7, -1), hp.choice("decay", [.001, .0001, .00001, 0]),
    optimizer_subspace = [hp.choice("lr", [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]), hp.choice("decay", [.001, .0001, .00001, 0]),
                          hp.choice("reduce_lr_factor", [0.5, 0.2, 0.1]), hp.choice("reduce_lr_patience", [5, 10])]

    subspace = [hp.choice(f"dense_output_{i:02d}", [250, 500, 1000]) for i in range(no_channels)]
    space = (optimizer_subspace, hp.choice("extra_dense_layers_no", [1, 2, 3]), subspace)
    result["mlpM"] = space
    result["mlpLstmM"] = space


    subspace1 = [hp.choice(f"filters_multiplier_{i:02d}", [0.5, 1, 2]) for i in range(no_channels)]
    subspace2 = [hp.choice(f"kernel_size_multipliers_{i:02d}", [0.5, 1, 2]) for i in range(no_channels)]
    space = (optimizer_subspace, subspace1, subspace2)
    result["mcdcnnM"] = space
    result["cnnM"] = space
    result["encoderM"] = space
    result["fcnM"] = space
    result["maml"] = space  # Add search space for maml

    space = (optimizer_subspace, subspace1, subspace2,
             [hp.choice(f"lstm_units_{i:02d}", [32, 64, 128]) for i in range(no_channels)])
    result["cnnLstmM"] = space

    subspace1 = [hp.choice(f"filters_{i:02d}", [32, 64, 128]) for i in range(no_channels)]
    subspace2 = [hp.choice(f"kernel_size_multiplier_{i:02d}", [1, 2, 4]) for i in range(no_channels)]
    space = (optimizer_subspace, hp.choice("depth", [2, 3, 4]), subspace1, subspace2)
    result["resnetM"] = space

    subspace1 = [hp.choice(f"filters_{i:02d}", [16, 32, 64]) for i in range(no_channels)]
    subspace2 = [hp.choice(f"kernel_size_{i:02d}", [21, 41, 81]) for i in range(no_channels)]
    space = (optimizer_subspace, hp.choice("depth", [5, 6, 7]), subspace1, subspace2)
    result["inceptionM"] = space

    subspace1 = [hp.choice(f"filters_{i:02d}", [32, 64, 128]) for i in range(no_channels)]
    subspace2 = [hp.choice(f"kernel_size_multiplier_{i:02d}", [0.5, 1, 2]) for i in range(no_channels)]
    space = (optimizer_subspace, hp.choice("depth", [5, 6, 7]), subspace1, subspace2)
    result["stresnetM"] = space

    return result[classifier_name]


def get_hyperparameters(classifier_name, x):
    lr, decay, reduce_lr_factor, reduce_lr_patience = x[0]

    if classifier_name in ["mcdcnnM", "cnnM", "encoderM", "fcnM"]:
        return Hyperparameters(lr, decay, reduce_lr_factor, reduce_lr_patience, filters_multipliers=x[1],
                               kernel_size_multipliers=x[2])
    if classifier_name == "cnnLstmM":
        return Hyperparameters(lr, decay, reduce_lr_factor, reduce_lr_patience, filters_multipliers=x[1],
                               kernel_size_multipliers=x[2], lstm_units=x[3])
    if classifier_name in ["resnetM", "stresnetM"]:
        return Hyperparameters(lr, decay, reduce_lr_factor, reduce_lr_patience, depth=x[1], filters=x[2],
                               kernel_size_multipliers=x[3])
    if classifier_name in ["mlpM", "mlpLstmM"]:
        return Hyperparameters(lr, decay, reduce_lr_factor, reduce_lr_patience, extra_dense_layers_no=x[1],
                               dense_outputs=x[2])
    if classifier_name == "inceptionM":
        return Hyperparameters(lr, decay, reduce_lr_factor, reduce_lr_patience, depth=x[1], filters=x[2],
                               kernel_sizes=x[3])
    if classifier_name == "maml":
        # FixMe: HR: Where can I set these values? Tentative values for now
        return Hyperparameters(lr, decay, reduce_lr_factor, reduce_lr_patience, filters_multipliers=x[1],
                               kernel_size_multipliers=x[2])
        # MAML-specific hyperparameters
        # lr = 0.001
        # decay = 1e-6
        # reduce_lr_factor = 0.5
        # reduce_lr_patience = 10
        # filters_multipliers = [1.0, 0.5, 0.2]  # Multipliers for filters in different layers
        # depth = 3  # Depth of the MAML model
        # filters = [64, 128, 256]  # Number of filters in each layer
        # kernel_sizes = [3, 3, 3]  # Kernel sizes for convolutional layers
        # dense_outputs = [128, 64]  # Output sizes for dense layers
        #return Hyperparameters(lr, decay, reduce_lr_factor, reduce_lr_patience, filters_multipliers, depth, filters, kernel_sizes, dense_outputs)
    raise NoSuchClassifier(classifier_name)


def wait_for_threads(threads, max_treads):
    while len(threads) > max_treads:
        threads = [t for t in threads if t.is_alive()]
        time.sleep(1)
    return threads


class HyperparameterTuning():
    def __init__(self, experiment: Experiment, gpus):
        self.experiment = experiment
        self.logger_obj = experiment.logger_obj

        self._gpus = gpus
        self._free_gpus = Queue()
        for gpu_id in range(len(self._gpus)):
            self._free_gpus.put(gpu_id)

        self.trials_path = f"{self.experiment.experiment_path}/trials_objs/"
        os.makedirs(self.trials_path, exist_ok=True)

    def tune_one(self, classifier_name, max_evals):
        self.logger_obj.info(f"Tuning of {classifier_name}, max_evals: {max_evals}")
        try:
            with FileLock(f"{self.trials_path}/{classifier_name}.lock", timeout=0):
                trials = self.load_trials(classifier_name)
                next_trial_no = len(trials.trials)
                shutil.rmtree(
                    f"{self.experiment.experiment_path}/tune_{next_trial_no:02d}/{classifier_name}",
                    ignore_errors=True)

                for trial_no in range(next_trial_no, max_evals):
                    fmin(fn=lambda x: self._objective(classifier_name, x, trial_no),
                         space=get_search_space(self.experiment.no_channels, classifier_name),
                         algo=tpe.suggest,
                         max_evals=trial_no + 1,
                         trials=trials)

                    self.save_trials(classifier_name, trials)
        except Timeout:
            self.logger_obj.info(
                f"Tuning of {classifier_name} (max_evals: {max_evals}) is being performed by another process")
        except Exception as e:
            GLOBAL_LOGGER.exception(e)

    def load_trials(self, classifier_name):
        trials_filename = f"{self.trials_path}/{classifier_name}.pkl"

        try:
            if os.path.exists(trials_filename):
                with open(trials_filename, "rb") as f:
                    return pickle.load(f)
        except EOFError:
            pass

        trials = Trials()
        self.save_trials(classifier_name, trials)
        return trials

    def save_trials(self, classifier_name, trials):
        with open(f"{self.trials_path}/{classifier_name}.pkl", "wb") as f:
            pickle.dump(trials, f)

    def _objective_with_error_handling(self, classifier_name, x, tuning_iteration):
        try:
            self._objective(classifier_name, x, tuning_iteration)
        except Exception as e:
            GLOBAL_LOGGER.exception(e)

    def _objective(self, classifier_name, x, tuning_iteration):
        iteration = f"tune_{tuning_iteration:02d}"

        gpu_id = self._free_gpus.get()
        self._experiment_on_gpu(classifier_name, iteration, x, gpu_id)

        loss = 0
        for setup in self.experiment.experimental_setups:
            best_model_stats_path = f"{self.experiment.experiment_path}/{iteration}/{classifier_name}/{setup.name}/df_best_model.csv"
            best_model_stats = pd.read_csv(best_model_stats_path)
            loss += best_model_stats.loc[0, "best_model_val_loss"]

        return {"status": STATUS_OK,
                "loss": loss,
                "iteration": iteration}

    def _experiment_on_gpu(self, classifier_name, iteration, x, gpu_id):
        try:
            GLOBAL_LOGGER.info(f"GPU {gpu_id} was taken for {classifier_name}, {iteration}.")
            self.experiment.perform_on_one_classifier(classifier_name, iteration,
                                                      get_hyperparameters(classifier_name, x), gpu=gpu_id)
        except Exception as e:
            GLOBAL_LOGGER.exception(f"For {classifier_name}, {iteration}, {gpu_id}, there was exception: {e}, x: {x}")
        finally:
            self._free_gpus.put(gpu_id)
            GLOBAL_LOGGER.info(f"GPU {gpu_id} was released by {classifier_name}, {iteration}.")
