import math
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from filelock import Timeout, FileLock
from tensorflow import Graph

import sys
sys.path.append('/home/haerang/git/dl-4-tsc/arpreprocessing')
sys.path.append('/home/haerang/git/dl-4-tsc/multimodal_classfiers')

from arpreprocessing.dataset import Dataset
from multimodal_classfiers.cnn_lstm import ClassifierCnnLstm
from multimodal_classfiers.encoder import ClassifierEncoder
from multimodal_classfiers.fcn import ClassifierFcn
from multimodal_classfiers.hyperparameters import Hyperparameters
from multimodal_classfiers.inception_time import ClassifierInception
from multimodal_classfiers.mcdcnn import ClassifierMcdcnn
from multimodal_classfiers.mlp import ClassifierMlp
from multimodal_classfiers.mlp_lstm import ClassifierMlpLstm
from multimodal_classfiers.resnet import ClassifierResnet
from multimodal_classfiers.stresnet import ClassifierStresnet
from multimodal_classfiers.time_cnn import ClassifierTimeCnn
from multimodal_classfiers.maml import MAML
from utils.utils import get_new_session, prepare_data, prepare_data_maml
import itertools as it

#CLASSIFIERS = ("mcdcnnM", "cnnM", "mlpM", "fcnM", "encoderM", "resnetM", "inceptionM", "stresnetM", "mlpLstmM",
#               "cnnLstmM", "maml")
CLASSIFIERS = ("maml")

# Constants for MAML
NUM_TASKS_PER_META_ITERATION = 5
META_EPOCHS = 10    # Number of meta-epochs (outer loop iter)
TASK_SPECIFIC_EPOCHS = 5    # Number of epochs for each task (inner loop iter)
META_TRAIN_ITERATIONS = 10  # Number of meta-iter
NUM_SHOTS_PER_CLASS = 5 # TODO:



class NoSuchClassifier(Exception):
    def __init__(self, classifier_name):
        self.message = "No such classifier: {}".format(classifier_name)


def create_classifier(classifier_name, input_shapes, nb_classes, output_directory, verbose=False,
                      sampling_rates=None, ndft_arr=None, hyperparameters=None, model_init=None):
    if classifier_name == 'fcnM':
        return ClassifierFcn(output_directory, input_shapes, nb_classes, verbose, hyperparameters,
                             model_init=model_init)
    if classifier_name == 'mlpM':
        return ClassifierMlp(output_directory, input_shapes, nb_classes, verbose, hyperparameters,
                             model_init=model_init)
    if classifier_name == 'resnetM':
        return ClassifierResnet(output_directory, input_shapes, nb_classes, verbose, hyperparameters,
                                model_init=model_init)
    if classifier_name == 'encoderM':
        return ClassifierEncoder(output_directory, input_shapes, nb_classes, verbose, hyperparameters,
                                 model_init=model_init)
    if classifier_name == 'mcdcnnM':
        return ClassifierMcdcnn(output_directory, input_shapes, nb_classes, verbose, hyperparameters,
                                model_init=model_init)
    if classifier_name == 'cnnM':
        return ClassifierTimeCnn(output_directory, input_shapes, nb_classes, verbose, hyperparameters,
                                 model_init=model_init)
    if classifier_name == 'inceptionM':
        depth = hyperparameters.depth if hyperparameters and hyperparameters.depth else 6
        return ClassifierInception(output_directory, input_shapes, nb_classes, depth=depth, verbose=verbose,
                                   hyperparameters=hyperparameters, model_init=model_init)
    if classifier_name == 'stresnetM':
        return ClassifierStresnet(output_directory, input_shapes, sampling_rates,
                                  ndft_arr, nb_classes, verbose=verbose,
                                  hyperparameters=hyperparameters,
                                  model_init=model_init)
    if classifier_name == 'cnnLstmM':
        return ClassifierCnnLstm(output_directory, input_shapes, nb_classes, hyperparameters=hyperparameters,
                                 model_init=model_init)
    if classifier_name == 'mlpLstmM':
        return ClassifierMlpLstm(output_directory, input_shapes, nb_classes, verbose, hyperparameters,
                                 model_init=model_init)
    if classifier_name == 'maml':
        return MAML(output_directory, input_shapes, nb_classes, verbose, hyperparameters, 
                                model_init=model_init)  # TODO

    raise NoSuchClassifier(classifier_name)


class ExperimentalSetup():
    def __init__(self, name, x_train, y_train, x_val, y_val, x_test, y_test, input_shapes, sampling_val, ndft_arr,
                 nb_classes, nb_ecpochs_fn, batch_size_fn):
        self.name = name
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.input_shapes = input_shapes
        self.sampling_val = sampling_val
        self.ndft_arr = ndft_arr
        self.nb_classes = nb_classes
        self.nb_epochs_fn = nb_ecpochs_fn
        self.batch_size_fn = batch_size_fn
        
        self.train_ids = x_train.keys()
        self.val_ids = x_val.keys()
        self.test_ids = x_test.keys()


class Experiment(ABC):
    def __init__(self, dataset_name: str, logger, no_channels, dataset_name_suffix=""):
        self._tuning_iteration = 0
        self.dataset_name = dataset_name
        self.logger_obj = logger
        self.experimental_setups = None
        self.no_channels = no_channels
        self.experiment_path = f"results/{self.dataset_name}{dataset_name_suffix}"

        self.prepare_experimental_setups()

    @abstractmethod
    def prepare_experimental_setups(self):
        pass

    def perform(self, iterations, classifiers=CLASSIFIERS):
        for iteration in range(iterations):
            for classifier_name in classifiers:
                self.perform_on_one_classifier(classifier_name, iteration)

    def perform_on_one_classifier(self, classifier_name, iteration, hyperparameters=None, gpu=0):
        with Graph().as_default():
            session = get_new_session()
            with session.as_default():
                with tf.device(f'/device:GPU:{gpu}'):
                    model_init = None

                    for setup in self.experimental_setups:
                        # Create an output directory based on the experiment details
                        output_directory = f"{self.experiment_path}/{iteration}/{classifier_name}/{setup.name}/"
                        os.makedirs(output_directory, exist_ok=True)

                        # Attempt to run a single experiment using the 'perform_single_experiment' method
                        # which includes creating, fitting, and saving a classifier model
                        try:
                            session.run(tf.compat.v1.global_variables_initializer())
                            model_init = self.perform_single_experiment(classifier_name, output_directory, setup,
                                                                        iteration, hyperparameters, model_init)
                        except Timeout:
                            self.logger_obj.info("Experiment is being performed by another process")

    def perform_single_experiment(self, classifier_name: str, output_directory: str, setup: ExperimentalSetup,
                                  iteration, hyperparameters: Hyperparameters, model_init):
        logging_message = "Experiment for {} dataset, classifier: {}, setup: {}, iteration: {}".format(
            self.dataset_name, classifier_name, setup.name, iteration)
        self.logger_obj.info(logging_message)

        SUPPORT_SPLIT = 0.7
        QUERY_SPLIT = 0.3
        
        with FileLock(output_directory + "DOING.lock", timeout=0):
            done_dict_path = output_directory + "DONE"
            if os.path.exists(done_dict_path):
                self.logger_obj.info("Experiment already performed")
                return
            
            # List to store tasks, each containing 'support' and 'query' data
            tasks = []
            
            for user_id in setup.train_ids:
                # Define 'support' and 'query' sets for each user (task)
                support_set = {'x': [], 'y': []}
                query_set = {'x': [], 'y': []}
                
                num_channels = len(setup.x_train[user_id])
                num_samples = len(setup.x_train[user_id][0])    # get # of sampels from an arbitrary channel
                num_support_samples = int(num_samples * SUPPORT_SPLIT)
                num_query_samples = num_samples - num_support_samples
                
                # print(f"{num_support_samples=}")  
                # print(f"{num_query_samples=}")    
                
                print(f"{len(setup.x_train)=}")
                
                support_set['x'].extend(setup.x_train[user_id][:][:num_support_samples])
                support_set['y'].extend(setup.y_train[user_id][:][:num_support_samples])
                query_set['x'].extend(setup.x_train[user_id][:][num_support_samples:])
                query_set['y'].extend(setup.y_train[user_id][:][num_support_samples:])
                
                # print(f"{setup.x_train[user_id][:][0]=}")
                
                tasks.append({'support': support_set, 'query': query_set})
            #input_shapes = tasks[0]['support']['x'][0].shape
    
            # Initialize MAML model
            print(f"{setup.input_shapes=}")
            
            
            
            model = create_classifier(classifier_name, setup.input_shapes, setup.nb_classes,
            #model = create_classifier(classifier_name, input_shapes, setup.nb_classes,
                                output_directory,
                                verbose=False,
                                sampling_rates=setup.sampling_val, ndft_arr=setup.ndft_arr,
                                hyperparameters=hyperparameters, model_init=model_init)
            
            # Meta-train the MAML model on the tasks
            model.maml_train(tasks, setup.batch_size_fn(classifier_name), META_EPOCHS, TASK_SPECIFIC_EPOCHS, META_TRAIN_ITERATIONS)
            
            
            # Evaluate the meta-trained model on a specific task
            x_test_task, y_true_task = setup.x_test, setup.y_test
            y_pred, y_pred_probabilities = classifier.evaluate(x_test_task, y_true_task)
            # classifier.fit(setup.x_train, setup.y_train, setup.x_val, setup.y_val, setup.y_test,
            #                x_test=setup.x_test, nb_epochs=setup.nb_epochs_fn(classifier_name),
            #                batch_size=setup.batch_size_fn(classifier_name))
            
            self.logger_obj.info(
                f"Fitted model for {self.dataset_name} dataset, classifier: {classifier_name}, setup: {setup.name}, iteration: {iteration}")

            os.makedirs(done_dict_path)
            self._clean_up_files(output_directory)
            self.logger_obj.info("Finished e" + logging_message[1:])

            return classifier.model

    @staticmethod
    def _clean_up_files(output_directory):
        best_model_path = output_directory + "best_model.hdf5"
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
  

def get_experimental_setup(logger_obj, channels_ids, test_ids, train_ids, val_ids, name, dataset_name, seed=5):
    path = "../archives/mts_archive"
    dataset = Dataset(dataset_name, None, logger_obj)
    
    random.seed(seed)
    
    # Initialize lists to hold data for each user/task
    x_test = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in test_ids}
    x_val = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in val_ids}
    x_train = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in train_ids}
    y_test = {user_id: [] for user_id in test_ids}
    y_val = {user_id: [] for user_id in val_ids}
    y_train = {user_id: [] for user_id in train_ids}
    
    all_user_ids = test_ids + val_ids + train_ids
    
    # Load data for each user/task
    for user_id in all_user_ids:
        x_test_single_user, x_val_single_user, x_train_single_user = [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)]
        y_test_single_user, y_val_single_user, y_train_single_user = [], [], []
        
        # Load data for the current user/task
        x, y, sampling_rate = dataset.load(path, [user_id], channels_ids)
        
        random.seed(seed)
        sampling_test, sampling_val, sampling_train = sampling_rate, sampling_rate, sampling_rate
        
        # Process input data x
        for channel_id in range(len(channels_ids)):
            signal = x[channel_id]
            
            num_rows = len(signal)
            # print(f"channel_id is {channel_id}, and {len(signal)=}")
            
            # Shuffle data
            combined_list = list(zip(signal, y))
            random.shuffle(combined_list)
            shuffled_signal, shuffled_y = zip(*combined_list)
            
            if user_id in train_ids:
                for i in range(num_rows):
                    x_train_single_user[channel_id].append(shuffled_signal[i])
            elif user_id in val_ids:
                for i in range(num_rows):
                    x_val_single_user[channel_id].append(shuffled_signal[i])
            elif user_id in test_ids:
                for i in range(num_rows):
                    x_test_single_user[channel_id].append(shuffled_signal[i])
            else:
                # TODO: Error handling
                print("Invalid user_id")
        
        if user_id in train_ids:
            for i in range(num_rows):
                y_train_single_user.append(shuffled_y[i])
            x_train_single_user = [np.expand_dims(np.array(x), 2) for x in x_train_single_user]
        elif user_id in val_ids:
            for i in range(num_rows):
                y_val_single_user.append(shuffled_y[i])
            x_val_single_user = [np.expand_dims(np.array(x), 2) for x in x_val_single_user]
        elif user_id in test_ids:
            for i in range(num_rows):
                y_test_single_user.append(shuffled_y[i])
            x_test_single_user = [np.expand_dims(np.array(x), 2) for x in x_test_single_user]
        else:
            # TODO: Error handling
            print("Invalid user_id")
        
        random.seed()
        
        x_train[user_id] = x_train_single_user
        x_val[user_id] = x_val_single_user
        x_test[user_id] = x_test_single_user
        y_train[user_id] = y_train_single_user
        y_val[user_id] = y_val_single_user
        y_test[user_id] = y_test_single_user
        
    
    input_shapes, nb_classes, y_val, y_train, y_test, y_true = prepare_data_maml(x_train, y_train, y_val, y_test)
    # input_shapes = [x.shape[1:] for x in x_train[user_id] for user_id in x_train.keys()]
    # concatenated_ys = []
    # concatenated_ys.extend([value for values in y_train.values() for value in values])
    # concatenated_ys.extend([value for values in y_val.values() for value in values])
    # concatenated_ys.extend([value for values in y_test.values() for value in values])
    # nb_classes = len(set(concatenated_ys))
    ndft_arr = [get_ndft(x) for x in sampling_test] # TODO: Q. Is this necessary?
    
    # if len(input_shapes) != len(ndft_arr):
    #     raise Exception("Different sizes of input_shapes and ndft_arr")

    # for i in range(len(input_shapes)):
    #     if input_shapes[i][0] < ndft_arr[i]:
    #         raise Exception(
    #             f"Too big ndft, i: {i}, ndft_arr[i]: {ndft_arr[i]}, input_shapes[i][0]: {input_shapes[i][0]}")
    
    experimental_setup = ExperimentalSetup(name, x_train, y_train, x_val, y_val, x_test, y_test, input_shapes,
                                           sampling_val, ndft_arr, nb_classes, lambda x: 150, get_batch_size)
    return experimental_setup

    '''
    path = "../archives/mts_archive"
    dataset = Dataset(dataset_name, None, logger_obj)
    signal_test, label_test, sampling_test = dataset.load(path, test_ids, channels_ids)
    signal_val, label_val, sampling_val = dataset.load(path, val_ids, channels_ids)
    signal_train, label_train, sampling_train = dataset.load(path, train_ids, channels_ids)
    
    # Initialize lists to hold data for each user/task
    x_test, x_val, x_train = [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)]
    y_test, y_val, y_train = [], [], []
    
    for channel_id in range(len(channels_ids)):
        signal_single_channel_test = signal_test[channel_id]
        signal_single_channel_val = signal_val[channel_id]
        signal_single_channel_train = signal_train[channel_id]

        num_rows_test = len(signal_single_channel_test)
        num_rows_val = len(signal_single_channel_val)
        num_rows_train = len(signal_single_channel_train)

        combined_list_test = list(zip(signal_single_channel_test, label_test))
        combined_list_val = list(zip(signal_single_channel_val, label_val))
        combined_list_train = list(zip(signal_single_channel_train, label_train))

        random.shuffle(combined_list_test)
        random.shuffle(combined_list_val)
        random.shuffle(combined_list_train)

        shuffled_signal_single_channel_test, shuffled_label_test = zip(*combined_list_test)
        shuffled_signal_single_channel_val, shuffled_label_val = zip(*combined_list_val)
        shuffled_signal_single_channel_train, shuffled_label_train = zip(*combined_list_train)

        # print(f"{len(shuffled_signal_single_channel_test)=}")

        # TODO: Confirm if this code works find
        for i in range(len(shuffled_signal_single_channel_test)):
            x_test[channel_id].append(shuffled_signal_single_channel_test[i])
        for i in range(len(shuffled_signal_single_channel_val)):
            x_val[channel_id].append(shuffled_signal_single_channel_val[i])
        for i in range(len(shuffled_signal_single_channel_train)):
            x_train[channel_id].append(shuffled_signal_single_channel_train[i])
    
    for i in range(len(shuffled_label_test)):
        y_test.append(shuffled_label_test[i])
    for i in range(len(shuffled_label_val)):
        y_val.append(shuffled_label_val[i])
    for i in range(len(shuffled_label_train)):
        y_train.append(shuffled_label_train[i])

    x_train = [np.expand_dims(np.array(x), 2) for x in x_train]
    x_val = [np.expand_dims(np.array(x), 2) for x in x_val]
    x_test = [np.expand_dims(np.array(x), 2) for x in x_test]
    input_shapes, nb_classes, y_val, y_train, y_test, y_true = prepare_data(x_train, y_train, y_val, y_test)
    ndft_arr = [get_ndft(x) for x in sampling_test]

    if len(input_shapes) != len(ndft_arr):
        raise Exception("Different sizes of input_shapes and ndft_arr")

    for i in range(len(input_shapes)):
        if input_shapes[i][0] < ndft_arr[i]:
            raise Exception(
                f"Too big ndft, i: {i}, ndft_arr[i]: {ndft_arr[i]}, input_shapes[i][0]: {input_shapes[i][0]}")
    experimental_setup = ExperimentalSetup(name, x_train, y_train, x_val, y_val, x_test, y_test, input_shapes,
                                           sampling_val, ndft_arr, nb_classes, lambda x: 150, get_batch_size)
    return experimental_setup
    '''

def get_ndft(sampling):
    if sampling <= 2:
        return 8
    if sampling <= 4:
        return 16
    if sampling <= 8:
        return 32
    if sampling <= 16:
        return 64
    if sampling <= 32:
        return 128
    if sampling in [70, 64, 65]:
        return 256
    raise Exception(f"No such sampling as {sampling}")


def get_batch_size(classifier_name):
    if classifier_name == "inceptionM":
        return 2
    if classifier_name == "resnetM":
        return 4
    if classifier_name == "encoderM":
        return 2
    if classifier_name == "mlpLstmM":
        return 16
    if classifier_name == "fcnM":
        return 4
    return 32


def n_fold_split(subject_ids, n, seed=5):
    """
    Custom n-fold cross-validation function for user-based meta-learning.

    Args:
    subject_ids (list or array): A list of user IDs to be split.
    n (int): The number of folds.
    seed (int): Random seed for reproducibility.

    Returns:
    result (list of dictionaries): A list of n-fold splits, each containing a unique combination of
    training, validation, and test users.
    """
    # In a user-based meta-learning setup, each user is treated as a separate task
    # Goal: train a model that can quickly adapt to the behaviors of different users.
    # => need to ensure that the samples from a particular user are distributed across different folds.
    # to make the model has opportunity to adapt to a diverse set of user during cross-validation.
    result = []

    random.seed(seed)
    user_ids = list(subject_ids)

    # Shuffle the user IDs randomly
    # to ensure randomness and then distributes users across different folds
    random.shuffle(user_ids)

    fold_size = len(user_ids) // n
    fold_remainder = len(user_ids) % n

    for fold in range(n):
        fold_start = fold * fold_size
        fold_end = (fold+1) * fold_size if fold < n - 1 else len(user_ids)
        test_users = user_ids[fold_start:fold_end]

        train_users = user_ids[:fold_start] + user_ids[fold_end:]
        
        val_fold = (fold + 1) % n   # Choose the next fold as the validation fold
        val_fold_start = val_fold * fold_size
        val_fold_end = (val_fold + 1) * fold_size if val_fold < n - 1 else len(user_ids)
        val_users = user_ids[val_fold_start:val_fold_end]
        # print(f"{fold}-fold:::")
        # print(f"{train_users=}")
        # print(f"{val_users=}")
        # print(f"{test_users=}")

        result.append({'train': train_users, 'val': val_users, 'test': test_users})
    
    random.seed()
    return result
    

# def prepare_experimental_setups_n_iterations(self_experiment: Experiment, train_ids, val_ids, test_ids, iterations=5):
# # For paper, we might need more iterations
def prepare_experimental_setups_n_iterations(self_experiment: Experiment, train_ids, val_ids, test_ids, iterations=1):
    self_experiment.experimental_setups = []

    for i in range(iterations):
        self_experiment.experimental_setups.append(
            get_experimental_setup(self_experiment.logger_obj, tuple(range(self_experiment.no_channels)),
                                   test_ids, train_ids, val_ids, f"it_{i:02d}", self_experiment.dataset_name, seed=5))
