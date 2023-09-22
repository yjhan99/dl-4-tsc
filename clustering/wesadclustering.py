import random
import math

import numpy as np

from utils.utils import prepare_data
from utils.loggerwrapper import GLOBAL_LOGGER
from arpreprocessing.dataset import Dataset

import keras
import keras_contrib

SIGNALS_LEN = 14

def n_fold_split_cluster_trait(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)
    subject_ids = list(subject_ids)
    subject_ids_female = list()
    subject_ids_male = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    if words[1] == 'male':
                        subject_ids_male.append(subject_id)
                    else:
                        subject_ids_female.append(subject_id)

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_set in test_sets:
        if test_set[0] in subject_ids_male:
            rest = [x for x in subject_ids if (x not in test_set) & (x in subject_ids_male)]
            val_set = random.sample(rest, math.ceil(len(rest) / 5))
            train_set = [x for x in rest if (x not in val_set) & (x in subject_ids_male)]
        else:
            rest = [x for x in subject_ids if (x not in test_set) & (x in subject_ids_female)]
            val_set = random.sample(rest, math.ceil(len(rest) / 5))
            train_set = [x for x in rest if (x not in val_set) & (x in subject_ids_female)]    
        result.append({"train": train_set, "val": val_set, "test": test_set})

    print(result)

    random.seed()
    return result


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


def n_fold_split_cluster_feature(subject_ids, n, seed=5):
    path = "../archives/mts_archive"
    dataset = Dataset("WESAD", None, GLOBAL_LOGGER)
    x, y, sampling_rate = dataset.load(path, subject_ids, SIGNALS_LEN)

    random.seed(seed)

    x_test, x_train = [[] for i in range(max(SIGNALS_LEN) + 1)], [[] for i in range(max(SIGNALS_LEN) + 1)]
    y_test, y_train = [], []
    sampling_test, sampling_train = sampling_rate, sampling_rate, sampling_rate

    for subject_id in subject_ids:
        for channel_id in range(len(SIGNALS_LEN)):
            signal = x[channel_id]

            num_rows = len(signal)
            split_size = num_rows // 5

            combined_list = list(zip(signal, y))
            random.shuffle(combined_list)
            shuffled_signal, shuffled_y = zip(*combined_list)

            for i in range(split_size):
                x_test[channel_id] += shuffled_signal[i]
            for i in range(split_size,num_rows):
                x_train[channel_id] += shuffled_signal[i]

        for i in range(split_size):
            y_test += shuffled_y[i]
        for i in range(split_size,num_rows):
            y_train += shuffled_y[i]

    random.seed()

    x_train = [np.expand_dims(np.array(x), 2) for x in x_train]
    x_test = [np.expand_dims(np.array(x), 2) for x in x_test]
    input_shapes, nb_classes, y_none, y_train, y_test, y_true = prepare_data(x_train, y_train, None, y_test)
    ndft_arr = [get_ndft(x) for x in sampling_test]

    result = []

    # random.seed(seed)
    # subject_ids = list(subject_ids)

    # test_sets = [subject_ids[i::n] for i in range(n)]

    # for test_set in test_sets:
    #     result.append({"train": test_set, "val": test_set, "test": test_set})

    # random.seed()
    return result

