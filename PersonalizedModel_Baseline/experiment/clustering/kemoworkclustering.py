import random
import math
import os

import numpy as np

from utils.utils import prepare_data, get_new_session
from utils.loggerwrapper import GLOBAL_LOGGER
from arpreprocessing.dataset import Dataset

import keras
import tensorflow as tf
from tensorflow import Graph

from multimodal_classfiers.mlp_lstm import ClassifierMlpLstm
from multimodal_classfiers.hyperparameters import Hyperparameters

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder


SIGNALS_LEN = 11


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

def reshape_samples(samples):
    return [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in samples]


def n_fold_split_cluster_trait(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)
    subject_ids = list(subject_ids)

    file_path = "./archives/{0}/{0}_PreSurvey.csv".format(dataset_name)

    file = pd.read_csv(file_path)

    # Including only age and BFI-15 information
    X = file.iloc[:, [1, 3] + list(range(80, 95))]
    X.columns = ['pnum', 'age'] + [f'bfi_{i}' for i in range(1, 16)]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.iloc[:,1:])

    n_components = 3
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    silhouette_scores = []
    possible_K_values = [i for i in range(2,6)]

    for each_value in possible_K_values:
        clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_pca)
        silhouette_scores.append(silhouette_score(X_pca, cluster_labels))

    k_value = silhouette_scores.index(min(silhouette_scores)) # k = 2
    clusterer = KMeans(n_clusters=possible_K_values[k_value], init='k-means++', n_init='auto', random_state=42)
    cluster_labels = clusterer.fit_predict(X_pca)

    X['cluster'] = list(cluster_labels)
    
    subject_ids_0 = list()
    subject_ids_1 = list()

    for idx, row in X.iterrows():
        if row['cluster'] == 0:
            subject_ids_0.append(row['pnum'])
        else:
            subject_ids_1.append(row['pnum'])

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_set in test_sets:
        if test_set[0] in subject_ids_0:
            rest = [x for x in subject_ids if (x not in test_set) & (x in subject_ids_0)]
            val_set = random.sample(rest, math.ceil(len(rest) / 5))
            train_set = [x for x in rest if (x not in val_set) & (x in subject_ids_0)]
        else:
            rest = [x for x in subject_ids if (x not in test_set) & (x in subject_ids_1)]
            val_set = random.sample(rest, math.ceil(len(rest) / 5))
            train_set = [x for x in rest if (x not in val_set) & (x in subject_ids_1)]    
        result.append({"train": train_set, "val": val_set, "test": test_set})
        print(result)

    random.seed()
    return result


def n_fold_split_cluster_feature(subject_ids, n, seed=5):
    path = "./archives/mts_archive"
    dataset = Dataset("KEmoWork", None, GLOBAL_LOGGER)
    X, y, sampling_rate = dataset.load(path, subject_ids, tuple(range(SIGNALS_LEN)))

    X = [np.expand_dims(np.array(x, dtype=object), 2) for x in X]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    nb_classes = len(np.unique(y))
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1,1)).toarray()
    if type(X) == list:
        input_shapes = [x.shape[1:] for x in X]
    else:
        input_shapes = X.shape[1:]

    ndft_arr = [get_ndft(x) for x in sampling_rate]

    if len(input_shapes) != len(ndft_arr):
        raise Exception("Different sizes of input_shapes and ndft_arr")

    for i in range(len(input_shapes)):
        if input_shapes[i][0] < ndft_arr[i]:
            raise Exception(
                f"Too big ndft, i: {i}, ndft_arr[i]: {ndft_arr[i]}, input_shapes[i][0]: {input_shapes[i][0]}")
        
    print(input_shapes)
                
    with Graph().as_default():
        session = get_new_session()
        with session.as_default():
            with tf.device('/device:GPU:0'):
                session.run(tf.compat.v1.global_variables_initializer())

                input_layers = []
                channel_outputs = []
                # extra_dense_layers_no = 2
                dense_outputs = len(input_shapes) * [500]
                
                for channel_id, input_shape in enumerate(input_shapes):
                    input_layer = keras.layers.Input(shape=(None, round(input_shape[0] / 2), 1), name=f"input_for_{channel_id}")
                    input_layers.append(input_layer)

                    input_layer_flattened = keras.layers.TimeDistributed(keras.layers.Flatten())(input_layer)

                    # layer_1 = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(input_layer_flattened)
                    layer = keras.layers.TimeDistributed(keras.layers.Dense(dense_outputs[channel_id], activation='relu'))(input_layer_flattened)
                    layer = keras.layers.TimeDistributed(keras.layers.Dense(8, activation='relu'))(layer)
                    layer = keras.layers.TimeDistributed(keras.layers.Dense(8, activation='relu'))(layer)

                    # for i in range(extra_dense_layers_no):
                    #     # layer = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(layer)
                    #     layer = keras.layers.TimeDistributed(keras.layers.Dense(dense_outputs[channel_id], activation='relu'))(
                    #         layer)

                    # output_layer = keras.layers.TimeDistributed(keras.layers.Dropout(0.3))(layer)
                    output_layer = keras.layers.LSTM(8)(layer)
                    channel_outputs.append(output_layer)

                flat = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]
                output_layer = keras.layers.Dense(nb_classes, activation='softmax')(flat)

                model = keras.models.Model(inputs=input_layers, outputs=output_layer)

                for layer in model.layers:
                    layer_input_shape = layer.input_shape
                    layer_output_shape = layer.output_shape
                    print(f"Layer: {layer.name}")
                    print(f"Input Shape: {layer_input_shape}")
                    print(f"Output Shape: {layer_output_shape}")
                    print('-' * 50)

                model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.legacy.Adam(lr=0.003, decay=math.exp(-6)),
                            metrics=['accuracy'])
                
                X = reshape_samples(X)
                
                mini_batch_size = int(min(X[0].shape[0] / 10, 16))

                hist = model.fit(X, y, batch_size=mini_batch_size, epochs=100, verbose=False,
                                    validation_data=(X, y), callbacks=[keras.callbacks.EarlyStopping(patience=30)], shuffle=True)

                y_pred_probabilities = model.predict(X)

                y_pred = np.argmax(y_pred_probabilities, axis=1)

                print(y_pred)

                
    # random.seed(seed)

    # x_test, x_train = [[] for i in range(max(SIGNALS_LEN) + 1)], [[] for i in range(max(SIGNALS_LEN) + 1)]
    # y_test, y_train = [], []
    # sampling_test, sampling_train = sampling_rate, sampling_rate, sampling_rate

    # for subject_id in subject_ids:
    #     for channel_id in range(len(SIGNALS_LEN)):
    #         signal = x[channel_id]

    #         num_rows = len(signal)
    #         split_size = num_rows // 5

    #         combined_list = list(zip(signal, y))
    #         random.shuffle(combined_list)
    #         shuffled_signal, shuffled_y = zip(*combined_list)

    #         for i in range(split_size):
    #             x_test[channel_id] += shuffled_signal[i]
    #         for i in range(split_size,num_rows):
    #             x_train[channel_id] += shuffled_signal[i]

    #     for i in range(split_size):
    #         y_test += shuffled_y[i]
    #     for i in range(split_size,num_rows):
    #         y_train += shuffled_y[i]

    # random.seed()

    # x_train = [np.expand_dims(np.array(x), 2) for x in x_train]
    # x_test = [np.expand_dims(np.array(x), 2) for x in x_test]
    # input_shapes, nb_classes, y_none, y_train, y_test, y_true = prepare_data(x_train, y_train, None, y_test)
    # ndft_arr = [get_ndft(x) for x in sampling_test]

    result = []

    return result

