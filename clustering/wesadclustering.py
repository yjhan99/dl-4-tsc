import random
import math
import os
from scipy import stats
from scipy.spatial.distance import euclidean

import numpy as np

from utils.utils import get_new_session
from utils.loggerwrapper import GLOBAL_LOGGER
from arpreprocessing.dataset import Dataset

import keras
import tensorflow as tf
from tensorflow import Graph

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

SIGNALS_LEN = 14


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


class GaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        return inputs
      

class Autoencoder(keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.input_shapes = input_shape

        e_input_layer = keras.layers.Input(input_shape)
        e_input_layer_flattened = keras.layers.Flatten()(e_input_layer)
        e_input_expanded = keras.layers.Reshape((1,-1))(e_input_layer_flattened)
        e_layer = keras.layers.Dense(4, activation='relu')(e_input_expanded)
        e_layer = keras.layers.Dense(4, activation='relu')(e_layer)
        e_lstm_layer = keras.layers.LSTM(4, return_sequences=False)(e_layer)

        self.encoder = keras.models.Model(inputs=e_input_layer, outputs=e_lstm_layer)

        d_input_layer = keras.layers.Input(e_lstm_layer.shape)
        d_input_layer_noise = GaussianNoiseLayer(stddev=0.1)(d_input_layer)
        d_input_layer_reshaped = keras.layers.Reshape(target_shape=(1,4))(d_input_layer_noise)
        d_lstm_layer = keras.layers.LSTM(4, return_sequences=False)(d_input_layer_reshaped)
        d_layer = keras.layers.Dense(4, activation='relu')(d_lstm_layer)
        d_layer = keras.layers.Dense(input_shape[0], activation='relu')(d_layer)
        d_reshaped_layer = keras.layers.Reshape(target_shape=input_shape)(d_layer)

        self.decoder = keras.models.Model(inputs=d_input_layer, outputs=d_reshaped_layer)

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded


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


def n_fold_split_cluster_feature(subject_ids, n, seed=5):
    test_sets = [subject_ids[i::n] for i in range(n)]

    result = []

    random.seed(seed)

    for test_subject in test_sets:
        print(test_subject)
        rest = [x for x in subject_ids if (x not in test_subject)]

        path = "./archives/mts_archive"
        dataset = Dataset("WESAD", None, GLOBAL_LOGGER)
        channels_ids = tuple(range(SIGNALS_LEN))

        X, s, y, sampling_rate = dataset.load_with_subjectid(path, rest, channels_ids)
        test_X, test_s, test_y, sampling_rate = dataset.load_with_subjectid(path, test_subject, channels_ids)

        try: 
            file_path = "./encodedresults/WESAD/encoded_results_restof_{1}.csv".format("WESAD",test_subject)
            file = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
            X_encoded_df = pd.DataFrame(file)
            file_path = "./encodedresults/WESAD/encoded_results_{1}.csv".format("WESAD",test_subject)
            file = pd.read_csv(file_path, usecols=lambda column: column != 'Unnamed: 0')
            test_X_encoded_df = pd.DataFrame(file)

        except FileNotFoundError: 
            print('file not found error')       
            x_val, x_train = [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)]
            s_val, s_train = [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)]

            for channel_id in range(len(channels_ids)):
                signal = X[channel_id]
                id = s[channel_id]

                num_rows = len(signal)
                split_size = num_rows // 4

                combined_list = list(zip(signal, id))
                random.shuffle(combined_list)
                shuffled_signal, shuffled_id = zip(*combined_list)

                for i in range(split_size):
                    x_val[channel_id].append(shuffled_signal[i])
                    s_val[channel_id].append(shuffled_id[i])
                for i in range(split_size,num_rows):
                    x_train[channel_id].append(shuffled_signal[i])
                    s_train[channel_id].append(shuffled_id[i])

            x_train = [np.expand_dims(np.array(x), 2) for x in x_train]
            x_val = [np.expand_dims(np.array(x), 2) for x in x_val]
            x_test = [np.expand_dims(np.array(x), 2) for x in test_X]

            s_train = [np.expand_dims(np.array(s), 2) for s in s_train]
            s_val = [np.expand_dims(np.array(s), 2) for s in s_val]

            if type(X) == list:
                input_shapes = [x.shape[1:] for x in x_train]
            else:
                input_shapes = x_train.shape[1:]

            ndft_arr = [get_ndft(x) for x in sampling_rate]

            if len(input_shapes) != len(ndft_arr):
                raise Exception("Different sizes of input_shapes and ndft_arr")

            for i in range(len(input_shapes)):
                if input_shapes[i][0] < ndft_arr[i]:
                    raise Exception(
                        f"Too big ndft, i: {i}, ndft_arr[i]: {ndft_arr[i]}, input_shapes[i][0]: {input_shapes[i][0]}")
        
            with Graph().as_default():
                session = get_new_session()
                with session.as_default():
                    with tf.device('/device:GPU:0'):
                        session.run(tf.compat.v1.global_variables_initializer())

                        X_encoded = []
                        test_X_encoded = []

                        for channel_id, input_shape in enumerate(input_shapes):
                            autoencoder = Autoencoder(input_shape)
                            autoencoder.compile(loss='mean_squared_error', 
                                optimizer=keras.optimizers.legacy.Adam(lr=0.003, decay=math.exp(-6)))
                            mini_batch_size = int(min(x_train[0].shape[0] / 10, 16))
                            history = autoencoder.fit(x_train[channel_id], x_train[channel_id], batch_size=mini_batch_size, epochs=100, verbose=False,
                                validation_data=(x_val[channel_id], x_val[channel_id]),
                                callbacks=[keras.callbacks.EarlyStopping(patience=30, monitor='val_loss')], shuffle=True)
                            encoded_x_train = autoencoder.encoder(x_train[channel_id]).eval()
                            encoded_x_val = autoencoder.encoder(x_val[channel_id]).eval()
                            encoded_x_test = autoencoder.encoder(x_test[channel_id]).eval()

                            X_encoded.append(np.vstack((encoded_x_train, encoded_x_val)))
                            test_X_encoded.append(encoded_x_test)

            X_encoded_df = pd.DataFrame(np.concatenate(X_encoded, axis=1))
            test_X_encoded_df = pd.DataFrame(np.concatenate(test_X_encoded, axis=1))
            s_train_list = [s_train[0][i,0,0] for i in range(s_train[0].shape[0])]
            s_val_list = [s_val[0][i,0,0] for i in range(s_val[0].shape[0])]
            s_list = s_train_list + s_val_list
            X_encoded_df['pnum'] = s_list
            test_X_encoded_df['pnum'] = test_subject[0]
            
            X_encoded_df.to_csv(f'./encodedresults/WESAD/encoded_results_restof_{test_subject}.csv', sep=',')
            test_X_encoded_df.to_csv(f'./encodedresults/WESAD/encoded_results_{test_subject}.csv', sep=',')

        scaler = MinMaxScaler()
        scaler.fit(X_encoded_df.iloc[:,:-1])
        X_encoded_scaled = scaler.transform(X_encoded_df.iloc[:,:-1])
        test_X_encoded_scaled = scaler.transform(test_X_encoded_df.iloc[:,:-1])

        silhouette_scores = []
        possible_K_values = [i for i in range(2,6)]

        for each_value in possible_K_values:
            clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
            cluster_labels = clusterer.fit_predict(X_encoded_scaled)
            silhouette_scores.append(silhouette_score(X_encoded_scaled, cluster_labels))

        k_value = silhouette_scores.index(min(silhouette_scores))
        clusterer = KMeans(n_clusters=possible_K_values[k_value], init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_encoded_scaled)
        X_encoded_df['cluster'] = list(cluster_labels)
        X_encoded_df.to_csv(f'./archives/WESAD/feature_clustering_results_{test_subject}.csv', sep=',')
        
        test_cluster_labels = clusterer.predict(test_X_encoded_scaled)
        test_X_encoded_df['cluster'] = list(test_cluster_labels)

        temp_cluster_X_df = X_encoded_df.groupby(['cluster', 'pnum']).size().reset_index(name='count')
        cluster_X_df = temp_cluster_X_df.pivot(index='pnum', columns='cluster', values='count').fillna(0).astype(int)
        cluster_X_df.reset_index(inplace=True)

        cluster_test_X = test_X_encoded_df['cluster'].value_counts().to_dict()
        cluster_df = pd.concat([cluster_X_df, pd.DataFrame([cluster_test_X])], ignore_index=True).fillna(0).astype(int)

        silhouette_scores = []
        possible_K_values = [i for i in range(2,6)]

        for each_value in possible_K_values:
            clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
            cluster_labels = clusterer.fit_predict(cluster_df.iloc[:-1,1:])
            silhouette_scores.append(silhouette_score(cluster_df.iloc[:-1,1:], cluster_labels))

        k_value = silhouette_scores.index(min(silhouette_scores))
        clusterer = KMeans(n_clusters=possible_K_values[k_value], init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit(cluster_df.iloc[:-1,1:])
        rest_cluster_label = clusterer.predict(cluster_df.iloc[:-1,1:])
        test_cluster_label = clusterer.predict(cluster_df.iloc[-1,1:].to_numpy().reshape(1,-1))
        print('rest cluster labels:', rest_cluster_label)
        print('test cluster labels:', test_cluster_label)

        same_cluster = []
        for subject, cluster in zip(rest, rest_cluster_label):
            if cluster == test_cluster_label:
                same_cluster.append(subject)
        print('same cluster:', same_cluster)

        if len(same_cluster) == 1:
            val_set = same_cluster
            train_set = same_cluster
        else:            
            # val_set = random.sample(same_cluster, math.ceil(len(same_cluster) / 4))
            val_set = random.sample(same_cluster, math.ceil(len(same_cluster) / 5))
            train_set = [x for x in rest if (x not in val_set) & (x in same_cluster)]
        print('final:', {"train": train_set, "val": val_set, "test": test_subject})
        result.append({"train": train_set, "val": val_set, "test": test_subject})

    print(result)
        
    return result


def n_fold_split_mtl_cluster_trait(subject_ids, n, dataset_name, seed=5):
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