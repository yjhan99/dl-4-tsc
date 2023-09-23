import random
import math
import os
from scipy import stats

import numpy as np

from utils.utils import get_new_session
from utils.loggerwrapper import GLOBAL_LOGGER
from arpreprocessing.dataset import Dataset

import keras
import tensorflow as tf
from tensorflow import Graph

from collections import Counter

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


# Cannot handle unseen subject
def n_fold_split_cluster_trait_old(subject_ids, n, dataset_name, seed=5):
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

def n_fold_split_cluster_trait(subject_ids, n, dataset_name, seed=5):
    test_sets = [subject_ids[i::n] for i in range(n)]

    result = []

    random.seed(seed)

    for test_subject in test_sets:
        rest = [x for x in subject_ids if (x not in test_subject)]

        file_path = "./archives/{0}/{0}_PreSurvey.csv".format(dataset_name)
        file = pd.read_csv(file_path)

        # Including only age and BFI-15 information
        X = file.iloc[:, [1, 3] + list(range(80, 95))]
        X.columns = ['pnum', 'age'] + [f'bfi_{i}' for i in range(1, 16)]

        X_test = X.loc[X['pnum']==test_subject[0]]
        X_rest = X.loc[X['pnum']!=test_subject[0]]

        scaler = MinMaxScaler()
        scaler.fit(X_rest.iloc[:,1:])
        X_test_scaled = scaler.transform(X_test.iloc[:,1:])
        X_rest_scaled = scaler.transform(X_rest.iloc[:,1:])

        n_components = 3
        pca = PCA(n_components=n_components)
        pca.fit(X_rest_scaled)
        X_test_pca = pd.DataFrame(pca.transform(X_test_scaled))
        X_rest_pca = pd.DataFrame(pca.transform(X_rest_scaled))

        silhouette_scores = []
        possible_K_values = [i for i in range(2,6)]

        for each_value in possible_K_values:
            clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
            cluster_labels = clusterer.fit_predict(X_rest_pca)
            silhouette_scores.append(silhouette_score(X_rest_pca, cluster_labels))

        k_value = silhouette_scores.index(min(silhouette_scores))
        clusterer = KMeans(n_clusters=possible_K_values[k_value], init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_rest_pca)
        X_rest_pca['cluster'] = list(cluster_labels)
        X_rest_pca['pnum'] = X_rest['pnum'].values.tolist()

        cluster_list = [[] for _ in range(possible_K_values[k_value])]

        for subject_id in rest:
            X_subject = X_rest_pca.loc[X_rest_pca['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_pca)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]
                val_set = random.sample(rest, math.ceil(len(rest) / 5))
                train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_subject})
            
    print(result)

    random.seed()
    return result


# Cannot handle unseen subject
def n_fold_split_cluster_feature_old(subject_ids, n, seed=5):
    path = "./archives/mts_archive"
    dataset = Dataset("KEmoWork", None, GLOBAL_LOGGER)
    channels_ids = tuple(range(SIGNALS_LEN))
    X, s, y, sampling_rate = dataset.load_with_subjectid(path, subject_ids, channels_ids)

    random.seed(seed)
    
    x_test, x_val, x_train = [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)]
    s_test, s_val, s_train = [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)]

    for channel_id in range(len(channels_ids)):
        signal = X[channel_id]
        id = s[channel_id]

        num_rows = len(signal)
        split_size = num_rows // 4

        combined_list = list(zip(signal, id))
        random.shuffle(combined_list)
        shuffled_signal, shuffled_id = zip(*combined_list)

        for i in range(split_size):
            x_test[channel_id].append(shuffled_signal[i])
            s_test[channel_id].append(shuffled_id[i])
        for i in range(split_size, split_size*2):
            x_val[channel_id].append(shuffled_signal[i])
            s_val[channel_id].append(shuffled_id[i])
        for i in range(split_size*2,num_rows):
            x_train[channel_id].append(shuffled_signal[i])
            s_train[channel_id].append(shuffled_id[i])

    x_train = [np.expand_dims(np.array(x), 2) for x in x_train]
    x_val = [np.expand_dims(np.array(x), 2) for x in x_val]
    x_test = [np.expand_dims(np.array(x), 2) for x in x_test]

    s_train = [np.expand_dims(np.array(s), 2) for s in s_train]
    s_val = [np.expand_dims(np.array(s), 2) for s in s_val]
    s_test = [np.expand_dims(np.array(s), 2) for s in s_test]

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
                # tf.compat.v1.enable_eager_execution()

                X_encoded = []

                for channel_id, input_shape in enumerate(input_shapes):
                    print('x test:', x_test[channel_id].shape)
                    autoencoder = Autoencoder(input_shape)
                    autoencoder.compile(loss='mean_squared_error', 
                        optimizer=keras.optimizers.legacy.Adam(lr=0.003, decay=math.exp(-6)))
                        # optimizer=keras.optimizers.legacy.Adam(lr=0.003, decay=math.exp(-6)), run_eagerly=True)
                    mini_batch_size = int(min(x_train[0].shape[0] / 10, 16))
                    autoencoder.fit(x_train[channel_id], x_train[channel_id], batch_size=mini_batch_size, epochs=100, verbose=False,
                        validation_data=(x_val[channel_id], x_val[channel_id]),
                        callbacks=[keras.callbacks.EarlyStopping(patience=30, monitor='val_loss')], shuffle=True)
                    # encoded_x_test = autoencoder.encoder(x_test[channel_id])
                    encoded_x_test = autoencoder.encoder(x_test[channel_id]).eval()
                    print('encoded x test:', encoded_x_test.shape)

                    X_encoded.append(encoded_x_test)

    X_encoded_df = pd.DataFrame(np.concatenate(X_encoded, axis=1))
    s_test_list = [s_test[0][i,0,0] for i in range(s_test[0].shape[0])]
    # valuecounts = Counter(s_test_list)
    # for value, count in valuecounts.items():
    #     print(value, ":", count)
    X_encoded_df['pnum'] = s_test_list
    
    # X_encoded_df.to_csv('./archives/KEmoWork/feature_clustering_results.csv', sep=',')

    # file_path = "./archives/{0}/feature_clustering_results.csv".format("KEmoWork")
    # file = pd.read_csv(file_path)
    # X_encoded_df = pd.DataFrame(file)

    scaler = MinMaxScaler()
    X_encoded_scaled = scaler.fit_transform(X_encoded_df.iloc[:,:-1])

    silhouette_scores = []
    possible_K_values = [i for i in range(2,6)]

    for each_value in possible_K_values:
        clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_encoded_scaled)
        silhouette_scores.append(silhouette_score(X_encoded_scaled, cluster_labels))

    k_value = silhouette_scores.index(min(silhouette_scores))
    clusterer = KMeans(n_clusters=possible_K_values[k_value], init='k-means++', n_init='auto', random_state=42)
    cluster_labels = clusterer.fit_predict(X_encoded_scaled)

    # print('cluster labels')
    # valuecounts = Counter(cluster_labels)
    # for value, count in valuecounts.items():
    #     print(value, ":", count)

    X_encoded_df['cluster'] = list(cluster_labels)

    cluster_list = [[] for _ in range(5)]

    for subject_id in subject_ids:
        X_subject = X_encoded_df.loc[X_encoded_df['pnum']==subject_id,:]
        cluster_mode = stats.mode(X_subject['cluster'])[0]
        cluster_list[cluster_mode].append(subject_id)

    test_sets = [subject_ids[i::n] for i in range(n)]

    result = []

    for test_set in test_sets:
        for cluster in cluster_list:
            if test_set[0] in cluster:
                rest = [x for x in subject_ids if (x not in test_set) & (x in cluster)]
                val_set = random.sample(rest, math.ceil(len(rest) / 5))
                train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_set})
    
    print(result)
        
    return result


def n_fold_split_cluster_feature(subject_ids, n, seed=5):
    test_sets = [subject_ids[i::n] for i in range(n)]

    result = []

    # random.seed(seed)

    for test_subject in test_sets:
        rest = [x for x in subject_ids if (x not in test_subject)]

        path = "./archives/mts_archive"
        dataset = Dataset("KEmoWork", None, GLOBAL_LOGGER)
        channels_ids = tuple(range(SIGNALS_LEN))

        X, s, y, sampling_rate = dataset.load_with_subjectid(path, rest, channels_ids)
        test_X, test_s, test_y, sampling_rate = dataset.load_with_subjectid(path, test_subject, channels_ids)

        random.seed(seed)
        
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
                    # s_list = []

                    for channel_id, input_shape in enumerate(input_shapes):
                        print('x test:', x_test[channel_id].shape)
                        autoencoder = Autoencoder(input_shape)
                        autoencoder.compile(loss='mean_squared_error', 
                            optimizer=keras.optimizers.legacy.Adam(lr=0.003, decay=math.exp(-6)))
                        mini_batch_size = int(min(x_train[0].shape[0] / 10, 16))
                        autoencoder.fit(x_train[channel_id], x_train[channel_id], batch_size=mini_batch_size, epochs=100, verbose=False,
                            validation_data=(x_val[channel_id], x_val[channel_id]),
                            callbacks=[keras.callbacks.EarlyStopping(patience=30, monitor='val_loss')], shuffle=True)
                        encoded_x_train = autoencoder.encoder(x_train[channel_id]).eval()
                        encoded_x_val = autoencoder.encoder(x_val[channel_id]).eval()
                        encoded_x_test = autoencoder.encoder(x_test[channel_id]).eval()
                        # print('encoded x train:', encoded_x_train.shape)
                        # print('encoded x val:', encoded_x_val.shape)
                        print('encoded x test:', encoded_x_test.shape)

                        X_encoded.append(np.vstack((encoded_x_train, encoded_x_val)))
                        test_X_encoded.append(encoded_x_test)

                        # s_train_list = [s_train[0][i,0,0] for i in range(s_train[channel_id].shape[0])]
                        # s_val_list = [s_val[0][i,0,0] for i in range(s_val[channel_id].shape[0])]
                        # s_list.extend(s_train_list+s_val_list)
                        # print(len(s_train_list + s_val_list))

        X_encoded_df = pd.DataFrame(np.concatenate(X_encoded, axis=1))
        test_X_encoded_df = pd.DataFrame(np.concatenate(test_X_encoded, axis=1))
        s_train_list = [s_train[0][i,0,0] for i in range(s_train[0].shape[0])]
        s_val_list = [s_val[0][i,0,0] for i in range(s_val[0].shape[0])]
        s_list = s_train_list + s_val_list
        X_encoded_df['pnum'] = s_list
        test_X_encoded_df['pnum'] = test_subject[0]
        
        X_encoded_df.to_csv(f'./archives/KEmoWork/encoding_results_{test_subject}.csv', sep=',')

        # file_path = "./archives/{0}/feature_clustering_results.csv".format("KEmoWork")
        # file = pd.read_csv(file_path)
        # X_encoded_df = pd.DataFrame(file)

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

        X_encoded_df.to_csv(f'./archives/KEmoWork/feature_clustering_results_{test_subject}.csv', sep=',')

        cluster_list = [[] for _ in range(possible_K_values[k_value])]

        for subject_id in rest:
            X_subject = X_encoded_df.loc[X_encoded_df['pnum']==subject_id,:]
            cluster_mode = stats.mode(X_subject['cluster'])[0]
            cluster_list[cluster_mode].append(subject_id)

        test_cluster_labels = clusterer.predict(test_X_encoded_scaled)
        test_cluster = stats.mode(test_cluster_labels)[0]

        print('cluster list:', cluster_list)
        print('test cluster:', test_cluster)

        for idx, cluster in enumerate(cluster_list):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]
                val_set = random.sample(rest, math.ceil(len(rest) / 5))
                train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_subject})

    print(result)
        
    return result