import random
import math
import os
from scipy import stats

import numpy as np

from utils.utils import get_new_session
from utils.loggerwrapper import GLOBAL_LOGGER
from arpreprocessing.dataset import Dataset
from clustering.autoencoder import Autoencoder1, Autoencoder2, Autoencoder3, Autoencoder4

import keras
import tensorflow as tf
from tensorflow import Graph

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.spatial import distance


SIGNALS_LEN = 14


# def get_ndft(sampling):
#     if sampling <= 2:
#         return 8
#     if sampling <= 4:
#         return 16
#     if sampling <= 8:
#         return 32
#     if sampling <= 16:
#         return 64
#     if sampling <= 32:
#         return 128
#     if sampling in [70, 64, 65, 50]:
#         return 256
#     raise Exception(f"No such sampling as {sampling}")


def n_fold_split_cluster_trait(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)

    subject_ids = list(subject_ids)
    gender_info = list()
    age_info = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    gender_info.append(words[1])
                elif len(words) > 0 and words[0].lower() == 'age:':
                    age_info.append(words[1])

    X = pd.DataFrame()
    X['pnum'] = subject_ids
    X['gender'] = gender_info
    X['age'] = age_info

    label_encoder = LabelEncoder()
    X['gender'] = label_encoder.fit_transform(X['gender'])
    X['age'] = label_encoder.fit_transform(X['age'])

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_subject in test_sets:
        X_test = X.loc[X['pnum']==test_subject[0]]
        X_rest = X.loc[X['pnum']!=test_subject[0]]

        scaler = MinMaxScaler()
        scaler.fit(X_rest.iloc[:,1:])
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:,1:]))
        X_rest_scaled = pd.DataFrame(scaler.transform(X_rest.iloc[:,1:]))

        silhouette_scores = []
        possible_K_values = [i for i in range(2,6)]

        for each_value in possible_K_values:
            clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
            cluster_labels = clusterer.fit_predict(X_rest_scaled)
            silhouette_scores.append(silhouette_score(X_rest_scaled, cluster_labels))

        k_value = silhouette_scores.index(min(silhouette_scores))
        clusterer = KMeans(n_clusters=possible_K_values[k_value], init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_rest_scaled)
        X_rest_scaled['cluster'] = list(cluster_labels)
        X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()

        cluster_list = [[] for _ in range(possible_K_values[k_value])]

        rest = [x for x in subject_ids if x not in test_subject]

        for subject_id in rest:
            X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]
                if len(rest) == 1:
                    val_set, train_set = rest, rest
                else:
                    val_set = random.sample(rest, math.ceil(len(rest) / 5))
                    train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_subject})

    print(result)

    random.seed()
    return result


def n_fold_split_cluster_trait_experiment(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)

    subject_ids = list(subject_ids)
    gender_info = list()
    age_info = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    gender_info.append(words[1])
                elif len(words) > 0 and words[0].lower() == 'age:':
                    age_info.append(words[1])

    X = pd.DataFrame()
    X['pnum'] = subject_ids
    X['gender'] = gender_info
    X['age'] = age_info

    label_encoder = LabelEncoder()
    X['gender'] = label_encoder.fit_transform(X['gender'])
    X['age'] = label_encoder.fit_transform(X['age'])

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_subject in test_sets:
        X_test = X.loc[X['pnum']==test_subject[0]]
        X_rest = X.loc[X['pnum']!=test_subject[0]]

        scaler = MinMaxScaler()
        scaler.fit(X_rest.iloc[:,1:])
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:,1:]))
        X_rest_scaled = pd.DataFrame(scaler.transform(X_rest.iloc[:,1:]))

        k_value = 2
        clusterer = KMeans(n_clusters=k_value, init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_rest_scaled)
        X_rest_scaled['cluster'] = list(cluster_labels)
        X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()

        cluster_list = [[] for _ in range(k_value)]

        rest = [x for x in subject_ids if x not in test_subject]

        for subject_id in rest:
            X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]
                if len(rest) == 1:
                    val_set, train_set = rest, rest
                else:
                    val_set = random.sample(rest, math.ceil(len(rest) / 5))
                    train_set = [x for x in rest if (x not in val_set) & (x in cluster)]

        result.append({"train": train_set, "val": val_set, "test": test_subject})

    print(result)


    random.seed()
    return result


def n_fold_split_cluster_trait_mtl(subject_ids, n, dataset_name, seed=5):
    result = []

    random.seed(seed)

    subject_ids = list(subject_ids)
    gender_info = list()
    age_info = list()

    path = "archives/{}".format(dataset_name)

    for subject_id in subject_ids:
        with open("{0}/S{1}/S{1}_readme.txt".format(path,subject_id)) as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0].lower() == 'gender:':
                    gender_info.append(words[1])
                elif len(words) > 0 and words[0].lower() == 'age:':
                    age_info.append(words[1])

    X = pd.DataFrame()
    X['pnum'] = subject_ids
    X['gender'] = gender_info
    X['age'] = age_info

    label_encoder = LabelEncoder()
    X['gender'] = label_encoder.fit_transform(X['gender'])
    X['age'] = label_encoder.fit_transform(X['age'])

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_subject in test_sets:
        X_test = X.loc[X['pnum']==test_subject[0]]
        X_rest = X.loc[X['pnum']!=test_subject[0]]

        scaler = MinMaxScaler()
        scaler.fit(X_rest.iloc[:,1:])
        X_test_scaled = pd.DataFrame(scaler.transform(X_test.iloc[:,1:]))
        X_rest_scaled = pd.DataFrame(scaler.transform(X_rest.iloc[:,1:]))

        # # User-as-task
        # X_test_scaled.columns = ['gender','age']
        # X_rest_scaled.columns = ['gender','age']
        # def calculate_distance(row, target):
        #     return distance.euclidean((row['gender'], row['age']), (target['gender'], target['age']))
        # target = X_test_scaled.iloc[0]
        # X_rest_scaled['distance'] = X_rest_scaled.apply(lambda row: calculate_distance(row, target), axis=1)
        # X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()
        # min_distance = X_rest_scaled['distance'].min()
        # closest_subjects = X_rest_scaled.loc[X_rest_scaled['distance'] == min_distance, 'pnum']
        # if len(closest_subjects) > 1:
        #     closest_subject = closest_subjects.sample(n=1).iloc[0]
        # else:
        #     closest_subject = closest_subjects.iloc[0]

        # result.append({"test": test_subject, "cluster": [closest_subject]})
        
        # Cluster-as-task
        silhouette_scores = []
        possible_K_values = [i for i in range(2,6)]

        for each_value in possible_K_values:
            clusterer = KMeans(n_clusters=each_value, init='k-means++', n_init='auto', random_state=42)
            cluster_labels = clusterer.fit_predict(X_rest_scaled)
            silhouette_scores.append(silhouette_score(X_rest_scaled, cluster_labels))

        k_value = silhouette_scores.index(min(silhouette_scores))
        k_value = possible_K_values[k_value]

        clusterer = KMeans(n_clusters=k_value, init='k-means++', n_init='auto', random_state=42)
        cluster_labels = clusterer.fit_predict(X_rest_scaled)
        X_rest_scaled['cluster'] = list(cluster_labels)
        X_rest_scaled['pnum'] = X_rest['pnum'].values.tolist()

        cluster_list = [[] for _ in range(k_value)]

        rest = [x for x in subject_ids if x not in test_subject]

        for subject_id in rest:
            X_subject = X_rest_scaled.loc[X_rest_scaled['pnum']==subject_id,'cluster'].values[0]
            cluster_list[X_subject].append(subject_id)

        test_cluster = clusterer.predict(X_test_scaled)

        for idx, cluster in enumerate(list(cluster_list)):
            if idx == test_cluster:
                rest = [x for x in rest if x in cluster]

        result.append({"test": test_subject, "cluster": rest})

    print(result)


    random.seed()
    return result

      

# def build_encoder(input_shape):
#     model = models.Sequential()
#     model.add(layers.Conv1D(16, 8, activation='relu', input_shape=input_shape))
#     model.add(layers.Conv1D(32, 6, activation='relu'))
#     model.add(layers.Conv1D(64, 5, activation='relu'))
#     model.add(layers.Conv1D(128, 3, activation='relu'))
#     model.add(layers.Conv1D(256, 3, activation='relu'))
#     model.add(layers.GlobalAveragePooling1D())
#     return model

# def build_decoder():
#     model = models.Sequential()
#     model.add(layers.Dense(256, activation='relu'))  # Start from a dense layer to expand from the compressed form
#     model.add(layers.Reshape((256, 1)))  # Reshape to match the expected input of Conv1DTranspose
#     model.add(layers.Conv1DTranspose(128, 3, activation='relu'))
#     model.add(layers.Conv1DTranspose(64, 3, activation='relu'))
#     model.add(layers.Conv1DTranspose(32, 5, activation='relu'))
#     model.add(layers.Conv1DTranspose(16, 6, activation='relu'))
#     model.add(layers.Conv1DTranspose(1, 8, activation='relu'))
#     return model

# class Autoencoder(keras.Model):
#     def __init__(self, input_shape):
#         super(Autoencoder, self).__init__()
#         self.input_shapes = input_shape

#         print('encoder input shape', input_shape)

#         e_input_layer = keras.layers.Input(input_shape)
#         e_layer = keras.layers.Conv1D(16, 8, activation='relu', input_shape=input_shape)(e_input_layer)
#         e_layer = keras.layers.Conv1D(32, 6, activation='relu')(e_layer)
#         e_layer = keras.layers.Conv1D(input_shape[0], 5, activation='relu')(e_layer)
#         # e_layer = keras.layers.Conv1D(64, 5, activation='relu')(e_layer)
#         # e_layer = keras.layers.Conv1D(128, 3, activation='relu')(e_layer)
#         # e_layer = keras.layers.Conv1D(256, 3, activation='relu')(e_layer)
#         e_output_layer = keras.layers.GlobalAveragePooling1D()(e_layer)

#         self.encoder = keras.models.Model(inputs=e_input_layer, outputs=e_output_layer)

#         d_input_layer = keras.layers.Input(e_output_layer.shape)
#         # d_layer = keras.layers.Reshape((256,1))(d_input_layer)
#         d_layer = keras.layers.Reshape((input_shape[0],1))(d_input_layer)
#         # d_layer = keras.layers.Conv1DTranspose(128, 3, activation='relu', padding='same')(d_layer)
#         # d_layer = keras.layers.Conv1DTranspose(64, 3, activation='relu', padding='same')(d_layer)
#         d_layer = keras.layers.Conv1DTranspose(input_shape[0], 3, activation='relu', padding='same')(d_layer)
#         d_layer = keras.layers.Conv1DTranspose(32, 5, activation='relu', padding='same')(d_layer)
#         d_layer = keras.layers.Conv1DTranspose(16, 6, activation='relu', padding='same')(d_layer)
#         d_output_layer = keras.layers.Conv1DTranspose(1, 8, activation='relu', padding='same')(d_layer)
#         print('decoder output shape', d_output_layer.shape)

#         self.decoder = keras.models.Model(inputs=d_input_layer, outputs=d_output_layer)

#     def call(self, x):
#       encoded = self.encoder(x)
#       decoded = self.decoder(encoded)
#       return decoded

# def build_classifier(input_shape, num_classes):
#     model = keras.models.Sequential()
#     model.add(build_encoder(input_shape))
#     model.add(keras.layers.Dense(128, activation='relu'))
#     model.add(keras.layers.Dropout(0.5))
#     model.add(keras.layers.Dense(num_classes, activation='softmax'))  # num_classes depends on your dataset
#     return model

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

            # ndft_arr = [get_ndft(x) for x in sampling_rate]

            # if len(input_shapes) != len(ndft_arr):
            #     raise Exception("Different sizes of input_shapes and ndft_arr")

            # for i in range(len(input_shapes)):
            #     if input_shapes[i][0] < ndft_arr[i]:
            #         raise Exception(
            #             f"Too big ndft, i: {i}, ndft_arr[i]: {ndft_arr[i]}, input_shapes[i][0]: {input_shapes[i][0]}")
        
            with Graph().as_default():
                session = get_new_session()
                with session.as_default():
                    with tf.device('/device:GPU:0'):
                        session.run(tf.compat.v1.global_variables_initializer())

                        X_encoded = []
                        test_X_encoded = []

                        for channel_id, input_shape in enumerate(input_shapes):
                            sampling = input_shape[0] // 10
                            if sampling <= 8:
                                autoencoder = Autoencoder1(input_shape)
                            elif sampling <= 16:
                                autoencoder = Autoencoder2(input_shape)
                            # elif sampling <= 32:
                            #     autoencoder = Autoencoder3(input_shape)
                            else:
                                autoencoder = Autoencoder3(input_shape)
                            
                            # print(input_shape)
                            # encoder = build_encoder(input_shape)
                            # decoder = build_decoder()

                            # input_signal = layers.Input(shape=input_shape)
                            # encoded_signal = encoder(input_signal)
                            # decoded_signal = decoder(encoded_signal)

                            # autoencoder = Model(inputs=encoded_signal, ouputs=decoded_signal)
                            # autoencoder.compile(loss='mean_squared_error', optimizer='adam')
                            # mini_batch_size = int(min(x_train[0].shape[0] / 10, 16))
                            # history = autoencoder.fit(x_train[channel_id], x_train[channel_id], batch_size=mini_batch_size, epochs=100, verbose=False, validation_data=(x_val[channel_id], x_val[channel_id]), shuffle=True)
                            # encoded_x_train = autoencoder.predict(x_train[channel_id]).eval()
                            # encoded_x_val = autoencoder.predic(x_val[channel_id]).eval()
                            # encoded_x_test = autoencoder.predict(x_test[channel_id]).eval()
                            
                            # autoencoder = Autoencoder(input_shape)
                            autoencoder.compile(loss='mean_squared_error', optimizer='adam')
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