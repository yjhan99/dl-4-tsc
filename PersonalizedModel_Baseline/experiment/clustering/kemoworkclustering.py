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

class GaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        return inputs


class Autoencoder_old(keras.Model):
  def __init__(self, input_shapes, nb_classes):
    super(Autoencoder_old, self).__init__()
    self.input_shapes = input_shapes
    self.nb_classes = nb_classes

    input_layers = []
    channel_outputs = []
    
    for channel_id, input_shape in enumerate(input_shapes):
        # input_layer = keras.layers.Input(shape=(None, round(input_shape[0] / 2), 1), name=f"input_for_{channel_id}")
        input_layer = keras.layers.Input(input_shape)
        input_layers.append(input_layer)
        input_layer_flattened = keras.layers.TimeDistributed(keras.layers.Flatten())(input_layer)
        layer = keras.layers.TimeDistributed(keras.layers.Dense(4, activation='relu'))(input_layer_flattened)
        layer = keras.layers.TimeDistributed(keras.layers.Dense(4, activation='relu'))(layer)
        lstm_layer = keras.layers.LSTM(4)(layer)
        channel_outputs.append(lstm_layer)

    flat = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]

    self.encoder = keras.models.Model(inputs=input_layers, outputs=flat)

    # encoded_output = keras.layers.Input(shape=(None, 4*len(input_shapes)), name="encoded_output")
    # decoded_inputs = keras.layers.Reshape(target_shape=[len(input_shapes)*(-1,4)])(encoded_output)

    encoded_output = keras.layers.Input(shape=(4*len(input_shapes),), name="encoded_output")
    encoded_output_noise = GaussianNoiseLayer(stddev=0.1)(encoded_output)
    split_tensors = tf.split(encoded_output_noise, num_or_size_splits=len(input_shapes), axis=-1)
    decoded_outputs = [keras.layers.Reshape(target_shape=(-1, 4))(split_tensor) for split_tensor in split_tensors]
    
    reshaped_layers = []

    for channel_id, (input_shape, decoded_output) in enumerate(zip(input_shapes, decoded_outputs)):
        print(channel_id)
        print(input_shape)
        print(decoded_output)
        # input_layer = keras.layers.Input(shape=(None, round(decoded_input[0] / 2), 1), name=f"input_for_{channel_id}")
        lstm_layer = keras.layers.LSTM(4, return_sequences=True)(decoded_output)
        layer = keras.layers.TimeDistributed(keras.layers.Dense(4, activation='relu'))(lstm_layer)
        layer = keras.layers.TimeDistributed(keras.layers.Dense(input_shape[0], activation='relu'))(layer)
        reshaped_layer = keras.layers.Reshape(target_shape=input_shape)(layer)
        reshaped_layers.append(reshaped_layer)

    self.decoder = keras.models.Model(inputs=encoded_output, outputs=reshaped_layers)

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  

class Autoencoder_old2(keras.Model):
  def __init__(self, input_shape):
    super(Autoencoder_old2, self).__init__()
    self.input_shapes = input_shape
    print('input shape:', input_shape)

    e_input_layer = keras.layers.Input((input_shape))
    print('e input layer:', e_input_layer.shape)
    e_input_layer_flattened = keras.layers.TimeDistributed(keras.layers.Flatten())(e_input_layer)
    print('e input flattened:', e_input_layer_flattened.shape)
    e_layer = keras.layers.TimeDistributed(keras.layers.Dense(4, activation='relu'))(e_input_layer_flattened)
    e_layer = keras.layers.TimeDistributed(keras.layers.Dense(4, activation='relu'))(e_layer)
    print('e layer:', e_layer.shape)
    e_lstm_layer = keras.layers.LSTM(4)(e_layer)
    print('e lstm layer:', e_lstm_layer.shape)

    self.encoder = keras.models.Model(inputs=e_input_layer, outputs=e_lstm_layer)

    # d_input_layer = keras.layers.Input(shape=e_lstm_layer.shape[1:])
    # d_input_layer_noise = GaussianNoiseLayer(stddev=0.1)(d_input_layer)
    
    d_input_layer = keras.layers.Input((4,))
    print('d input layer:', d_input_layer.shape)
    d_input_layer_noise = GaussianNoiseLayer(stddev=0.1)(d_input_layer)
    print('d input layer noise:', d_input_layer_noise.shape)
    d_repeat_layer = keras.layers.RepeatVector(input_shape[0])(d_input_layer_noise)
    print('d repeat layer:', d_input_layer.shape)
    d_lstm_layer = keras.layers.LSTM(4, return_sequences=True, input_shape=d_input_layer.shape)(d_repeat_layer)
    print('d lstm layer:', d_lstm_layer.shape)

    # d_lstm_layer = keras.layers.LSTM(4, return_sequences=True)(d_input_layer_noise)
    d_layer = keras.layers.TimeDistributed(keras.layers.Dense(4, activation='relu'))(d_lstm_layer)
    d_layer = keras.layers.TimeDistributed(keras.layers.Dense(4, activation='relu'))(d_layer)
    d_reshaped_layer = keras.layers.Reshape(target_shape=input_shape)(d_layer)

    self.decoder = keras.models.Model(inputs=d_input_layer, outputs=d_reshaped_layer)

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  

class Autoencoder(keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.input_shapes = input_shape

        e_input_layer = keras.layers.Input(input_shape)
        print('e input layer:', e_input_layer.shape)
        e_input_layer_flattened = keras.layers.Flatten()(e_input_layer)
        print('e input layer flattended:', e_input_layer_flattened.shape)
        e_input_expanded = keras.layers.Reshape((1,-1))(e_input_layer_flattened)
        print('e input expanded:', e_input_expanded.shape)
        e_layer = keras.layers.Dense(4, activation='relu')(e_input_expanded)
        e_layer = keras.layers.Dense(4, activation='relu')(e_layer)
        print('e layer:', e_layer.shape)
        e_lstm_layer = keras.layers.LSTM(4, return_sequences=False)(e_layer)

        self.encoder = keras.models.Model(inputs=e_input_layer, outputs=e_lstm_layer)

        d_input_layer = keras.layers.Input(e_lstm_layer.shape)
        print('d input layer:', d_input_layer.shape)
        d_input_layer_noise = GaussianNoiseLayer(stddev=0.1)(d_input_layer)
        print('d input layer noise:', d_input_layer_noise.shape)
        d_input_layer_reshaped = keras.layers.Reshape(target_shape=(1,4))(d_input_layer_noise)
        print('d input layer reshaped:', d_input_layer_reshaped.shape)
        d_lstm_layer = keras.layers.LSTM(4, return_sequences=False)(d_input_layer_reshaped)
        print('d lstm layer:', d_lstm_layer.shape)
        d_layer = keras.layers.Dense(4, activation='relu')(d_lstm_layer)
        d_layer = keras.layers.Dense(input_shape[0], activation='relu')(d_layer)
        print('d layer:', d_layer.shape)
        d_reshaped_layer = keras.layers.Reshape(target_shape=input_shape)(d_layer)
        print('d reshaped layer:', d_reshaped_layer.shape)

        self.decoder = keras.models.Model(inputs=d_input_layer, outputs=d_reshaped_layer)

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded

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

    # X = [np.expand_dims(np.array(x, dtype=object), 2) for x in X]
    X = [np.expand_dims(np.array(x), 2) for x in X]
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
        
    # print(input_shapes)
    # print(X[0].shape)
                
    with Graph().as_default():
        session = get_new_session()
        with session.as_default():
            with tf.device('/device:GPU:0'):
                session.run(tf.compat.v1.global_variables_initializer())

                encoded_Xs = []

                for channel_id, input_shape in enumerate(input_shapes):
                    print(X[channel_id].shape)
                    autoencoder = Autoencoder(input_shape)
                    autoencoder.compile(loss='categorical_crossentropy', 
                        optimizer=keras.optimizers.legacy.Adam(lr=0.003, decay=math.exp(-6)), metrics=['accuracy'])
                    mini_batch_size = int(min(X[0].shape[0] / 10, 16))
                    autoencoder.fit(X[channel_id], X[channel_id], batch_size=mini_batch_size, epochs=100, verbose=False,
                        callbacks=[keras.callbacks.EarlyStopping(patience=30, monitor='loss')], shuffle=True)
                    encoded_X = autoencoder.encoder(X[channel_id])
                    encoded_Xs.append(encoded_X)

                    print(encoded_X.shape)
                    
                # TODO: validation 어떻게 할지 ㅎ
                
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

