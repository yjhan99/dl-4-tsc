from arpreprocessing.dataset import Dataset

import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
import random
import os

from utils.loggerwrapper import GLOBAL_LOGGER   # logger_obj
from arpreprocessing.wesad import Wesad

SIGNALS_LEN = 14    # no_channels
# Wesad.SUBJECTS_IDS

NUM_EPOCHS = 100
META_BATCH_SIZE = 3
INNER_UPDATE_STEPS = 5

# Load Dataset
def load_dataset(logger_obj, channels_ids, test_ids, train_ids, val_ids, dataset_name, seed=5):
    path = "../archives/mts_archive"
    path = os.path.abspath('../archives/mts_archive')
    dataset = Dataset(dataset_name, None, logger_obj)


    random.seed(seed)

    # Initialize lists to hold data for each user/task
    x_test = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in test_ids}
    x_val = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in val_ids}
    x_train = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in train_ids}
    y_test = {user_id: [] for user_id in test_ids}
    y_val = {user_id: [] for user_id in val_ids}
    y_train = {user_id: [] for user_id in train_ids}
    
    all_user_ids = Wesad.SUBJECTS_IDS #test_ids + val_ids + train_ids
    
    # Load data for each user/task
    for user_id in all_user_ids:
        # Load data for the current user/task
        x, y, sampling_rate = dataset.load(path, [user_id], channels_ids)
        
        random.seed(seed)

        x_test_single_user, x_val_single_user, x_train_single_user = [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)], [[] for i in range(max(channels_ids) + 1)]
        y_test_single_user, y_val_single_user, y_train_single_user = [], [], []
        sampling_test, sampling_val, sampling_train = sampling_rate, sampling_rate, sampling_rate
        
        # Process input data x
        for channel_id in range(len(channels_ids)):
            signal = x[channel_id]
            
            num_rows = len(signal)
            split_size = num_rows // 10
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
        
    
    # input_shapes, nb_classes, y_val, y_train, y_test, y_true = prepare_data_maml(x_train, y_train, y_val, y_test)
    #ndft_arr = [get_ndft(x) for x in sampling_test] # TODO: Q. Is this necessary?
    
    return x_train, x_val, x_test, y_train, y_val, y_test


def split_users(users, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    np.random.seed(seed)

    # Shuffle the list of users
    shuffled_users = np.random.permutation(users)

    # Calculate the number of users for each set
    num_users = len(users)
    num_train = int(train_ratio * num_users)
    num_val = int(val_ratio * num_users)

    # Split the shuffled list into meta-training, meta-validation, and meta-testing sets
    meta_train_set = shuffled_users[:num_train]
    meta_val_set = shuffled_users[num_train:num_train + num_val]
    meta_test_set = shuffled_users[num_train + num_val:]

    return meta_train_set, meta_val_set, meta_test_set

def split_data(user_data, user_label, split_ratio=0.7):
    num_samples = len(user_data)
    split_index = int(split_ratio * num_samples)

    support_data = user_data[:split_index]
    support_label = user_label[:split_index]
    query_data = user_data[split_index:]
    query_label = user_label[split_index:]

    return support_data, support_label, query_data, query_label

# def compute_loss(model, data, label):
#     inputs = data
#     labels = label

#     # Forward pass to obtain predictions
#     predictions = model(inputs)

#     # Calculate categorical cross-entropy loss
#     loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=True)

#     return loss

class MAML:
    def __init__(self, input_shapes, num_classes, meta_lr=0.001, base_lr=0.01):
        self.models = [self.create_model(input_shape, num_classes) for input_shape in input_shapes]
        self.meta_optimizer = optimizers.Adam(learning_rate=meta_lr)
        self.base_lr = base_lr

    def create_model(self, input_shape, num_classes):
        input_layer = layers.Input(input_shape)
        x = layers.Flatten()(input_layer)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(500, activation='relu')(x)

        for _ in range(2):
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(500, activation='relu')(x)

        output_layer = layers.Dropout(0.3)(x)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])
        return model

    def inner_update(self, model, loss, step_size):
        grads = tf.gradients(loss, model.trainable_variables)
        updated_model = self._update_model(model, grads, step_size)
        return updated_model

    def _update_model(self, model, grads, step_size):
        updated_model = model
        for param, grad in zip(updated_model.trainable_variables, grads):
            param.assign_sub(step_size * grad)
        return updated_model

    def meta_update(self, task_gradients):
        self.meta_optimizer.apply_gradients(zip(task_gradients, self.models[0].trainable_variables))

    def late_fusion(self, x_list):
        outputs = [model(x) for model, x in zip(self.models, x_list)]
        flat = layers.concatenate(outputs, axis=-1) if len(outputs) > 1 else outputs[0]
        output_layer = layers.Dense(num_classes, activation='softmax')(flat)
        return output_layer

    def get_meta_model(self):
        input_layers = [layers.Input(input_shape) for input_shape in input_shapes]
        late_fusion_output = self.late_fusion(input_layers)
        meta_model = tf.keras.models.Model(inputs=input_layers, outputs=late_fusion_output)
        return meta_model
    
    def compute_loss(self, model, data, label):
        inputs = data
        labels = label

        # Convert labels to NumPy array if they are a list
        if isinstance(labels, list):
            labels = np.array(labels)

        # Convert labels to one-hot encoding if not already
        if len(labels.shape) == 1:
            labels = tf.one_hot(labels, depth=model.output_shape[-1])

        # Ensure that the shapes are compatible
        labels = tf.image.resize_with_pad(labels, model.output_shape[1], model.output_shape[2]) if len(model.output_shape) == 4 else labels

        # Reshape data to (batch_size * num_time_steps, num_features)
        data = tf.reshape(data, (-1, data.shape[-1]))

        # Forward pass to obtain predictions
        predictions = model(inputs)

        # Slice predictions to match the length of labels
        predictions = predictions[:labels.shape[0]]

        # Calculate categorical cross-entropy loss
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions, from_logits=True)

        return loss

meta_train_set, meta_val_set, meta_test_set = split_users(Wesad.SUBJECTS_IDS, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)


for epoch in range(NUM_EPOCHS):
    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(GLOBAL_LOGGER, tuple(range(SIGNALS_LEN)), meta_test_set, meta_train_set, meta_val_set, 'WESAD')

    meta_batch_x_train = {user: x_train[user] for user in meta_train_set}
    meta_batch_y_train = {user: y_train[user] for user in meta_train_set}

    #input_shapes = [np.array(user_data[0]).shape[1:] for user_data in x_train.values()]#[user_data.shape[1:] for user_data in x_train.values()]
    for user_id, values in x_train.items():
        if type(values) == list:
            input_shapes = [x.shape[1:] for x in x_train[user_id]]
            break
    num_classes = 2 # TODO: num_classes for each dataset

    maml = MAML(input_shapes, num_classes)  

    for user_id in meta_batch_x_train.keys():
        user_models = [maml.create_model(input_shape, num_classes=2) for input_shape in input_shapes]
        user_optimizers = [optimizers.SGD(learning_rate=maml.base_lr) for _ in input_shapes]

        support_data, support_label, query_data, query_label = split_data(meta_batch_x_train[user_id], meta_batch_y_train[user_id], split_ratio=0.7)

        for step in range(INNER_UPDATE_STEPS):
            for channel_id in range(len(input_shapes)):
                # Correct the way to access support_data and query_data
                support_data_channel = support_data[channel_id]
                query_data_channel = query_data[channel_id]

                with tf.GradientTape() as tape:
                    support_loss = maml.compute_loss(user_models[channel_id], support_data_channel, support_label)
                grads = tape.gradient(support_loss, user_models[channel_id].trainable_variables)
                user_optimizers[channel_id].apply_gradients(zip(grads, user_models[channel_id].trainable_variables))

        with tf.GradientTape() as tape:
            # Correct the way to access query_data
            query_outputs = [user_models[channel_id](query_data[channel_id]) for channel_id in range(len(input_shapes))]
            query_loss = maml.compute_loss(maml.late_fusion(query_outputs), query_label)
        task_gradients = tape.gradient(query_loss, maml.models[0].trainable_variables)

        maml.meta_update(task_gradients)
