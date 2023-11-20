from arpreprocessing.dataset import Dataset

import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
import random
import os

from fcn import FCN

from tensorflow import keras

from utils.loggerwrapper import GLOBAL_LOGGER   # logger_obj
from arpreprocessing.wesad import Wesad

from collections import defaultdict

from tqdm import tqdm

SIGNALS_LEN = 14    # no_channels
# Wesad.SUBJECTS_IDS

NUM_EPOCHS = 100
META_BATCH_SIZE = 3
BATCH_SIZE = 16
INNER_UPDATE_STEPS = 5

# Load Dataset

def load_dataset(logger_obj, channels_ids, test_ids, train_ids, dataset_name, seed=5):
    path = "../archives/mts_archive"
    path = os.path.abspath('../archives/mts_archive')
    dataset = Dataset(dataset_name, None, logger_obj)

    random.seed(seed)

    # Initialize lists to hold data for each user/task
    x_train = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in train_ids}
    y_train = {user_id: [] for user_id in train_ids}
    
    x_test = {user_id: [[] for _ in range(max(channels_ids) + 1)] for user_id in test_ids}
    y_test = {user_id: [] for user_id in test_ids}
    
    for user_id in train_ids:
        # Load data for the current user/task
        x, y, sampling_rate = dataset.load(path, [user_id], channels_ids)
        
        random.seed(seed)

        x_train_single_user = [[] for i in range(max(channels_ids) + 1)]
        y_train_single_user = []
        sampling_train = sampling_rate
        
        # Process input data x
        for channel_id in range(len(channels_ids)):
            signal = x[channel_id]
            
            num_rows = len(signal)
            # print(f"channel_id is {channel_id}, and {len(signal)=}")
            
            # Shuffle data
            combined_list = list(zip(signal, y))
            random.shuffle(combined_list)
            shuffled_signal, shuffled_y = zip(*combined_list)
            
            for i in range(num_rows):
                x_train_single_user[channel_id].append(shuffled_signal[i])
        
        for i in range(num_rows):
            y_train_single_user.append(shuffled_y[i])
        x_train_single_user = [np.expand_dims(np.array(x), 2) for x in x_train_single_user]
        
        random.seed()
        
        x_train[user_id] = x_train_single_user
        y_train[user_id] = y_train_single_user
    
    for user_id in test_ids:
        # Load data for the current user/task
        x, y, sampling_rate = dataset.load(path, [user_id], channels_ids)
        
        random.seed(seed)

        x_test_single_user = [[] for i in range(max(channels_ids) + 1)]
        y_test_single_user = []
        sampling_test = sampling_rate
        
        # Process input data x
        for channel_id in range(len(channels_ids)):
            signal = x[channel_id]
            
            num_rows = len(signal)
            # print(f"channel_id is {channel_id}, and {len(signal)=}")
            
            # Shuffle data
            combined_list = list(zip(signal, y))
            random.shuffle(combined_list)
            shuffled_signal, shuffled_y = zip(*combined_list)
            
            for i in range(num_rows):
                x_test_single_user[channel_id].append(shuffled_signal[i])
        
        for i in range(num_rows):
            y_test_single_user.append(shuffled_y[i])
        x_test_single_user = [np.expand_dims(np.array(x), 2) for x in x_test_single_user]
        
        random.seed()
        
        x_test[user_id] = x_test_single_user
        y_test[user_id] = y_test_single_user
    
    return x_train, x_test, y_train, y_test
        

def split_users(users, seed=None):
    np.random.seed(seed)
    # Shuffle the list of users
    shuffled_users = np.random.permutation(users)

    # Calculate the number of users for each set
    num_users = len(users)
    num_test = 1
    num_train = num_users - num_test
    

    # Split the shuffled list into meta-training, and meta-testing sets
    meta_train_set = shuffled_users[:num_train]
    meta_test_set = shuffled_users[num_train:]

    return meta_train_set, meta_test_set

def split_data(user_data, user_label, split_ratio=0.8):
    indices_by_label = defaultdict(list)
    for i, label in enumerate(user_label):
        indices_by_label[label].append(i)

    # print(f"{user_label=}")
    # Get the minimum number of data 
    samples_per_label = min(len(indices_by_label[0]), len(indices_by_label[1]))
    num_support = int(samples_per_label * split_ratio)
    num_query = samples_per_label - num_support

    label_0_data = []
    label_0_label = []
    label_1_data = []
    label_1_label = []

    support_data = []
    support_label = []
    query_data = []
    query_label = []

    # TODO: shuffle indices_by_label[0] and [1]

    for idx, x_test_i in enumerate(user_data):
        temp_label_0_data = np.array(x_test_i[indices_by_label[0][:samples_per_label],:,:])
        label_0_data.append(temp_label_0_data)
        temp_label_1_data = np.array(x_test_i[indices_by_label[1][:samples_per_label],:,:])
        label_1_data.append(temp_label_1_data)

        # print(f"{len(label_1_data)=}")
        #print(label_1_data[idx].shape)
        
    for idx, x_test_i in enumerate(label_0_data):
        temp_support_data = np.array(x_test_i[:num_support,:,:])
        support_data.append(temp_support_data)
        

        temp_query_data = np.array(x_test_i[num_support:,:,:])
        query_data.append(temp_query_data)
    
    for idx, x_test_i in enumerate(label_1_data):
        temp_support_data = np.array(x_test_i[:num_support,:,:])
        support_data[idx] = np.concatenate((support_data[idx], temp_support_data), axis=0)

        temp_query_data = np.array(x_test_i[num_support:,:,:])
        query_data[idx] = np.concatenate((query_data[idx], temp_query_data), axis=0)
    
    for l in range(num_support):
        support_label.append(0)
    for l in range(num_support):
        support_label.append(1)

    for l in range(num_query):
        query_label.append(0)
    for l in range(num_query):
        query_label.append(1)
    
    return support_data, support_label, query_data, query_label

def load_test_data(original_test_data, original_test_label):
    indices_by_label = defaultdict(list)
    label_0_data = []
    label_0_label = []
    label_1_data = []
    label_1_label = []
    
    test_data = []
    test_label = []
    
    for i, label in enumerate(original_test_label):
        indices_by_label[label].append(i)
    
    for idx, x_test_i in enumerate(original_test_data):
        temp_label_0_data = np.array(x_test_i[np.asarray(indices_by_label[0]), :, :])
        label_0_data.append(temp_label_0_data)
        temp_label_1_data = np.array(x_test_i[np.asarray(indices_by_label[1]), :, :])
        label_1_data.append(temp_label_1_data)
    
    for idx, x_test_i in enumerate(label_0_data):
        temp_test_data = np.array(x_test_i[:,:,:])
        test_data.append(temp_test_data)
    
    for idx, x_test_i in enumerate(label_1_data):
        temp_test_data = np.array(x_test_i[:,:,:])
        test_data[idx] = np.concatenate((test_data[idx], temp_test_data), axis=0)
    
    for l in range(len(indices_by_label[0])):
        test_label.append(0)
    for l in range(len(indices_by_label[1])):
        test_label.append(1)
    
    return test_data, test_label
        
class MAML:
    def __init__(self, input_shapes, num_classes, meta_lr=0.001, base_lr=0.01):
        self.meta_optimizer = optimizers.Adam(learning_rate=meta_lr)
        self.model = self.create_model(input_shapes, num_classes)
        self.base_lr = base_lr

    # TODO: Model 종류: mlp_lstm.py fcn.py resnet.py
    def create_model(self, input_shapes, num_classes):
        return FCN(input_shapes,num_classes).model

def compute_loss(logits, labels):
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=None)
    mse = tf.keras.losses.categorical_crossentropy(one_hot_labels, logits, from_logits=True)
    return mse #, logits

def compute_accuracy(logits, labels):
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(logits, axis=-1)

    # Get predicted class indices
    predicted_classes = tf.argmax(probabilities, axis=-1)

    # Compare predicted classes with true labels
    correct_predictions = tf.equal(predicted_classes, labels)

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy

meta_train_set, meta_test_set = split_users(Wesad.SUBJECTS_IDS, seed=42) # meta_val_set, 

x_train, x_test, y_train, y_test = load_dataset(GLOBAL_LOGGER, tuple(range(SIGNALS_LEN)), meta_test_set, meta_train_set, 'WESAD')

meta_batch_x_train = {user: x_train[user] for user in meta_train_set}
meta_batch_y_train = {user: y_train[user] for user in meta_train_set}

for user_id, values in x_train.items():
    if type(values) == list:
        input_shapes = [x.shape[1:] for x in x_train[user_id]]
        break
# print(f"{input_shapes=}")
num_classes = 2 # TODO: num_classes for each dataset

maml = MAML(input_shapes, num_classes)  
for epoch in tqdm(range(NUM_EPOCHS)):
    # Meta-batch construction
    for user_id in tqdm(meta_batch_x_train.keys()):
        model = maml.model
        optimizer = maml.meta_optimizer #optimizers.SGD(learning_rate=model.base_lr)
        
        # Randomly sample a set of tasks, and for each task (user/cluster/random-domain), sample support and query sets based on the ratios defined
        support_data, support_label, query_data, query_label = split_data(meta_batch_x_train[user_id], meta_batch_y_train[user_id], split_ratio=0.7)
        
        # Inner loop (support set optimization)
        with tf.GradientTape(persistent=True) as inner_tape:
            # Forward pass on the support set
            support_logits = model(support_data)
            support_loss = compute_loss(support_logits, support_label)
        
        # Compute gradients and perform a gradient descent step on the support set
        gradients = inner_tape.gradient(support_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Outer loop (meta-parameters optimizatino based on the query set)
        with tf.GradientTape() as outer_tape:
            # Forward pass on the query set
            query_logits = model(query_data)
            query_loss = compute_loss(query_logits, query_label)
        
        # Compute gradients for the meta-parameters
        meta_gradients = outer_tape.gradient(query_loss, model.trainable_variables)

        # Apply meta-optimization step
        optimizer.apply_gradients(zip(meta_gradients, model.trainable_variables))
    
# Test
for user_id, values in x_test.items():
    test_data, test_label = load_test_data(x_test[user_id], y_test[user_id])
    test_logits = model(test_data)
    test_loss = compute_loss(test_logits, test_label)
    test_accuracy = compute_accuracy(test_logits, test_label)
    print(f"Final Test for User #{user_id} Loss: {test_loss}, Accuracy: {test_accuracy}")
