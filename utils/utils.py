import os

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


class NoSuchClassifier(Exception):
    def __init__(self, classifier_name):
        self.message = "No such classifier: {}".format(classifier_name)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False, sampling_rates=None,
                      ndft_arr=None):
    if classifier_name == 'fcnM':
        from multimodal_classfiers import fcn_multimodal
        return fcn_multimodal.ClassifierFcn(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlpM':
        from multimodal_classfiers import mlp_multimodal
        return mlp_multimodal.ClassifierMlp(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnetM':
        from multimodal_classfiers import resnet
        return resnet.ClassifierResnet(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'encoderM':
        from multimodal_classfiers import encoder_multimodal
        return encoder_multimodal.ClassifierEncoder(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnnM':
        from multimodal_classfiers import mcdcnn_multimodal
        return mcdcnn_multimodal.ClassifierMcdcnn(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnnM':
        from multimodal_classfiers import cnn_multimodal
        return cnn_multimodal.ClassifierTimeCnn(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inceptionM':
        from multimodal_classfiers import inceptionTime
        return inceptionTime.ClassifierInception(output_directory, input_shape, nb_classes, verbose=verbose)
    if classifier_name == 'stresnetM':
        from multimodal_classfiers import spectroTemporalResNet_multimodal
        return spectroTemporalResNet_multimodal.ClassifierStresnet(output_directory, input_shape, sampling_rates,
                                                                   ndft_arr,
                                                                   nb_classes, verbose=verbose)

    raise NoSuchClassifier(classifier_name)


def prepare_data(x_train, y_train, y_val, y_test):
    y_train, y_val, y_test = transform_labels(y_train, y_test, y_val=y_val)
    y_true = y_val.astype(np.int64)
    concatenated_ys = np.concatenate((y_train, y_val, y_test), axis=0)
    nb_classes = len(np.unique(concatenated_ys))
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(concatenated_ys.reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
    if type(x_train) == list:
        input_shapes = [x.shape[1:] for x in x_train]
    else:
        input_shapes = x_train.shape[1:]
    return input_shapes, nb_classes, y_val, y_train, y_test, y_true

def prepare_data_maml(x_train, y_train, y_val, y_test):
    y_true = {user_id: [] for user_id in y_val.keys()}
    for user_id in y_val.keys():
        y_true[user_id] = y_val[user_id]
    y_all = []
    y_all.extend([value for values in y_train.values() for value in values])
    y_all.extend([value for values in y_test.values() for value in values])
    y_all.extend([value for values in y_val.values() for value in values])
    y_all = np.array(y_all)
    nb_classes = len(np.unique(y_all))
    for user_id, values in x_train.items():
        if type(values) == list:
            input_shapes = [x.shape[1:] for x in x_train[user_id]]
            break
    return input_shapes, nb_classes, y_val, y_train, y_test, y_true
    

def prepare_data_mtl(x_test, y_task_rest, y_task_test, y_test):
    y_task_rest, y_task_test, y_test = transform_labels_mtl(y_task_rest, y_test, y_task_test)
    y_true = y_test.astype(np.int64)
    concatenated_ys = np.concatenate((y_test, y_task_test), axis=0)
    for y_task_rest_i in y_task_rest:
        concatenated_ys = np.concatenate((concatenated_ys, y_task_rest_i), axis=0)
    nb_classes = len(np.unique(concatenated_ys))
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(concatenated_ys.reshape(-1, 1))
    y_task_test = enc.transform(y_task_test.reshape(-1, 1)).toarray()
    for idx, y_task_rest_i in enumerate(y_task_rest):
        y_task_rest[idx] = enc.transform(y_task_rest_i.reshape(-1, 1)).toarray()
    if type(x_test) == list:
        input_shapes = [x.shape[1:] for x in x_test]
    else:
        input_shapes = x_test.shape[1:]
    return input_shapes, nb_classes, y_task_rest, y_task_test, y_test, y_true

def set_available_gpus(gpu_ids):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def get_new_session():
    # config = tf.ConfigProto(allow_soft_placement=True,
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def interpolate_for_length(values, new_length):
    if len(values) == new_length:
        return values

    idx = range(len(values))
    idx_new = np.linspace(0, len(values) - 1, new_length)

    f = interp1d(idx, values, kind='cubic')
    return f(idx_new)


def transform_labels(y_train, y_test, y_val=None):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None:
        # index for when resplitting the concatenation
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_val_test = np.concatenate((y_train, y_val, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val, new_y_test
    else:
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test
    

def transform_labels_mtl(y_task_rest, y_test, y_task_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    idx_y_test = len(y_test)
    idx_y_task_test = idx_y_test + len(y_task_test)
    idx_y_task_rest = []
    encoder = LabelEncoder()
    y_all = np.concatenate((y_test, y_task_test), axis=0)
    for y_task_rest_i in y_task_rest:
        y_all = np.concatenate((y_all, y_task_rest_i), axis=0)
        idx_y_task_rest.append(len(y_task_rest_i))
    encoder.fit(y_all)
    new_y_all = encoder.transform(y_all)
    new_y_test = new_y_all[0:idx_y_test]
    new_y_task_test = new_y_all[idx_y_test:idx_y_task_test]
    new_y_task_rest = []
    upto = idx_y_task_test
    for idx in idx_y_task_rest:
        new_y_task_rest.append(new_y_all[idx_y_task_test:upto+idx])
        upto += idx
    return new_y_task_rest, new_y_test, new_y_task_test


def save_logs(output_directory, hist, y_pred, y_pred_probabilities, y_true, duration, lr=True, y_true_val=None,
              y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, y_pred_probabilities, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['val_loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    accuracy_name = 'accuracy' if 'accuracy' in row_best_model else 'acc'
    df_best_model['best_model_train_acc'] = row_best_model[accuracy_name]
    df_best_model['best_model_val_acc'] = row_best_model['val_' + accuracy_name]
    # if lr == True:
    #     df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')
    plot_predictions(y_pred, y_true, output_directory + 'predictions.png')
    save_predictions(y_true, y_pred, y_pred_probabilities, f"{output_directory}predictions.txt")

    return df_metrics


def plot_predictions(y_pred, y_true, filename):
    fig, ax = plt.subplots()
    t = list(range(len(y_pred)))
    ax.plot(t, y_true, "b-", t, y_pred, "r.")
    fig.savefig(filename)
    plt.close(fig)


def save_predictions(y_true, y_pred, y_pred_probabilities, filename):
    with open(filename, "w+") as file:
        for line in [y_true, y_pred, y_pred_probabilities]:
            for elem in line:
                file.write(f"{elem} ")
            file.write("\n")


def plot_epochs_metric(hist, file_name, metric='loss'):
    fig, ax = plt.subplots()
    ax.plot(hist.history[metric])
    ax.plot(hist.history['val_' + metric])
    ax.set_title('model ' + metric)
    ax.set_ylabel(metric, fontsize='large')
    ax.set_xlabel('epoch', fontsize='large')
    ax.legend(['train', 'val'], loc='upper left')
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)


def calculate_metrics(y_true, y_pred, y_pred_probabilities, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration', 'f1', 'auc'])
    res['duration'] = duration

    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['f1'] = f1_score(y_true, y_pred, average='macro')

    try:
        print(y_pred_probabilities.shape)
        if y_pred_probabilities.shape[1] == 2:
            res['auc'] = roc_auc_score(y_true, y_pred_probabilities[:, 0], multi_class="ovo")
        else:
            res['auc'] = roc_auc_score(y_true, y_pred_probabilities, multi_class="ovo")
    except:
        res['auc'] = None

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)
    return res
