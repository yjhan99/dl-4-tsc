#import torch.utils.data
import pandas as pd
import numpy as np
import time
from itertools import islice
import itertools
import multiprocessing as mp
import io, os
import pickle
import tensorflow as tf
import sys
from sklearn.preprocessing import MinMaxScaler
import options

opt = options.WESADOpt  # TODO
WIN_LEN = opt['seq_len']
filePath = '../dataset/wesad/'

manager = mp.Manager()
sys.path.append('..')

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def getColumnNameDict(df):
    columnNameDict = {}

    for i in range(len(list(df)):
        columnNameDict[i] = list(df)[i]
    return columnNameDict

def minmaxScaling(df):
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

def sampling(df):
    columnName = getColumnNameDict(df)
    sampledDf = pd.DataFrame(columns=columnName.values())
    i = 0
    while 1:
        tmpLabel = df["label"][i]
        for j in range(1, WIN_LEN):
            if i + j >= len(df):
                break
            if tmpLabel != df["label"][i + j]:
                i += j - 1  # break -> i+=1
                break
            if j == WIN_LEN - 1:
                newDf = df.iloc[i:i + WIN_LEN, :]
                newDf = newDf.reset_index(drop=True)
                sampledDf = sampledDf.append(newDf, ignore_index=True)
                # sampledDf.insert(loc= sampleNum*16, column = columnName, value=newDf)
                i += WIN_LEN - 1  # -> i+=1
        i += 1
        if i >= len(df):
            break

    return sampledDf

#class KSHOTTensorDataset(torch.utils.data.Dataset):
class KSHOTTensorDataset(tf.data.Dataset):
    # class for MAML
    def __init__(self, num_classes, features, classes, domains):
        assert (features.shape[0] == classes.shape[0] == domains.shape[0])

        self.num_classes = num_classes
        self.features_per_class = []
        self.classes_per_class = []
        self.domains_per_class = []

        for class_idx in range(self.num_classes):
            indices = np.where(classes == class_idx)
            self.features_per_class.append(np.random.permutation(features[indices]))
            self.classes_per_class.append(np.random.permutation(classes[indices]))
            self.domains_per_class.append(np.random.permutation(domains[indices]))

        self.data_num = min(
            [len(feature_per_class) for feature_per_class in self.features_per_class])  # get min number of classes

        for i in range(self.num_classes):
            self.features_per_class[i] = tf.constant(self.features_per_class[i][:self.data_num], dtype=tf.float32)
            self.classes_per_class[i] = tf.constant(self.classes_per_class[i][:self.data_num], dtype=tf.int64)
            self.domains_per_class[i] = tf.constant(self.domains_per_class[i][:self.data_num], dtype=tf.int64)
            # self.features_per_class[i] = torch.from_numpy(self.features_per_class[i][:self.data_num]).float()
            # self.classes_per_class[i] = torch.from_numpy(self.classes_per_class[i][:self.data_num])
            # self.domains_per_class[i] = torch.from_numpy(self.domains_per_class[i][:self.data_num])

    def __iter__(self):
        return self
    
    def __next__(self):
        return self

    def __getitem__(self, index):
        features = tf.TensorArray(dtype=tf.float32, size=self.num_classes, element_shape=tf.TensorShape(
            self.features_per_class[0][0].shape))
        classes = tf.TensorArray(dtype=tf.int64, size=self.num_classes)
        domains = tf.TensorArray(dtype=tf.int64, size=self.num_classes)
        # features = torch.FloatTensor(self.num_classes, *(self.features_per_class[0][0].shape)) # make FloatTensor with shape num_classes x F-dim1 x F-dim2...
        # classes = torch.LongTensor(self.num_classes)
        # domains = torch.LongTensor(self.num_classes)

        rand_indices = np.random.permutation(self.num_classes)
        #rand_indices = [i for i in range(self.num_classes)]
        #np.random.shuffle(rand_indices)

        for i in range(self.num_classes):
            features = features.write(i, self.features_per_class[rand_indices[i]][index])
            classes = classes.write(i, self.classes_per_class[rand_indices[i]][index])
            domains = domains.write(i, self.domains_per_class[rand_indices[i]][index])
            # features[i] = self.features_per_class[rand_indices[i]][index]
            # classes[i] = self.classes_per_class[rand_indices[i]][index]
            # domains[i] = self.domains_per_class[rand_indices[i]][index]
        return (features.stack(), classes.stack(), domains.stack())
        #return (features, classes, domains)

    def __len__(self):
        return self.data_num

#class WESADDataset(torch.utils.data.Dataset):
class WESADDataset(tf.data.Dataset):
    def __init__(self, file='./dataset/wesad/both_all.csv',
                 transform=None, domain=None, activity=None, complementary=False, get_calculated_features=False,
                 max_source=100, ecdf=False, num_bin=5):
        st = time.time()
        self.domain = domain
        self.activity = activity
        self.complementary = complementary
        self.max_source = max_source

        self.df = pd.read_csv(file)
        if complementary:   # for multi domain
            if domain:
                self.df = self.df[self.df['domain'] != domain]
            if activity:
                self.df = self.df[self.df['label'] != activity]
        else:
            if domain:
                self.df = self.df[self.df['domain'] == domain]
            if activity:
                self.df = self.df[self.df['label'] == activity]
            #print(len(self.df))

        self.transform = transform
        self.num_bin = num_bin
        ppt = time.time()

        self.preprocessing(get_calculated_features, ecdf)

    def preprocessing(self, get_calculated_feature, ecdf):
        self.num_domains = 0
        self.features = []
        self.class_labels = []
        self.domain_labels = []

        self.datasets = []  # list of dataset per each domain
        self.kshot_datasets = []  # list of dataset per each domain



        if self.complementary:
            domains = set(options.WESADOpt['domains']) - set(self.domain)  # currently supports only user and position
        else:
            domains = set([self.domain])  # bracket is required to append a tuple
        domains = list(domains)
        # domain_superset = list(itertools.product(positions, users))
        valid_domains = []

        for idx in range(max(len(self.df) // WIN_LEN, 0)):
            domain = self.df.iloc[idx * WIN_LEN, 11]
            class_label = self.df.iloc[idx * WIN_LEN, 10]
            domain_label = -1

            for i in range(len(domains)):
                if domains[i] == domain and domains[i] not in valid_domains:
                    valid_domains.append(domains[i])
                    break

            if domain in valid_domains:
                domain_label = valid_domains.index(domain)
            else:
                continue
            feature = self.df.iloc[idx * WIN_LEN:(idx + 1) * WIN_LEN, 0:10].values
            # print('feature: ')
            # print(feature)
            feature = feature.T

            self.features.append(feature)
            self.class_labels.append(self.class_to_number(class_label))
            self.domain_labels.append(domain_label)

        self.num_domains = len(valid_domains) if len(valid_domains) < self.max_source else self.max_source
        self.domain_indices = [i for i in range(len(valid_domains))]
        np.random.shuffle(self.domain_indices)
        self.domain_indices = self.domain_indices[:self.num_domains]

        for i in self.domain_indices:
            print(str(valid_domains[i]), )

        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)
        self.domain_labels = np.array(self.domain_labels)
        '''
        print('self.features: ')
        print(self.features)
        print('self.class_labels: ')
        print(self.class_labels)
        print('self.domain_labels: ')
        print(self.domain_labels)
        '''

        # append dataset for each domain
        for domain_idx in self.domain_indices:
            indices = np.where(self.domain_labels == domain_idx)[0]
            self.datasets.append((self.features[indices], self.class_labels[indices], self.domain_labels[indices]))
            # self.datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(self.features[indices]).float(),
            #                                                     torch.from_numpy(self.class_labels[indices]),
            #                                                     torch.from_numpy(self.domain_labels[indices])))
            kshot_dataset = KSHOTTensorDataset(len(np.unique(self.class_labels)),
                                               self.features[indices],
                                               self.class_labels[indices],
                                               self.domain_labels[indices])
            self.kshot_datasets.append(kshot_dataset)

            # print("i:{:d}, len:{:d}".format(domain_idx, len(kshot_dataset))) # print number of available shots per domain
        # concated dataset
        self.dataset = tf.data.Dataset.from_tensor_slices(self.datasets)
        #self.dataset = torch.utils.data.ConcatDataset(self.datasets)

    def __len__(self):
        # return max(len(self.df) // OVERLAPPING_WIN_LEN - 1, 0)
        return len(self.dataset)

    def get_num_domains(self):
        return self.num_domains

    def get_datasets_per_domain(self):
        return self.kshot_datasets

    def class_to_number(self, label):
        # 'baseline','stress','amusement','mediation'
        dic = {'baseline': 0,
               'stress': 1,
               'amusement': 2,
               'mediation': 3,
               }
        return dic[label]

    def __getitem__(self, idx):
        #if isinstance(idx, torch.Tensor):
        if isinstance(idx, tf.Tensor):
            idx = idx.numpy()
            #idx = idx.item()

        return self.dataset[idx]






# def initial_preprocessing_extract_required_data(file_path):
#     domains = opt['domains']
#     # filePath = "../dataset/wesad/"
#     filePath = '../dataset/wesad/'
#     currentExtractedData = pd.read_csv(file_path +  'both_all.csv')

def numShot():
    df = pd.read_csv(filePath+'both_all.csv')
    df = df.drop(['Unnamed: 0'], axis=1)

    domain = opt['domains']
    activities = opt['classes']

    # Initialize
    num_shot_actset = {key: 0 for key in activities}
    num_shot = {key: num_shot_actset.copy() for key in domain}
    i = 0
    while 1:
        j = 0
        while 1:
            if i+j >= len(df):
                break
            if j is not 0:
                if df['domain'][i+j] != df['domain'][i+j-1]:    # Different Domain
                    i=i+j
                    j=0
                    break
                if df['label'][i+j] != df['label'][i+j-1]:    # Different Activity
                    i=i+j
                    j=0
                    break
            if j is WIN_LEN-1:     # consistent domain and activity with 8 rows
                num_shot[df['domain'][i+j]][df['label'][i+j]] += 1
                i = i+j
                j=0
                break
            j+=1

        if i >= len(df):
            break
        i += 1

    print("num_shot is =========")
    print(num_shot)
    df_num_shot = pd.DataFrame.from_dict(num_shot, orient="index", columns=activities)
    df_num_shot.to_csv(filePath+'wesad_num_shot.csv', index=True)

def pkl_to_csv():
    colDomainList = opt['domains']
    #filePath = '../dataset/wesad/'
    filePath = '../dataset/wesad/'
    def readPKL():
        for domain in colDomainList:
            df = pd.read_pickle('../dataset/wesad/'+domain+'/'+domain+'.pkl')
            #print(df)
            signalChest = df['signal']['chest'].copy()
            signalWrist = df['signal']['wrist'].copy()
            label = df['label'] # array
            # (1) ACC processing
            signalChest['chestAcc']=[]
            signalWrist['wristAcc']=[]
            for i in range(len(signalChest['ACC'])):
                signalChest['chestAcc'].append(np.sqrt(np.power(signalChest['ACC'][i][0], 2) + np.power(signalChest['ACC'][i][1], 2) + np.power(signalChest['ACC'][i][2], 2)))
            for i in range(len(signalWrist['ACC'])):
                signalWrist['wristAcc'].append(np.sqrt(np.power(signalWrist['ACC'][i][0], 2) + np.power(signalWrist['ACC'][i][1], 2) + np.power(signalChest['ACC'][i][2], 2)))
            # Remove [ACC] from signalChest and signalWrist
            signalChest.pop('ACC')
            signalWrist.pop('ACC')

            print("--------Acc processing completed--------")
            # (2) Flat columns
            signalChest['chestECG'] = np.array(signalChest['ECG'][:,0])
            signalChest['chestEMG'] = np.array(signalChest['EMG'][:,0])
            signalChest['chestEDA'] = np.array(signalChest['EDA'][:,0])
            signalChest['chestTemp'] = np.array(signalChest['Temp'][:,0])
            signalChest['chestResp'] = np.array(signalChest['Resp'][:,0])
            signalChest.pop('ECG')
            signalChest.pop('EMG')
            signalChest.pop('EDA')
            signalChest.pop('Temp')
            signalChest.pop('Resp')
            signalWrist['wristBVP'] = np.array(signalWrist['BVP'])
            signalWrist['wristEDA'] = np.array(signalWrist['EDA'])
            signalWrist['wristTemp'] = np.array(signalWrist['TEMP'])
            signalWrist.pop('BVP')
            signalWrist.pop('EDA')
            signalWrist.pop('TEMP')

            print("--------First flatting completed--------")

            # (3) Downsampling
            print("-------Before downsampling-----------")
            '''
            print(len(signalWrist['wristAcc'])) # 32Hz : 8row
            print(len(signalWrist['wristTemp']))   # 4Hz : 1row
            print(len(signalWrist['wristBVP']))  # 64Hz : 16row
            print(len(signalWrist['wristEDA'])) # 4Hz : 1row
            '''
            def downsample_to_proportion(rows, proportion=1):
                return list(islice(rows, 0, len(rows), int(1 / proportion)))
            signalWrist['wristAcc'] = downsample_to_proportion(rows=signalWrist['wristAcc'], proportion=0.125)
            signalWrist['wristBVP'] = downsample_to_proportion(rows=signalWrist['wristBVP'], proportion=0.0625)
            signalChest['chestAcc'] = downsample_to_proportion(rows=signalChest['chestAcc'], proportion=0.005714285714)
            signalChest['chestECG'] = downsample_to_proportion(rows=signalChest['chestECG'], proportion=0.005714285714)
            signalChest['chestEMG'] = downsample_to_proportion(rows=signalChest['chestEMG'], proportion=0.005714285714)
            signalChest['chestEDA'] = downsample_to_proportion(rows=signalChest['chestEDA'], proportion=0.005714285714)
            signalChest['chestTemp'] = downsample_to_proportion(rows=signalChest['chestTemp'], proportion=0.005714285714)
            signalChest['chestResp'] = downsample_to_proportion(rows=signalChest['chestResp'], proportion=0.005714285714)


            print("-------After downsampling-----------")
            # Flat dimension
            l = []
            m = []
            n = []
            for sublist in signalWrist['wristTemp']:
                for item in sublist:
                    l.append(item)
            for sublist in signalWrist['wristBVP']:
                for item in sublist:
                    m.append(item)
            for sublist in signalWrist['wristEDA']:
                for item in sublist:
                    n.append(item)
            signalWrist['wristTemp'] = l
            signalWrist['wristBVP'] = m
            signalWrist['wristEDA'] = n

            # (4) Dict to dataframe
            dfSignalChest = pd.DataFrame.from_dict(signalChest)
            dfSignalWrist = pd.DataFrame.from_dict(signalWrist)

            # minmax Scaling
            minmaxScaledSignalChest = minmaxScaling(dfSignalChest)
            minmaxScaledSignalWrist = minmaxScaling(dfSignalWrist)



            # label and domain fields
            minmaxScaledSignalChest['label'] = downsample_to_proportion(rows=label, proportion=0.005714285714)
            minmaxScaledSignalWrist['label'] = downsample_to_proportion(rows=label, proportion=0.005714285714)
            l = []
            m = []
            for _ in range(len(signalWrist['wristAcc'])):
                l.append(domain)
            for _ in range(len(signalChest['chestAcc'])):
                m.append(domain)
            minmaxScaledSignalChest['domain'] = m
            minmaxScaledSignalWrist['domain'] = l
            minmaxScaledSignalChest = minmaxScaledSignalChest.loc[dfSignalChest['label'].isin([1,2,3,4])]
            minmaxScaledSignalWrist = minmaxScaledSignalWrist.loc[dfSignalWrist['label'].isin([1,2,3,4])]


            # sampling
            minmaxScaledSignalChest = minmaxScaledSignalChest.reset_index(drop=True)
            minmaxScaledSignalWrist = minmaxScaledSignalWrist.reset_index(drop=True)
            minmaxScaledSignalChest = sampling(minmaxScaledSignalChest)
            minmaxScaledSignalWrist = sampling(minmaxScaledSignalWrist)


            minmaxScaledSignalBoth = pd.concat([minmaxScaledSignalChest, minmaxScaledSignalWrist], axis = 1)
            minmaxScaledSignalBoth = minmaxScaledSignalBoth.loc[:,~minmaxScaledSignalBoth.columns.duplicated()]
            minmaxScaledSignalBoth = minmaxScaledSignalBoth[['chestAcc','chestECG','chestEMG','chestEDA','chestTemp','chestResp','wristAcc','wristBVP','wristEDA','wristTemp','label','domain']]

            minmaxScaledSignalChest = minmaxScaledSignalChest.reset_index(drop=True)
            minmaxScaledSignalWrist = minmaxScaledSignalWrist.reset_index(drop=True)
            minmaxScaledSignalBoth = minmaxScaledSignalBoth.reset_index(drop=True)

            minmaxScaledSignalChest['label'] = minmaxScaledSignalChest['label'].replace({1:"baseline", 2:"stress", 3:"amusement", 4:"mediation"})
            minmaxScaledSignalWrist['label'] = minmaxScaledSignalWrist['label'].replace({1:"baseline", 2:"stress", 3:"amusement", 4:"mediation"})
            minmaxScaledSignalBoth['label'] = minmaxScaledSignalBoth['label'].replace({1:"baseline", 2:"stress", 3:"amusement", 4:"mediation"})

            minmaxScaledSignalChest.to_csv('../dataset/wesad/' + domain + '/' + domain + '_chest.csv')
            minmaxScaledSignalWrist.to_csv('../dataset/wesad/'+domain+'/'+domain+'_wrist.csv')
            minmaxScaledSignalBoth.to_csv('../dataset/wesad/'+domain+'/'+domain+'_both.csv')
            print("---------"+domain+" saved------------")

    #readPKL()

    def combineCSV():
        #colChestLabel = ['chestAcc','chestECG','chestEMG','chestEDA','chestTemp','chestResp', 'label','domain']
        #colWristLabel = ['wristAcc','wristBVP','wristEDA','wristTemp','label','domain']
        colBothLabel = ['chestAcc','chestECG','chestEMG','chestEDA','chestTemp','chestResp','wristAcc','wristBVP','wristEDA','wristTemp','label','domain']

        #chestAll = pd.DataFrame(columns=colChestLabel)
        #wristAll = pd.DataFrame(columns=colWristLabel)
        bothAll = pd.DataFrame(columns = colBothLabel)

        #extractedData_chest = pd.DataFrame(columns=colChestLabel)
        #extractedData_wrist = pd.DataFrame(columns = colWristLabel)
        extractedData_both = extractedData_both.sort_values(['domain', 'label'], ascending=[True, True])
        extractedData_both = pd.DataFrame(columns=colBothLabel)

        for domain in domains:
            # currentExtractedData_chest = pd.read_csv(filePath + domain + '/'+domain+'_chest.csv', usecols=colChestLabel)
            # currentExtractedData_wrist = pd.read_csv(filePath + domain + '/'+domain+'_wrist.csv', usecols=colWristLabel)
            currentExtractedData_both = pd.read_csv(filePath+domain+'/'+domain+'_both.csv', usecols=colBothLabel)

            # extractedData_chest = extractedData_chest.append(currentExtractedData_chest, ignore_index=True)
            # extractedData_wrist = extractedData_wrist.append(currentExtractedData_wrist, ignore_index=True)
            extractedData_both = extractedData_both.append(currentExtractedData_both, ignore_index=True)

            print("----------"+domain+" completed----------")

        # extractedData_chest.to_csv('../dataset/wesad/chest_all.csv', index=False)
        # extractedData_wrist.to_csv('../dataset/wesad/wrist_all.csv', index=False)
        extractedData_both.to_csv('../dataset/wesad/both_all.csv', index=False)

    combineCSV()

if __name__ == '__main__':
    is_minmax_scaling = True
    domains = opt['domains']
    activities = opt['classes']
    pkl_to_csv()
    #numShot()
    #initial_preprocessing_extract_required_data('../dataset/wesad/')

#initial_preprocessing_extract_required_data('../dataset/wesad/')
#numShot()