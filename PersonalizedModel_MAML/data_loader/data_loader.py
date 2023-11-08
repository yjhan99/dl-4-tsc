import tensorflow as tf
from tensorflow.data import Dataset
import numpy as np
import time
import math

from .wesadDataset import wesadDataset
from .kemoworkDataset import kemoworkDataset

def split_data(entire_data, valid_split, test_split, train_max_rows, valid_max_rows, test_max_rows):
    valid_size = math.floor(len(entire_data) * valid_split)
    test_size = math.floor(len(entire_data) * test_split)

    train_size = len(entire_data) - valid_size - test_size
    if valid_size > 0 and test_size > 0 : # this meets when it's mostly target condition
        if train_size < 20: # for 'src = all' && meta learning cases, source should have at least 5 supports and queries, while target has at least 10 shots for evaluation. so at least "22 shots" are required
            gap = 20 - train_size
            train_size = 20
            # valid_size -= gap # decrease valid size
            total_remain = valid_size + test_size - gap
            valid_size = math.floor(total_remain * (valid_split/(valid_split+test_split)))
            test_size = total_remain - valid_size

            if valid_size == 0 or test_size == 0:
                print(train_size, valid_size, test_size)
                exit(1)


    assert(train_size >=0 and valid_size >=0 and test_size >=0)

    train_data, valid_data, test_data = torch.utils.data.random_split(entire_data,
                                                                      [train_size, valid_size, test_size])

    if len(entire_data) > train_max_rows:
        train_data = torch.utils.data.Subset(train_data, range(train_max_rows))
    if len(valid_data) > valid_max_rows:
        valid_data = torch.utils.data.Subset(valid_data, range(valid_max_rows))
    if len(test_data) > test_max_rows:
        test_data = torch.utils.data.Subset(test_data, range(test_max_rows))

    return train_data, valid_data, test_data

def domain_data_loader(args, domains, file_path, batch_size, train_max_rows=np.inf, valid_max_rows=np.inf,
                       test_max_rows=np.inf, valid_split=0.1, test_split=0.1, separate_domains=False, is_src=True):
    entire_datasets = []
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    st = time.time()
    
    if isinstance(domains, (list,)):
        processed_domains = domains
    else:
        processed_domains = [domains]
    
    ##-- preprocessing 'rest' domain
    if domains == ['all']:
        domains = ['rest'] # same processing for 'all' case
    if domains == ['rest']:  # convert 'rest' into list of domains
        test_domains = []

        if args.dataset in ['wesad_scaled']:
            test_domains.append(args.tgt)
        elif args.dataset in ['kemowork']:
            test_domains.append(args.tgt)

        processed_domains = test_domains

    ##-- load dataset per each domain
    print('Domains:{}'.format(processed_domains))
    for domain in processed_domains:
        if args.dataset in ['wesad', 'wesad_scaled']:
            options = domain
            if separate_domains:
                for train_data in WESADDataset(file=file_path, domain=options, complementary=True if domains == ['rest'] else False,
                                                      max_source = args.num_source).get_datasets_per_domain():
                    entire_datasets.append(train_data)
            else:
                train_data = WESADDataset(file=file_path, domain=options, complementary=True if domains == ['rest'] else False,
                                                 max_source = args.num_source)

                if len(train_data) == 0:
                    print('Zero train data: {:s}'.format(domain))
                    continue
                else:
                    entire_datasets.append(train_data)

        elif args.dataset in ['kemowork']:  # TODO
            print("Not implemented yet!")

    ##-- split each dataset into train, valid, and test
    for train_data in entire_datasets:
        train_data, valid_data, test_data = split_data(train_data, valid_split, test_split, train_max_rows,
                                                       valid_max_rows, test_max_rows)
        train_datasets.append(train_data)
        valid_datasets.append(valid_data)
        test_datasets.append(test_data)

        print('#Multi?:{:d} data_loader len:{:d} Train: {:d}\t# Valid: {:d}\t# Test: {:d}'.format(
            True if domains == ['rest'] else False, len(train_data), len(train_data), len(valid_data),
            len(test_data)))

    train_datasets = train_datasets[:args.num_source]
    valid_datasets = valid_datasets[:args.num_source]
    test_datasets = test_datasets[:args.num_source]

    print('# Time: {:f} secs'.format(time.time() - st))

    if separate_domains:
        # actual batch size is multiplied by num_class
        train_data_loaders = [train_dataset.batch(batch_size) for train_dataset in train_datasets]
        valid_data_loaders = [valid_dataset.batch(32) for valid_dataset in valid_datasets]
        test_data_loaders = [test_dataset.batch(batch_size) for test_dataset in test_datasets]

        #train_data_loaders = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=False, drop_last=True)
        #valid_data_loaders = datasets_to_dataloader(valid_datasets, batch_size=1,
        #                                            concat=False)  # set validation batch_size = 32 to boost validation speed
        # if is_src:
        #     test_data_loaders = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=False, drop_last=True) #for query data, shape should be matched among them
        # else:
        #     test_data_loaders = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=False)
        # assert (len(train_data_loaders) == len(valid_data_loaders) == len(test_data_loaders))
        
        data_loaders = [{
            'train': train_data_loader,
            'valid': valid_data_loader if len(valid_data_loaders) == len(train_data_loaders) else None,
            'test': test_data_loader,
            'num_domains': len(train_data_loaders)
        } for train_data_loader, valid_data_loader, test_data_loader in zip(train_data_loaders, valid_data_loaders, test_data_loaders)]
        # data_loaders = []
        # for i in range(len(train_data_loaders)):
        #     data_loader = {
        #         'train': train_data_loaders[i],
        #         'valid': valid_data_loaders[i] if len(valid_data_loaders) == len(train_data_loaders) else None,
        #         'test': test_data_loaders[i],
        #         'num_domains': len(train_data_loaders)
        #     }
        #     data_loaders.append(data_loader)


        print('num_domains:' + str(len(train_data_loaders)))

        return data_loaders
    else:
        train_data_loader = tf.concat(train_datasets, axis=0).batch(batch_size)
        valid_data_loader = tf.concat(valid_datasets, axis=0).batch(batch_size)
        test_data_loader = tf.concat(test_datasets, axis=0).batch(batch_size)

        #train_data_loader = datasets_to_dataloader(train_datasets, batch_size=batch_size, concat=True, drop_last=True)
        #valid_data_loader = datasets_to_dataloader(valid_datasets, batch_size=batch_size, concat=True)
        #test_data_loader = datasets_to_dataloader(test_datasets, batch_size=batch_size, concat=True)

        data_loader = {
            'train': train_data_loader,
            'valid': valid_data_loader,
            'test': test_data_loader,
            'num_domains': sum([len(dataset) for dataset in train_datasets]),
            #'num_domains': sum([dataset.dataset.get_num_domains() for dataset in train_datasets]),
        }

        print('num_domains:' + str(data_loader['num_domains']))
        return data_loader

def support_query_data_loader(target_data_loader, query_split,  batch_size, nshot, src_separate_domains = True):
    '''
    :param target_data_loader: original target_data_loader
    :param query_split: src_query_split
    :param batch_size: trg_batch_size
    :param nshot: src_batch_size
    :param src_separate_domains:
    :return: (support+query)_data_loader, final_target_data_loader
    '''
    # Distribute original train data of target_data_loader to source(support+query) and target(train)
    source_dataset = []
    empty_dataset = []
    train_target_dataset = []

    for i in range(len(target_data_loader)):
        total_samples = len(target_data_loader[i]['train'].get_data().shape[0])
        temp_source_size = math.floor(total_samples * 0.5) # Split by 50:50
        #temp_source_size = math.floor(len(target_data_loader[i]['train'].dataset.indices) * 0.5) # split by 50:50

        temp_source_data = tf.data.Dataset.from_tensor_slices(target_data_loader[i]['train'].get_data())
        temp_source_data = temp_source_data.take(temp_source_size)
        temp_target_train_data = tf.data.Dataset.from_tensor_slices(target_data_loader[i]['train'].get_data())
        temp_target_train_data = temp_target_train_data.skip(temp_source_size)

        #temp_source_data = copy.copy(target_data_loader[i]['train'].dataset)
        #temp_source_data_indices = target_data_loader[i]['train'].dataset.indices[:temp_source_size]
        #temp_source_data.indices = temp_source_data_indices
        #temp_target_train_data = copy.copy(target_data_loader[i]['train'].dataset)
        #temp_target_data_indices = target_data_loader[i]['train'].dataset.indices[temp_source_size:]
        #temp_target_train_data.indices = temp_target_data_indices

        source_dataset.append(temp_source_data)
        train_target_dataset.append(temp_target_train_data)

    # Distribute source_dataset to support_set, valid_set(empty), query_set
    support_sets=[]
    valid_sets=[]
    query_sets=[]
    for i in range(len(source_dataset)):
        total_samples = len(source_datasets[i])
        temp_query_size = math.floor(total_samples * query_split)
        #temp_query_size = math.floor(len(source_dataset[i].indices)*query_split)
        
        if query_split > 0: # for meta-learning methods
            temp_query_data = source_datasets[i].take(temp_query_size)
            query_sets.append(temp_query_data)
            # temp_query_data = copy.copy(source_dataset[i])
            # temp_query_data_indices = source_dataset[i].indices[:temp_query_size]
            # temp_query_data.indices = temp_query_data_indices
            # query_sets.append(temp_query_data)

        temp_support_data = source_datasets[i].skip(temp_query_size)
        support_sets.append(temp_support_data)
        # temp_support_data = copy.copy(source_dataset[i])
        # temp_support_data_indices = source_dataset[i].indices[temp_query_size:]
        # temp_support_data.indices = temp_support_data_indices
        # support_sets.append(temp_support_data)

    temp_source_data_loaders = []

    # each data_loader for temp_source_data_loaders
    if src_separate_domains:

        # support_data_loaders = datasets_to_dataloader(support_sets, batch_size=nshot, concat=False, drop_last=True)
        # valid_data_loaders = datasets_to_dataloader(valid_sets, batch_size=32, concat=False)
        # query_data_loaders = datasets_to_dataloader(query_sets, batch_size=nshot, concat=False, drop_last=True)

        ## temp_source_data_loader = ['train'//support]:0.5 ['valid']:0 ['test'//query]:0.5
        # temp_source_data_loaders = []
        for i in range(len(support_data_loaders)):
            temp_support_data_loader = support_sets[i].batch(nshot)
            valid_data_loader = valid_sets[i].batch(32)
            query_data_loader = query_sets[i].batch(nshot) if len(query_sets) != 0 else None

            temp_source_data_loader = {
                'train': temp_support_data_loader,
                'valid': valid_data_loader if len(valid_data_loader) == len(temp_support_data_loader) else None,
                'test': query_data_loader if query_data_loader is not None else None,
                'num_domains': len(temp_support_data_loader)
            }

            temp_source_data_loaders.append(temp_source_data_loader)
            # temp_source_data_loader = {
            #     'train': support_data_loaders[i],
            #     'valid': valid_data_loaders[i] if len(valid_data_loaders) == len(support_data_loaders) else None,
            #     'test': query_data_loaders[i] if len(query_data_loaders) != 0 else None,
            #     'num_domains': len(support_data_loaders)
            # }
            # temp_source_data_loaders.append(temp_source_data_loader)
        print('num_domains: ' + str(len(support_sets)))
        #print('num_domains:' + str(len(support_data_loaders)))
    else:
        support_data_loader = support_sets[0].batch(nshot)
        valid_data_loader = valid_sets[0].batch(nshot)
        query_data_loader = query_sets[0].batch(nshot)

        temp_source_data_loaders = {
            'train': support_data_loader,
            'valid': valid_data_loader,
            'test': query_data_loader,
            'num_domains': len(support_sets)
        }

        print('num_domains:' + str(temp_source_data_loaders['num_domains']))
        # support_data_loaders = datasets_to_dataloader(support_sets, batch_size=nshot, concat=True, drop_last=True)
        # valid_data_loaders = datasets_to_dataloader(valid_sets, batch_size=nshot, concat=True)
        # query_data_loaders = datasets_to_dataloader(query_sets, batch_size=nshot, concat=True)

        # temp_source_data_loaders = {
        #     'train': support_data_loaders,
        #     'valid': valid_data_loaders,
        #     'test': query_data_loaders,
        #     'num_domains': sum([dataset.dataset.get_num_domains() for dataset in support_sets]),
        # }
        # print('num_domains:' + str(temp_source_data_loaders['num_domains']))

    # Update target_data_loader[0]['train']
    target_valid_set=[]
    target_test_set=[]
    for i in range(len(target_data_loader)):
        temp_valid_set = tf.data.Dataset.from_tensor_slices(target_data_loader[i]['valid'].get_data())
        target_valid_sets.append(temp_valid_set)

        temp_test_set = tf.data.Dataset.from_tensor_slices(target_data_loader[i]['test'].get_data())
        target_test_sets.append(temp_test_set)
        # temp_valid_set = target_data_loader[i]['valid'].dataset
        # temp_test_set = target_data_loader[i]['test'].dataset

        # target_valid_set.append(temp_valid_set)
        # target_test_set.append(temp_test_set)

    target_train_data_loaders = tf.data.Dataset.from_tensor_slices(train_target_datasets[0].get_data()).batch(batch_size)
    target_valid_data_loaders = target_valid_sets[0].batch(32)
    target_test_data_loaders = target_test_sets[0].batch(batch_size)
    # target_train_data_loaders = datasets_to_dataloader(train_target_dataset, batch_size=batch_size, concat=False, drop_last=True)
    # target_valid_data_loaders = datasets_to_dataloader(target_valid_set, batch_size=32, concat=False)
    # target_test_data_loaders = datasets_to_dataloader(target_test_set, batch_size=batch_size, concat=False)

    temp_target_data_loaders = []
    
    for i in range(len(target_train_data_loaders)):
        temp_target_data_loader = {
            'train': target_train_data_loaders[i],
            'valid': target_valid_data_loaders[i] if len(target_valid_data_loaders) == len(target_train_data_loaders) else None,
            'test': target_test_data_loaders[i],
            'num_domains': len(target_train_data_loaders)
        }
        temp_target_data_loaders.append(temp_target_data_loader)
    print('num_domains:' + str(len(target_train_data_loaders)))

    return temp_source_data_loaders, temp_target_data_loaders

# if __name__ == '__main__':
#     import argparse
#     import src_query_split
#     sys.path.append("..")
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--dataset', type=str, default='wesad', help='Dataset to be used')
#     args = parser.parse_args()

#     domains = []