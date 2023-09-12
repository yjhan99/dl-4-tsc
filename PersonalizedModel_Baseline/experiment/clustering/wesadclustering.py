import random
import math

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
        result.append({"train": test_set, "val": val_set, "test": train_set})

    random.seed()
    return result


def n_fold_split_cluster_feature(subject_ids, n, seed=5):
    result = []

    random.seed(seed)
    subject_ids = list(subject_ids)

    test_sets = [subject_ids[i::n] for i in range(n)]

    for test_set in test_sets:
        result.append({"train": test_set, "val": test_set, "test": test_set})

    random.seed()
    return result