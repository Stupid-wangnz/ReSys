import numpy as np
from collections import defaultdict


def load_data(path='./data/'):
    train_data = defaultdict(dict)
    item_set = set()
    id_index_dict = {}

    train_path = path + 'train.txt'
    test_path = path + 'test.txt'

    with open(train_path, 'r') as f:
        user_items = f.readline()
        while user_items:
            user, items = [int(x) for x in user_items.split('|')]
            for item in range(items):
                line = f.readline()
                item_id, item_score = [int(x) for x in line.split('  ')]

                if item_id not in item_set:
                    id_index_dict[item_id] = len(item_set)
                    item_set.add(item_id)

                item_index = id_index_dict[item_id]
                train_data[user][item_index] = item_score

            user_items = f.readline()

    test_data = defaultdict(dict)
    with open(test_path, 'r') as f:
        user_items = f.readline()
        while user_items:
            user, items = [int(x) for x in user_items.split('|')]
            for item in range(items):
                item_id = f.readline()
                item_id = int(item_id)

                if item_id not in item_set:
                    id_index_dict[item_id] = len(item_set)
                    item_set.add(item_id)

                item_index = id_index_dict[item_id]
                test_data[user][item_index] = 0

            user_items = f.readline()

    print(f"Total number of user: {len(train_data)}")
    print(f"Total number of items: {len(item_set)}")
    return train_data, test_data, id_index_dict, len(train_data), len(item_set)


def load_attribute(id_index_dict, path='./data/'):
    item_attribute = np.full((len(id_index_dict), 2), np.nan)
    attribute_path = path + 'itemAttribute.txt'

    with open(attribute_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            line = line.replace('None', '0')
            item, attribute1, attribute2 = line.split('|')
            item_id = int(item)
            if item_id in id_index_dict.keys():
                item_index = id_index_dict[item_id]
                if attribute1 == 0:
                    item_attribute[item_index][0] = np.nan
                else:
                    item_attribute[item_index][0] = int(attribute1)
                if attribute2 == 0:
                    item_attribute[item_index][1] = np.nan
                else:
                    item_attribute[item_index][1] = int(attribute2)

    return item_attribute


def transform_data(data):
    score_num = sum(len(items) for items in data.values())

    reconstruct_data = np.zeros((score_num, 3))
    n = 0
    for u, items in data.items():
        for i in items.keys():
            reconstruct_data[n][0] = u
            reconstruct_data[n][1] = i
            reconstruct_data[n][2] = items[i]
            n += 1

    return reconstruct_data


def split_validate_train(data, validate_size=0.1, scale=1.):
    validate_data = defaultdict(dict)
    train_data = defaultdict(dict)
    for user, items in data.items():
        items_list = list(items.keys())
        np.random.shuffle(items_list)
        validate_num = int(len(items_list) * validate_size)
        validate_items = {item_index: items[item_index] for item_index in items_list[:validate_num]}
        train_items = {item_index: (items[item_index] / scale) for item_index in items_list[validate_num:]}
        validate_data[user] = validate_items
        train_data[user] = train_items
    return train_data, validate_data


def split_validate_train_for_svd(data, n=4, scale=1., id_index_dict=None):
    validate_data = defaultdict(dict)
    train_data = defaultdict(dict)
    for user, items in data.items():
        items_list = list(items.keys())
        np.random.shuffle(items_list)
        validate_items_list = items_list[:n]
        train_items_list = items_list[n:]
        validate_items = {item_index: (items[item_index]) for item_index in validate_items_list}
        train_items = {item_index: (items[item_index] / scale) for item_index in train_items_list}
        validate_data[user] = validate_items
        train_data[user] = train_items
    return train_data, validate_data


def split_validate_train_for_svdknn(data, n=4, scale=1., id_index_dict=None):
    index_id_dict = {v: k for k, v in id_index_dict.items()}

    validate_data = defaultdict(dict)
    train_data = defaultdict(dict)
    for user, items in data.items():
        items_list = list(items.keys())
        np.random.shuffle(items_list)
        validate_items_list = items_list[:n]
        train_items_list = items_list[n:]
        validate_items = {index_id_dict[item_index]: (items[item_index]) for item_index in validate_items_list}
        train_items = {index_id_dict[item_index]: (items[item_index] / scale) for item_index in train_items_list}
        validate_data[user] = validate_items
        train_data[user] = train_items

    id_index_dict = {}
    train_item = set()
    r_train_data = defaultdict(dict)
    r_validate_data = defaultdict(dict)
    for user, items in train_data.items():
        for item_id in items.keys():
            if item_id not in train_item:
                id_index_dict[item_id] = len(train_item)
                train_item.add(item_id)

            item_index = id_index_dict[item_id]
            r_train_data[user][item_index] = train_data[user][item_id]

    train_item_len = len(train_item)

    for user, items in validate_data.items():
        for item_id in items.keys():
            if item_id not in train_item:
                id_index_dict[item_id] = len(train_item)
                train_item.add(item_id)
            item_index = id_index_dict[item_id]
            r_validate_data[user][item_index] = validate_data[user][item_id]

    print(f"Total number of items: {len(train_item)}. Train number of items: {train_item_len}")
    return r_train_data, r_validate_data, id_index_dict, len(r_train_data), train_item_len


def output_test_result(test_result, path='./data/result.txt', id_index_dict=None):
    index_id_dict = {v: k for k, v in id_index_dict.items()}
    with open(path, 'w') as f:
        for u, items in test_result.items():
            f.write(f"{u}|{len(items)}\n")
            for i in items.keys():
                f.write(f"{index_id_dict[i]}  {test_result[u][i]}\n")
