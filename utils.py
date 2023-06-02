import random
import numpy as np
import pandas as pd
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

def load_data_mean_std(path='./data/'):
    train_data = defaultdict(dict)
    item_set = set()
    id_index_dict = {}

    train_path = path + 'train.txt'
    test_path = path + 'test.txt'

    score_list = []

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
                score_list.append(item_score)
            user_items = f.readline()
    score_list = np.array(score_list)

    mean = np.mean(score_list)
    std = 10
    
    for u, items in train_data.items():
        for i in items.keys():
            items[i] = items[i] / std

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
    return train_data, test_data, id_index_dict, len(train_data), len(item_set), mean, std

def split_validate_train(data, validate_size=0.1):
    validate_data = defaultdict(dict)
    train_data = defaultdict(dict)
    for user, items in data.items():
        items_list = list(items.keys())
        random.shuffle(items_list)
        validate_num = int(len(items_list) * validate_size)
        validate_items = {item_index: items[item_index] for item_index in items_list[:validate_num]}
        train_items = {item_index: items[item_index] for item_index in items_list[validate_num:]}
        validate_data[user] = validate_items
        train_data[user] = train_items
    return train_data, validate_data


"""
def load_test_data(path='./data/test.txt', id_index_dict=None):
    test_data = defaultdict(dict)
    with open(path, 'r') as f:
        user_items = f.readline()
        while user_items:
            user, items = [int(x) for x in user_items.split('|')]
            for item in range(items):
                item_id = f.readline()
                item_id = int(item_id)
                item_index = id_index_dict[item_id]
                test_data[user][item_index] = 0

            user_items = f.readline()
    return test_data
"""


def output_test_result(test_result, path='./data/result.txt', id_index_dict=None):
    index_id_dict = {v: k for k, v in id_index_dict.items()}
    with open(path, 'w') as f:
        for u, items in test_result.items():
            f.write(f"{u}|{len(items)}\n")
            for i in items.keys():
                f.write(f"{index_id_dict[i]}  {test_result[u][i]}\n")
