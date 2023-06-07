import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import load_data, split_validate_train, output_test_result, load_attribute, transform_data, \
    split_validate_train_for_svdknn, split_validate_train_for_svd
from SVD import FunkSVD, BiasSVD
from ItemSVD import ItemSVD, SVDkNN
import argparse

np.random.seed(3407)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dim', type=int, default=250)
    parser.add_argument('-scale', type=int, default=20)
    args = parser.parse_args()
    # data, test_data, id_index_dict, n_users, n_items, mean, std = load_data_mean_std()
    data, test_data, id_index_dict, n_users, n_items = load_data()
    train_data, validate_data = split_validate_train_for_svd(data, scale=args.scale)

    item_attribute = load_attribute(id_index_dict)
    scaler = MinMaxScaler()
    scaler.fit(item_attribute)
    item_attribute = scaler.transform(item_attribute)
    item_attribute = np.nan_to_num(item_attribute)

    train_data = transform_data(train_data)
    validate_data = transform_data(validate_data)

    resys = BiasSVD(factors=args.dim, scale=args.scale)
    resys.fit(train_data, validate_data, n_users, n_items)

    # test_result = resys.predict(test_data)
    # output_test_result(test_result, id_index_dict=id_index_dict)


def svd_plus_knn():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dim', type=int, default=250)
    parser.add_argument('-scale', type=int, default=20)
    args = parser.parse_args()
    # data, test_data, id_index_dict, n_users, n_items, mean, std = load_data_mean_std()
    data, test_data, id_index_dict, n_users, n_items = load_data()
    train_data_dict, validate_data_dict, id_index_dict, n_users, n_items = split_validate_train_for_svdknn(
        data,
        scale=args.scale,
        id_index_dict=id_index_dict)

    item_attribute = load_attribute(id_index_dict)
    scaler = MinMaxScaler()
    scaler.fit(item_attribute)
    item_attribute = scaler.transform(item_attribute)
    item_attribute = np.nan_to_num(item_attribute)

    train_data = transform_data(train_data_dict)
    validate_data = transform_data(validate_data_dict)

    resys = SVDkNN(factors=args.dim, scale=args.scale, u_i_dict=train_data_dict,
                   item_attribute=item_attribute, train_n_item=n_items)
    resys.fit(train_data, validate_data, n_users, n_items)


if __name__ == "__main__":
    svd_plus_knn()
