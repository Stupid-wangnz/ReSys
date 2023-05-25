from utils import load_data, split_validate_train, output_test_result
from SVD import FunkSVD


def main():
    data, test_data, id_index_dict, n_users, n_items = load_data()
    train_data, validate_data = split_validate_train(data)
    resys = FunkSVD()
    resys.fit(train_data, validate_data, n_users, n_items)
    test_result = resys.predict(test_data)
    output_test_result(test_result, id_index_dict=id_index_dict)


if __name__ == "__main__":
    main()
