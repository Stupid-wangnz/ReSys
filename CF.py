import numpy as np
from sklearn.metrics import mean_squared_error
from utils import load_data, split_validate_train
from collections import defaultdict


class MatrixFactorization:
    """
    基于矩阵分解的协同过滤算法
    """

    def __init__(self, factors=10, epochs=10, lr=0.01, reg=0.1, init_mean=0, init_std=0.1):
        self.factors = factors  # 潜在特征数量
        self.epochs = epochs  # 迭代次数
        self.lr = lr  # 学习率
        self.reg = reg  # 正则化系数
        self.init_mean = init_mean  # 随机初始化的均值
        self.init_std = init_std  # 随机初始化的标准差

    def fit(self, data):
        """
        训练模型
        :param data: 数据字典，格式为 {user_id: {item_id: rating, ...}, ...}
        """
        self.global_mean = np.mean(list(rating for user in data for _, rating in data[user].items()))
        self.user_factors = {}
        self.item_factors = {}
        for user in data:
            self.user_factors[user] = np.random.normal(self.init_mean, self.init_std, self.factors)
        for item in item_attrs:
            self.item_factors[item] = np.random.normal(self.init_mean, self.init_std, self.factors)
        self.user_bias = {}
        self.item_bias = {}
        for user in data:
            self.user_bias[user] = 0
        for item in item_attrs:
            self.item_bias[item] = 0
        for epoch in range(self.epochs):
            for user in data:
                for item in data[user]:
                    rating = data[user][item]
                    error = rating - (
                            self.global_mean + self.user_bias[user] + self.item_bias[item] + self.user_factors[user].dot(self.item_factors[item]))
                    self.user_factors[user] += self.lr * (
                            error * self.item_factors[item] - self.reg * self.user_factors[user])
                    self.item_factors[item] += self.lr * (
                            error * self.user_factors[user] - self.reg * self.item_factors[item])
                    self.user_bias[user] += self.lr * (error - self.reg * self.user_bias[user])
                    self.item_bias[item] += self.lr * (error - self.reg * self.item_bias[item])

    def predict(self, user, item):
        """
        预测用户对物品的评分
        :param user: 用户id
        :param item: 物品id
        :return: 预测得分
        """
        if user not in self.user_factors:
            return self.global_mean
        elif item not in self.item_factors:
            return self.global_mean
        else:
            return self.global_mean + self.user_bias[user] + self.item_bias[item] + self.user_factors[user].dot(
                self.item_factors[item])


if __name__ == '__main__':
    data, test_data, _, _, _ = load_data()

    train_data, validate_data = split_validate_train(data)
    model = MatrixFactorization(factors=10, epochs=20, lr=0.03, reg=0.1)
    model.fit(train_data, )

    # 使用RMSE评价指标评估模型
    y_true = [validate_data[user][item] for user in validate_data for item in validate_data[user]]
    y_pred = [model.predict(user, item) for user in validate_data for item in validate_data[user]]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print('测试集上的RMSE为{}'.format(rmse))
