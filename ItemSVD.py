import numpy as np
from collections import defaultdict

from SVD import BiasSVD


class ItemSVD(BiasSVD):
    def __init__(self, learning_rate=0.003, reg_param=0.02, bias_reg_param=0.02, n_iters=40, factors=200, scale=1,
                 item_attribute=None):
        super().__init__(learning_rate, reg_param, bias_reg_param, n_iters, factors, scale)
        self.item_attribute = item_attribute

    def fit(self, X, validate_data, n_users, n_items):
        n_attribute = self.item_attribute.shape[1]
        self.user_vecs = np.random.rand(n_users, self.factors + n_attribute) / (self.factors ** 0.5)
        self.item_vecs = np.random.rand(n_items, self.factors + n_attribute) / (self.factors ** 0.5)
        self.item_vecs[:, self.factors:] = self.item_attribute

        self.global_bias = np.mean([r for _, _, r in X])
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        for iter in range(self.n_iters):
            lr = self.lr
            _X = X[np.random.permutation(X.shape[0])]
            for u, i, r in _X:
                u = int(u)
                i = int(i)
                e = r - self._score(u, i)  # Compute error residuals

                self.user_bias[u] += lr * (e - self.bias_reg_param * self.user_bias[u])
                self.item_bias[i] += lr * (e - self.bias_reg_param * self.item_bias[i])
                uv = self.user_vecs[u, :self.factors]
                iv = self.item_vecs[i, :]
                self.user_vecs[u, :] += lr * (e * iv - self.reg * self.user_vecs[u, :])
                self.item_vecs[i, :self.factors] += lr * (e * uv - self.reg * self.item_vecs[i, :self.factors])

            train_rmse = self.validate(X, True)
            validate_rmse = self.validate(validate_data, False)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def _score(self, u, i):
        return self.global_bias + self.user_bias[u] + self.item_bias[i] + np.dot(self.user_vecs[u, :],
                                                                                 self.item_vecs[i, :].T)


class SVDkNN(BiasSVD):
    def __init__(self, learning_rate=0.003, reg_param=0.02, bias_reg_param=0.02, n_iters=40, factors=200, scale=1,
                 u_i_dict=None, item_attribute=None, train_n_item=0):
        super().__init__(learning_rate, reg_param, bias_reg_param, n_iters, factors, scale)
        self.u_i_dict = u_i_dict
        self.item_attribute = item_attribute
        self.train_n_item = train_n_item

    def fit(self, X, validate_data, n_users, n_items):
        self.user_vecs = np.random.rand(n_users, self.factors) / (self.factors ** 0.5)
        self.item_vecs = np.random.rand(n_items, self.factors) / (self.factors ** 0.5)

        self.global_bias = np.mean([r for _, _, r in X])
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        for iter in range(self.n_iters):
            lr = self.lr
            _X = X[np.random.permutation(X.shape[0])]
            for u, i, r in _X:
                u = int(u)
                i = int(i)
                e = r - self._score(u, i)  # Compute error residuals

                self.user_bias[u] += lr * (e - self.bias_reg_param * self.user_bias[u])
                self.item_bias[i] += lr * (e - self.bias_reg_param * self.item_bias[i])
                uv = self.user_vecs[u, :]
                iv = self.item_vecs[i, :]
                self.user_vecs[u, :] += lr * (e * iv - self.reg * uv)
                self.item_vecs[i, :] += lr * (e * uv - self.reg * iv)

            train_rmse = self.validate(X, True)
            validate_rmse = self.validate(validate_data, False)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def _score(self, u, i):
        return self.global_bias + self.user_bias[u] + self.item_bias[i] + np.dot(self.user_vecs[u, :],
                                                                                 self.item_vecs[i, :].T)

    def _find_knn(self, u, i, k=3):
        d_u_i = {}
        for item in self.u_i_dict[u].keys():
            distance = (self.item_attribute[i][0] - self.item_attribute[item][0]) ** 2 + (
                        self.item_attribute[i][1] - self.item_attribute[item][1]) ** 2
            d_u_i[item] = distance

        sorted_items = sorted(d_u_i.items(), key=lambda x: x[1])
        k_nearest_items = [item for item, distance in sorted_items[:k]]

        s = 0
        for item in k_nearest_items:
            s += self.u_i_dict[u][item]

        return s / k

    def validate(self, validate_data, train=False):
        sse_sum = 0
        count = 0
        if train:
            for u, i, r in validate_data:
                u = int(u)
                i = int(i)
                pred = self._score(u, i)
                sse_sum += (pred - r) ** 2
                count += 1
        else:
            for u, i, r in validate_data:
                u = int(u)
                i = int(i)
                if i in range(self.train_n_item):
                    pred = self._score(u, i) * self.scale
                else:
                    pred = self._find_knn(u, i) * self.scale
                sse_sum += (pred - r) ** 2
                count += 1
        return np.sqrt(sse_sum / count)
