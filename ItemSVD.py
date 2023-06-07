import numpy as np
from collections import defaultdict

from SVD import BiasSVD


class ItemSVD(BiasSVD):
    def __init__(self, learning_rate=0.005, reg_param=0.01, bias_reg_param=0.01, n_iters=120, factors=250, scale=1,
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
            self.lr *= 0.95
            train_rmse = self.validate(X, True)
            validate_rmse = self.validate(validate_data, False)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def _score(self, u, i):
        r = self.global_bias + self.user_bias[u] + self.item_bias[i] + np.dot(self.user_vecs[u, :],
                                                                              self.item_vecs[i, :].T)
        r = max(0, min(100 / self.scale, r))
        return r


class KnnSVD(ItemSVD):
    def __init__(self, learning_rate=0.005, reg_param=0.01, bias_reg_param=0.01, n_iters=200, factors=250, scale=1,
                 item_attribute=None, test_item_knn=None, train_n_item=0):
        super().__init__(learning_rate, reg_param, bias_reg_param, n_iters, factors, scale, item_attribute)
        self.test_item_knn = test_item_knn
        self.train_n_item = train_n_item

    def _knn_item_compute(self, u, i):
        r = 0
        for k_i in self.test_item_knn[i]:
            r += self._score(u, k_i)
        return r / len(self.test_item_knn[i])

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
                    pred = self._knn_item_compute(u, i) * self.scale
                sse_sum += (pred - r) ** 2
                count += 1
        return np.sqrt(sse_sum / count)

    def predict(self, test_data):
        res = test_data.copy()
        for u, items in test_data.items():
            for i in items.keys():
                if i in range(self.train_n_item):
                    pred = self._score(u, i) * self.scale
                else:
                    pred = self._knn_item_compute(u, i) * self.scale
                pred = max(0, min(100, pred))
                res[u][i] = pred
        return res
