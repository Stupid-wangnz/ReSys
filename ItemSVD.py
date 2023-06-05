import numpy as np
from collections import defaultdict

from SVD import BiasSVD


class ItemSVD(BiasSVD):
    def __init__(self, learning_rate=0.003, reg_param=0.02, bias_reg_param=0.02, n_iters=40, factors=200, scale=1,
                 item_attribute=None):
        super().__init__(learning_rate, reg_param, bias_reg_param, n_iters, factors, scale)
        self.item_attribute = item_attribute

    def fit(self, X, validate_data, n_users, n_items):
        # Initialize user and item vectors, user_bias and item_bias
        n_attributes = self.item_attribute.shape[1]
        self.user_vecs = np.random.rand(n_users, self.factors + n_attributes) / (self.factors ** 0.5)
        self.item_vecs = np.random.rand(n_items, self.factors + n_attributes) / (self.factors ** 0.5)
        self.item_vecs[:, self.factors:] = self.item_attribute
        users_score = defaultdict(list)
        items_score = defaultdict(list)

        for u, items in X.items():
            for item_id, item_score in items.items():
                users_score[u].append(item_score)
                items_score[item_id].append(item_score)

        self.global_bias = np.mean(list(score for user in X for item_id, score in X[user].items()))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        for iter in range(self.n_iters):
            lr = self.lr
            for u, items in X.items():
                for i in items.keys():
                    e = items[i] - self._score(u, i)  # Compute error residuals

                    self.user_bias[u] += lr * (e - self.bias_reg_param * self.user_bias[u])
                    self.item_bias[i] += lr * (e - self.bias_reg_param * self.item_bias[i])
                    uv = self.user_vecs[u, :]
                    iv = self.item_vecs[i, :]
                    self.user_vecs[u, :] += lr * (e * iv - self.reg * uv)
                    self.item_vecs[i, :] += lr * (e * uv - self.reg * iv)
                    self.item_vecs[i, self.factors:] -= lr * self.reg * (
                             self.item_attribute[i, :] - self.item_vecs[i, self.factors:])

            train_rmse = self.validate(X, True)
            validate_rmse = self.validate(validate_data, False)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def _score(self, u, i):
        return self.global_bias + self.user_bias[u] + self.item_bias[i] + np.dot(self.user_vecs[u, :],
                                                                                 self.item_vecs[i, :].T)
