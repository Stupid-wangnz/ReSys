import numpy as np
from collections import defaultdict

np.seterr(all='raise')


class FunkSVD:
    def __init__(self, learning_rate=0.0005, reg_param=0.02, n_iters=20, factors=200):
        self.item_vecs = None
        self.user_vecs = None
        self.lr = learning_rate  # learning rate for gradient descent
        self.reg = reg_param  # regularization parameter for L2 regularization
        self.n_iters = n_iters
        self.factors = factors

    def fit(self, X, validate_data, n_users, n_items):
        # Initialize user and item vectors
        self.user_vecs = np.random.rand(n_users, self.factors) / (self.factors ** 0.5)
        self.item_vecs = np.random.rand(n_items, self.factors) / (self.factors ** 0.5)

        for _ in range(self.n_iters):
            for u, items in X.items():
                for i in items.keys():
                    e = items[i] - np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)  # Compute error residuals
                    self.user_vecs[u, :] += self.lr * (e * self.item_vecs[i, :] - self.reg * self.user_vecs[u, :])
                    self.item_vecs[i, :] += self.lr * (e * self.user_vecs[u, :] - self.reg * self.item_vecs[i, :])
            rmse = self.validate(validate_data)
            print(f"train iter: {_}, rmse: {rmse}")

    def _score(self, u, i):
        return np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)

    def validate(self, validate_data):
        sse_sum = 0
        count = 0
        for u, items in validate_data.items():
            for i in items.keys():
                pred = np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)
                sse_sum += ((pred - items[i])) ** 2
                count += 1
        return np.sqrt(sse_sum / count)

    def predict(self, test_data):
        res = test_data.copy()
        for u, items in test_data.items():
            for i in items.keys():
                pred = np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)
                res[u][i] = pred
        return res


class BiasSVD(FunkSVD):
    def __init__(self, learning_rate=0.002, reg_param=1e-4, bias_reg_param=1e-4, n_iters=20, factors=250):
        super().__init__(learning_rate, reg_param, n_iters, factors)
        self.global_bias = 0
        self.user_bias = None
        self.item_bias = None
        self.bias_reg_param = bias_reg_param

    def fit(self, X, validate_data, n_users, n_items):
        # Initialize user and item vectors, user_bias and item_bias
        self.user_vecs = np.random.rand(n_users, self.factors) / (self.factors ** 0.5)
        self.item_vecs = np.random.rand(n_items, self.factors) / (self.factors ** 0.5)
        users_score = defaultdict(list)
        items_score = defaultdict(list)

        for u, items in X.items():
            for item_id, item_score in items.items():
                users_score[u].append(item_score)
                items_score[item_id].append(item_score)
        self.global_bias = np.mean(list(score for user in X for item_id, score in X[user].items()))
        self.user_bias = np.full(n_users, self.global_bias)
        self.item_bias = np.full(n_items, self.global_bias)
        for u, scores in users_score.items():
            self.user_bias[u] = np.mean(scores)
        for i, scores in items_score.items():
            self.item_bias[i] = np.mean(scores)
        for iter in range(self.n_iters):
            for u, items in X.items():
                for i in items.keys():
                    e = items[i] - self._score(u, i)  # Compute error residuals

                    self.user_bias[u] += self.lr * (e - self.bias_reg_param * self.user_bias[u])
                    self.item_bias[i] += self.lr * (e - self.bias_reg_param * self.item_bias[i])

                    self.user_vecs[u, :] += self.lr * (e * self.item_vecs[i, :] - self.reg * self.user_vecs[u, :])
                    self.item_vecs[i, :] += self.lr * (e * self.user_vecs[u, :] - self.reg * self.item_vecs[i, :])
            train_rmse = self.validate((X))
            validate_rmse = self.validate(validate_data)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def _score(self, u, i):
        return self.global_bias + self.user_bias[u] + self.item_bias[i] + np.dot(self.user_vecs[u, :],
                                                                                 self.item_vecs[i, :].T)
