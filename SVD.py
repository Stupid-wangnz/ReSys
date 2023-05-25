import numpy as np
from collections import defaultdict

np.seterr(all='raise')


class FunkSVD:
    def __init__(self, learning_rate=0.0005, reg_param=0.01, n_iters=20, factors=150):
        self.item_vecs = None
        self.user_vecs = None
        self.lr = learning_rate  # learning rate for gradient descent
        self.reg = reg_param  # regularization parameter for L2 regularization
        self.n_iters = n_iters
        self.factors = factors

    def fit(self, X: defaultdict[dict], validate_data: defaultdict[dict], n_users, n_items):
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

    def validate(self, validate_data: defaultdict[dict]):
        sse_sum = 0
        count = 0
        for u, items in validate_data.items():
            for i in items.keys():
                pred = np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)
                sse_sum += (pred - items[i]) ** 2
                count += 1
        return np.sqrt(sse_sum / count)

    def predict(self, test_data: defaultdict[dict]):
        res = test_data.copy()
        for u, items in test_data.items():
            for i in items.keys():
                pred = np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)
                res[u][i] = pred
        return res