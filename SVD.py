import numpy as np
from collections import defaultdict

np.seterr(all='raise')


class FunkSVD:
    def __init__(self, learning_rate=0.005, reg_param=0.01, n_iters=120, factors=250, scale=1):
        self.item_vecs = None
        self.user_vecs = None
        self.lr = learning_rate  # learning rate for gradient descent
        self.reg = reg_param  # regularization parameter for L2 regularization
        self.n_iters = n_iters
        self.factors = factors
        self.scale = scale

    def fit(self, X, validate_data, n_users, n_items):
        # Initialize user and item vectors
        self.user_vecs = np.random.rand(n_users, self.factors) / (self.factors ** 0.5)
        self.item_vecs = np.random.rand(n_items, self.factors) / (self.factors ** 0.5)

        for epoch in range(self.n_iters):
            _X = X[np.random.permutation(X.shape[0])]
            for u, i, r in _X:
                u = int(u)
                i = int(i)
                e = r - np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)  # Compute error residuals
                uv = self.user_vecs[u, :]
                iv = self.item_vecs[i, :]
                self.user_vecs[u, :] += self.lr * (e * iv - self.reg * uv)
                self.item_vecs[i, :] += self.lr * (e * uv - self.reg * iv)

            train_rmse = self.validate(X, True)
            validate_rmse = self.validate(validate_data, False)
            print(f"train iter: {epoch}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def warmup(self, X, warm_up_iters):
        warm_up_lr = 0
        per_iter = len(X)
        for warm_up_iter in range(warm_up_iters):
            for u, i, r in X:
                e = r - self._score(u, i)
                self.user_vecs[u, :] += warm_up_lr * (e * self.item_vecs[i, :] - self.reg * self.user_vecs[u, :])
                self.item_vecs[i, :] += warm_up_lr * (e * self.user_vecs[u, :] - self.reg * self.item_vecs[i, :])
                warm_up_lr += self.lr / (per_iter * warm_up_iters)

    def _score(self, u, i):
        return np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)

    def validate(self, validate_data, train=False):
        sse_sum = 0
        count = 0
        for u, i, r in validate_data:
            u = int(u)
            i = int(i)
            if train:
                pred = self._score(u, i)
            else:
                pred = self._score(u, i) * self.scale
            sse_sum += (pred - r) ** 2
            count += 1
        return np.sqrt(sse_sum / count)

    def predict(self, test_data):
        res = test_data.copy()
        for u, items in test_data.items():
            for i in items.keys():
                pred = self._score(u, i) * self.scale
                pred = max(0, min(100, pred))
                res[u][i] = pred
        return res


class BiasSVD(FunkSVD):
    def __init__(self, learning_rate=0.005, reg_param=0.01, bias_reg_param=0.01, n_iters=120, factors=250, scale=1):
        super().__init__(learning_rate, reg_param, n_iters, factors, scale)
        self.global_bias = 0
        self.user_bias = None
        self.item_bias = None
        self.bias_reg_param = bias_reg_param

    def fit(self, X, validate_data, n_users, n_items):
        # Initialize user and item vectors, user_bias and item_bias
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


class SVDPlusPlus(BiasSVD):
    def __init__(self, learning_rate=0.005, reg_param=0.01, bias_reg_param=0.01, n_iters=120, factors=250, scale=1):
        super().__init__(learning_rate, reg_param, bias_reg_param, n_iters, factors, scale)
        self.y = None
        self.X = None

    def fit(self, X, validate_data, n_users, n_items):
        self.X = X
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
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.y = np.zeros([n_items, self.factors])

        for iter in range(self.n_iters):
            lr = self.lr
            for u, items in X.items():
                implict_feedback = self._implicit_feedback(u)
                for i in items.keys():
                    e = items[i] - self._score(u, i, implict_feedback)

                    self.user_bias[u] += self.lr * (e - self.bias_reg_param * self.user_bias[u])
                    self.item_bias[i] += self.lr * (e - self.bias_reg_param * self.item_bias[i])
                    self.user_vecs[u, :] += self.lr * (e * self.item_vecs[i, :] - self.reg * self.user_vecs[u, :])
                    self.item_vecs[i, :] += self.lr * (
                            e * (self.user_vecs[u, :] + implict_feedback) - self.reg * self.item_vecs[i, :])
                    # Update implicit feedback vectors
                    self.y[i, :] += self.lr * (
                            e * (1 / np.sqrt(len(self.X[u]))) * self.item_vecs[i, :] - self.reg * self.y[i, :])

            train_rmse = self.validate(X, True)
            validate_rmse = self.validate(validate_data, False)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def _implicit_feedback(self, u):
        implicit_feedback = np.zeros(self.factors)
        for i in self.X[u].keys():
            implicit_feedback += self.y[i, :]
        return implicit_feedback / np.sqrt(len(self.X[u]))

    def _score(self, u, i, implict_feedback):
        return self.global_bias + self.user_bias[u] + self.item_bias[i] + np.dot(
            self.user_vecs[u, :] + implict_feedback,
            self.item_vecs[i, :].T)

    def validate(self, validate_data, train=False):
        sse_sum = 0
        count = 0
        for u, items in validate_data.items():
            implict_feedback = self._implicit_feedback(u)
            for i in items.keys():
                if train:
                    pred = self._score(u, i, implict_feedback)
                else:
                    pred = self._score(u, i, implict_feedback) * self.scale
                sse_sum += ((pred - items[i])) ** 2
                count += 1
        return np.sqrt(sse_sum / count)
