import numpy as np
from collections import defaultdict

np.seterr(all='raise')


class FunkSVD:
    def __init__(self, learning_rate=0.0005, reg_param=0.02, n_iters=50, factors=20):
        self.item_vecs = None
        self.user_vecs = None
        self.lr = learning_rate  # learning rate for gradient descent
        self.reg = reg_param  # regularization parameter for L2 regularization
        self.n_iters = n_iters
        self.factors = factors

        self.mean = None
        self.std = None

    def init_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

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

            train_rmse = self.validate((X))
            validate_rmse = self.validate(validate_data)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def warmup(self, X, warm_up_iters, per_iter):
        warm_up_lr = 0
        for warm_up_iter in range(warm_up_iters):
            for u, items in X.items():
                for i in items.keys():
                    e = items[i] - self._score(u, i)
                    self.user_vecs[u, :] += warm_up_lr * (e * self.item_vecs[i, :] - self.reg * self.user_vecs[u, :])
                    self.item_vecs[i, :] += warm_up_lr * (e * self.user_vecs[u, :] - self.reg * self.item_vecs[i, :])
                warm_up_lr += self.lr / (per_iter * warm_up_iters)

    def _score(self, u, i):
        return np.dot(self.user_vecs[u, :], self.item_vecs[i, :].T)

    def validate(self, validate_data):
        sse_sum = 0
        count = 0
        for u, items in validate_data.items():
            for i in items.keys():
                pred = self._score(u, i)
                if self.std is not None:
                    sse_sum += ((pred - items[i])*self.std) ** 2
                else:
                    sse_sum += ((pred - items[i])) ** 2
                count += 1
        return np.sqrt(sse_sum / count)

    def predict(self, test_data):
        res = test_data.copy()
        for u, items in test_data.items():
            for i in items.keys():
                if self.mean is not None:
                    pred = self._score(u, i) * self.std
                else:
                    pred = self._score(u, i)
                res[u][i] = pred
        return res


class BiasSVD(FunkSVD):
    def __init__(self, learning_rate=5e-4, reg_param=0.02, bias_reg_param=0.02, n_iters=50, factors=100):
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
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)


        for iter in range(self.n_iters):
            lr = self.lr
            #lr = self.lr / (2.0 ** iter)
            for u, items in X.items():
                for i in items.keys():
                    e = items[i] - self._score(u, i)  # Compute error residuals

                    self.user_bias[u] += lr * (e - self.bias_reg_param * self.user_bias[u])
                    self.item_bias[i] += lr * (e - self.bias_reg_param * self.item_bias[i])

                    self.user_vecs[u, :] += lr * (e * self.item_vecs[i, :] - self.reg * self.user_vecs[u, :])
                    self.item_vecs[i, :] += lr * (e * self.user_vecs[u, :] - self.reg * self.item_vecs[i, :])

            train_rmse = self.validate((X))
            validate_rmse = self.validate(validate_data)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def _score(self, u, i):
        return self.global_bias + self.user_bias[u] + self.item_bias[i] + np.dot(self.user_vecs[u, :],
                                                                                            self.item_vecs[i, :].T)

    def warmup(self, X, warm_up_iters, per_iter):
        warm_up_lr = 0
        for warm_up_iter in range(warm_up_iters):
            for u, items in X.items():
                for i in items.keys():
                    e = items[i] - self._score(u, i)  # Compute error residuals

                    self.user_bias[u] += warm_up_lr * (e - self.bias_reg_param * self.user_bias[u])
                    self.item_bias[i] += warm_up_lr * (e - self.bias_reg_param * self.item_bias[i])

                    self.user_vecs[u, :] += warm_up_lr * (e * self.item_vecs[i, :] - self.reg * self.user_vecs[u, :])
                    self.item_vecs[i, :] += warm_up_lr * (e * self.user_vecs[u, :] - self.reg * self.item_vecs[i, :])

                    warm_up_lr += self.lr / (per_iter * warm_up_iters)
        print(f"warm up lr: {warm_up_lr}")


class SVDPlusPlus(BiasSVD):
    def __init__(self, learning_rate=0.0005, reg_param=0.01, bias_reg_param=0.005, n_iters=20, factors=150):
        super().__init__(learning_rate, reg_param, bias_reg_param, n_iters, factors)
        self.y = None
        self.X = None

    def fit(self, X, validate_data, n_users, n_items):
        self.X=X
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
        # self.y = np.random.rand(n_items, self.factors) / (self.factors ** 0.5)
        self.y = np.zeros([n_items, self.factors])

        for iter in range(self.n_iters):
            lr = self.lr
            #lr = self.lr / (2.0 ** iter)
            for u, items in X.items():
                implict_feedback = self._implicit_feedback(u)
                num = len(items)
                for i in items.keys():
                    e = items[i] - self._score(u, i, num, implict_feedback)

                    self.user_bias[u] += self.lr * (e - self.bias_reg_param * self.user_bias[u])
                    self.item_bias[i] += self.lr * (e - self.bias_reg_param * self.item_bias[i])
                    self.user_vecs[u, :] += self.lr * (e * self.item_vecs[i, :] - self.reg * self.user_vecs[u, :])
                    self.item_vecs[i, :] += self.lr * (e * (self.user_vecs[u, :] + (1/np.sqrt(num)) * implict_feedback) - self.reg *self.item_vecs[i, :])
                    # Update implicit feedback vectors
                    self.y[i, :] += self.lr * (e * (1/np.sqrt(num)) * self.item_vecs[i, :] - self.reg * self.y[i, :])

            train_rmse = self.validate((X))
            validate_rmse = self.validate(validate_data)
            print(f"train iter: {iter}, train rmse:{train_rmse}, validate rmse: {validate_rmse}")

    def _implicit_feedback(self, u):
        implicit_feedback = np.zeros(self.factors)
        for i in self.X[u].keys():
            implicit_feedback += self.y[i, :]
        return implicit_feedback

    def _score(self, u, i, num, implict_feedback):
        return self.global_bias + self.user_bias[u] + self.item_bias[i] + np.dot(self.user_vecs[u, :] + ( 1/np.sqrt(num) * implict_feedback),
                                                                                            self.item_vecs[i, :].T)

    def validate(self, validate_data):
        sse_sum = 0
        count = 0
        for u, items in validate_data.items():
            implict_feedback = self._implicit_feedback(u)
            num = len(items)
            for i in items.keys():
                pred = self._score(u, i, num, implict_feedback)
                if self.std is not None:
                    sse_sum += ((pred - items[i])*self.std) ** 2
                else:
                    sse_sum += ((pred - items[i])) ** 2
                count += 1
        return np.sqrt(sse_sum / count)
            