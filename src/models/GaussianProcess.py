import numpy as np

class GaussianProcess:
    def __init__(self, kernel, noise=1e-5):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        K_ss = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K_ss[i, j] = self.kernel.compute(X[i], X[j])

        K_s = np.zeros((X.shape[0], self.X_train.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.X_train.shape[0]):
                K_s[i, j] = self.kernel.compute(X[i], self.X_train[j])

        K = np.zeros((self.X_train.shape[0], self.X_train.shape[0]))
        for i in range(self.X_train.shape[0]):
            for j in range(self.X_train.shape[0]):
                K[i, j] = self.kernel.compute(self.X_train[i], self.X_train[j])

        K_inv = np.linalg.inv(K + self.noise * np.identity(K.shape[0]))

        mu_s = K_s.dot(K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.dot(K_inv).dot(K_s.T)

        return mu_s, cov_s