import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import models.helper as helper


# Two stage least squares
def train_twosls(data):
    tsls = twosls()
    tsls.train(data)
    return tsls


class twosls():
    def __init__(self):
        super().__init__()
        self.lmA = None

    def train(self, data):
        Y, A, Z, X = helper.split_data(data)

        # First stage regression and fitted values
        covZ = np.concatenate((np.expand_dims(Z, 1), X), 1)
        lmA = LinearRegression().fit(covZ, A)
        fittedA = lmA.predict(covZ)
        # Second stage regression
        covA = np.concatenate((np.expand_dims(fittedA, 1), X), 1)
        lmA = LinearRegression().fit(covA, Y)
        self.lmA = lmA

    def predict_cf(self, x_np, nr):
        n = x_np.shape[0]
        a_x = np.concatenate((np.full((n, 1), nr), x_np), 1)
        return self.lmA.predict(a_x)

    def predict_ite(self, x_np):
        y_hat1 = self.predict_cf(x_np, 1)
        y_hat0 = self.predict_cf(x_np, 0)
        tau = y_hat1 - y_hat0
        return tau


def train_Wald_linear(data):
    wald_lin = Wald_linear()
    wald_lin.train(data)
    return wald_lin


class Wald_linear():
    def __init__(self):
        super().__init__()
        self.lmY1 = LinearRegression()
        self.lmA1 = LogisticRegression()
        self.lmY0 = LinearRegression()
        self.lmA0 = LogisticRegression()

    def train(self, data):
        data_1 = data[data[:, 2] == 1, :]
        data_0 = data[data[:, 2] == 0, :]

        self.lmA1.fit(data_1[:, 3:], data_1[:, 1])
        self.lmA0.fit(data_0[:, 3:], data_0[:, 1])
        self.lmY1.fit(data_1[:, 3:], data_1[:, 0])
        self.lmY0.fit(data_0[:, 3:], data_0[:, 0])

    def predict_ite(self, x_np):
        y1 = self.lmY1.predict(x_np)
        y0 = self.lmY0.predict(x_np)
        a1 = self.lmA1.predict_proba(x_np)[:, 1]
        a0 = self.lmA0.predict_proba(x_np)[:, 1]
        tau = (y1 -y0) / (a1 - a0)
        return tau
