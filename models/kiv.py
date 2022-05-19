# Implementation of Kernel IV (Singh 2019 NeurIPS paper)
import numpy as np
import numpy.linalg
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split


def train_kiv(data, config):
    kiv = KIV(config)
    kiv.train(data)
    return kiv


class KIV():
    def __init__(self, config):
        super().__init__()
        self.lamb = config["lambda"]
        self.xi = config["xi"]
        self.alpha = None
        self.X = None

    @staticmethod
    def exp_kernel(a, b):
        n = a.shape[0]
        dim_a = a.shape[1]
        assert dim_a == b.shape[1]
        # Set length scales to median inter-point distances
        kernel = np.ones((n, b.shape[0]))
        for i in range(dim_a):
            pairwise_dist = euclidean_distances(a[:, i:i + 1], b[:, i:i + 1])
            len_scale = np.median(pairwise_dist)
            # Set length scale = 1/2 for binary input (instruments)
            if len_scale == 0:
                len_scale = 1 / 2
            kernel_i = RBF(length_scale=len_scale)
            kernel *= kernel_i(a[:, i:i + 1], b[:, i:i + 1])
        return kernel

    def train(self, data):
        d_first, d_second = train_test_split(data, test_size=0.5, shuffle=False)
        n = d_first.shape[0]
        m = d_second.shape[0]

        #Add covariates to both treatment and instrument
        z_1 = d_first[:, 2:]
        z_2 = d_second[:, 2:]
        a_1 = np.concatenate((d_first[:, 1:2], d_first[:, 3:]), 1)
        y_2 = d_second[:, 0:1]

        # First stage
        kxx = KIV.exp_kernel(a_1, a_1)
        kzz = KIV.exp_kernel(z_1, z_1)
        bracket1 = kzz + n * self.lamb * np.identity(n)
        inv1 = numpy.linalg.inv(bracket1)
        kzz_bar = KIV.exp_kernel(z_1, z_2)
        W = numpy.matmul(numpy.matmul(kxx, inv1), kzz_bar)

        # Second stage
        inv2 = numpy.linalg.inv(numpy.matmul(W, numpy.transpose(W)) + m * self.xi * kxx)
        self.alpha = numpy.transpose(numpy.matmul(numpy.matmul(inv2, W), y_2))
        self.X = a_1

    def predict_cf(self, x_np, nr):
        n = x_np.shape[0]
        if isinstance(nr, int):
            ax = np.concatenate((np.full((n, 1), nr), x_np), 1)
        else:
            ax = np.concatenate((np.expand_dims(nr, 1), x_np), 1)
        k_xx = KIV.exp_kernel(self.X, ax)
        return np.squeeze(numpy.matmul(self.alpha, k_xx))

    def predict_ite(self, x_np):
        y_hat0 = self.predict_cf(x_np, 0)
        y_hat1 = self.predict_cf(x_np, 1)
        tau_hat = y_hat1 - y_hat0
        return tau_hat
