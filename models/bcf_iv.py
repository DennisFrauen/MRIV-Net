#Bayesian causal forest IV (only using steps 1+2)
#Using causal forests and random forest classification for wald estimator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from bartpy.sklearnmodel import SklearnModel
#from xbcausalforest import XBCF



def train_bcf_iv(data, config):
    bcfiv = Bcf_iv(config)
    bcfiv.train(data)
    return bcfiv

class Bcf_iv():
    def __init__(self, config):
        super().__init__()
        self.pic_forest = RandomForestClassifier(n_estimators=config["n_trees_pic"], max_features=None, random_state=1234)
        self.itt_bcf1 = SklearnModel(n_trees=config["n_trees_itt"])
        self.itt_bcf0 = SklearnModel(n_trees=config["n_trees_itt"])

    def train(self, data):
        a = data[:, 1]
        z = data[:, 2]
        x = data[:, 3:]
        #PIC
        self.pic_forest.fit(X=np.concatenate((np.expand_dims(z, 1), x), 1), y=a)
        #ITT
        data1 = data[data[:, 2] == 1, :]
        data0 = data[data[:, 2] == 0, :]
        self.itt_bcf1.fit(data1[:, 3:], data1[:, 0])
        self.itt_bcf0.fit(data0[:, 3:], data0[:, 0])

    def predict_cf(self, x_np, nr):
        n = x_np.shape[0]
        mu1 = self.itt_bcf1.predict(X=x_np)
        mu0 = self.itt_bcf0.predict(X=x_np)
        mu = nr * mu1 + (1 - nr) * mu0
        delta1 = self.pic_forest.predict_proba(np.concatenate((np.ones((n, 1)), x_np), 1))
        delta0 = self.pic_forest.predict_proba(np.concatenate((np.zeros((n, 1)), x_np), 1))
        pic = delta1[:, 1] - delta0[:, 1]
        return mu / pic

    def predict_ite(self, x_np):
        n = x_np.shape[0]
        mu1 = self.itt_bcf1.predict(X=x_np)
        mu0 = self.itt_bcf0.predict(X=x_np)
        itt = mu1 - mu0
        delta1 = self.pic_forest.predict_proba(np.concatenate((np.ones((n, 1)), x_np), 1))
        delta0 = self.pic_forest.predict_proba(np.concatenate((np.zeros((n, 1)), x_np), 1))
        pic = delta1[:, 1] - delta0[:, 1]
        #Make sure no division by zero happens
        pic[np.absolute(pic) < 0.1] = 0.1
        tau = itt / pic
        #Stabilize ITE
        tau[np.absolute(tau - np.mean(tau)) > 1.5] = np.mean(tau)
        return tau
