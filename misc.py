from pathlib import Path
import os
import yaml
import torch
import numpy as np
import data.sim_gp as sim_gp
import data.sim_semi as sim_semi
from data.load_real import load_oregon


# Some useful functions

def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.absolute())


def load_hyper_yaml(path, base_model_name, meta_learner_name="", type="base_model"):
    if type == "base_model":
        return yaml.safe_load(open(get_project_path() + "/hyperparam" + path + "params/base_models/params_" + base_model_name
                                   + ".yaml", 'r'))
    elif type == "meta_learner":
        return yaml.safe_load(open(get_project_path() + "/hyperparam" + path + "params/meta_learners/params_" + meta_learner_name
                                   + "_" + base_model_name + ".yaml", 'r'))
    else:
        raise ValueError('Invalid type for hyperparameter loading')

def get_device():
    if torch.cuda.is_available():
        gpu = 1
    else:
        gpu = 0
    return gpu


def binary_cross_entropy(yhat, y):
    return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()


#Output: d_train, d_test, groundtruth train, groundtruth test
def load_data(config, scale=True):
    print("Create dataset ------------------------")
    # Check whether simulated or real-world data is used
    if config["dataset"] == "sim":
        n = config["n"]
        p = config["p"]
        sigma_U = config["sigma_U"]
        alpha_U = config["alpha_U"]
        sigma_A = config["sigma_A"]
        sigma_Y = config["sigma_Y"]
        beta = config["beta"]
        gamma = config["gamma"]
        delta = config["delta"]
        # Generate data
        data, comp, scaler = sim_gp.simulate_data(n, p, sigma_U=sigma_U, alpha_U=alpha_U, sigma_Y=sigma_Y, sigma_A=sigma_A,
                                          plot=config["plotting"], scale=True, beta=beta, gamma=gamma, delta=delta)
        d_train, d_test, c_train, c_test = sim_gp.train_test_split(data, comp)
        return d_train, d_test, c_train, c_test, scaler
    elif config["dataset"] == "real":
        return load_oregon(scale=scale)
    elif config["dataset"] == "sim_semi":
        n = config["n"]
        sigma_U = config["sigma_U"]
        alpha_U = config["alpha_U"]
        sigma_A = config["sigma_A"]
        sigma_Y = config["sigma_Y"]
        # Generate data
        data, comp, scaler = sim_semi.simulate_data(n, sigma_U=sigma_U, alpha_U=alpha_U, sigma_Y=sigma_Y, sigma_A=sigma_A,
                                          plot=config["plotting"], scale=True)
        d_train, d_test, c_train, c_test = sim_gp.train_test_split(data, comp)
        return d_train, d_test, c_train, c_test, scaler



def check_nuisance_need(model_configs):
    nuisance_mr = False
    nuisance_driv = False
    for model_config in model_configs:
        if "meta_learners" in model_config:
            if model_config["meta_learners"] is not None:
                if model_config["name"] != "ncnet" and "mriv" in model_config["meta_learners"]:
                    nuisance_mr = True
                if "driv" in model_config["meta_learners"]:
                    nuisance_driv = True
    return nuisance_mr, nuisance_driv


def get_model_names(model_configs):
    model_names = []
    for model_config in model_configs:
        model_names.append(model_config["name"])
    return model_names


def get_meta_names(model_configs):
    meta_names = []
    for model_config in model_configs:
        base_model_name = model_config["name"]
        if "meta_learners" in model_config:
            if model_config["meta_learners"] is not None:
                for meta_learner in model_config["meta_learners"]:
                    meta_names.append(meta_learner + "_" + base_model_name)
    return meta_names

