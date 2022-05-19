import pandas as pd

import data.sim_gp as sim_gp
import misc
import yaml
import models.mr_learner as mr
import models.helper as helper
import numpy as np
import matplotlib.pyplot as plt
import models.dml_dr_iv as dml
import models.standard_ite as standard_ite
import random
import torch
import joblib



def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(config, return_results=False):
    #Initial seed for reproducibility
    set_seeds(config["seed"])
    #Generate dataset
    [d_train, [pi, sd_info], scaler] = misc.load_data(config)

    models = []
    meta_learners = []

    #Test data for case study
    p = d_train.shape[1] - 3
    #Age range
    age_range = np.array(range(20, 70))
    age_range_scaled = (age_range - sd_info[0][0]) / sd_info[0][1]
    n_test = age_range.shape[0]
    #other covariates
    num_sign_up = (np.full(n_test, 1) - sd_info[1][0]) / sd_info[1][1]
    num_visits = (np.full(n_test, 1) - sd_info[2][0]) / sd_info[2][1]
    gender = np.ones(n_test)
    english = np.ones(n_test)

    x_test = np.zeros((n_test, p))
    x_test[:, 0] = age_range_scaled
    x_test[:, 1] = num_sign_up
    x_test[:, 2] = num_visits
    x_test[:, 3] = gender
    x_test[:, 4] = english

    #Dataframes to store results
    model_names = []
    for model_config in config["models"]:
        model_names.append(model_config["name"])
    df_results_models = pd.DataFrame(columns=model_names, index=range(n_test))

    meta_names = []
    for model_config in config["models"]:
        if "meta_learners" in model_config:
            if model_config["meta_learners"] is not None:
                for meta_learner in model_config["meta_learners"]:
                    meta_names.append(meta_learner + "_" + model_config["name"])
    df_results_meta = pd.DataFrame(columns=meta_names, index=range(n_test))

    #Estimate nuisance parameters for meta learners
    nuisance_mriv = None
    nuisance_driv = None
    nuisance_need_mr, nuisance_need_driv = misc.check_nuisance_need(config["models"])
    if nuisance_need_mr:
        set_seeds(config["seed"])
        print("Training | Nuisance parameter MRIV")
        params_nuisance_mr = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name="nuisance",
                                      meta_learner_name="mriv", type="meta_learner")
        nuisance_mriv = mr.get_nuisance_full(d_train, params_nuisance_mr)
    if nuisance_need_driv:
        set_seeds(config["seed"])
        print("Training | Nuisance parameter DRIV")
        params_nuisance_driv = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name="nuisance",
                                      meta_learner_name="driv", type="meta_learner")
        nuisance_driv = dml.get_nuisance_full(d_train, params_nuisance_driv)
    print("Model training ------------------------")

    for model_config in config["models"]:
        set_seeds(config["seed"])
        # Train base model
        params = None
        if model_config["name"] not in ["tsls", "waldlinear"]:
            params = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name=model_config["name"])
        trained_model = train_model(model_config, d_train, params)
        models.append(trained_model)
        # Train meta learners on top of base model
        if "meta_learners" in model_config:
            if model_config["meta_learners"] is not None:
                for meta_learner in model_config["meta_learners"]:
                    set_seeds(config["seed"])
                    params = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name=model_config["name"],
                                                  meta_learner_name=meta_learner, type="meta_learner")
                    if meta_learner in ["mriv", "mrivsingle"]:
                        meta_learners.append(train_meta_learner(meta_learner, trained_model, d_train, params,
                                                                nuisance_mriv, pi))
                    elif meta_learner in ["driv", "dr"]:
                        meta_learners.append(train_meta_learner(meta_learner, trained_model, d_train, params,
                                                                nuisance_driv, pi))
                    else:
                        raise ValueError('Meta learner not recognized')


    print("Predict ITEs for models ------------------------")
    # Computing test mse for each trained model
    for model in models:
        name = model["name"]
        # Test prediction
        tau_hat = model["trained_model"].predict_ite(x_test)
        df_results_models.loc[:, name] = tau_hat

    print("Predict ITEs for meta learners ------------------------")
    # Computing test mse for each traind meta learner
    for meta_learner in meta_learners:
        meta_name = meta_learner["meta_learner_name"]
        base_name = meta_learner["base_model_name"]
        # Test prediction
        tau_hat = meta_learner["trained_meta_learner"].predict(x_test)
        df_results_meta.loc[:, meta_name + "_" + base_name] = tau_hat

    print("Results--------------------------------------")
    print("Results Base models--------------------------------------------------------")
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #print(df_results_models)

    print("Results Meta learners-------------------------------------------------------")
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #print(df_results_meta)
    # Save results to file
    path_results = misc.get_project_path() + "/results/real/"
    joblib.dump(df_results_models, path_results + "results_models1.pkl")
    joblib.dump(df_results_meta, path_results + "results_meta1.pkl")




def train_model(model_config, d_train, params=None):
    model_name = model_config["name"]
    print(f"Training | Base model: {model_name} | Meta learner: -")
    model = helper.train_base_model(model_name, d_train, params=params)
    return {"name": model_name, "trained_model": model}


def train_meta_learner(meta_learner_name, trained_model, d_train, params, nuisance, pi):
    base_model = trained_model["trained_model"]
    base_model_name = trained_model["name"]
    x_train = d_train[:, 3:]

    meta_learner = None
    print(f"Training | Base model: {base_model_name} | Meta learner: {meta_learner_name}")
    # MR Learner
    if meta_learner_name in ["mriv", "mrivsingle"]:

        if base_model_name == "ncnet":
            if meta_learner_name == "mriv":
                mr_input = list(base_model.predict_mr_input(x_train))
                mr_input[0] = pi
            elif meta_learner_name == "mrivsingle":
                mr_input = list(base_model.predict_mr_input_single(x_train))
                mr_input[0] = pi
        else:
            if meta_learner_name == "mrivsingle":
                raise ValueError('MR single only works with ncnet')
            tau = base_model.predict_ite(x_train)

            [mu_0Y, mu_0A, delta_A, _] = nuisance
            mr_input = [pi, mu_0A, mu_0Y, delta_A, tau]
        # Train MR Learner
        print(f"Train MR learner")
        meta_learner = mr.train_mr_learner(data=d_train, init_estimates=mr_input, config=params,
                                           validation=False, logging=False)

    # DRIV
    if meta_learner_name == "driv":
        if base_model_name == "dmliv":
            [tau, q, p] = base_model.predict_driv_input(x_train)
            [_, _, _, f] = nuisance
            driv_input = [tau, q, p, pi, f]
        else:
            tau = base_model.predict_ite(x_train)
            [q, p, _, f] = nuisance
            driv_input = [tau, q, p, pi, f]
        # Train DRIV
        print(f"Train DRIV learner")
        meta_learner = dml.train_driv(data=d_train, init_estimates=driv_input, config=params,
                                      validation=False, logging=False)

    # DR Learner
    if meta_learner_name == "dr":
        if base_model_name == "tarnet":
            mu_1 = base_model.predict_cf(x_train, 1)
            mu_0 = base_model.predict_cf(x_train, 0)
            pi = base_model.predict_pi(x_train)
            print(f"Train DRIV learner")
            meta_learner = standard_ite.train_dr_learner(data=np.delete(d_train, 2, 1), config=params, init_estimates=[pi, mu_1, mu_0],
                                          validation=False, logging=False)
        else:
            raise ValueError('DR Learner only works with TARNet')
    return {"meta_learner_name": meta_learner_name, "base_model_name": base_model_name,
            "trained_meta_learner": meta_learner}


if __name__ == "__main__":
    # Select configuration file here
    stream = open(misc.get_project_path() + "/experiments/real/exp_real_config.yaml", 'r')
    run_config = yaml.safe_load(stream)
    run_experiment(run_config)

