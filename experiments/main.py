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
from datetime import datetime



def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(config, return_results=False):
    number_exp = config["number_experiments"]#
    if "cross_fitting" in config:
        cf = config["cross_fitting"]
    else:
        cf = False
    #Initial seed for reproducibility
    set_seeds(config["seed"])
    #Save results
    results_models = []
    results_meta = []
    for i in range(number_exp):
        print(f"Experiment Nr {i}")
        #Sample random seed
        seed = random.randint(0, 1000000)
        #Generate dataset
        np.random.seed(seed)
        [d_train, d_test, truth_train, truth_test, scaler] = misc.load_data(config)
        tau = truth_test[0]

        #Check for cross-fitting
        if cf:
            d_cf3 = misc.sample_split_cf(d_train, n_datasets=3)
            d_cf2 = misc.sample_split_cf(d_train, n_datasets=2)

        models = []
        meta_learners = []
        results_models_i = []
        results_meta_i = []

        #Estimate nuisance parameters for meta learners
        nuisance_mriv = None
        nuisance_driv = None
        nuisance_need_mr, nuisance_need_driv = misc.check_nuisance_need(config["models"])
        if nuisance_need_mr:
            print("Training | Nuisance parameter MRIV")
            params_nuisance_mr = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name="nuisance",
                                          meta_learner_name="mriv", type="meta_learner")
            #Check for cross-fitting
            if not cf:
                nuisance_mriv, _ = mr.get_nuisance_full(d_train, params_nuisance_mr, d_val=d_test)
            else:
                nuisance_mriv = mr.get_nuisance_full_cf(d_cf3, params_nuisance_mr)
        if nuisance_need_driv:
            print("Training | Nuisance parameter DRIV")
            params_nuisance_driv = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name="nuisance",
                                          meta_learner_name="driv", type="meta_learner")
            if not cf:
                nuisance_driv, _ = dml.get_nuisance_full(d_train, params_nuisance_driv, d_val=d_test)
            else:
                nuisance_driv = dml.get_nuisance_full_cf(d_cf2, params_nuisance_driv)
        print("Model training ------------------------")

        for model_config in config["models"]:
            set_seeds(seed)
            # Train base model
            params = None
            if model_config["name"] not in ["tsls", "waldlinear"]:
                params = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name=model_config["name"])
            #time1 = datetime.now()
            #Check cross-fitting
            if not cf:
                trained_model = train_model(model_config, d_train, params)
                models.append(trained_model)
            else:
                trained_model_mr = train_model_cf(model_config, d_cf3, params)
                trained_model_dr = train_model_cf(model_config, d_cf2, params)
                trained_model = [trained_model_mr, trained_model_dr]

            #time2 = datetime.now()
            #runtime = (time2 - time1).total_seconds()
            #print(f"Runtime: {runtime}")
            # Train meta learners on top of base model
            if "meta_learners" in model_config:
                if model_config["meta_learners"] is not None:
                    for meta_learner in model_config["meta_learners"]:
                        set_seeds(seed)
                        params = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name=model_config["name"],
                                                      meta_learner_name=meta_learner, type="meta_learner")
                        if meta_learner in ["mriv", "mrivsingle"]:
                            if not cf:
                                meta_learners.append(train_meta_learner(meta_learner, trained_model, d_train, params,
                                                                    nuisance_mriv))
                            else:
                                meta_learners.append(train_meta_learner_cf(meta_learner, trained_model, d_cf3, params,
                                                                    nuisance_mriv))

                        elif meta_learner in ["driv", "dr"]:
                            if not cf:
                                meta_learners.append(train_meta_learner(meta_learner, trained_model, d_train, params,
                                                                    nuisance_driv))
                            else:
                                meta_learners.append(train_meta_learner_cf(meta_learner, trained_model, d_cf3, params,
                                                                    nuisance_driv))
                        else:
                            raise ValueError('Meta learner not recognized')

        print("Test set RMSE for base ITE models ------------------------")
        # Computing test mse for each trained model
        for model in models:
            name = model["name"]
            # Test prediction
            tau_hat = model["trained_model"].predict_ite(d_test[:, 3:])
            rmse = helper.rmse(tau_hat, tau, scaler=scaler)
            #print(np.where(np.absolute(tau_hat*scaler - tau) > 4))
            print(f"MSE {name} : {rmse}")
            results_models_i.append([name, rmse])
            if config["plotting"]:
                plot_ite_hat(X=d_test[:, 3:4], tau_hat=scaler*tau_hat, tau=tau, title=name)

        print("Test set RMSE for meta learners ------------------------")
        # Computing test mse for each traind meta learner
        for meta_learner in meta_learners:
            meta_name = meta_learner["meta_learner_name"]
            base_name = meta_learner["base_model_name"]
            # Test prediction
            if not cf:
                tau_hat = meta_learner["trained_meta_learner"].predict(d_test[:, 3:])
            else:
                tau_hat = np.zeros(shape=(d_test.shape[0]))
                counter = 0
                for learner in meta_learner["trained_meta_learner"]:
                    tau_hat_i = learner.predict(d_test[:, 3:])
                    #Check for explosion in small sample sizes (remove)
                    if helper.rmse(tau_hat_i, tau, scaler=scaler) <= 1:
                        tau_hat += learner.predict(d_test[:, 3:])
                        counter += 1
                tau_hat = tau_hat / counter
            rmse = helper.rmse(tau_hat, tau, scaler=scaler)
            print(f"MSE Meta learner - base model: {meta_name} - {base_name} : {rmse}")
            results_meta_i.append([meta_name, base_name, rmse])
            if config["plotting"]:
                plot_ite_hat(X=d_test[:, 3:4], tau_hat=tau_hat*scaler, tau=tau, title=meta_name + " + " + base_name)

        results_models.append(results_models_i)
        results_meta.append(results_meta_i)

    #Save results in Dataframes
    model_names = []
    for i in range(len(results_models[0])):
        model_names.append(results_models[0][i][0])
    df_results_models = pd.DataFrame(columns=model_names, index=range(number_exp))

    meta_names = []
    for i in range(len(results_meta[0])):
        meta_names.append(results_meta[0][i][0] + "_" + results_meta[0][i][1])
    df_results_meta = pd.DataFrame(columns=meta_names, index=range(number_exp))

    for i in range(number_exp):
        for j in range(len(model_names)):
            df_results_models.iloc[i, j] = results_models[i][j][1]
        for j in range(len(meta_names)):
            df_results_meta.iloc[i, j] = results_meta[i][j][2]

    print("Results--------------------------------------")
    print("Results Base models--------------------------------------------------------")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_results_models)

    print("Results Meta learners-------------------------------------------------------")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_results_meta)

    print("Results mean/ std ---------------------------")

    #Calculate means and standard deviations
    means_models = df_results_models.mean()
    means_meta = df_results_meta.mean()
    if number_exp > 1:
        sd_models = df_results_models.std()
        means_models = pd.concat((means_models, sd_models), axis=1).transpose()
        means_models = means_models.rename(index={0: "mean", 1: "sdt"})
        sd_meta = df_results_meta.std()
        means_meta = pd.concat((means_meta, sd_meta), axis=1).transpose()
        means_meta = means_meta.rename(index={0: "mean", 1: "sdt"})
    print("Results Base models---------------------------------------------------------")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(means_models)
    print("Results Meta learners-------------------------------------------------------")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(means_meta)
    print("finished ------------------------")

    if return_results:
        return means_models, means_meta


def train_model(model_config, d_train, params=None):
    model_name = model_config["name"]
    print(f"Training | Base model: {model_name} | Meta learner: -")
    model = helper.train_base_model(model_name, d_train, params=params)
    return {"name": model_name, "trained_model": model}

#cross-fitting
def train_model_cf(model_config, d_train_list, params=None):
    model_name = model_config["name"]
    if model_name == "ncnet":
        model_name = "ncnet_cf"
    models = []
    print(f"Training cross-fitting | Base model: {model_name} | Meta learner: -")
    for data in d_train_list:
        models.append(helper.train_base_model(model_name, data, params=params))
    return {"name": model_name, "trained_models": models}


def train_meta_learner(meta_learner_name, trained_model, d_train, params, nuisance):
    base_model = trained_model["trained_model"]
    base_model_name = trained_model["name"]
    x_train = d_train[:, 3:]

    meta_learner = None
    print(f"Training | Base model: {base_model_name} | Meta learner: {meta_learner_name}")
    # MR Learner
    if meta_learner_name in ["mriv", "mrivsingle"]:

        if base_model_name == "ncnet":
            if meta_learner_name == "mriv":
                mr_input = base_model.predict_mr_input(x_train)
            elif meta_learner_name == "mrivsingle":
                mr_input = base_model.predict_mr_input_single(x_train)
        else:
            if meta_learner_name == "mrivsingle":
                raise ValueError('MR single only works with ncnet')
            tau = base_model.predict_ite(x_train)

            [mu_0Y, mu_0A, delta_A, pi] = nuisance
            mr_input = [pi, mu_0A, mu_0Y, delta_A, tau]
        # Train MR Learner
        print(f"Train MR learner")
        meta_learner = mr.train_mr_learner(data=d_train, init_estimates=mr_input, config=params,
                                           validation=False, logging=False)

    # DRIV
    if meta_learner_name == "driv":
        if base_model_name == "dmliv":
            [tau, q, p] = base_model.predict_driv_input(x_train)
            [_, _, r, f] = nuisance
            driv_input = [tau, q, p, r, f]
        else:
            tau = base_model.predict_ite(x_train)
            [q, p, r, f] = nuisance
            driv_input = [tau, q, p, r, f]
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


def train_meta_learner_cf(meta_learner_name, trained_models, d_train_list, params, nuisance_models):
    learners = []
    base_model_name = trained_models[0]["name"]
    if meta_learner_name == "mriv":
        models_init = trained_models[0]["trained_models"]
        def fit_nuisance_mr(test_ind, nuisance1_ind, nuisance2_ind):
            data_test = d_train_list[test_ind]
            tau = models_init[nuisance1_ind].predict_ite(data_test[:, 3:])
            if base_model_name == "ncnet":
                mu_0Y, mu_0A = models_init[nuisance1_ind].predict_components(data_test[:, 3:], 0)
            else:
                mu_0Y = nuisance_models[nuisance1_ind][0].predict_cf(data_test[:, 3:], 0)
                mu_0A = nuisance_models[nuisance1_ind][1].predict_cf(data_test[:, 3:], 0)
            mu_1A_d = nuisance_models[nuisance2_ind][2].predict_cf(data_test[:, 3:], 1)
            mu_0A_d = nuisance_models[nuisance2_ind][2].predict_cf(data_test[:, 3:], 0)
            delta_A = mu_1A_d - mu_0A_d
            pi = nuisance_models[nuisance2_ind][2].predict_pi(data_test[:, 3:])
            mr_input = [pi, mu_0A, mu_0Y, delta_A, tau]
            # Train MR Learner
            print(f"Train MR learner cf")
            return mr.train_mr_learner(data=data_test, init_estimates=mr_input, config=params,
                                               validation=False, logging=False)
        #cross-fitting
        learners.append(fit_nuisance_mr(0, 1, 2))
        learners.append(fit_nuisance_mr(1, 2, 1))
        learners.append(fit_nuisance_mr(2, 0, 1))

    if meta_learner_name == "driv":
        models_init = trained_models[1]["trained_models"]

        def fit_nuisance_driv(test_ind, nuisance_ind):
            data_test = d_train_list[test_ind]
            tau = models_init[nuisance_ind].predict_ite(data_test[:, 3:])
            q = nuisance_models[nuisance_ind][0].predict(data_test[:, 3:])
            p = nuisance_models[nuisance_ind][1].predict(data_test[:, 3:])
            r = nuisance_models[nuisance_ind][2].predict(data_test[:, 3:])
            f = nuisance_models[nuisance_ind][3].predict(data_test[:, 3:])
            driv_input = [tau, q, p, r, f]
            # Train DRIV
            print(f"Train DRIV learner")
            return dml.train_driv(data=data_test, init_estimates=driv_input, config=params,
                                      validation=False, logging=False)
        learners.append(fit_nuisance_driv(0, 1))
        learners.append(fit_nuisance_driv(1, 0))

    if meta_learner_name == "dr":
        models_init = trained_models[0]["trained_models"]
        if base_model_name == "tarnet":
            def fit_nuisance_dr(test_ind, nuisance1_ind, nuisance2_ind):
                data_test = d_train_list[test_ind]
                mu_1 = models_init[nuisance1_ind].predict_cf(data_test[:, 3:], 1)
                mu_0 = models_init[nuisance1_ind].predict_cf(data_test[:, 3:], 1)
                pi = models_init[nuisance2_ind].predict_pi(data_test[:, 3:])
                print(f"Train DRIV learner")
                return standard_ite.train_dr_learner(data=np.delete(data_test, 2, 1), config=params, init_estimates=[pi, mu_1, mu_0],
                                          validation=False, logging=False)

            # cross-fitting
            learners.append(fit_nuisance_dr(0, 1, 2))
            learners.append(fit_nuisance_dr(1, 2, 1))
            learners.append(fit_nuisance_dr(2, 0, 1))
        else:
            raise ValueError('DR Learner only works with TARNet')
    return {"meta_learner_name": meta_learner_name, "base_model_name": base_model_name,
            "trained_meta_learner": learners}


def plot_ite_hat(X, tau, tau_hat, title):
    data = np.concatenate((X, np.expand_dims(tau, axis=1), np.expand_dims(tau_hat, axis=1)), axis=1)
    data = data[data[:, 0].argsort()]
    plt.plot(data[:, 0], data[:, 1], label="tau")
    plt.plot(data[:, 0], data[:, 2], label="tau_hat")
    plt.title(title)
    plt.legend()
    plt.show()


