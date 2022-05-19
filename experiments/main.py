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



def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(config, return_results=False):
    number_exp = config["number_experiments"]
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
            nuisance_mriv, _ = mr.get_nuisance_full(d_train, params_nuisance_mr, d_val=d_test)
        if nuisance_need_driv:
            print("Training | Nuisance parameter DRIV")
            params_nuisance_driv = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name="nuisance",
                                          meta_learner_name="driv", type="meta_learner")
            nuisance_driv, _ = dml.get_nuisance_full(d_train, params_nuisance_driv, d_val=d_test)
        print("Model training ------------------------")

        for model_config in config["models"]:
            set_seeds(seed)
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
                        set_seeds(seed)
                        params = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name=model_config["name"],
                                                      meta_learner_name=meta_learner, type="meta_learner")
                        if meta_learner in ["mriv", "mrivsingle"]:
                            meta_learners.append(train_meta_learner(meta_learner, trained_model, d_train, params,
                                                                    nuisance_mriv))
                        elif meta_learner in ["driv", "dr"]:
                            meta_learners.append(train_meta_learner(meta_learner, trained_model, d_train, params,
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
            tau_hat = meta_learner["trained_meta_learner"].predict(d_test[:, 3:])
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


def plot_ite_hat(X, tau, tau_hat, title):
    data = np.concatenate((X, np.expand_dims(tau, axis=1), np.expand_dims(tau_hat, axis=1)), axis=1)
    data = data[data[:, 0].argsort()]
    plt.plot(data[:, 0], data[:, 1], label="tau")
    plt.plot(data[:, 0], data[:, 2], label="tau_hat")
    plt.title(title)
    plt.legend()
    plt.show()


