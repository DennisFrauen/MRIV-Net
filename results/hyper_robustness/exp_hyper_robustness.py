import pandas as pd
import misc
import yaml
from experiments.main import train_model, train_meta_learner, set_seeds
import joblib
import random
import numpy as np
import models.helper as helper
import models.mr_learner as mr

def create_hyper_list(params, range, name, caption):
    hyper_list = []
    for x in range:
        params_x = params.copy()
        params_x[name] = x
        params_x["name"] = caption
        hyper_list.append(params_x)
    return hyper_list

def get_rmse_hyper(config, hyper_lists):
    set_seeds(config["seed"])
    # Save results
    hyper_names = []
    for hyper_list in hyper_lists:
        hyper_names.append(hyper_list[0]["name"])

    df_results = pd.DataFrame(0, columns=hyper_names, index=range(len(hyper_lists[0])))

    for k in range(config["number_experiments"]):
        print(f"Experiment Nr {k}")
        # Sample random seed
        seed = random.randint(0, 1000000)
        # Generate dataset
        np.random.seed(seed)
        [d_train, d_test, truth_train, truth_test, scaler] = misc.load_data(config)
        tau = truth_test[0]

        print("Model training ------------------------")
        for i, hyper_list in enumerate(hyper_lists):
            for j, params in enumerate(hyper_list):
                print(i)
                print(j)
                # Estimate nuisance parameters for meta learners
                params_mriv = {}
                params_mriv["hidden_size"] = params["hidden_size_mriv"]
                params_mriv["lr"] = params["lr_mriv"]
                params_mriv["batch_size"] = params["batch_size_mriv"]
                params_mriv["dropout"] = params["dropout_mriv"]
                params_ncnet = {}
                params_ncnet["hidden_size1"] = params["hidden_size1_ncnet"]
                params_ncnet["hidden_size2"] = params["hidden_size2_ncnet"]
                params_ncnet["lr"] = params["lr_ncnet"]
                params_ncnet["batch_size"] = params["batch_size_ncnet"]
                params_ncnet["dropout"] = params["dropout_ncnet"]
                # Training
                ncnet = helper.train_base_model("ncnet", d_train, params=params_ncnet)
                mr_input = ncnet.predict_mr_input(d_train[:, 3:])
                mriv = mr.train_mr_learner(data=d_train, init_estimates=mr_input, config=params_mriv,
                                           validation=False, logging=False)
                tau_hat = mriv.predict(d_test[:, 3:])
                rmse = helper.rmse(tau_hat, tau, scaler=scaler)
                df_results.iloc[j, i] += rmse

    return df_results / config["number_experiments"]

if __name__ == "__main__":
    # Select configuration file here
    stream = open(misc.get_project_path() + "/results/hyper_robustness/exp_hyper_robustness_config.yaml", 'r')
    config = yaml.safe_load(stream)

    params_ncnet = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name="ncnet")
    params_mriv = misc.load_hyper_yaml(path=config["hyper_path"], base_model_name="ncnet",
                                       meta_learner_name="mriv", type="meta_learner")

    params = {}
    params["hidden_size_mriv"] = params_mriv["hidden_size"]
    params["lr_mriv"] = params_mriv["lr"]
    params["batch_size_mriv"] = params_mriv["batch_size"]
    params["dropout_mriv"] = params_mriv["dropout"]
    params["hidden_size1_ncnet"] = params_ncnet["hidden_size1"]
    params["hidden_size2_ncnet"] = params_ncnet["hidden_size2"]
    params["lr_ncnet"] = params_ncnet["lr"]
    params["batch_size_ncnet"] = params_ncnet["batch_size"]
    params["dropout_ncnet"] = params_ncnet["dropout"]

    hyper_list_hidden = create_hyper_list(params=params, range=[20, 25, 30, 35], name="hidden_size_ncnet", caption= "Hidden size")
    hyper_list_lr = create_hyper_list(params=params, range=[0.0005, 0.001, 0.005, 0.01], name="lr_ncnet", caption= "Learning rate")
    hyper_list_dropout = create_hyper_list(params=params, range=[0, 0.1, 0.2, 0.3], name="dropout_ncnet", caption= "Dropout prob.")
    hyper_list_batch = create_hyper_list(params=params, range=[32, 64, 128, 256], name="batch_ncnet", caption= "Batch size")

    hyper_lists = [hyper_list_hidden, hyper_list_lr, hyper_list_dropout, hyper_list_batch]

    df_results = get_rmse_hyper(config, hyper_lists)

    #Save results to file
    path_results = misc.get_project_path() + "/results/hyper_robustness/"
    joblib.dump(df_results, path_results + "results_hyper_robustness.pkl")






