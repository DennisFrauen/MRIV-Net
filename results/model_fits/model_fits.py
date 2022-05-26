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

from experiments.main import train_model
from experiments.main import train_meta_learner

def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    # Select configuration file here
    stream = open(misc.get_project_path() + "/results/model_fits/model_fits_config.yaml", 'r')
    config = yaml.safe_load(stream)
    sample_size = config["n"]
    hyper_path = config["hyper_path"]

    # Initial seed for reproducibility
    seed = config["seed"]
    set_seeds(config["seed"])

    # Generate dataset
    np.random.seed(seed)
    [d_train, d_test, truth_train, truth_test, scaler] = misc.load_data(config)
    tau = truth_test[0]

    models = []
    meta_learners = []
    results_models = []
    results_meta = []

    # Estimate nuisance parameters for meta learners
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

    print("Compute predicted ITEs for base ITE models ------------------------")
    # Computing test mse for each trained model
    model_names = ["TARNet", "2SLS", "KIV", "DFIV", "DeepIV", "DeepGMM", "DMLIV", "Wald (linear)", "Wald (BART)", "MRIV-Net (network only)"]
    for i, model in enumerate(models):
        name = model["name"]
        # Test prediction
        tau_hat = model["trained_model"].predict_ite(d_test[:, 3:])
        d_sort = np.concatenate((d_test[:, 3:], np.expand_dims(tau, axis=1), np.expand_dims(tau_hat, axis=1)), axis=1)
        d_sort = d_sort[d_sort[:, 0].argsort()]
        results_models.append([model_names[i], d_sort[:, 2]])

    print("Compute predicted ITEs for meta learners ------------------------")
    # Computing test mse for each traind meta learner
    meta_names = ["MRIV-Net"]
    for i, meta_learner in enumerate(meta_learners):
        meta_name = meta_learner["meta_learner_name"]
        base_name = meta_learner["base_model_name"]
        # Test prediction
        tau_hat = meta_learner["trained_meta_learner"].predict(d_test[:, 3:])
        d_sort = np.concatenate((d_test[:, 3:], np.expand_dims(tau, axis=1), np.expand_dims(tau_hat, axis=1)), axis=1)
        d_sort = d_sort[d_sort[:, 0].argsort()]
        results_meta.append([meta_name, base_name, d_sort[:, 2]])


    # Save results in Dataframes
    #model_names = []
    #for i in range(len(results_models)):
    #    model_names.append(results_models[i][0])
    df_results_models = pd.DataFrame(columns=["x", "tau"] + model_names, index=range(d_test.shape[0]))

    #meta_names = []
    #for i in range(len(results_meta)):
    #    meta_names.append(results_meta[i][0] + "_" + results_meta[i][1])
    df_results_meta = pd.DataFrame(columns=["x", "tau"] + meta_names, index=range(d_test.shape[0]))

    df_results_models.loc[:, "x"] = d_sort[:, 0]
    df_results_meta.loc[:, "x"] = d_sort[:, 0]
    df_results_models.loc[:, "tau"] = d_sort[:, 1]
    df_results_meta.loc[:, "tau"] = d_sort[:, 1]

    for i in range(d_test.shape[0]):
        for j in range(len(model_names)):
            df_results_models.iloc[i, j+2] = results_models[j][1][i]
        for j in range(len(meta_names)):
            df_results_meta.iloc[i, j+2] = results_meta[j][2][i]
    #Save results to file
    path_results = misc.get_project_path() + "/results/model_fits/"
    joblib.dump(df_results_models, path_results + "results_models.pkl")
    joblib.dump(df_results_meta, path_results + "results_meta.pkl")

