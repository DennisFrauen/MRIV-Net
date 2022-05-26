import sklearn as sk
import optuna
import joblib
from pathlib import Path
import os
import numpy as np
import random
import torch
import yaml
from optuna.samplers import RandomSampler
from sklearn.model_selection import train_test_split
import models.helper as helper
import misc
import hyperparam.hyper_objectives as objectives
import models.mr_learner as mriv
import models.dml_dr_iv as driv


def tune_objective(objective, study_name, path, num_samples=10, sampler=None):
    if sampler is not None:
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=num_samples)

    print("Finished. Best trial:")
    trial_best = study.best_trial

    print("  Value: ", trial_best.value)

    print("  Params: ")
    for key, value in trial_best.params.items():
        print("    {}: {}".format(key, value))

    #save_dir = path + study_name + ".pkl"
    #joblib.dump(study, save_dir)
    return study


def tune_basemodel(method, data, path, num_samples=10, sampler=None, dataset="sim"):
    study_name = "study_" + method
    d_train, d_test = train_test_split(data, test_size=0.2, shuffle=False)
    if method == "ncnet":
        obj = objectives.get_objective_ncnet(d_train, d_test, dataset=dataset)
        return tune_objective(obj, study_name=study_name, path=path, num_samples=num_samples, sampler=sampler)
    if method == "tarnet":
        obj = objectives.get_objective_tarnet(d_train, d_test, dataset=dataset)
        return tune_objective(obj, study_name=study_name, path=path, num_samples=num_samples, sampler=sampler)
    if method == "dmliv":
        # Nuisance tuning
        obj_nuisance = objectives.get_objective_dmliv_nuisance(d_train, d_test, dataset=dataset)
        study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)
        study.optimize(obj_nuisance, n_trials=int(num_samples / 2))
        params_best = study.best_trial.params
        obj_dml = objectives.get_objective_dmliv(d_train, d_test, params_best, dataset=dataset)
        return tune_objective(obj_dml, study_name=study_name, path=path, num_samples=num_samples, sampler=sampler)
    if method == "tarnet":
        obj = objectives.get_objective_tarnet(d_train, d_test, dataset=dataset)
        return tune_objective(obj, study_name=study_name, path=path, num_samples=num_samples, sampler=sampler)
    if method == "deepiv":
        obj1 = objectives.get_objective_deepiv1(d_train, d_train, dataset=dataset)
        study = optuna.create_study(direction="minimize", study_name="First stage deepiv study", sampler=sampler)
        study.optimize(obj1, n_trials=int(num_samples / 2))
        params_best = study.best_trial.params
        obj2 = objectives.get_objective_deepiv2(d_train, d_train, params_best, dataset=dataset)
        return tune_objective(obj2, study_name=study_name, path=path, num_samples=int(num_samples / 2), sampler=sampler)
    if method == "deepgmm":
        obj = objectives.get_objective_deepgmm(d_train, d_test, dataset=dataset)
        return tune_objective(obj, study_name=study_name, path=path, num_samples=num_samples, sampler=sampler)
    if method == "dfiv":
        # First stage
        obj1 = objectives.get_objective_dfiv_1(data, dataset=dataset)
        study = optuna.create_study(direction="minimize", study_name="First stage dfiv study", sampler=sampler)
        study.optimize(obj1, n_trials=int(num_samples / 2))
        params_best = study.best_trial.params
        obj2 = objectives.get_objective_dfiv_2(data, params_best, dataset=dataset)
        return tune_objective(obj2, study_name=study_name, path=path, num_samples=int(num_samples / 2), sampler=sampler)
    if method == "kiv":
        # First stage
        obj1 = objectives.get_objective_kiv_1(data, dataset=dataset)
        study = optuna.create_study(direction="minimize", study_name="First stage kiv study", sampler=sampler)
        study.optimize(obj1, n_trials=int(num_samples / 2))
        params_best = study.best_trial.params
        obj2 = objectives.get_objective_kiv_2(data, lamb=params_best["lambda"], dataset=dataset)
        return tune_objective(obj2, study_name=study_name, path=path, num_samples=int(num_samples / 2), sampler=sampler)
    if method == "bcfiv":
        obj = objectives.get_objective_bcfiv(d_train, d_test, dataset=dataset)
        return tune_objective(obj, study_name=study_name, path=path, num_samples=num_samples, sampler=sampler)


def tune_meta_learner(base_model, base_model_name, method, data, path, nuisance_train=None, nuisance_val=None,
                      num_samples=10, sampler=None, dataset="sim"):
    study_name = "study_" + method + "_" + base_model_name
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    obj = None
    if method == "mriv":
        obj = objectives.get_objective_mriv(d_train=d_train, d_val=d_val, base_model=base_model,
                                            base_model_name=base_model_name, mrsingle=False, dataset=dataset,
                                            nuisance_train=nuisance_train, nuisance_val=nuisance_val)

    if method == "mrivsingle":
        obj = objectives.get_objective_mriv(d_train=d_train, d_val=d_val, base_model=base_model, dataset=dataset,
                                            base_model_name=base_model_name, mrsingle=True)
    if method == "driv":
        obj = objectives.get_objective_driv(d_train=d_train, d_val=d_val, base_model=base_model, dataset=dataset,
                                            base_model_name=base_model_name, nuisance_train=nuisance_train,
                                            nuisance_val=nuisance_val)

    if method == "dr":
        obj = objectives.get_objective_dr(d_train=d_train, d_val=d_val, base_model=base_model, dataset=dataset)

    return tune_objective(obj, study_name=study_name, path=path, num_samples=num_samples, sampler=sampler)


def tune_nuisance_models(method, data, path, num_samples=10, sampler=None, pi=None, dataset="sim"):
    study_name = "study_" + method
    path_best_config = path + "params/meta_learners/params_" + method + ".yaml"
    d_train, d_val = train_test_split(data, test_size=0.2, shuffle=False)
    if pi is not None:
        pi_train, pi_val = train_test_split(pi, test_size=0.2, shuffle=False)
    if method == "mriv_nuisance":
        obj = objectives.get_objective_mriv_nuisance(d_train=d_train, d_val=d_val, dataset=dataset)
        study = tune_objective(obj, study_name=study_name, path=path + "studies/meta_learners/",
                               num_samples=num_samples, sampler=sampler)
        best_config = study.best_trial.params
        nuisance_train, nuisance_val = mriv.get_nuisance_full(data=d_train, config=best_config, d_val=d_val)
        if pi is not None:
            nuisance_train[3] = pi_train
            nuisance_val[3] = pi_val
    elif method == "driv_nuisance":
        obj = objectives.get_objective_driv_nuisance(d_train=d_train, d_val=d_val, dataset=dataset)
        study = tune_objective(obj, study_name=study_name, path=path + "studies/meta_learners/", num_samples=num_samples,
                               sampler=sampler)
        best_config = study.best_trial.params
        nuisance_train, nuisance_val = driv.get_nuisance_full(data=d_train, config=best_config, d_val=d_val)
        if pi is not None:
            nuisance_train[2] = pi_train
            nuisance_val[2] = pi_val
    else:
        raise ValueError('Error in Nuisance model name')
    # Save best nuisance hyperparameter
    with open(path_best_config, 'w') as outfile:
        yaml.dump(best_config, outfile, default_flow_style=False)
    return nuisance_train, nuisance_val


def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tune_sampler = RandomSampler(seed=seed)
    return tune_sampler


def run_hyper_tuning(config):
    np.random.seed(config["seed"])
    # Load data
    pi = None
    if config["dataset"]=="sim":
        [d_train, _, _, _, _] = misc.load_data(config)
    else:
        d_train, [pi, _], _ = misc.load_data(config)
    # Construct paths
    path = misc.get_project_path() + "/hyperparam" + config["path"]

    print("Start hyperparameter tuning ------------------------")

    # First, check whether nuisance parameter for meta learners are needed. If yes, tune them
    nuisance_need_mr, nuisance_need_driv = misc.check_nuisance_need(config["models"])
    mriv_nuisance_train = None
    mriv_nuisance_val = None
    driv_nuisance_train = None
    driv_nuisance_val = None
    if nuisance_need_mr:
        print(f"################Tuning | Nuisance model for MRIV")
        tune_sampler = set_seeds(config["seed"])
        mriv_nuisance_train, mriv_nuisance_val = tune_nuisance_models("mriv_nuisance", d_train, path, dataset=config["dataset"],
                                                                      num_samples=config["num_samples"],
                                                                      sampler=tune_sampler,  pi=pi)
    if nuisance_need_driv:
        print(f"################Tuning | Nuisance model for DRIV")
        tune_sampler = set_seeds(config["seed"])
        driv_nuisance_train, driv_nuisance_val = tune_nuisance_models("driv_nuisance", d_train, path, dataset=config["dataset"],
                                                                      num_samples=config["num_samples"],
                                                                      sampler=tune_sampler, pi=pi)

    # Paths for saving studies
    base_path_study = path + "studies/base_models/"
    meta_path_study = path + "studies/meta_learners/"

    # Hyperparameter tuning of base models + meta learners
    for model_config in config["models"]:
        tune_sampler = set_seeds(config["seed"])
        base_model_name = model_config["name"]
        if base_model_name not in ["tsls", "waldlinear"]:
            print(f"################Tuning | Base model: {base_model_name} | Meta learner: -")
            study = tune_basemodel(method=base_model_name, data=d_train, path=base_path_study,
                                   num_samples=config["num_samples"],
                                   sampler=tune_sampler, dataset=config["dataset"])
            # Load and save optimal parameter
            base_path_param = path + "params/base_models/params_" + base_model_name + ".yaml"
            best_params = study.best_trial.params
            with open(base_path_param, 'w') as outfile:
                yaml.dump(best_params, outfile, default_flow_style=False)
        else:
            best_params = None
        if "meta_learners" in model_config:
            if model_config["meta_learners"] is not None:
                # Train base model with optimal hyperparameters
                _ = set_seeds(config["seed"])
                base_model = helper.train_base_model(model_name=base_model_name, d_train=d_train, params=best_params,
                                                     validation=False,
                                                     logging=False)
                # Meta learner
                for meta_learner in model_config["meta_learners"]:
                    print(f"################Tuning | Base model: {base_model_name} | Meta learner: {meta_learner}")
                    tune_sampler = set_seeds(config["seed"])
                    if meta_learner in ["mriv", "mrivsingle"]:
                        study = tune_meta_learner(base_model=base_model, base_model_name=base_model_name,
                                                  method=meta_learner, data=d_train, path=meta_path_study,
                                                  num_samples=config["num_samples"], sampler=tune_sampler,
                                                  nuisance_train=mriv_nuisance_train, nuisance_val=mriv_nuisance_val,
                                                  dataset=config["dataset"])
                    elif meta_learner in ["driv", "dr"]:
                        study = tune_meta_learner(base_model=base_model, base_model_name=base_model_name,
                                                  method=meta_learner, data=d_train, path=meta_path_study,
                                                  num_samples=config["num_samples"], sampler=tune_sampler,
                                                  nuisance_train=driv_nuisance_train, nuisance_val=driv_nuisance_val,
                                                  dataset=config["dataset"])
                    else:
                        raise ValueError('Meta learner name not recognized')
                    # Load and save optimal parameter
                    meta_path_param = path + "params/meta_learners/params_" + meta_learner + "_" + base_model_name + ".yaml"
                    best_params = study.best_trial.params
                    with open(meta_path_param, 'w') as outfile:
                        yaml.dump(best_params, outfile, default_flow_style=False)
