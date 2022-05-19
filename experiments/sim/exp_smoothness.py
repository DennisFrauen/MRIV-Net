import pandas as pd
import misc
import yaml
from experiments.main import run_experiment
import joblib


if __name__ == "__main__":
    # Select configuration file here
    stream = open(misc.get_project_path() + "/experiments/sim/exp_smoothness_config.yaml", 'r')
    run_config = yaml.safe_load(stream)

    beta_range = [0.5, 1.5, 3, 5, 10, 20]
    sample_size = run_config["n"]

    model_names = misc.get_model_names(run_config["models"])
    meta_names = misc.get_meta_names(run_config["models"])
    #Data Frames to store results
    means_models = pd.DataFrame(columns=model_names, index=beta_range)
    sd_models = pd.DataFrame(columns=model_names, index=beta_range)
    means_meta = pd.DataFrame(columns=meta_names, index=beta_range)
    sd_meta = pd.DataFrame(columns=meta_names, index=beta_range)



    for beta in beta_range:
        print(f"##################Run experiment with smoothness level beta = {beta}, n = {sample_size}")
        run_config["beta"] = beta
        result_models, result_meta = run_experiment(run_config, return_results=True)
        #Store results
        for model in model_names:
            means_models.loc[beta, model] = result_models.loc["mean", model]
            sd_models.loc[beta, model] = result_models.loc["sdt", model]
        for meta in meta_names:
            means_meta.loc[beta, meta] = result_meta.loc["mean", meta]
            sd_meta.loc[beta, meta] = result_meta.loc["sdt", meta]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("BASE MODEL RESULTS------------------------------------------------------------")
        print("Means")
        print(means_models)
        print("Standard deviations")
        print(sd_models)
        print("META LEARNER RESULTS------------------------------------------------------------")
        print("Means")
        print(means_meta)
        print("Standard deviations")
        print(sd_meta)

    #Save results to file
    path_results = misc.get_project_path() + "/results/exp_smoothness/"
    joblib.dump(means_models, path_results + "means_models_" + str(sample_size) + ".pkl")
    joblib.dump(sd_models, path_results + "sd_models_" + str(sample_size) + ".pkl")
    joblib.dump(means_meta, path_results + "means_meta_" + str(sample_size) + ".pkl")
    joblib.dump(sd_meta, path_results + "sd_meta_" + str(sample_size) + ".pkl")