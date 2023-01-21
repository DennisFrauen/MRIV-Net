import misc
import yaml
from experiments.main import run_experiment
import joblib

if __name__ == "__main__":
    # Select configuration file here
    stream = open(misc.get_project_path() + "/experiments/sim/exp_sim_cross_fitting.yaml", 'r')
    run_config = yaml.safe_load(stream)
    sample_size = run_config["n"]
    hyper_path = run_config["hyper_path"]
    print(f"Configuration hyper path {hyper_path}, sample size n = {sample_size}")
    result_models, result_meta = run_experiment(run_config, return_results=True)

    #Save results to file
    path_results = misc.get_project_path() + "/results/exp_sim/"
    joblib.dump(result_meta, path_results + "results_meta_" + str(sample_size) + "_cf.pkl")
