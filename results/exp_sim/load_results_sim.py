import joblib
import misc
import pandas as pd

if __name__ == "__main__":
    path = misc.get_project_path() + "/results/exp_sim/"
    results_models = joblib.load(path + "results_models_5000.pkl")
    results_meta = joblib.load(path + "results_meta_5000.pkl")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_models)
        print(results_meta)