import misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

if __name__ == "__main__":
    path = misc.get_project_path() + "/results/model_fits/"
    results_models = joblib.load(path + "results_models.pkl")
    results_meta = joblib.load(path + "results_meta.pkl")
    # Convert to long format
    df_models = results_models.melt(var_name="Method", id_vars=["x", "tau"], value_name="Est. ITE")
    df_meta = results_meta.melt(var_name="Method", id_vars=["x", "tau"], value_name="Est. ITE")

    df_plot = pd.concat([df_models, df_meta], ignore_index=True, sort=False)

    grid = sns.FacetGrid(df_plot, col="Method", col_wrap=4, height=3)
    grid.map(plt.plot, "x", "Est. ITE")
    grid.map(plt.plot, "x", "tau", color="darkred")
    grid.set(xlabel='X', ylabel="Y")

    grid.add_legend()
    plt.savefig(path + "plot_model_fit.pdf", bbox_inches='tight')
    plt.show()
