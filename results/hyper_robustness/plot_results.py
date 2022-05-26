import joblib
import misc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    path = misc.get_project_path() + "/results/hyper_robustness/"
    results = joblib.load(path + "results_hyper_robustness.pkl")

    df_plot = results.melt(var_name="Parameter", value_name="Value")
    df_plot.loc[:, "x"] = [20, 25, 30, 35, 0.0005, 0.001, 0.005, 0.01, 0, 0.1, 0.2, 0.3, 32, 64, 128, 256]

    sns.set_theme(style="darkgrid")
    grid = sns.FacetGrid(df_plot, col="Parameter", hue="Parameter", col_wrap=2, height=3, sharex=False)
    grid.map(plt.plot, "x", "Value")
    grid.set(ylim=(0, 0.5), xlabel='Value', ylabel="RMSE")

    #grid.add_legend()
    plt.savefig(path + "plot_hyper_robustness.pdf", bbox_inches='tight')
    plt.show()