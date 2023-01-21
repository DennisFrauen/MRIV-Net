import joblib
import misc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    path = misc.get_project_path() + "/results/exp_smoothness/"
    means_models3 = joblib.load(path + "means_models_3000.pkl")
    means_meta3 = joblib.load(path + "means_meta_3000.pkl")
    sd_models3 = joblib.load(path + "sd_models_3000.pkl")
    sd_meta3 = joblib.load(path + "sd_meta_3000.pkl")
    means_models5 = joblib.load(path + "means_models_5000.pkl")
    means_meta5 = joblib.load(path + "means_meta_5000.pkl")
    sd_models5 = joblib.load(path + "sd_models_5000.pkl")
    sd_meta5 = joblib.load(path + "sd_meta_5000.pkl")
    means_models8 = joblib.load(path + "means_models_8000.pkl")
    means_meta8 = joblib.load(path + "means_meta_8000.pkl")
    sd_models8 = joblib.load(path + "sd_models_8000.pkl")
    sd_meta8 = joblib.load(path + "sd_meta_8000.pkl")

    smooth_levels = list(means_models3.index)[0:3]
    names = ["deepiv", "ncnet", "mriv_ncnet"]
    captions = ["DeepIV", "MRIV-Net\w network only", "MRIV-Net (ours)"]
    colors = ["darkred", "deepskyblue", "darkblue"]
    names_means = [name + "_means" for name in names]
    names_sd = [name + "_sd" for name in names]

    data_plot = pd.DataFrame(columns=["mean", "ci_lower", "ci_upper", "Method", "n", "beta"], index=range(3 * len(smooth_levels) * len(names)))
    data_plot.loc[:, "beta"] = smooth_levels * 3 * len(names)

    data_plot.loc[:, "n"] = (["3000"] * len(smooth_levels) + ["5000"] * len(smooth_levels) + ["8000"] * len(smooth_levels)) * len(names)

    for i, name in enumerate(names):
        if name in means_models3.columns:
            means3 = means_models3.loc[smooth_levels, name]
            sd3 = sd_models3.loc[smooth_levels, name]
            means5 = means_models5.loc[smooth_levels, name]
            sd5 = sd_models5.loc[smooth_levels, name]
            means8 = means_models8.loc[smooth_levels, name]
            sd8 = sd_models8.loc[smooth_levels, name]
        elif name in means_meta3.columns:
            means3 = means_meta3.loc[smooth_levels, name]
            sd3 = sd_meta3.loc[smooth_levels, name]
            means5 = means_meta5.loc[smooth_levels, name]
            sd5 = sd_meta5.loc[smooth_levels, name]
            means8 = means_meta8.loc[smooth_levels, name]
            sd8 = sd_meta8.loc[smooth_levels, name]
        else:
            raise ValueError('name not recognized')
        means = np.concatenate((means3, means5, means8))
        sds = np.concatenate((sd3, sd5, sd8))
        index_range = list(range(i * len(smooth_levels) * 3, (i + 1) * len(smooth_levels) * 3))

        data_plot.loc[data_plot.index[index_range], "Method"] = captions[i]
        data_plot.loc[data_plot.index[index_range], "mean"] = means
        data_plot.loc[data_plot.index[index_range], "ci_lower"] = means - sds
        data_plot.loc[data_plot.index[index_range], "ci_upper"] = means + sds
    # Initialize a grid of plots with an Axes for each n
    #grid = sns.relplot(data=data_plot, x="confounding", y="mean", hue="name", col="n", kind="line")
    #grid.map(sns.lineplot, y="ci_lower", hue="name")
    #Plot for n=8000
    data_plot = data_plot[data_plot.n == "8000"]

    grid = sns.FacetGrid(data_plot, col="n", hue="Method", col_wrap=1, height=3, palette=colors)
    grid.map(plt.plot, "beta", "mean")
    grid.set(ylim=(0.1, 0.35), xlabel=r'Smoothness $\alpha$', ylabel="RMSE")


    for i, ax in enumerate(grid.axes):
        for j, caption in enumerate(captions):
            data = data_plot[data_plot.Method == caption]
            data = data[data.n == "8000"]
            y1 = data.ci_upper.to_numpy()
            y1 = [float(y) for y in y1]
            y2 = data.ci_lower.to_numpy()
            y2 = [float(y) for y in y2]
            ax.fill_between(smooth_levels, y1, y2, alpha=0.1, color=colors[j])


    grid.add_legend()
    grid.set_titles("")
    plt.savefig(path + "plot_smoothness.pdf", bbox_inches='tight')
    plt.show()