import joblib
import misc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    path = misc.get_project_path() + "/results/real/"
    df_models = joblib.load(path + "results_models.pkl")
    df_meta = joblib.load(path + "results_meta.pkl")
    df_models1 = joblib.load(path + "results_models1.pkl")
    df_meta1 = joblib.load(path + "results_meta1.pkl")


    ages = np.array(list(df_models.index)) + 20
    names = ["tarnet", "ncnet", "mriv_ncnet", "driv_dmliv"]
    captions = ["TARNet", "MRIV-Net\w netowrk only", "MRIV-Net (ours)", "DMLIV + DRIV"]
    colors = ["darkred", "deepskyblue", "darkblue", "black"]
    data_plot = pd.DataFrame(columns=["age", "ite", "Method", "gender"], index=range(2*len(ages)*len(names)))

    for i, name in enumerate(names):
        if name in df_models.columns:
            ites = df_models.loc[:, name].to_numpy()
            ites1 = df_models1.loc[:, name].to_numpy()
        elif name in df_meta.columns:
            ites = df_meta.loc[:, name].to_numpy()
            ites1 = df_meta1.loc[:, name].to_numpy()
        else:
            raise ValueError('name not recognized')
        ites = np.concatenate((ites, ites1))
        index_range = np.array(list(range(2*i*len(ages),2*(i+1)*len(ages))))
        data_plot.loc[data_plot.index[index_range], "age"] = np.concatenate((ages, ages))
        data_plot.loc[data_plot.index[index_range], "ite"] = ites
        data_plot.loc[data_plot.index[index_range], "Method"] = captions[i]
        data_plot.loc[data_plot.index[index_range], "gender"] = ["male"] * ages.shape[0] + ["female"] * ages.shape[0]

    # Initialize a grid of plots with an Axes for each n
    grid = sns.FacetGrid(data_plot,col= "gender", hue="Method", height=3, palette=colors)
    grid.map(plt.plot, "age", "ite")
    grid.set(xlabel="Age", ylabel="Estimated ITE")

    grid.add_legend()
    #p = sns.lineplot(data=data_plot, x="age", y="ite", hue="name")
    plt.savefig(path + "plot_real.pdf")
    plt.show()