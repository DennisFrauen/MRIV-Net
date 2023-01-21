import joblib
import misc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    path = misc.get_project_path() + "/results/real_appendix/"
    df_models_visits_age50 = joblib.load(path + "results_models_visits_age50.pkl")
    df_models_visits_age60 = joblib.load(path + "results_models_visits_age60.pkl")
    df_models_visits_age70 = joblib.load(path + "results_models_visits_age70.pkl")
    df_meta_visits_age50 = joblib.load(path + "results_meta_visits_age50.pkl")
    df_meta_visits_age60 = joblib.load(path + "results_meta_visits_age60.pkl")
    df_meta_visits_age70 = joblib.load(path + "results_meta_visits_age70.pkl")

    visits = num_visit_range = np.array([0, 2, 4, 6, 8, 10, 12])
    names = ["tarnet", "ncnet", "mriv_ncnet"]
    captions = ["TARNet", "MRIV-Net\w network only", "MRIV-Net (ours)"]
    colors = ["darkred", "deepskyblue", "darkblue", "black"]
    data_plot = pd.DataFrame(columns=["age", "ite", "Method", "visits"], index=range(3*len(visits)*len(names)))

    for i, name in enumerate(names):
        if name in df_models_visits_age50.columns:
            ites50 = df_models_visits_age50.loc[:, name].to_numpy()
            ites60 = df_models_visits_age60.loc[:, name].to_numpy()
            ites70 = df_models_visits_age70.loc[:, name].to_numpy()
        elif name in df_meta_visits_age50.columns:
            ites50 = df_meta_visits_age50.loc[:, name].to_numpy()
            ites60 = df_meta_visits_age60.loc[:, name].to_numpy()
            ites70 = df_meta_visits_age70.loc[:, name].to_numpy()
        else:
            raise ValueError('name not recognized')
        ites = np.concatenate((ites50, ites60, ites70))
        index_range = np.array(list(range(3*i*len(visits),3*(i+1)*len(visits))))
        data_plot.loc[data_plot.index[index_range], "visits"] = np.concatenate((visits, visits, visits))
        data_plot.loc[data_plot.index[index_range], "ite"] = ites
        data_plot.loc[data_plot.index[index_range], "Method"] = captions[i]
        data_plot.loc[data_plot.index[index_range], "age"] = ["50"] * visits.shape[0] + ["60"] * visits.shape[0] + ["70"] * visits.shape[0]

    # Initialize a grid of plots with an Axes for each n
    grid = sns.FacetGrid(data_plot, col="age", hue="Method", height=3, palette=colors)
    grid.map(plt.plot, "visits", "ite")
    grid.set(xlabel="Number of emergency visits", ylabel="Estimated ITE")

    grid.add_legend()
    #p = sns.lineplot(data=data_plot, x="age", y="ite", hue="name")
    plt.savefig(path + "plot_real_visits.pdf")
    plt.show()


    #Language results

    df_models_language_age50 = joblib.load(path + "results_models_language_age50.pkl")
    df_models_language_age60 = joblib.load(path + "results_models_language_age60.pkl")
    df_models_language_age70 = joblib.load(path + "results_models_language_age70.pkl")
    df_meta_language_age50 = joblib.load(path + "results_meta_language_age50.pkl")
    df_meta_language_age60 = joblib.load(path + "results_meta_language_age60.pkl")
    df_meta_language_age70 = joblib.load(path + "results_meta_language_age70.pkl")

    language = num_visit_range = np.array([0, 1])
    xlabels = ["", "Other", "", "English"]
    names = ["mriv_ncnet"]
    captions = ["MRIV-Net (ours)"]
    colors = ["darkblue"]
    data_plot = pd.DataFrame(columns=["age", "ite", "Method", "language"], index=range(3 * len(language) * len(names)))

    for i, name in enumerate(names):
        if name in df_models_visits_age50.columns:
            ites50 = df_models_language_age50.loc[:, name].to_numpy()
            ites60 = df_models_language_age60.loc[:, name].to_numpy()
            ites70 = df_models_language_age70.loc[:, name].to_numpy()
        elif name in df_meta_visits_age50.columns:
            ites50 = df_meta_language_age50.loc[:, name].to_numpy()
            ites60 = df_meta_language_age60.loc[:, name].to_numpy()
            ites70 = df_meta_language_age70.loc[:, name].to_numpy()
        else:
            raise ValueError('name not recognized')
        ites = np.concatenate((ites50, ites60, ites70))
        index_range = np.array(list(range(3 * i * len(language), 3 * (i + 1) * len(language))))
        data_plot.loc[data_plot.index[index_range], "language"] = np.concatenate((language, language, language))
        data_plot.loc[data_plot.index[index_range], "ite"] = ites
        data_plot.loc[data_plot.index[index_range], "Method"] = captions[i]
        data_plot.loc[data_plot.index[index_range], "age"] = ["50"] * language.shape[0] + ["60"] * language.shape[0] + [
            "70"] * language.shape[0]

    # Initialize a grid of plots with an Axes for each n
    grid = sns.FacetGrid(data_plot, col="age", hue="Method", height=3, palette=colors)
    grid.map(plt.bar, "language", "ite")
    grid.set(xlabel="Language", ylabel="Estimated ITE", xticklabels=xlabels)
    #for ax in grid.axes.flat:
    #    ax.set_xticklabels(xlabels)
    grid.add_legend()
    # p = sns.lineplot(data=data_plot, x="age", y="ite", hue="name")
    plt.savefig(path + "plot_real_language.pdf")
    plt.show()

