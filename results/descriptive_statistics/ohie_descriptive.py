import random
import numpy as np
import pandas as pd
import yaml
import misc
import seaborn as sns
import matplotlib.pyplot as plt

def set_seeds(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    # Select configuration file here
    stream = open(misc.get_project_path() + "/experiments/real/exp_real_config.yaml", 'r')
    config = yaml.safe_load(stream)
    set_seeds(config["seed"])
    #Generate dataset
    data = misc.load_data(config, scale=False)
    #Create dataframe for plotting
    df_plotting = pd.DataFrame(columns=["Y", "A", "Z", "Age", "Num. signed up", "Num. visits", "Gender", "Language"],
                               index=range(data.shape[0]))
    df_plotting.loc[:, "Y"] = data[:, 0]
    df_plotting.loc[:, "A"] = data[:, 1]
    df_plotting.loc[:, "Z"] = data[:, 2]
    df_plotting.loc[:, "Age"] = data[:, 3]
    df_plotting.loc[:, "Num. signed up"] = data[:, 4]
    df_plotting.loc[:, "Num. visits"] = data[:, 5]
    df_plotting.loc[:, "Gender"] = data[:, 6]
    df_plotting.loc[:, "Language"] = data[:, 7]

    #df_plotting = df_plotting.melt(var_name="Variable", value_name="Value")
    sns.set_theme(style="darkgrid")
    #grid = sns.FacetGrid(df_plotting, col="Variable", col_wrap=4, height=3)

    # define plotting region (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # create boxplot in each subplot
    sns.histplot(data=df_plotting, x='Y', ax=axes[0, 0], stat="frequency", discrete=False).set(yticklabels=[])
    sns.histplot(data=df_plotting, x='A', ax=axes[0, 1], stat="frequency", discrete=True).set(yticklabels=[], xticks=[0, 1], xticklabels=["0", "1"])
    sns.histplot(data=df_plotting, x='Z', ax=axes[0, 2], stat="frequency", discrete=True).set(yticklabels=[], xticks=[0, 1], xticklabels=["0", "1"])
    sns.histplot(data=df_plotting, x='Age', ax=axes[0, 3], stat="frequency", discrete=False).set(yticklabels=[])
    sns.histplot(data=df_plotting, x='Num. signed up', ax=axes[1, 0], stat="frequency", discrete=True).set(yticklabels=[], xticks=[1, 2, 3], xticklabels=["1", "2", "3"])
    sns.histplot(data=df_plotting, x='Num. visits', ax=axes[1, 1], stat="frequency", discrete=True).set(yticklabels=[])
    sns.histplot(data=df_plotting, x='Gender', ax=axes[1, 2], stat="frequency", discrete=True).set(yticklabels=[], xticks=[0, 1], xticklabels=["Male", "Female"])
    sns.histplot(data=df_plotting, x='Language', ax=axes[1, 3], stat="frequency", discrete=True).set(yticklabels=[], xticks=[0, 1], xticklabels=["Other", "English"])





        #ax.fill_between(conf_levels, y1, y2, alpha=0.1, color=colors[j])
    #grid = sns.displot(df_plotting, x="Value", stat="probability", kind="hist", col="Variable", binwidth=0.3, height=3,
    #                   facet_kws=dict(margin_titles=True))


    #grid.add_legend()
    plt.savefig(misc.get_project_path() + "/results/descriptive_statistics/plot_hist.pdf", bbox_inches='tight')
    plt.show()