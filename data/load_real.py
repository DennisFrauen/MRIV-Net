import numpy as np
import pandas as pd
import misc

def load_oregon(scale=True):
    # Select configuration file here
    path = misc.get_project_path() + "/data/oregon_health_exp/OHIE_Data/"
    data_descr = pd.read_stata(path + "oregonhie_descriptive_vars.dta")
    data_state = pd.read_stata(path + "oregonhie_stateprograms_vars.dta")[["person_id", "ohp_all_ever_matchn_30sep2009"]]
    data_ed = pd.read_stata(path + "oregonhie_ed_vars.dta")[["person_id", "num_visit_pre_cens_ed"]]
    outcome = pd.read_stata(path + "oregonhie_inperson_vars.dta")[["person_id", "pcs8_score_inp"]]

    #Preprocess desciptive data
    data_descr.loc[data_descr.numhh_list == "signed self up", "num_signed_up"] = 1
    data_descr.loc[data_descr.numhh_list == "signed self up + 1 additional person", "num_signed_up"] = 2
    data_descr.loc[data_descr.numhh_list == "signed self up + 2 additional people", "num_signed_up"] = 3

    #Single/ Double/ Triple sign ups
    num_single = data_descr[data_descr.num_signed_up == 1].shape[0]
    num_double = data_descr[data_descr.num_signed_up == 2].shape[0]
    num_triple = data_descr[data_descr.num_signed_up == 3].shape[0]
    pseudo_n = num_single + 2*num_double + 3*num_triple

    #Calculated propensity scores (from R script propensity_scores.R)
    data_descr.loc[data_descr.numhh_list == "signed self up", "pi"] = 0.345
    data_descr.loc[data_descr.numhh_list == "signed self up + 1 additional person", "pi"] = 0.571
    data_descr.loc[data_descr.numhh_list == "signed self up + 2 additional people", "pi"] = 0.719

    data_descr["age"] = 2009 - data_descr["birthyear_list"]
    data_descr["gender"] = 1
    data_descr["gender"][data_descr.female_list == "0: Male"] = 0
    data_descr["city"] = 0
    data_descr["city"][data_descr.zip_msa_list == "Zip code of residence in a MSA"] = 1
    data_descr["language"] = 0
    data_descr["language"][data_descr.english_list == "Requested English materials"] = 1
    data_descr = data_descr.merge(data_ed, on="person_id")
    data_descr = data_descr.merge(data_state, on="person_id")
    data_descr = data_descr.merge(outcome, on="person_id")
    #Treatment
    data_descr["treat"] = 0
    data_descr["treat"][data_descr.ohp_all_ever_matchn_30sep2009 == "Enrolled"] = 1
    #Instrument
    data_descr["instrument"] = 0
    data_descr["instrument"][data_descr.treatment == "Selected"] = 1
    data_descr["outcome"] = data_descr["pcs8_score_inp"]


    data = data_descr[["outcome", "treat", "instrument",
                       "age", "num_signed_up", "num_visit_pre_cens_ed", "gender", "language", "pi"]]
    data = data.dropna(axis=0)
    data = data.to_numpy()
    np.random.shuffle(data)
    #Extract propensity score
    pi = data[:, -1]
    data = np.delete(data, -1, 1)

    if scale:
        #Standardize outcome
        sd = np.std(data[:, 0])
        data[:, 0] = (data[:, 0] - np.mean(data[:, 0])) / sd
        standardize_info = []
        #Standardize covariates
        for i in [3, 4, 5]:
            mean_i = np.mean(data[:, i])
            sd_i = np.std(data[:, i])
            standardize_info.append([mean_i, sd_i])
            data[:, i] = (data[:, i] - mean_i) / sd_i

        return data, [pi, standardize_info], sd
    else:
        return data


if __name__ == "__main__":
    load_oregon()


