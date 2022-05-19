def get_hidden_sizes(p, dataset="sim"):
    if dataset == "sim":
        return [p, 5 * p, 10 * p, 20 * p, 30 * p]
    if dataset == "real":
        return [p, 3 * p, 5 * p, 8 * p, 10 * p]

def get_lrs():
    return [0.0001, 0.0005, 0.001, 0.005, 0.01]

# Base models------------------------------------------------------------------

def sample_hyper_ncnet(trial, p, dataset="sim"):
    config = {
        "hidden_size1": trial.suggest_categorical("hidden_size1", get_hidden_sizes(p, dataset=dataset)),
        "hidden_size2": trial.suggest_categorical("hidden_size2", get_hidden_sizes(p, dataset=dataset)),
        "lr": trial.suggest_categorical("lr", get_lrs()),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_tarnet(trial, p, dataset="sim"):
    config = {
        "hidden_size1": trial.suggest_categorical("hidden_size1", get_hidden_sizes(p, dataset=dataset)),
        "hidden_size2": trial.suggest_categorical("hidden_size2", get_hidden_sizes(p, dataset=dataset)),
        "hidden_size_pi": trial.suggest_categorical("hidden_size_pi", get_hidden_sizes(p, dataset=dataset)),
        "lr": trial.suggest_categorical("lr", get_lrs()),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_dmliv(trial, p, dataset="sim"):
    config = {
        "hidden_size_dml": trial.suggest_categorical("hidden_size_dml", get_hidden_sizes(p, dataset=dataset)),
        "lr_dml": trial.suggest_categorical("lr_dml", get_lrs()),
        "batch_size_dml": trial.suggest_categorical("batch_size_dml", [64, 128, 256]),
        "dropout_dml": trial.suggest_categorical("dropout_dml", [0, 0.1, 0.2, 0.3]),
        "hidden_size_nuisance": trial.suggest_categorical("hidden_size_nuisance", get_hidden_sizes(p, dataset=dataset)),
        "lr_nuisance": trial.suggest_categorical("lr_nuisance", get_lrs()),
        "batch_size_nuisance": trial.suggest_categorical("batch_size_nuisance", [64, 128, 256]),
        "dropout_nuisance": trial.suggest_categorical("dropout_nuisance", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_deepiv(trial, p, dataset="sim"):
    config = {
        "hidden_size": trial.suggest_categorical("hidden_size", get_hidden_sizes(p + 1, dataset=dataset)),
        "hidden_size2": trial.suggest_categorical("hidden_size2", get_hidden_sizes(p + 1, dataset=dataset)),
        "lr": trial.suggest_categorical("lr", get_lrs()),
        "lr2": trial.suggest_categorical("lr2", get_lrs()),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "batch_size2": trial.suggest_categorical("batch_size2", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3]),
        "dropout2": trial.suggest_categorical("dropout2", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_deepgmm(trial, p, dataset="sim"):
    config = {
        "hidden_size_f": trial.suggest_categorical("hidden_size_f", get_hidden_sizes(p + 1, dataset=dataset)),
        "hidden_size_g": trial.suggest_categorical("hidden_size_g", get_hidden_sizes(p + 1, dataset=dataset)),
        "lr_g": trial.suggest_categorical("lr_g", get_lrs()),
        "lambda_f": trial.suggest_categorical("lambda_f", [0.5, 1, 1.5, 2, 5]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_dfiv(trial, p, dataset="sim"):
    lambda1 = [0.0001, 0.001, 0.01, 0.1]
    lambda2 = [0.0001, 0.001, 0.01, 0.1]
    if dataset == "real":
        lambda1 = [0.01, 0.05, 0.1]
        lambda2 = [0.01, 0.05, 0.1]
    config = {
        "hidden_size_psi": trial.suggest_categorical("hidden_size_psi", get_hidden_sizes(p + 1, dataset=dataset)),
        "hidden_size_phi": trial.suggest_categorical("hidden_size_phi", get_hidden_sizes(1, dataset=dataset)),
        "hidden_size_xi": trial.suggest_categorical("hidden_size_xi", get_hidden_sizes(p, dataset=dataset)),
        "lr1": trial.suggest_categorical("lr1", get_lrs()),
        "lr2": trial.suggest_categorical("lr2", get_lrs()),
        "lambda1": trial.suggest_categorical("lambda1", lambda1),
        "lambda2": trial.suggest_categorical("lambda2", lambda2),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_kiv(trial, dataset="sim"):
    config = {
        "lambda": trial.suggest_categorical("lambda", [5, 6, 7, 8, 9, 10, 12]),
        "xi": trial.suggest_categorical("xi", [5, 6, 7,  8, 9, 10, 12])
    }
    return config


def sample_hyper_bcfiv(trial, dataset="sim"):
    config = {
        "n_trees_pic": trial.suggest_categorical("n_trees_pic", [20, 30, 40, 50]),
        "n_trees_itt": trial.suggest_categorical("n_trees_itt", [20, 30, 40, 50])
    }
    return config


# Meta learner------------------------------------------------------------------------
def sample_hyper_mr_nuisance(trial, p, dataset="sim"):
    config = {
        "hidden_size1": trial.suggest_categorical("hidden_size1", get_hidden_sizes(p, dataset=dataset)),
        "hidden_size2": trial.suggest_categorical("hidden_size2", get_hidden_sizes(p, dataset=dataset)),
        "hidden_size_pi": trial.suggest_categorical("hidden_size_pi", get_hidden_sizes(p, dataset=dataset)),
        "lr": trial.suggest_categorical("lr", get_lrs()),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_mr(trial, p, dataset="sim"):
    config = {
        "hidden_size": trial.suggest_categorical("hidden_size", get_hidden_sizes(p, dataset=dataset)),
        "lr": trial.suggest_categorical("lr", get_lrs()),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_driv_nuisance(trial, p, dataset="sim"):
    config = {
        "hidden_size": trial.suggest_categorical("hidden_size", get_hidden_sizes(p, dataset=dataset)),
        "lr": trial.suggest_categorical("lr", get_lrs()),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config

def sample_hyper_driv(trial, p, dataset="sim"):
    config = {
        "hidden_size": trial.suggest_categorical("hidden_size", get_hidden_sizes(p, dataset=dataset)),
        "lr": trial.suggest_categorical("lr", get_lrs()),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper_dr(trial, p, dataset="sim"):
    config = {
        "hidden_size": trial.suggest_categorical("hidden_size", get_hidden_sizes(p, dataset=dataset)),
        "lr": trial.suggest_categorical("lr", get_lrs()),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3])
    }
    return config


def sample_hyper(method, trial, p, dataset="sim"):
    if method == "ncnet":
        return sample_hyper_ncnet(trial, p, dataset=dataset)
    if method == "tarnet":
        return sample_hyper_tarnet(trial, p, dataset=dataset)
    if method == "dmliv":
        return sample_hyper_dmliv(trial, p, dataset=dataset)
    if method == "deepiv":
        return sample_hyper_deepiv(trial, p, dataset=dataset)
    if method == "deepgmm":
        return sample_hyper_deepgmm(trial, p, dataset=dataset)
    if method == "dfiv":
        return sample_hyper_dfiv(trial, p, dataset=dataset)
    if method == "kiv":
        return sample_hyper_kiv(trial, dataset=dataset)
    if method == "bcfiv":
        return sample_hyper_bcfiv(trial, dataset=dataset)
    if method == "mr":
        return sample_hyper_mr(trial, p, dataset=dataset)
    if method == "mr_nuisance":
        return sample_hyper_mr_nuisance(trial, p, dataset=dataset)
    if method == "driv":
        return sample_hyper_driv(trial, p, dataset=dataset)
    if method == "driv_nuisance":
        return sample_hyper_driv_nuisance(trial, p, dataset=dataset)
    if method == "dr":
        return sample_hyper_dr(trial, p, dataset=dataset)
