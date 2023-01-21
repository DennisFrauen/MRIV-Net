# implementation of DML IV and DR IV
import torch
import torch.nn as nn
import torch.nn.functional as fctnl
import pytorch_lightning as pl
import numpy as np

import misc
import models.helper as helper


def change_config_keys_dml(config):
    params_nuisance = config.copy()
    params_nuisance["hidden_size"] = params_nuisance.pop("hidden_size_nuisance")
    params_nuisance["lr"] = params_nuisance.pop("lr_nuisance")
    params_nuisance["batch_size"] = params_nuisance.pop("batch_size_nuisance")
    params_nuisance["dropout"] = params_nuisance.pop("dropout_nuisance")
    params_dml = config.copy()
    params_dml["hidden_size"] = params_dml.pop("hidden_size_dml")
    params_dml["lr"] = params_dml.pop("lr_dml")
    params_dml["batch_size"] = params_dml.pop("batch_size_dml")
    params_dml["dropout"] = params_dml.pop("dropout_dml")
    return params_dml, params_nuisance

# DML IV
def train_dmliv(data, config, validation=True, logging=False):
    Y, A, Z, X = helper.split_data(data)
    params_dml, params_nuisance = change_config_keys_dml(config)
    # Train nuisance models
    data_yx = np.concatenate((np.expand_dims(Y, 1), X), 1)
    data_azx = np.concatenate((np.expand_dims(A, 1), np.expand_dims(Z, 1), X), 1)
    data_ax = np.concatenate((np.expand_dims(A, 1), X), 1)
    model_yx, _ = helper.train_nn(data=data_yx, config=params_nuisance, model_class=helper.ffnn, input_size=X.shape[1],
                                  validation=False, logging=False, output_type="continuous")
    model_azx, _ = helper.train_nn(data=data_azx, config=params_nuisance, model_class=helper.ffnn, input_size=X.shape[1] + 1,
                                   validation=False, logging=False, output_type="binary")
    model_ax, _ = helper.train_nn(data=data_ax, config=params_nuisance, model_class=helper.ffnn, input_size=X.shape[1],
                                  validation=False, logging=False, output_type="binary")
    #Train dml model
    dml_iv, _ = helper.train_nn(data=data, config=params_dml, model_class=dml_nn, input_size=X.shape[1],
                                validation=validation, logging=logging, model_ax=model_ax, model_azx=model_azx,
                                model_yx=model_yx)
    return dml_iv


# Feed forward neural network, either binary or continuous output
class dml_nn(pl.LightningModule):
    def __init__(self, config, input_size, model_yx, model_azx, model_ax):
        super().__init__()
        self.layer1 = nn.Linear(input_size, config["hidden_size"])
        self.layer2 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.layer3 = nn.Linear(config["hidden_size"], 1)
        self.dropout = nn.Dropout(config["dropout"])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.model_yx = model_yx
        self.model_azx = model_azx
        self.model_ax = model_ax
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def format_input(self, batch_torch):
        Y = batch_torch[:, 0]
        A = batch_torch[:, 1]
        Z = batch_torch[:, 2]
        X = batch_torch[:, 3:]
        return [Y, A, Z, X]

    # Orthogonal loss
    def obj(self, Y_hat, Y, Z, X):
        # Generate fitted values
        q = self.model_yx.forward(X).detach()
        h = self.model_azx.forward(torch.concat((torch.unsqueeze(Z, 1), X), 1)).detach()
        p = self.model_ax.forward(X).detach()
        loss = torch.mean(((Y - q - Y_hat) * (h - p)) ** 2)
        return loss

    def forward(self, X):
        out = self.dropout(fctnl.relu(self.layer1(X)))
        out = self.dropout(fctnl.relu(self.layer2(out)))
        out = torch.squeeze(self.layer3(out))
        return out

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Format data
        [Y, A, Z, X] = self.format_input(train_batch)
        # Forward pass
        y_hat = self.forward(X)
        # Loss
        loss = self.obj(y_hat, Y, Z, X)
        # Logging
        self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Format data
        [Y, A, Z, X] = self.format_input(train_batch)
        # Forward pass
        y_hat = self.forward(X)
        # Loss
        loss = self.obj(y_hat, Y, Z, X)
        # Logging
        self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def predict_ite(self, x_np):
        self.eval()
        X = torch.from_numpy(x_np.astype(np.float32))
        tau_hat = self.forward(X)
        return tau_hat.detach().numpy()

    def predict_driv_input(self, x_np):
        tau = self.predict_ite(x_np)
        q = self.model_yx.predict(x_np)
        p = self.model_ax.predict(x_np)
        return [tau, q, p]

def create_pseudo_outcomes(z, a, y, tau, q, p, r, f):
    # Create pseudo-outcomes
    Y_bar = y - q
    A_bar = a - p
    Z_bar = z - r
    beta = f - p * r
    Y0 = tau + (((Y_bar - tau * A_bar) * Z_bar) / beta)
    return Y0

# DR IV using DML IV as initial estimator, also returns DML IV
# comp are input components
def train_driv(data, init_estimates, config, validation=True, logging=False):
    [tau, q, p, r, f] = init_estimates
    Y, A, Z, X = helper.split_data(data)
    # Create pseudo-outcomes
    Y0 = create_pseudo_outcomes(z=Z, a=A, y=Y, tau=tau, q=q, p=p, r=r, f=f)
    # Create data for DR IV
    data_dr = np.concatenate((np.expand_dims(Y0, 1), X), 1)
    driv, _ = helper.train_nn(data=data_dr, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                              validation=validation, logging=logging, output_type="continuous")
    return driv


# Estimation of nuisance parameters------------------------------------------------------

def get_nuisance_full(data, config, d_val=None):
    # Create datasets
    Y, A, Z, X = helper.split_data(data)
    data_yx = np.concatenate((np.expand_dims(Y, 1), X), 1)
    data_ax = np.concatenate((np.expand_dims(A, 1), X), 1)
    data_zx = np.concatenate((np.expand_dims(Z, 1), X), 1)
    data_az_x = np.concatenate((np.expand_dims(A * Z, 1), X), 1)
    # Train nuisance models
    model_zx, _ = helper.train_nn(data=data_zx, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                  validation=False, logging=False, output_type="binary")
    model_az_x, _ = helper.train_nn(data=data_az_x, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                    validation=False, logging=False, output_type="binary")
    model_yx, _ = helper.train_nn(data=data_yx, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                  validation=False, logging=False, output_type="continuous")
    model_ax, _ = helper.train_nn(data=data_ax, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                  validation=False, logging=False, output_type="binary")
    q = model_yx.predict(X)
    p = model_ax.predict(X)
    r = model_zx.predict(X)
    f = model_az_x.predict(X)

    if d_val is not None:
        X = d_val[:, 3:]
        q_val = model_yx.predict(X)
        p_val = model_ax.predict(X)
        r_val = model_zx.predict(X)
        f_val = model_az_x.predict(X)
        return [q, p, r, f], [q_val, p_val, r_val, f_val]
    else:
        return [q, p, r, f]

#Cross-fitting
def get_nuisance_full_cf(data_list, config):
    nuisance = []
    for data in data_list:
        # Create datasets
        Y, A, Z, X = helper.split_data(data)
        data_yx = np.concatenate((np.expand_dims(Y, 1), X), 1)
        data_ax = np.concatenate((np.expand_dims(A, 1), X), 1)
        data_zx = np.concatenate((np.expand_dims(Z, 1), X), 1)
        data_az_x = np.concatenate((np.expand_dims(A * Z, 1), X), 1)
        # Train nuisance models
        model_zx, _ = helper.train_nn(data=data_zx, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                      validation=False, logging=False, output_type="binary")
        model_az_x, _ = helper.train_nn(data=data_az_x, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                        validation=False, logging=False, output_type="binary")
        model_yx, _ = helper.train_nn(data=data_yx, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                      validation=False, logging=False, output_type="continuous")
        model_ax, _ = helper.train_nn(data=data_ax, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                      validation=False, logging=False, output_type="binary")
        nuisance.append([model_yx, model_ax, model_zx, model_az_x])
    return nuisance
