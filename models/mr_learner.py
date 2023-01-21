import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fctnl
import pytorch_lightning as pl
import models.helper as helper
from models.standard_ite import TARNet


class ncnet(pl.LightningModule):
    def __init__(self, config, input_size):
        super().__init__()
        # Joint representations
        self.repr11 = nn.Linear(input_size, config["hidden_size1"])
        self.repr12 = nn.Linear(config["hidden_size1"], config["hidden_size1"])
        self.repr13 = nn.Linear(config["hidden_size1"], config["hidden_size1"])

        self.repr21 = nn.Linear(input_size, config["hidden_size1"])
        self.repr22 = nn.Linear(config["hidden_size1"], config["hidden_size1"])
        self.repr23 = nn.Linear(config["hidden_size1"], config["hidden_size1"])

        # Heads for representation 1
        self.pi1 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.pi2 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.pi3 = nn.Linear(config["hidden_size2"], 1)

        self.mu1A11 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu1A12 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu1A13 = nn.Linear(config["hidden_size2"], 1)

        self.mu0A11 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu0A12 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu0A13 = nn.Linear(config["hidden_size2"], 1)

        # Heads for representation 2
        self.mu1Y1 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu1Y2 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu1Y3 = nn.Linear(config["hidden_size2"], 1)

        self.mu0Y1 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu0Y2 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu0Y3 = nn.Linear(config["hidden_size2"], 1)

        self.mu1A21 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu1A22 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu1A23 = nn.Linear(config["hidden_size2"], 1)

        self.mu0A21 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu0A22 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu0A23 = nn.Linear(config["hidden_size2"], 1)

        self.dropout = nn.Dropout(config["dropout"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def format_input(self, batch_torch):
        Y = batch_torch[:, 0]
        A = batch_torch[:, 1]
        Z = batch_torch[:, 2]
        X = batch_torch[:, 3:]
        return [Y, A, Z, X]

    def forward(self, X, Z):
        # Shared representations
        repr1 = self.dropout(fctnl.relu(self.repr11(X)))
        repr1 = self.dropout(fctnl.relu(self.repr12(repr1)))
        repr1 = self.repr13(repr1)

        repr2 = self.dropout(fctnl.relu(self.repr21(X)))
        repr2 = self.dropout(fctnl.relu(self.repr22(repr2)))
        repr2 = self.repr23(repr2)

        # Heads for representation 1
        pi = self.dropout(fctnl.relu(self.pi1(repr1)))
        pi = self.dropout(fctnl.relu(self.pi2(pi)))
        pi = torch.sigmoid(torch.squeeze(self.pi3(pi)))

        mu1A1 = self.dropout(fctnl.relu(self.mu1A11(repr1)))
        mu1A1 = self.dropout(fctnl.relu(self.mu1A12(mu1A1)))
        mu1A1 = torch.sigmoid(torch.squeeze(self.mu1A13(mu1A1)))

        mu0A1 = self.dropout(fctnl.relu(self.mu0A11(repr1)))
        mu0A1 = self.dropout(fctnl.relu(self.mu0A12(mu0A1)))
        mu0A1 = torch.sigmoid(torch.squeeze(self.mu0A13(mu0A1)))

        A_hat1 = Z * mu1A1 + (1 - Z) * mu0A1

        # Heads for representation 2
        mu1Y = self.dropout(fctnl.relu(self.mu1Y1(repr2)))
        mu1Y = self.dropout(fctnl.relu(self.mu1Y2(mu1Y)))
        mu1Y = torch.squeeze(self.mu1Y3(mu1Y))

        mu0Y = self.dropout(fctnl.relu(self.mu0Y1(repr2)))
        mu0Y = self.dropout(fctnl.relu(self.mu0Y2(mu0Y)))
        mu0Y = torch.squeeze(self.mu0Y3(mu0Y))

        Y_hat = Z * mu1Y + (1 - Z) * mu0Y

        mu1A2 = self.dropout(fctnl.relu(self.mu1A21(repr2)))
        mu1A2 = self.dropout(fctnl.relu(self.mu1A22(mu1A2)))
        mu1A2 = torch.sigmoid(torch.squeeze(self.mu1A23(mu1A2)))

        mu0A2 = self.dropout(fctnl.relu(self.mu0A21(repr2)))
        mu0A2 = self.dropout(fctnl.relu(self.mu0A22(mu0A2)))
        mu0A2 = torch.sigmoid(torch.squeeze(self.mu0A23(mu0A2)))

        A_hat2 = Z * mu1A2 + (1 - Z) * mu0A2

        return Y_hat, A_hat1, A_hat2, pi

    # Loss function
    def obj_ncnet(self, est, Y, A, Z):
        Y_hat = est[0]
        A_hat1 = est[1]
        A_hat2 = est[2]
        pi = est[3]
        # Loss components
        loss_y = torch.mean((Y_hat - Y) ** 2)
        loss_a1 = fctnl.binary_cross_entropy(A_hat1, A, reduction='mean')
        loss_a2 = fctnl.binary_cross_entropy(A_hat2, A, reduction='mean')
        loss_pi = fctnl.binary_cross_entropy(pi, Z, reduction='mean')
        # Overall loss
        loss = loss_y + loss_a1 + loss_a2 + loss_pi
        return loss, [loss_y, loss_a1, loss_a2, loss_pi]

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Formnat data
        [Y, A, Z, X] = self.format_input(train_batch)
        # Forward pass
        est = self.forward(X, Z)
        # Loss
        loss, [loss_y, loss_a1, loss_a2, loss_pi] = self.obj_ncnet(est, Y, A, Z)

        # Logging
        self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('train_loss_y', loss_y.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('train_loss_a1', loss_a1.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('train_loss_a2', loss_a2.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('train_loss_pi', loss_pi.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Formnat data
        [Y, A, Z, X] = self.format_input(train_batch)
        # Forward pass
        est = self.forward(X, Z)
        # Loss
        loss, [loss_y, loss_a1, loss_a2, loss_pi] = self.obj_ncnet(est, Y, A, Z)

        # Logging
        self.log('val_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_y', loss_y.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_a1', loss_a1.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_a2', loss_a2.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_pi', loss_pi.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def predict_components(self, x_np, nr):
        self.eval()
        X = torch.from_numpy(x_np.astype(np.float32))
        est = self.forward(X, nr)

        mu_Y = est[0].detach().numpy()
        mu_A1 = est[1].detach().numpy()
        mu_A2 = est[2].detach().numpy()
        pi = est[3].detach().numpy()

        return mu_Y, mu_A1, mu_A2, pi

    def predict_cf(self, x_np, nr):
        [mu_Y, mu_A1, mu_A2, pi] = self.predict_components(x_np, nr)
        return mu_Y

    def predict_ite(self, x_np):
        [mu_1Y, mu_1A1, mu_1A2, pi] = self.predict_components(x_np, 1)
        [mu_0Y, mu_0A1, mu_0A2, _] = self.predict_components(x_np, 0)
        delta_A = mu_1A2 - mu_0A2
        #print(np.where(np.absolute(delta_A) < 0.2))
        return (mu_1Y - mu_0Y) / delta_A

    def predict_mr_input(self, x_np):
        [mu_1Y, mu_1A1, mu_1A2, pi] = self.predict_components(x_np, 1)
        [mu_0Y, mu_0A1, mu_0A2, _] = self.predict_components(x_np, 0)
        delta_A = mu_1A1 - mu_0A1
        tau = (mu_1Y - mu_0Y) / (mu_1A2 - mu_0A2)
        return pi, mu_0A2, mu_0Y, delta_A, tau

    def predict_mr_input_single(self, x_np):
        [mu_1Y, mu_1A1, mu_1A2, pi] = self.predict_components(x_np, 1)
        [mu_0Y, mu_0A1, mu_0A2, _] = self.predict_components(x_np, 0)
        delta_A = mu_1A2 - mu_0A2
        tau = (mu_1Y - mu_0Y) / (mu_1A2 - mu_0A2)
        return pi, mu_0A2, mu_0Y, delta_A, tau

    def validation_mse(self, d_val):
        [mu_Y, mu_A1, mu_A2, pi] = self.predict_components(d_val[:, 3:], d_val[:, 2])
        loss_y = np.mean((mu_Y - d_val[:, 0]) ** 2)
        loss_a1 = fctnl.binary_cross_entropy(mu_A1, d_val[:, 1], reduction='mean')
        loss_a2 = fctnl.binary_cross_entropy(mu_A2, d_val[:, 1], reduction='mean')
        loss_pi = fctnl.binary_cross_entropy(pi, d_val[:, 2], reduction='mean')
        return loss_y + loss_a1 + loss_a2 + loss_pi


#Model for cross-fitting
class ncnet_cf(pl.LightningModule):
    def __init__(self, config, input_size):
        super().__init__()
        # Joint representations
        self.repr21 = nn.Linear(input_size, config["hidden_size1"])
        self.repr22 = nn.Linear(config["hidden_size1"], config["hidden_size1"])
        self.repr23 = nn.Linear(config["hidden_size1"], config["hidden_size1"])

        # Heads for representation 2
        self.mu1Y1 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu1Y2 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu1Y3 = nn.Linear(config["hidden_size2"], 1)

        self.mu0Y1 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu0Y2 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu0Y3 = nn.Linear(config["hidden_size2"], 1)

        self.mu1A21 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu1A22 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu1A23 = nn.Linear(config["hidden_size2"], 1)

        self.mu0A21 = nn.Linear(config["hidden_size1"], config["hidden_size2"])
        self.mu0A22 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.mu0A23 = nn.Linear(config["hidden_size2"], 1)

        self.dropout = nn.Dropout(config["dropout"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def format_input(self, batch_torch):
        Y = batch_torch[:, 0]
        A = batch_torch[:, 1]
        Z = batch_torch[:, 2]
        X = batch_torch[:, 3:]
        return [Y, A, Z, X]

    def forward(self, X, Z):
        #Shared representation
        repr2 = self.dropout(fctnl.relu(self.repr21(X)))
        repr2 = self.dropout(fctnl.relu(self.repr22(repr2)))
        repr2 = self.repr23(repr2)

        # Heads for representation 2
        mu1Y = self.dropout(fctnl.relu(self.mu1Y1(repr2)))
        mu1Y = self.dropout(fctnl.relu(self.mu1Y2(mu1Y)))
        mu1Y = torch.squeeze(self.mu1Y3(mu1Y))

        mu0Y = self.dropout(fctnl.relu(self.mu0Y1(repr2)))
        mu0Y = self.dropout(fctnl.relu(self.mu0Y2(mu0Y)))
        mu0Y = torch.squeeze(self.mu0Y3(mu0Y))

        Y_hat = Z * mu1Y + (1 - Z) * mu0Y

        mu1A2 = self.dropout(fctnl.relu(self.mu1A21(repr2)))
        mu1A2 = self.dropout(fctnl.relu(self.mu1A22(mu1A2)))
        mu1A2 = torch.sigmoid(torch.squeeze(self.mu1A23(mu1A2)))

        mu0A2 = self.dropout(fctnl.relu(self.mu0A21(repr2)))
        mu0A2 = self.dropout(fctnl.relu(self.mu0A22(mu0A2)))
        mu0A2 = torch.sigmoid(torch.squeeze(self.mu0A23(mu0A2)))

        A_hat2 = Z * mu1A2 + (1 - Z) * mu0A2

        return Y_hat, A_hat2

    # Loss function
    def obj_ncnet(self, est, Y, A, Z):
        Y_hat = est[0]
        A_hat2 = est[1]
        # Loss components
        loss_y = torch.mean((Y_hat - Y) ** 2)
        loss_a2 = fctnl.binary_cross_entropy(A_hat2, A, reduction='mean')
        # Overall loss
        loss = loss_y + loss_a2
        return loss, [loss_y, loss_a2]

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Formnat data
        [Y, A, Z, X] = self.format_input(train_batch)
        # Forward pass
        est = self.forward(X, Z)
        # Loss
        loss, [loss_y, loss_a2] = self.obj_ncnet(est, Y, A, Z)

        # Logging
        self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('train_loss_y', loss_y.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('train_loss_a2', loss_a2.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Formnat data
        [Y, A, Z, X] = self.format_input(train_batch)
        # Forward pass
        est = self.forward(X, Z)
        # Loss
        loss, [loss_y, loss_a2] = self.obj_ncnet(est, Y, A, Z)

        # Logging
        self.log('val_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_y', loss_y.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_a2', loss_a2.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def predict_components(self, x_np, nr):
        self.eval()
        X = torch.from_numpy(x_np.astype(np.float32))
        est = self.forward(X, nr)

        mu_Y = est[0].detach().numpy()
        mu_A2 = est[1].detach().numpy()

        return mu_Y, mu_A2

    def predict_cf(self, x_np, nr):
        [mu_Y, mu_A2] = self.predict_components(x_np, nr)
        return mu_Y

    def predict_ite(self, x_np):
        [mu_1Y, mu_1A2] = self.predict_components(x_np, 1)
        [mu_0Y, mu_0A2] = self.predict_components(x_np, 0)
        delta_A = mu_1A2 - mu_0A2
        #print(np.where(np.absolute(delta_A) < 0.2))
        return (mu_1Y - mu_0Y) / delta_A

    def validation_mse(self, d_val):
        [mu_Y, mu_A2] = self.predict_components(d_val[:, 3:], d_val[:, 2])
        loss_y = np.mean((mu_Y - d_val[:, 0]) ** 2)
        loss_a2 = fctnl.binary_cross_entropy(mu_A2, d_val[:, 1], reduction='mean')
        return loss_y + loss_a2

def create_pseudo_outcomes(z, a, y, pi, mu_A, mu_Y, delta_A, tau):
    n = np.shape(z)[0]
    z0 = np.ones(n) - z
    pi0 = np.ones(n) - pi

    denom = delta_A * (z * pi + z0 * pi0)
    nom = (z - z0) * (y - a * tau - mu_Y + mu_A * tau)
    return (nom / denom) + tau

def train_mr_learner(data, init_estimates, config, validation=True, logging=False):
    [pi, mu_0A2, mu_0Y, delta_A, tau] = init_estimates


    # Train MR Learner
    y0 = create_pseudo_outcomes(z=data[:, 2], a=data[:, 1], y=data[:, 0], pi=pi, mu_A=mu_0A2, mu_Y=mu_0Y,
                                   delta_A=delta_A, tau=tau)
    data_mr = np.concatenate((np.expand_dims(y0, 1), data[:, 3:]), axis=1)
    mr_learner, _ = helper.train_nn(data=data_mr, config=config, model_class=helper.ffnn, output_type="continuous",
                                    input_size=data_mr.shape[1] - 1, validation=validation, logging=logging)
    return mr_learner


def get_nuisance_full(data, config, d_val=None):
    X = data[:, 3:]
    data_yzx = np.delete(data, 1, 1)
    data_azx = data[:, 1:]
    model_yzx, _ = helper.train_nn(data=data_yzx, config=config, model_class=TARNet, input_size=X.shape[1],
                                   validation=False, logging=False, output_type="continuous", learn_pi=False)
    model_azx, _ = helper.train_nn(data=data_azx, config=config, model_class=TARNet, input_size=X.shape[1],
                                  validation=False, logging=False, output_type="binary")

    mu_0Y = model_yzx.predict_cf(X, 0)
    pi = model_azx.predict_pi(X)
    mu_1A = model_azx.predict_cf(X, 1)
    mu_0A = model_azx.predict_cf(X, 0)
    delta_A = mu_1A - mu_0A

    if d_val is not None:
        X = d_val[:, 3:]
        mu_0Y_val = model_yzx.predict_cf(X, 0)
        pi_val = model_azx.predict_pi(X)
        mu_1A_val = model_azx.predict_cf(X, 1)
        mu_0A_val = model_azx.predict_cf(X, 0)
        delta_A_val = mu_1A_val - mu_0A_val
        return [mu_0Y, mu_0A, delta_A, pi], [mu_0Y_val, mu_0A_val, delta_A_val, pi_val]
    else:
        return [mu_0Y, mu_0A, delta_A, pi]

#cross fitting
def get_nuisance_full_cf(data_splitted, config):
    nuisance = []
    for data in data_splitted:
        X = data[:, 3:]
        data_yzx = np.delete(data, 1, 1)
        data_azx = data[:, 1:]
        model_yzx, _ = helper.train_nn(data=data_yzx, config=config, model_class=TARNet, input_size=X.shape[1],
                                       validation=False, logging=False, output_type="continuous", learn_pi=False)
        model_azx1, _ = helper.train_nn(data=data_azx, config=config, model_class=TARNet, input_size=X.shape[1],
                                       validation=False, logging=False, output_type="binary", learn_pi=False)
        model_azx2, _ = helper.train_nn(data=data_azx, config=config, model_class=TARNet, input_size=X.shape[1],
                                       validation=False, logging=False, output_type="binary", learn_pi=True)
        nuisance.append([model_yzx, model_azx1, model_azx2])
    return nuisance