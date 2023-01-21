import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fctnl
import pytorch_lightning as pl
import models.helper as helper


class TARNet(pl.LightningModule):
    def __init__(self, config, input_size, output_type="continuous", learn_pi=True):
        super().__init__()

        self.repr = nn.Sequential(
            nn.Linear(input_size, config["hidden_size1"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size1"], config["hidden_size1"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size1"], config["hidden_size1"]),
            nn.ReLU()
        )

        self.y1_net = nn.Sequential(
            nn.Linear(config["hidden_size1"], config["hidden_size2"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size2"], config["hidden_size2"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size2"], 1),
        )

        self.y0_net = nn.Sequential(
            nn.Linear(config["hidden_size1"], config["hidden_size2"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size2"], config["hidden_size2"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size2"], 1),
        )

        self.pi_net = nn.Sequential(
            nn.Linear(input_size, config["hidden_size_pi"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_pi"], config["hidden_size_pi"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_pi"], 1),
            nn.Sigmoid()
        )

        self.learn_pi = learn_pi
        self.output_type = output_type
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    @staticmethod
    def format_input(batch_torch):
        Y = batch_torch[:, 0]
        A = batch_torch[:, 1]
        X = batch_torch[:, 2:]
        return [Y, A, X]

    def obj(self, pi_hat, A, y_hat, Y):
        loss_pi = fctnl.binary_cross_entropy(pi_hat, A, reduction='mean')
        if self.output_type == "continuous":
            loss_y = torch.mean((y_hat - Y) ** 2)
        else:
            loss_y = fctnl.binary_cross_entropy(y_hat, Y, reduction='mean')
        if self.learn_pi==True:
            loss = loss_y + loss_pi
        else:
            loss = loss_y
        return loss, loss_y, loss_pi

    def forward(self, x, A):
        pi_hat = torch.squeeze(self.pi_net(x))
        repr = self.repr(x)
        y1 = torch.squeeze(self.y1_net(repr))
        y0 = torch.squeeze(self.y0_net(repr))
        if self.output_type == "binary":
            y1 = torch.sigmoid(y1)
            y0 = torch.sigmoid(y0)
        y_hat = A * y1 + (1 - A) * y0
        return y_hat, pi_hat

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Formnat data
        x_format = self.format_input(train_batch)
        # Forward pass
        y_hat, pi_hat = self.forward(x_format[2], x_format[1])
        # Loss
        loss, loss_y, loss_pi = self.obj(pi_hat, x_format[1], y_hat, x_format[0])
        # Logging
        self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('train_loss_y', loss_y.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('train_loss_pi', loss_pi.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Formnat data
        x_format = self.format_input(train_batch)
        # Forward pass
        y_hat, pi_hat = self.forward(x_format[2], x_format[1])
        # Loss
        loss, loss_y, loss_pi = self.obj(pi_hat, x_format[1], y_hat, x_format[0])
        # Logging
        self.log('val_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_y', loss_y.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_loss_pi', loss_pi.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def predict_cf(self, x_np, nr):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        if isinstance(nr, int):
            A = torch.squeeze(torch.full((n, 1), nr))
        else:
            A = torch.from_numpy(nr.astype(np.float32))
        y_hat, _ = self.forward(X, A)
        return y_hat.detach().numpy()

    def predict_ite(self, x_np):
        self.eval()
        y_hat1 = self.predict_cf(x_np, 1)
        y_hat0 = self.predict_cf(x_np, 0)
        return y_hat1 - y_hat0

    def predict_pi(self, x_np):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        A = torch.full((n, 1), 1)
        _, pi_hat = self.forward(X, A)
        return pi_hat.detach().numpy()


def create_pseudo_outcomes(a, y, pi, mu_1, mu_0):
    n = np.shape(a)[0]
    a0 = np.ones(n) - a
    pi0 = np.ones(n) - pi

    y0 = ((a / pi) - (a0 / pi0)) * y
    y0 += (np.ones(n) - (a / pi)) * mu_1
    y0 -= (np.ones(n) - (a0 / pi0)) * mu_0
    return y0

#DR Learner
def train_dr_learner(data, init_estimates, config, validation=False, logging=False):
    [pi, mu_1, mu_0] = init_estimates

    # Train DR Learner
    y0 = create_pseudo_outcomes(a=data[:, 1], y=data[:, 0], pi=pi, mu_1=mu_1, mu_0=mu_0)
    data_dr = np.concatenate((np.expand_dims(y0, 1), data[:, 2:]), axis=1)
    dr_learner, _ = helper.train_nn(data=data_dr, config=config, model_class=helper.ffnn, output_type="continuous",
                                    input_size=data_dr.shape[1] - 1, validation=validation, logging=logging)
    return dr_learner

#nuisance parameters for cross-fitting
def get_nuisance_full_cf(data_list, config):
    nuisance = []
    for data in data_list:
        Y, A, Z, X = helper.split_data(data)
        data_ax = np.concatenate((np.expand_dims(A, 1), X), 1)
        model_pi, _ = helper.train_nn(data=data_ax, config=config, model_class=helper.ffnn, input_size=X.shape[1],
                                      validation=False, logging=False, output_type="binary")
        nuisance.append(model_pi)
