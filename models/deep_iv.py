#DeepIV implementation for binary treatments (see Hartford paper)
import torch
import torch.nn as nn
import torch.nn.functional as fctnl
import pytorch_lightning as pl
import numpy as np
import models.helper as helper


def train_DeepIV(data, config, validation=True, logging=False):
    first_stage = train_first_stage(data, config, validation=validation, logging=logging)
    second_stage = train_second_stage(data, config, first_stage, validation=validation, logging=logging)
    return second_stage


def train_first_stage(data, config, validation=True, logging=False):
    Y, A, Z, X = helper.split_data(data)
    data1 = np.concatenate((np.expand_dims(A, 1), np.expand_dims(Z, 1), X), 1)
    first_stage, _ = helper.train_nn(data=data1, config=config, model_class= helper.ffnn, input_size=X.shape[1] + 1,
                                  validation=validation, logging=logging, output_type="binary")
    return first_stage

def train_second_stage(data, config, first_stage, validation=True, logging=False):
    X = data[:, 3:]
    config2 = config.copy()
    config2["batch_size"] = config["batch_size2"]
    second_stage, _ = helper.train_nn(data=data, config=config2, model_class=second_stage_nn, input_size=X.shape[1] + 1,
                                  validation=validation, logging=logging, first_stage_nn=first_stage)
    return second_stage

#Feed forward neural network for second stage regression
class second_stage_nn(pl.LightningModule):
    def __init__(self, config, input_size, first_stage_nn):
        super().__init__()
        self.layer1 = nn.Linear(input_size, config["hidden_size2"])
        self.layer2 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.layer3 = nn.Linear(config["hidden_size2"], 1)
        self.dropout = nn.Dropout(config["dropout2"])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr2"])
        self.first_stage_nn = first_stage_nn
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def format_input(self, batch_torch):
        n = batch_torch.shape[0]
        Y = batch_torch[:, 0]
        x0 = torch.concat((torch.zeros(n, 1).type_as(batch_torch), batch_torch[:, 3:]), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(batch_torch), batch_torch[:, 3:]), dim=1)
        return [Y, x0, x1]

    def obj(self, Y_hat0, Y_hat1, data):
        pi = self.first_stage_nn.forward(data[:, 2:]).detach()
        loss = torch.mean((data[:, 0] - (1 - pi) * Y_hat0 - pi*Y_hat1)**2)
        return loss

    def forward(self, x0, x1):
        out0 = self.dropout(fctnl.relu(self.layer1(x0)))
        out0 = self.dropout(fctnl.relu(self.layer2(out0)))
        out0 = torch.squeeze(self.layer3(out0))
        out1 = self.dropout(fctnl.relu(self.layer1(x1)))
        out1 = self.dropout(fctnl.relu(self.layer2(out1)))
        out1 = torch.squeeze(self.layer3(out1))
        return out0, out1

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Formnat data
        [Y, x0, x1] = self.format_input(train_batch)
        # Forward pass
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        # Loss
        loss = self.obj(Y_hat0, Y_hat1, train_batch)
        # Logging
        self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Formnat data
        [Y, x0, x1] = self.format_input(train_batch)
        # Forward pass
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        # Loss
        loss = self.obj(Y_hat0, Y_hat1, train_batch)
        # Logging
        self.log('val_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def predict_cf(self, x_np, nr):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        x0 = torch.concat((torch.zeros(n, 1).type_as(X), X), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(X), X), dim=1)
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        if nr==1:
            return Y_hat1.detach().numpy()
        else:
            return Y_hat0.detach().numpy()

    def predict_ite(self, x_np):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        x0 = torch.concat((torch.zeros(n, 1).type_as(X), X), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(X), X), dim=1)
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        tau_hat = Y_hat1 - Y_hat0
        return tau_hat.detach().numpy()

