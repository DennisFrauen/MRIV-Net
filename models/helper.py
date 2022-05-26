from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fctnl
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger


def rmse(y_hat, y, scaler=1):
    return np.sqrt(np.mean(((y - (y_hat*scaler)) ** 2)))


def create_loaders(data, batch_size, validation=True):
    if validation:
        d_train, d_val = train_test_split(data, test_size=0.1, shuffle=False)
        d_train = torch.from_numpy(d_train.astype(np.float32))
        d_val = torch.from_numpy(d_val.astype(np.float32))
    else:
        d_train = torch.from_numpy(data.astype(np.float32))

    train_loader = DataLoader(dataset=d_train, batch_size=batch_size, shuffle=False)
    if validation:
        val_loader = DataLoader(dataset=d_val, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    return train_loader, val_loader


def train_nn(data, config, model_class, epochs=100, validation=True, logging=True, **kwargs):
    # Data
    train_loader, val_loader = create_loaders(data, config["batch_size"], validation=validation)
    # Model
    model = model_class(config=config, **kwargs)
    # Check for available GPUs
    if torch.cuda.is_available():
        gpu = -1
    else:
        gpu = 0
    # Train
    if logging:
        neptune_logger = NeptuneLogger(project='dennisfrauen/ite-noncomplience')
        Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, gpus=gpu,
                              enable_model_summary=False, logger=neptune_logger)
    else:
        Trainer1 = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False, gpus=gpu,
                              logger=False, enable_checkpointing=False)

    if validation:
        Trainer1.fit(model, train_loader, val_loader)
        # Validation error after training
        val_results = Trainer1.validate(model=model, dataloaders=val_loader, verbose=False)
        val_err = val_results[0]['val_loss']
    else:
        Trainer1.fit(model, train_loader)
        val_err = None

    return model, val_err


def split_data(data):
    Y = data[:, 0]
    A = data[:, 1]
    Z = data[:, 2]
    X = data[:, 3:]
    return Y, A, Z, X


# Feed forward neural network, either binary or continuous output
class ffnn(pl.LightningModule):
    def __init__(self, config, input_size, output_type, weights=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.layer1 = nn.Linear(input_size, config["hidden_size"])
        self.layer2 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.layer3 = nn.Linear(config["hidden_size"], 1)
        self.dropout = nn.Dropout(config["dropout"])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.output_type = output_type
        if weights is None:
            self.weights = 1
        else:
            self.weights = torch.from_numpy(weights.astype(np.float32))

    def configure_optimizers(self):
        return self.optimizer

    def format_input(self, batch_torch):
        Y = batch_torch[:, 0]
        X = batch_torch[:, 1:]
        return [Y, X]

    def obj(self, y_hat, y):
        if self.output_type == "continuous":
            loss_y = torch.mean(((y_hat - y) * self.weights) ** 2)
        else:
            loss_y = fctnl.binary_cross_entropy(y_hat, y, reduction='mean')
        return loss_y

    def forward(self, x):
        out = self.dropout(fctnl.relu(self.layer1(x)))
        out = self.dropout(fctnl.relu(self.layer2(out)))
        out = torch.squeeze(self.layer3(out))
        if self.output_type == "binary":
            out = torch.sigmoid(out)
        return out

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Formnat data
        [Y, X] = self.format_input(train_batch)
        # Forward pass
        y_hat = self.forward(X)
        # Loss
        loss = self.obj(y_hat, Y)
        # Logging
        self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Formnat data
        [Y, X] = self.format_input(train_batch)
        # Forward pass
        y_hat = self.forward(X)
        # Loss
        loss = self.obj(y_hat, Y)
        # Logging
        self.log('val_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def predict(self, x_np):
        self.eval()
        X = torch.from_numpy(x_np.astype(np.float32))
        tau_hat = self.forward(X)
        return tau_hat.detach().numpy()

    def validation_mse(self, d_val):
        y_hat = self.predict(d_val[:, 1:])
        return np.mean((y_hat - d_val[:, 0])**2)


def train_base_model(model_name, d_train, params=None, validation=False, logging=False):
    import models.linear_methods as linear_iv
    import models.deep_iv as deepiv
    import models.dml_dr_iv as dml
    import models.df_iv as dfiv
    import models.deepgmm as deepgmm
    import models.standard_ite as standard_ite
    import models.bcf_iv as bcfiv
    import models.kiv as kiv
    import models.mr_learner as mr

    model = None
    if model_name == "ncnet":
        model, _ = train_nn(data=d_train, config=params, model_class=mr.ncnet,
                                   input_size=d_train.shape[1] - 3, validation=validation, logging=logging)
    if model_name == "dmliv":
        model = dml.train_dmliv(data=d_train, config=params, validation=validation, logging=logging)
    if model_name == "tsls":
        model = linear_iv.train_twosls(d_train)
    if model_name == "waldlinear":
        model = linear_iv.train_Wald_linear(d_train)
    if model_name == "deepiv":
        model = deepiv.train_DeepIV(d_train, params, validation=validation, logging=logging)
    if model_name == "dfiv":
        model = dfiv.train_dfiv(d_train, params, epochs=200, logging=logging)
    if model_name == "deepgmm":
        model, _ = train_nn(data=d_train, config=params, model_class=deepgmm.DeepGMM, epochs=200,
                                   xdim=d_train.shape[1] - 3, validation=validation, logging=logging)
    if model_name == "kiv":
        model = kiv.train_kiv(data=d_train, config=params)
    if model_name == "tarnet":
        model, _ = train_nn(data=np.delete(d_train, 2, 1), config=params, model_class=standard_ite.TARNet,
                                   input_size=d_train.shape[1] - 3, validation=validation, logging=logging)
    if model_name == "bcfiv":
        model = bcfiv.train_bcf_iv(d_train, params)
    return model
