import math
import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as fctnl
import pytorch_lightning as pl
import numpy as np


class OAdam(Optimizer):
    r"""Implements optimistic Adam algorithm.
    It has been proposed in `Training GANs with Optimism`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Training GANs with Optimism:
        https://arxiv.org/abs/1711.00141
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(OAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Optimistic update :)
                p.data.addcdiv_(tensor1=exp_avg, tensor2=exp_avg_sq.sqrt().add(group['eps']), value=step_size)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(alpha=1 - beta1, other=grad)
                exp_avg_sq.mul_(beta2).addcmul_(value=1 - beta2, tensor1=grad, tensor2=grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(tensor1=exp_avg, tensor2=denom, value=-2.0 * step_size)

        return loss


class DeepGMM(pl.LightningModule):
    def __init__(self, config, xdim):
        super().__init__()
        self.dropout = nn.Dropout(config["dropout"])

        # Neural networks
        self.f = nn.Sequential(
            nn.Linear(xdim + 1, config["hidden_size_f"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_f"], config["hidden_size_f"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_f"], 1)
        )

        self.g = nn.Sequential(
            nn.Linear(xdim + 1, config["hidden_size_f"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_f"], config["hidden_size_f"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_f"], 1)
        )

        self.automatic_optimization = False
        self.optimizer_g = OAdam(self.g.parameters(), lr=config["lr_g"])
        self.optimizer_f = OAdam(self.f.parameters(), lr=config["lr_g"] * config["lambda_f"])

    def configure_optimizers(self):
        return self.optimizer_g, self.optimizer_f

    @staticmethod
    def format_input(data):
        Y = data[:, 0]
        A = data[:, 1]
        Z = data[:, 2]
        X = data[:, 3:]
        return Y, A, Z, X

    @staticmethod
    def obj_f(f_hat, g_hat, Y):
        batch_size = Y.size(0)
        sum1 = torch.mean(f_hat * (Y - g_hat))
        sum2 = (1 / (4 * batch_size)) * torch.sum(torch.square(f_hat) * torch.square(Y - g_hat))
        return sum2 - sum1

    @staticmethod
    def obj_g(f_hat, g_hat, Y):
        return torch.mean(f_hat * (Y - g_hat))

    def forward_f(self, x, z):
        outf = torch.squeeze(self.f(torch.concat((torch.unsqueeze(z, 1), x), 1)))
        return outf

    def forward_g(self, x, a):
        outg = torch.squeeze(self.g(torch.concat((torch.unsqueeze(a, 1), x), 1)))
        return outg

    def training_step(self, train_batch, batch_idx):
        self.train()
        input = DeepGMM.format_input(train_batch)
        f_hat = self.forward_f(input[3], input[2])
        g_hat = self.forward_g(input[3], input[1])
        # Optimize f
        loss_f = DeepGMM.obj_f(f_hat, g_hat.detach(), input[0])
        self.optimizer_f.zero_grad()
        self.manual_backward(loss_f)
        self.optimizer_f.step()
        # Optimize g
        loss_g = DeepGMM.obj_g(f_hat.detach(), g_hat, input[0])
        self.optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        self.optimizer_g.step()

        # Logging
        self.log('loss_f', loss_f.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('loss_g', loss_g.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)

    def predict_cf(self, x_np, nr):
        self.eval()
        n = x_np.shape[0]
        x = torch.from_numpy(x_np.astype(np.float32))
        if isinstance(nr, int):
            a = torch.squeeze(torch.full((n, 1), nr))
        else:
            a = torch.from_numpy(nr.astype(np.float32))
        y_hat = self.forward_g(x, a)
        return y_hat.detach().numpy()

    def predict_ite(self, x_np):
        y_hat0 = self.predict_cf(x_np, 0)
        y_hat1 = self.predict_cf(x_np, 1)
        tau_hat = y_hat1 - y_hat0
        return tau_hat
