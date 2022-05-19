import hyperparam.parameter_sampling as param_sampling
import models.helper as helper
import numpy as np
import misc
import torch
from sklearn.model_selection import train_test_split
import models.kiv as kiv
import models.mr_learner as mr
import models.dml_dr_iv as dml
import models.standard_ite as standard_ite
import models.deep_iv as deepiv


def train_model(trial, d_train, model, dataset="sim"):
    config = param_sampling.sample_hyper(model, trial, d_train.shape[1] - 3, dataset=dataset)
    model = helper.train_base_model(model_name=model, d_train=d_train, params=config, validation=False,
                                    logging=False)
    return model


def get_objective_ncnet(d_train, d_val, dataset="sim"):
    def obj(trial):
        model = train_model(trial, d_train, "ncnet", dataset=dataset)
        # Validation error
        [mu_Y, mu_A1, mu_A2, pi] = model.predict_components(d_val[:, 3:],
                                                            torch.from_numpy(d_val[:, 2].astype(np.float32)))
        loss_y = np.mean((mu_Y - d_val[:, 0]) ** 2)
        loss_a1 = misc.binary_cross_entropy(mu_A1, d_val[:, 1])
        loss_a2 = misc.binary_cross_entropy(mu_A2, d_val[:, 1])
        loss_pi = misc.binary_cross_entropy(pi, d_val[:, 2])
        return loss_y + loss_a1 + loss_a2 + loss_pi

    return obj


def get_objective_tarnet(d_train, d_val, dataset="sim"):
    def obj(trial):
        model = train_model(trial, d_train, "tarnet", dataset=dataset)
        # Validation error
        y_hat = model.predict_cf(d_val[:, 3:], d_val[:, 1])
        return np.mean((y_hat - d_val[:, 0]) ** 2)

    return obj


def get_objective_dmliv_nuisance(d_train, d_val, dataset="sim"):
    def obj(trial):
        x_dim = d_train.shape[1] - 3
        config = param_sampling.sample_hyper("dmliv", trial, d_train.shape[1] - 3, dataset=dataset)
        config_dml, config_nuisance = dml.change_config_keys_dml(config)
        Y_train, A_train, Z_train, X_train = helper.split_data(d_train)
        # Train nuisance models
        data_yx_train = np.concatenate((np.expand_dims(Y_train, 1), X_train), 1)
        data_azx_train = np.concatenate((np.expand_dims(A_train, 1), np.expand_dims(Z_train, 1), X_train), 1)
        data_ax_train = np.concatenate((np.expand_dims(A_train, 1), X_train), 1)
        model_yx, _ = helper.train_nn(data=data_yx_train, config=config_nuisance, model_class=helper.ffnn,
                                      input_size=x_dim,
                                      validation=False, logging=False, output_type="continuous")
        model_azx, _ = helper.train_nn(data=data_azx_train, config=config_nuisance, model_class=helper.ffnn,
                                       input_size=x_dim + 1,
                                       validation=False, logging=False, output_type="binary")
        model_ax, _ = helper.train_nn(data=data_ax_train, config=config_nuisance, model_class=helper.ffnn,
                                      input_size=x_dim,
                                      validation=False, logging=False, output_type="binary")
        # Predictions
        q = model_yx.predict(d_val[:, 3:])
        h = model_azx.predict(d_val[:, 2:])
        p = model_ax.predict(d_val[:, 3:])

        # Errors
        err_q = np.mean((q - d_val[:, 0]) ** 2)
        err_h = misc.binary_cross_entropy(h, d_val[:, 1])
        err_p = misc.binary_cross_entropy(p, d_val[:, 1])
        return err_q + err_h + err_p

    return obj


def get_objective_dmliv(d_train, d_val, config_nuisance, dataset="sim"):
    def obj(trial):
        config = param_sampling.sample_hyper("dmliv", trial, d_train.shape[1] - 3, dataset=dataset)
        config["lr_nuisance"] = config_nuisance["lr_nuisance"]
        config["batch_size_nuisance"] = config_nuisance["batch_size_nuisance"]
        config["hidden_size_nuisance"] = config_nuisance["hidden_size_nuisance"]
        config["dropout_nuisance"] = config_nuisance["dropout_nuisance"]
        model = helper.train_base_model(model_name="dmliv", d_train=d_train, params=config, validation=False,
                                        logging=False)
        # Validation error
        d_val_torch = torch.from_numpy(d_val.astype(np.float32))
        q = model.model_yx.forward(d_val_torch[:, 3:]).detach().numpy()
        h = model.model_azx.forward(
            torch.concat((torch.unsqueeze(d_val_torch[:, 2], 1), d_val_torch[:, 3:]), 1)).detach().numpy()
        p = model.model_ax.forward(d_val_torch[:, 3:]).detach().numpy()
        y_hat = model.forward(d_val_torch[:, 3:]).detach().numpy()
        loss = np.mean(((d_val[:, 0] - q - y_hat) * (h - p)) ** 2)
        return loss

    return obj


def get_objective_deepiv1(d_train, d_val, dataset="sim"):
    def obj(trial):
        config = param_sampling.sample_hyper("deepiv", trial, d_train.shape[1] - 3, dataset=dataset)
        model = deepiv.train_first_stage(d_train, config, validation=False, logging=False)
        # Validation error
        pi_hat = model.predict(d_val[:, 2:])
        # Return oos deviance critereon
        return -1 * np.sum(np.log(pi_hat * d_val[:, 1] + (1 - pi_hat) * (1 - d_val[:, 1])))

    return obj


def get_objective_deepiv2(d_train, d_val, config1, dataset="sim"):
    def obj(trial):
        config = param_sampling.sample_hyper("deepiv", trial, d_train.shape[1] - 3, dataset=dataset)
        config["lr"] = config1["lr"]
        config["batch_size"] = config1["batch_size"]
        config["hidden_size"] = config1["hidden_size"]
        config["dropout"] = config1["dropout"]
        model = helper.train_base_model(model_name="deepiv", d_train=d_train, params=config, validation=False,
                                        logging=False)
        # Validation error
        d_val_torch = torch.from_numpy(d_val.astype(np.float32))
        n = d_val_torch.size(0)
        x0 = torch.concat((torch.zeros(n, 1).type_as(d_val_torch), d_val_torch[:, 3:]), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(d_val_torch), d_val_torch[:, 3:]), dim=1)
        y_hat0, y_hat1 = model.forward(x0, x1)
        pi = model.first_stage_nn.forward(torch.from_numpy(d_val[:, 2:].astype(np.float32)))
        loss = torch.mean((d_val_torch[:, 0] - (1 - pi) * y_hat0 - pi * y_hat1) ** 2)
        return loss.detach().numpy()

    return obj


def get_objective_deepgmm(d_train, d_val, dataset="sim"):
    def obj(trial):
        model = train_model(trial, d_train, "deepgmm", dataset=dataset)
        # Validation error
        y_hat = model.predict_cf(d_val[:, 3:], d_val[:, 1])
        return np.mean((y_hat - d_val[:, 0]) ** 2)

    return obj


def get_objective_dfiv_1(data, dataset="sim"):
    def obj(trial):
        d_first, d_second = train_test_split(data, test_size=0.5, shuffle=False)
        d_second_torch = torch.from_numpy(d_second.astype(np.float32))
        model = train_model(trial, data, "dfiv", dataset=dataset)
        # Validation error
        repr_psi1 = model.psi(d_second_torch[:, 2:])
        repr_phi1 = model.phi(d_second_torch[:, 1:2])
        V_hat1 = model.compute_V(phi=repr_phi1, psi=repr_psi1)
        loss1 = model.loss_1(V_hat=V_hat1, phi=repr_phi1.detach(), psi=repr_psi1).detach().numpy()
        return loss1

    return obj


def get_objective_dfiv_2(data, config1, dataset="sim"):
    def obj(trial):
        d_first, d_second = train_test_split(data, test_size=0.5, shuffle=False)
        d_first_torch = torch.from_numpy(d_first.astype(np.float32))
        config = param_sampling.sample_hyper("dfiv", trial, data.shape[1] - 3, dataset=dataset)
        config["lr1"] = config1["lr1"]
        config["lambda1"] = config1["lambda1"]
        config["hidden_size_psi"] = config1["hidden_size_psi"]
        config["hidden_size_phi"] = config1["hidden_size_phi"]
        config["dropout"] = config1["dropout"]
        config["batch_size"] = config1["batch_size"]
        model = helper.train_base_model(model_name="dfiv", d_train=data, params=config, validation=False,
                                        logging=False)
        # Validation error
        repr_psi2 = model.psi(d_first_torch[:, 2:])
        repr_phi2 = model.phi(d_first_torch[:, 1:2])
        repr_xi2 = model.xi(d_first_torch[:, 3:])
        V_hat2 = model.compute_V(phi=repr_phi2, psi=repr_psi2)
        mu_hat2 = model.compute_mu(V=V_hat2, psi=repr_psi2, xi=repr_xi2, Y=d_first_torch[:, 0])
        loss2 = model.loss_2(mu_hat=mu_hat2, V_hat=V_hat2, psi=repr_psi2, xi=repr_xi2.detach(), Y=d_first_torch[:, 0])
        return loss2.detach().numpy()

    return obj


def get_objective_kiv_1(data, dataset="sim"):
    def obj(trial):
        # Sample lambda
        config = param_sampling.sample_hyper_kiv(trial, dataset=dataset)
        lamb = config["lambda"]
        # Validation error
        d_first, d_second = train_test_split(data, test_size=0.5, shuffle=False)
        n = d_first.shape[0]
        m = d_second.shape[0]
        z_1 = d_first[:, 2:]
        z_2 = d_second[:, 2:]
        a_1 = np.concatenate((d_first[:, 1:2], d_first[:, 3:]), 1)
        a_2 = np.concatenate((d_second[:, 1:2], d_second[:, 3:]), 1)
        kzz = kiv.KIV.exp_kernel(z_1, z_1)
        kzz_bar = kiv.KIV.exp_kernel(z_1, z_2)
        gamma = np.matmul(np.linalg.inv(kzz + n * lamb * np.identity(n)), kzz_bar)

        kxx = kiv.KIV.exp_kernel(a_1, a_1)
        kxbarx = kiv.KIV.exp_kernel(a_2, a_1)
        kxbarxbar = kiv.KIV.exp_kernel(a_2, a_2)
        l1 = kxbarxbar - 2 * np.matmul(kxbarx, gamma) + np.matmul(np.matmul(np.transpose(gamma), kxx), gamma)
        l1 = (1 / m) * np.trace(l1)
        return l1

    return obj


def get_objective_kiv_2(data, lamb, dataset="sim"):
    def obj(trial):
        config = param_sampling.sample_hyper_kiv(trial, dataset=dataset)
        config["lambda"] = lamb
        model = helper.train_base_model(model_name="kiv", d_train=data, params=config, validation=False,
                                        logging=False)
        # Validation error
        d_first, d_second = train_test_split(data, test_size=0.5, shuffle=False)
        h_hat = model.predict_cf(d_first[:, 3:], d_first[:, 1])
        return np.mean((d_first[:, 0] - h_hat) ** 2)

    return obj


def get_objective_bcfiv(d_train, d_val, dataset="sim"):
    def obj(trial):
        model = train_model(trial, d_train, "bcfiv", dataset=dataset)
        # Validation error
        y_hat = model.predict_cf(d_val[:, 3:], d_val[:, 1])
        return np.mean((y_hat - d_val[:, 0]) ** 2)

    return obj


# Meta learner---------------------------------------------------------------
def get_objective_mriv_nuisance(d_train, d_val, dataset="sim"):
    def obj(trial):
        p = d_train.shape[1] - 3
        config = param_sampling.sample_hyper_mr_nuisance(trial, d_train.shape[1] - 3, dataset=dataset)
        model_yzx, _ = helper.train_nn(data=np.delete(d_train, 1, 1), config=config, model_class=standard_ite.TARNet,
                                       input_size=p,
                                       validation=False, logging=False, output_type="continuous", learn_pi=False)
        model_azx, _ = helper.train_nn(data=d_train[:, 1:], config=config, model_class=standard_ite.TARNet,
                                       input_size=p,
                                       validation=False, logging=False, output_type="binary")
        y_hat = model_yzx.predict_cf(d_val[:, 3:], d_val[:, 2])
        a_hat = model_azx.predict_cf(d_val[:, 3:], d_val[:, 2])
        pi_hat = model_azx.predict_pi(d_val[:, 3:])
        return np.mean((y_hat - d_val[:, 0]) ** 2) + misc.binary_cross_entropy(a_hat,
                                                                               d_val[:, 1]) + misc.binary_cross_entropy(
            pi_hat, d_val[:, 1])

    return obj


def get_objective_mriv(d_train, d_val, base_model, base_model_name, nuisance_train=None, nuisance_val=None,
                       mrsingle=False, dataset="sim"):
    def obj(trial):
        config = param_sampling.sample_hyper_mr(trial, d_train.shape[1] - 3, dataset=dataset)

        if base_model_name == "ncnet":
            if not mrsingle:
                mr_input_train = base_model.predict_mr_input(d_train[:, 3:])
                mr_input_val = base_model.predict_mr_input(d_val[:, 3:])
            else:
                mr_input_train = base_model.predict_mr_input_single(d_train[:, 3:])
                mr_input_val = base_model.predict_mr_input(d_val[:, 3:])
        else:
            tau_train = base_model.predict_ite(d_train[:, 3:])
            tau_val = base_model.predict_ite(d_val[:, 3:])
            [mu_0Y_train, mu_0A_train, delta_A_train, pi_train] = nuisance_train
            mr_input_train = [pi_train, mu_0A_train, mu_0Y_train, delta_A_train, tau_train]
            [mu_0Y_val, mu_0A_val, delta_A_val, pi_val] = nuisance_val
            mr_input_val = [pi_val, mu_0A_val, mu_0Y_val, delta_A_val, tau_val]
        # Train MRIV
        meta_learner = mr.train_mr_learner(data=d_train, init_estimates=mr_input_train, config=config,
                                           validation=False, logging=False)
        # Train validation pseudo outcomes
        y0_val = mr.create_pseudo_outcomes(z=d_val[:, 2], a=d_val[:, 1], y=d_val[:, 0], pi=mr_input_val[0],
                                           mu_A=mr_input_val[1], mu_Y=mr_input_val[2], delta_A=mr_input_val[3],
                                           tau=mr_input_val[4])
        y0_hat = meta_learner.predict(d_val[:, 3:])
        return np.mean((y0_hat - y0_val) ** 2)
    return obj


def get_objective_driv_nuisance(d_train, d_val, dataset="sim"):
    def obj(trial):
        config = param_sampling.sample_hyper_driv_nuisance(trial, d_train.shape[1] - 3, dataset=dataset)
        _, [q, p, r, f] = dml.get_nuisance_full(d_train, config, d_val)
        # Errors
        err_q = np.mean((q - d_val[:, 0]) ** 2)
        err_p = misc.binary_cross_entropy(p, d_val[:, 1])
        err_r = misc.binary_cross_entropy(r, d_val[:, 2])
        err_f = misc.binary_cross_entropy(f, d_val[:, 1] * d_val[:, 2])
        return err_q + err_p + err_r + err_f

    return obj


def get_objective_driv(d_train, d_val, base_model, base_model_name, nuisance_train=None, nuisance_val=None, dataset="sim"):
    def obj(trial):
        config = param_sampling.sample_hyper_driv(trial, d_train.shape[1] - 3, dataset=dataset)


        if base_model_name == "dmliv":
            [tau_train, q_train, p_train] = base_model.predict_driv_input(d_train[:, 3:])
            [tau_val, q_val, p_val] = base_model.predict_driv_input(d_val[:, 3:])
            [_, _, r_train, f_train] = nuisance_train
            [_, _, r_val, f_val] = nuisance_val
            driv_input_train = [tau_train, q_train, p_train, r_train, f_train]
            driv_input_val = [tau_val, q_val, p_val, r_val, f_val]
        else:
            tau_train = base_model.predict_ite(d_train[:, 3:])
            tau_val = base_model.predict_ite(d_val[:, 3:])
            [q_train, p_train, r_train, f_train] = nuisance_train
            [q_val, p_val, r_val, f_val] = nuisance_val
            driv_input_train = [tau_train, q_train, p_train, r_train, f_train]
            driv_input_val = [tau_val, q_val, p_val, r_val, f_val]
        # Train DRIV
        meta_learner = dml.train_driv(data=d_train, init_estimates=driv_input_train, config=config,
                                      validation=False, logging=False)
        # Create validation pseudo outcomes
        y0_val = dml.create_pseudo_outcomes(z=d_val[:, 2], a=d_val[:, 1], y=d_val[:, 0], tau=driv_input_val[0],
                                            q=driv_input_val[1], p=driv_input_val[2], r=driv_input_val[3],
                                            f=driv_input_val[4])
        y0_hat = meta_learner.predict(d_val[:, 3:])
        return np.mean((y0_hat - y0_val) ** 2)
    return obj


def get_objective_dr(d_train, d_val, base_model, dataset="sim"):
    def obj(trial):
        config = param_sampling.sample_hyper_dr(trial, d_train.shape[1] - 3, dataset=dataset)
        # Nuisance parameters
        mu_1_train = base_model.predict_cf(d_train[:, 3:], 1)
        mu_1_val = base_model.predict_cf(d_val[:, 3:], 1)
        mu_0_train = base_model.predict_cf(d_train[:, 3:], 0)
        mu_0_val = base_model.predict_cf(d_val[:, 3:], 0)
        pi_train = base_model.predict_pi(d_train[:, 3:])
        pi_val = base_model.predict_pi(d_val[:, 3:])
        print(f"Train DRIV learner")
        meta_learner = standard_ite.train_dr_learner(data=np.delete(d_train, 2, 1), config=config,
                                                     init_estimates=[pi_train, mu_1_train, mu_0_train],
                                                     validation=False, logging=False)
        # Create validation pseudo outcomes
        y0_val = standard_ite.create_pseudo_outcomes(a=d_val[:, 1], y=d_val[:, 0], pi=pi_val,
                                                     mu_1=mu_1_val, mu_0=mu_0_val)
        y0_hat = meta_learner.predict(d_val[:, 3:])
        return np.mean((y0_hat - y0_val) ** 2)

    return obj
