import numpy as np
from sklearn.gaussian_process.kernels import Matern
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import norm


def simulate_data(n, p=1, sigma_U=0.05, alpha_U=1, sigma_A=1, sigma_Y=1, plot=True, scale=True,
                  beta=0.5, gamma=50, delta=50):

    # Observed confounders
    oversupply = 3000
    X = np.random.normal(size=(n + oversupply, p))
    # Check for too small distances (lead to nan in matern kernel)
    dist = euclidean_distances(X, X)
    dist += np.identity(n + oversupply)
    index = np.argwhere(dist < 0.0001)
    index = np.sort(index, axis=1)
    index = np.unique(index[:, 0])

    leftover = oversupply - len(index)
    assert leftover > 0

    X = np.delete(X, index, axis=0)
    X = X[:n, :]

    delta_Y = sample_gp_prior(X, nu=gamma)
    mu_Y0 = sample_gp_prior(X, nu=beta)
    f1 = sample_gp_prior(X, nu=delta)
    f0 = sample_gp_prior(X, nu=delta)
    g = sample_gp_prior(X, nu=delta)

    mu_A1 = 0.3 * expit(f1) + 0.7
    mu_A0 = 0.3 * expit(f0)
    delta_A = mu_A1 - mu_A0
    mu_Y1 = delta_Y + mu_Y0
    tau = delta_Y / delta_A
    pi = 0.8*expit(g) + 0.1



    # Check for correct support
    assert (delta_A != 0).all()
    assert (0.7 <= mu_A1).all()
    assert (mu_A1 <= 1).all()
    assert (0 <= mu_A0).all()
    assert (mu_A0 <= 0.3).all()
    assert np.all(tau - ((mu_Y1 - mu_Y0) / (mu_A1 - mu_A0)) < 0.000001)

    # Create dataset------------------------------------------------------------
    #Unobserved confounders
    U = np.random.normal(loc=0, scale=sigma_U, size=n)
    #Instruments
    Z = np.random.binomial(1, pi)
    #Treatments
    epsilon_A = np.random.normal(size=n, scale=sigma_A)
    #Calculate quantiles
    alpha_1 = norm.ppf((1 - mu_A1)) * np.sqrt((sigma_A**2) + (sigma_U**2))
    alpha_0 = norm.ppf((1 - mu_A0)) * np.sqrt((sigma_A**2) + (sigma_U**2))
    A = Z * np.where(epsilon_A + U > alpha_1, 1, 0) + (1 - Z) * np.where(epsilon_A + U > alpha_0, 1, 0)

    # Outcomes
    epsilon_Y = np.random.normal(size=n, scale=sigma_Y)
    Y = A * (((mu_A1 - 1) * mu_Y0 - mu_A0 * mu_Y1 + mu_Y1) / delta_A) + (1 - A) * (
            (mu_A1 * mu_Y0 - mu_A0 * mu_Y1) / delta_A) + alpha_U*U + epsilon_Y

    # Counterfactuals
    A_cf = 1 - A
    Y_cf = A_cf * (((mu_A1 - 1) * mu_Y0 - mu_A0 * mu_Y1 + mu_Y1) / delta_A) + (1 - A_cf) * (
            (mu_A1 * mu_Y0 - mu_A0 * mu_Y1) / delta_A) + alpha_U*U + epsilon_Y

    # Scale outcomes / ITE
    sigma = 1
    if scale == True:
        sigma = np.std(Y)
    Y = Y / sigma
    Y_cf = Y_cf / sigma
    #tau = tau / sigma
    #mu_Y1 = mu_Y1 / sigma
    #mu_Y0 = mu_Y0 / sigma

    # Create data matrix
    data = np.concatenate((np.expand_dims(Y, 1), np.expand_dims(A, 1), np.expand_dims(Z, 1), X), axis=1)
    comp = [tau, mu_Y1, mu_Y0, mu_A1, mu_A0, pi, Y_cf]
    if plot:
        plot_sim(data, comp)

    # Checks-------------------------------------------------------------
    # Check instrument/ treatment proportions
    propZ = np.sum(Z) / n
    propA = np.sum(A) / n
    assert (0.1 < propZ < 0.9)
    assert (0.1 < propA < 0.9)

    return data, comp, sigma


def sample_gp_prior(X, nu=1.5, l=1.0):
    n = X.shape[0]
    kernel = Matern(length_scale=l, length_scale_bounds="fixed", nu=nu)
    sigma = kernel(X)
    #print(np.argwhere(np.isnan(sigma)))
    assert not np.isnan(sigma).any()
    y = np.squeeze(np.random.multivariate_normal(mean=np.zeros(n), cov=sigma, size=1, check_valid="raise"))
    return y


def train_test_split(data, comp, ratio=0.9):
    n = data.shape[0]
    train_ind = np.random.choice(np.arange(0, n), replace=False, size=int(n * ratio))
    test_ind = np.setdiff1d(np.arange(0, n), train_ind, assume_unique=True)
    d_train = data[train_ind, :]
    d_test = data[test_ind, :]

    c_train = []
    c_test = []

    for c in comp:
        c_train.append(c[train_ind])
        c_test.append(c[test_ind])
    return d_train, d_test, c_train, c_test


def plot_sim(data, comp):
    X = data[:, 3:]
    Y_1 = data[data[:, 2] == 1, 0]
    Y_0 = data[data[:, 2] == 0, 0]
    X_1 = data[data[:, 2] == 1, 3]
    X_0 = data[data[:, 2] == 0, 3]
    tau = comp[0]
    p = X.shape[1]
    mu_Y1 = comp[1]
    mu_Y0 = comp[2]
    mu_A1 = comp[3]
    mu_A0 = comp[4]
    data = np.concatenate((X, np.expand_dims(tau, axis=1), np.expand_dims(mu_Y1, axis=1), np.expand_dims(mu_Y0, axis=1),
                           np.expand_dims(mu_A1, axis=1), np.expand_dims(mu_A0, axis=1)), axis=1)
    data = data[data[:, 0].argsort()]
    plt.plot(X_1, Y_1, 'o', label="y1", alpha=0.08, color="mediumblue")
    plt.plot(X_0, Y_0, 'o', label="y0", alpha=0.08, color="orchid")
    plt.plot(data[:, 0], data[:, p+1], label="mu_Y1", color="mediumblue")
    plt.plot(data[:, 0], data[:, p+2], label="mu_Y0", color="orchid")
    plt.plot(data[:, 0], data[:, p+3], label="mu_A1", color="yellowgreen")
    plt.plot(data[:, 0], data[:, p+4], label="mu_A0", color="lime")
    plt.plot(data[:, 0], data[:, p], label="tau", color="red")
    plt.legend()
    plt.show()

# def standardize(data, tau):
