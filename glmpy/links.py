import numpy as np
from scipy import stats

DBL_EPS = np.finfo(np.float64).eps


def array_of_x_shape_of_y(x, y):
    try:
        length = len(y)
    except Exception:
        length = 1
    return np.array([x] * length)


class LogitLink:
    def link_function(self, mu):
        return np.log(mu / (1 - mu))

    def link_inverse(self, eta):
        return np.exp(eta) / (1 + np.exp(eta))

    def d_mu_d_eta(self, eta):
        return np.exp(eta) / ((1 + np.exp(eta)) * (1 + np.exp(eta)))

    def valid_eta(self, eta):
        return array_of_x_shape_of_y(x=True, y=eta)


class ProbitLink:
    def link_function(self, mu):
        return stats.norm.ppf(mu)

    def link_inverse(self, eta):
        threshold = -stats.norm.ppf(DBL_EPS)
        eta = np.maximum(eta, -threshold)
        eta = np.minimum(eta, threshold)
        return stats.norm.cdf(eta)

    def d_mu_d_eta(self, eta):
        return np.maximum(stats.norm.pdf(eta), DBL_EPS)

    def valid_eta(self, eta):
        return array_of_x_shape_of_y(x=True, y=eta)


class CauchitLink:
    pass


class CLogLogLink:
    pass


class IdentityLink:
    def link_function(self, mu):
        return mu

    def link_inverse(self, eta):
        return eta

    def d_mu_d_eta(self, eta):
        return array_of_x_shape_of_y(x=1, y=eta)

    def valid_eta(self, eta):
        return array_of_x_shape_of_y(x=True, y=eta)


class LogLink:
    def link_function(self, mu):
        return np.log(mu)

    def link_inverse(self, eta):
        return np.exp(eta)

    def d_mu_d_eta(self, eta):
        return np.exp(eta)

    def valid_eta(self, eta):
        return array_of_x_shape_of_y(x=True, y=eta)


class SqrtLink:
    def link_function(self, mu):
        return np.sqrt(mu)

    def link_inverse(self, eta):
        return eta * eta

    def d_mu_d_eta(self, eta):
        return 2 * eta

    def valid_eta(self, eta):
        return np.isfinite(eta) & (eta > 0.0)


class OneOverMuSquaredLink:
    def link_function(self, mu):
        return 1 / (mu * mu)

    def link_inverse(self, eta):
        return 1 / np.sqrt(eta)

    def d_mu_d_eta(self, eta):
        return -1 / (2 * np.power(eta, 1.5))

    def valid_eta(self, eta):
        return np.isfinite(eta) & (eta > 0.0)


class InverseLink:
    def link_function(self, mu):
        return 1 / mu

    def link_inverse(self, eta):
        return 1 / eta

    def d_mu_d_eta(self, eta):
        return -1 / (eta * eta)

    def valid_eta(self, eta):
        return np.isfinite(eta) & (eta != 0.0)
