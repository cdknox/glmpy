import numpy as np
from scipy import stats

import glmpy.links


class Poisson:
    allowed_links = [
        glmpy.links.LogLink,
        glmpy.links.IdentityLink,
        glmpy.links.SqrtLink,
    ]

    def variance(self, mu):
        return mu

    def valid_mu(self, mu):
        return np.isfinite(mu) & (mu > 0)

    def deviance_residuals(self, y, mu, weight):
        output = np.zeros(y.shape[0])
        positive = weight > 0
        output[~positive] = (mu * weight)[~positive]
        output[positive] = (weight * (y * np.log(y / mu) - (y - mu)))[positive]
        output = 2 * output
        return output

    def aic(self, y, n, mu, weight, total_deviance):
        return -2 * np.sum(stats.poisson.logpmf(y, mu) * weight)

    def mu_start_from_y(self, y):
        return y + 0.1


class QuasiPoisson(Poisson):
    def aic(self, y, n, mu, weight, total_deviance):
        return np.NaN


class Gaussian:
    allowed_links = [
        glmpy.links.IdentityLink,
        glmpy.links.InverseLink,
        glmpy.links.LogLink,
    ]

    def variance(self, mu):
        return np.ones(mu.shape[0])

    def valid_mu(self, mu):
        return np.array([True] * mu.shape[0])

    def deviance_residuals(self, y, mu, weight):
        return weight * (y - mu) * (y - mu)

    def aic(self, y, n, mu, weight, total_deviance):
        return (
            n * (np.log(2 * np.pi * total_deviance / n) + 1)
            + 2
            - np.sum(np.log(weight))
        )

    def mu_start_from_y(self, y):
        return y


class Binomial:
    pass
