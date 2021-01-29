import glmpy.links
import numpy as np
from scipy import stats


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



