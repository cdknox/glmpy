import numpy as np
from scipy import linalg

import glmpy.families
import glmpy.links


class GLM:
    def __init__(
        self,
        family=glmpy.families.Poisson,
        link=glmpy.links.LogLink,
        add_intercept=True,
    ):
        self.family = family()
        self.link = link()
        self.add_intercept = add_intercept
        if link not in family.allowed_links:
            raise ValueError("Incompatible family and link combination")

    def fit(
        self,
        X,
        y,
        weights=None,
        offset=None,
        max_iter=25,
        epsilon=10 * np.finfo(np.float64).eps,
    ):

        family = self.family
        link = self.link

        # could create a new mu_start here from coefficients
        # coefficients_prior = None
        mu_start = family.mu_start_from_y(y)
        eta = link.link_function(mu_start)
        mu = link.link_inverse(eta)
        offset = offset if offset else np.zeros(y.shape[0])
        weights = weights if weights else np.ones(y.shape[0])

        all_valid_mu = np.all(family.valid_mu(mu))
        all_valid_eta = np.all(link.valid_eta(eta))
        if not all_valid_mu:
            # providing different starting parameters may help
            raise ValueError("Invalid means for the distribution")

        if not all_valid_eta:
            # providing different starting parameters may help
            raise ValueError("Invalid eta for the link")

        deviance_old = np.sum(family.deviance_residuals(y, mu, weights))
        boundary = False
        converged = False

        for i in range(max_iter):
            variances = family.variance(mu)
            d_mu_d_eta = link.d_mu_d_eta(eta)
            z = (eta - offset) + (y - mu) / d_mu_d_eta
            z = z.reshape(-1, 1)
            w = np.sqrt(weights * d_mu_d_eta * d_mu_d_eta / variances)
            w = w.reshape(-1, 1)
            Q, R, P = linalg.qr(X * w, pivoting=True)
            # TODO: handle finding how many parameters
            # to solve for after householder routine
            # moves problematic (colinear) columns out of the way
            # in a better manner
            to_keep = (np.abs(R).sum(axis=1) > 0.001).sum()
            params = linalg.solve_triangular(
                R[:to_keep], np.matmul(Q.T, z * w)[:to_keep]
            )
            params = params[np.argsort(P)]
            eta = np.matmul(X, params).reshape(-1)
            mu = link.link_inverse(eta + offset).reshape(-1)
            deviance = np.sum(family.deviance_residuals(y, mu, weights))
            if boundary or converged:
                # TODO: handle boundary conditions and divergence better
                # decrease step size if problematic
                continue
            relative_deviance_change = np.abs(deviance - deviance_old) / (
                0.1 + np.abs(deviance)
            )
            if relative_deviance_change < epsilon:
                converged = True
                break
            else:
                deviance_old = deviance

        if not converged:
            raise ValueError("failed to converge")

        self.P = P
        self.params = params.reshape(-1)
