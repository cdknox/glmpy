import numpy as np
from scipy import linalg, stats

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
        epsilon=1e-8,
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

        self.rank = to_keep
        self.n_ok = X.shape[0] - np.sum(weights == 0)
        self.deviance = deviance
        self.P = P
        self.params = params.reshape(-1)
        self.residual_dof = self.n_ok - self.rank

        self.handle_dispersion_parameter(y, mu, weights)
        self.calculate_vcov(R, P)
        self.calculate_p_values()

    def handle_dispersion_parameter(self, y, mu, weights):
        estimated_dispersion = False

        no_dispersion_families = [
            glmpy.families.Poisson,
            glmpy.families.Binomial,
        ]
        family_in_no_dispersion = any(
            [isinstance(self.family, family) for family in no_dispersion_families]
        )
        if family_in_no_dispersion:
            dispersion = 1
        elif self.residual_dof:
            estimated_dispersion = True
            residuals = y - mu
            numerator = np.sum((weights * residuals * residuals)[weights > 0])
            dispersion = numerator / self.residual_dof
        else:
            estimated_dispersion = True
            dispersion = np.NaN

        self.estimated_dispersion = estimated_dispersion
        self.dispersion = dispersion

    def calculate_vcov(self, R, P):
        # TODO address when rank isn't full / aliasing happens
        undo = np.argsort(P)
        # rows/columns now also in order in which originally passed
        covmat_unscaled = np.linalg.inv(np.matmul(R.T, R))[undo, :][:, undo]
        covmat = self.dispersion * covmat_unscaled
        var_coef = np.diag(covmat)
        standard_error = np.sqrt(var_coef)

        self.covmat_unscaled = covmat_unscaled
        self.covmat = covmat
        self.var_coef = var_coef
        self.standard_error = standard_error

    def calculate_p_values(self):
        t_values = self.params / self.standard_error
        if not self.estimated_dispersion:
            p_values = 2 * stats.norm.cdf(-np.abs(t_values))
        elif self.residual_dof:
            p_values = 2 * stats.t.cdf(-np.abs(t_values), df=self.residual_dof)
        else:
            p_values = np.array([np.NaN] * len(self.params))
        self.p_values = p_values
        self.t_values = t_values
