import warnings
from typing import Optional, Callable, Tuple, Dict
from operator import itemgetter

import numpy as np
from scipy import optimize
from sklearn import linear_model

from spatial_nets.locations import LocationsDataClass
from spatial_nets.base import Model
from spatial_nets.metrics import CPC
from spatial_nets.models.constraints import (
    UnconstrainedModel,
    ProductionConstrained,
    AttractionConstrained,
    DoublyConstrained,
)


# A couple of utility functions
first, second, third = itemgetter(0), itemgetter(1), itemgetter(2)
GetterType = Callable[[float], float]


def zero(*args):
    return 0


def kwargs_from_vec(
    x: np.ndarray,
    template: Dict[str, GetterType],
) -> Dict[str, float]:
    kwargs = dict()
    for k, v in template.items():
        kwargs[k] = v(x)
    return kwargs


class GravityModel(Model):
    def __init__(
        self,
        constraint: Optional[str] = None,
        coef: Tuple[float, float, float] = None,
        method: str = "nlls",
        use_log: bool = False,
        maxiters: int = 500,
        verbose: bool = False,
    ):
        super().__init__(constraint=constraint)
        self.verbose = verbose
        self.routine = None

        if method is None:
            if coef is None:
                raise ValueError(
                    "when `method` is set to None `coef` should be provided"
                )
            if len(coef) == 3:
                if not isinstance(coef, dict):
                    template = {"γ": first, "α": second, "β": third}
                    self.coef_ = kwargs_from_vec(coef, template)
                elif all(k in coef for k in ["γ", "α", "β"]):
                    self.coef_ = coef
                else:
                    raise ValueError("invalid keys for coefficients")

            else:
                raise ValueError("invalid number of coefficients")

        elif method not in ("nlls", "cpc", "linreg"):
            raise ValueError("invalid method")

        else:
            if method == "nlls":
                routine = getattr(self, "_nonlinear_leastsquares")
            elif method == "cpc":
                routine = getattr(self, "_max_cpc")
            else:  # "linreg"
                routine = getattr(self, "_linreg")

            self.routine = routine
            self.use_log = use_log

            # Auxiliary model for constraints that will during the call to fit
            if constraint is None:
                aux_constraint = UnconstrainedModel()
            elif constraint == "production":
                aux_constraint = ProductionConstrained()
            elif constraint == "attraction":
                aux_constraint = AttractionConstrained()
            else:  # doubly
                aux_constraint = DoublyConstrained(
                    maxiters=maxiters, verbose=verbose)

            self.aux_constraint = aux_constraint

    def transform(self, mat=None):
        """
        Compute the gravity law predictions.

        TODO: Mathematical formula goes here.

        """
        return self._gravity_matrix(**self.coef_)

    def _gravity_matrix(self, γ: float, α: float = 1.0, β: float = 1.0) -> np.ndarray:
        """
        Calculate the gravity matrix.

        Parameters
        ----------
        γ : float
            Decay exponent.
        α, β : float, optional
            Respetively the power of the origin and destination terms.

        Returns
        -------
        np.array
            NxN unconstrained gravity matrix.

        """
        m = self.production[:, np.newaxis]  # column vector
        n = self.attraction[:, np.newaxis]  # column vector

        with np.errstate(divide="ignore", invalid="raise"):
            # 0**(-γ) raises divide error, 0 * ∞ raises invalid error in multiply
            dmat_power = self.dmat ** (-γ)
            dmat_power[np.diag_indices(self.N)] = 0.0
            # note this doesn't fix problems outside diagonal, this is on purpose

            out = (m ** α) * (n.T ** β) * dmat_power
        return out

    def fit(self, data: LocationsDataClass):
        """
        Fit the model to the observations and compute the model parameters.

        This method sets the `coef_` attribute and overwrites any data already
        stored there.

        wARNING: this method overrides the parameters already stored under `_coef`

        Parameters
        ----------
        data : LocationsDataClass
            The custom object which we defined to store the data. Note that
            this object needs to have its `flow_data` attribute set.

        Returns
        -------
        self

        """
        super().fit(data)

        self.production = data.production
        self.attraction = data.attraction
        self.dmat = data.dmat

        if self.routine is not None:
            self.aux_constraint.fit(data)
            try:
                self.coef_ = self.routine(
                    constraint=self.constraint,
                    use_log=self.use_log,
                    verbose=self.verbose,
                )
            except AssertionError as e:
                warnings.warn(str(e))

        return self

    def _nonlinear_leastsquares(
        self,
        constraint: str,
        use_log=False,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Calibrate the gravity model using nonlinear least squares."""

        def cost_fun(x, template, y, idx, use_log):
            kwargs = kwargs_from_vec(x, template)
            fmat = self._gravity_matrix(**kwargs)
            predict = self.aux_constraint.transform(fmat)
            diff = np.log(
                y) - np.log(predict[idx]) if use_log else y - predict[idx]
            return diff

        if constraint is None:
            # to find: gamma, alpha, beta
            x0 = [2, 1, 1]
            bounds = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])
            template_args = {"γ": first, "α": second, "β": third}

        elif constraint == "production":
            # to find: gamma, beta
            x0 = [1, 1]
            bounds = ([-np.inf, 0], [np.inf, np.inf])
            template_args = {"γ": first, "α": zero, "β": second}

        elif constraint == "attraction":
            # to find: gamma, alpha
            x0 = [1, 1]
            bounds = ([-np.inf, 0], [np.inf, np.inf])
            template_args = {"γ": first, "α": second, "β": zero}

        elif constraint == "doubly":
            # to find: gamma
            x0 = [2]
            bounds = (-np.inf, np.inf)
            template_args = {"γ": first, "α": zero, "β": zero}

        # The observations
        idx = self.flow_data.nonzero()
        y = np.asarray(self.flow_data[idx]).flatten()

        res = optimize.least_squares(
            cost_fun, x0, bounds=bounds, args=(template_args, y, idx, use_log)
        )
        st = res.status
        assert st > 0, "optimization routine failed"

        st_dict = {
            -1: "improper input parameters status returned from MINPACK.",
            0: "the maximum number of function evaluations is exceeded.",
            1: "gtol termination condition is satisfied.",
            2: "ftol termination condition is satisfied.",
            3: "xtol termination condition is satisfied.",
            4: "Both ftol and xtol termination conditions are satisfied.",
        }

        if verbose:
            print("Status: ", st_dict[st])

        return kwargs_from_vec(res.x, template_args)

    def _max_cpc(
        self,
        constraint: str,
        verbose: bool = False,
        **kwargs,
    ) -> Dict[str, float]:
        """Calibrate the gravity model using CPC maximisation."""

        def cost_fun(x, template):
            kwargs = kwargs_from_vec(x, template)
            fmat = self._gravity_matrix(**kwargs)
            predict = self.aux_constraint.transform(fmat)
            return 1 - CPC(predict, self.flow_data)

        if constraint is None:
            x0 = [2, 1, 1]
            bounds = [(0, None)] * 3
            template_args = {"γ": first, "α": second, "β": third}

        elif constraint == "production":
            x0 = [1, 1]
            bounds = [(0, None)] * 2
            template_args = {"γ": first, "α": zero, "β": second}

        elif constraint == "attraction":
            x0 = [1, 1]
            bounds = [(0, None)] * 2
            template_args = {"γ": first, "α": second, "β": zero}

        elif constraint == "doubly":
            x0 = [2]
            bounds = [(0, None)]
            template_args = {"γ": first, "α": zero, "β": zero}

        res = optimize.minimize(
            cost_fun, x0, bounds=bounds, args=(template_args))
        assert res.success, "optimization routine failed"

        if verbose:
            print("Status: ", res.message)

        return kwargs_from_vec(res.x, template_args)

    def _linreg(self, verbose: bool = False, **kwargs) -> Dict[str, float]:
        """Calibrate the gravity parameters using linear least squares regression."""

        # The observations
        i, j = self.flow_data.nonzero()
        y = np.asarray(self.flow_data[i, j]).flatten()

        # The model data
        m = self.target_rows_[:, np.newaxis]  # col. vec.
        n = self.target_cols_[np.newaxis, :]  # row vec.

        m_repeated = np.repeat(m, self.N, axis=1)
        n_repeated = np.repeat(n, self.N, axis=0)

        X = np.vstack((m_repeated[i, j], n_repeated[i, j], self.dmat[i, j])).T
        logX = np.log(X)

        reg = linear_model.LinearRegression()
        reg.fit(logX, np.log(y))
        #  not used: reg.intercept_
        coefs = reg.coef_ * [1, 1, -1]

        return kwargs_from_vec(coefs, {"γ": third, "α": first, "β": second})
