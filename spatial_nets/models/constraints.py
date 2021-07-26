from abc import ABC
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp
from scipy import stats
from tqdm import tqdm

from spatial_nets.base import Model, PValues, DataNotSet


class UnconstrainedModel(Model):
    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def transform(self, mat: np.ndarray) -> np.ndarray:
        K = self.total_flow_ / mat.sum()
        return K * mat

    #  transform.__doc__ = Model.transform.__doc__


class ConstrainedModel(Model, ABC):
    def __init__(
        self,
        constraint: str,
        approx_pvalues: bool = False,
        extend_left: bool = False,
        verbose: bool = False,
    ):
        if constraint is None:
            raise ValueError("invalid constraint")

        super().__init__(constraint=constraint)
        self.approx_pvalues = approx_pvalues
        self.extend_left = extend_left
        self.verbose = verbose

        self.probabilities_: np.ndarray = None
        self.balancing_factors_: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # TODO: maybe move out balancing factors and put in doubly constrained?

    def pvalues(self) -> PValues:
        """
        Compute the p-values using the binomial PMF or the normal approximation.

        Returns
        -------
        PValues

        """
        method_name = "_pvalues_" + \
            ("approx" if self.approx_pvalues else "exact")
        method = getattr(self, method_name)
        return method()

    def _pvalues_approx(self) -> PValues:
        if (pmat := self.probabilities_) is None:
            raise DataNotSet(
                "the `probabilities_` attribute is unset",
                "this issue is normally resolved by calling the `transform` method",
            )

        # Entry-wise first and second moments (binomial model)
        if self.constraint in ("production", "doubly"):
            Exp = self.target_rows_[:, np.newaxis] * pmat
        elif self.constraint == "attraction":
            Exp = pmat * self.target_cols_[np.newaxis, :]

        Std = np.sqrt(Exp * (1 - pmat))

        with np.errstate(invalid="ignore"):
            Z_score = (self.flow_data.toarray() - Exp) / Std

        data_plus = stats.norm.cdf(-Z_score)
        data_minus = stats.norm.cdf(Z_score)

        shp = self.flow_data.shape
        i, j = self.flow_data.nonzero()
        plus = sp.csr_matrix((data_plus[i, j], (i, j)), shape=shp)
        minus = sp.csr_matrix((data_minus[i, j], (i, j)), shape=shp)

        pvals = PValues(
            constraint=self.constraint,
            right=plus,
            left=minus,
            N=self.N,
            verbose=self.verbose,
        )

        if self.extend_left:
            # Manually chang problematic entries
            data_minus[np.diag_indices(self.N)] = 1.0
            data_minus[(data_minus == 0).nonzero()] = 1e-16
            log_not_pmat = np.log(data_minus)
            # NB: the target_vec is not used for Z-scores
            target_vec = np.ones(self.N)
            pvals._set_left_data(log_not_pmat, target_vec)

        return pvals

    def _pvalues_exact(self) -> PValues:
        if (pmat := self.probabilities_) is None:
            raise DataNotSet(
                "the `probabilities_` attribute is unset",
                "this issue is normally resolved by calling the `transform` method",
            )

        ii, jj = self.flow_data.nonzero()
        n = len(ii)
        data_plus, data_minus = np.zeros(n), np.zeros(n)

        # The target n in the binom. distrib. is either the row or the column sum
        target_vec = (
            self.target_rows_
            if self.constraint in ("production", "doubly")
            else self.target_cols_
        )
        idx = ii if self.constraint in ("production", "doubly") else jj

        for k in tqdm(range(n)):
            i, j = ii[k], jj[k]
            x, n, p = self.flow_data[i, j], target_vec[idx][k], pmat[i, j]
            data_plus[k] = stats.binom_test(x, n=n, p=p, alternative="greater")
            data_minus[k] = stats.binom_test(x, n=n, p=p, alternative="less")

        shp = self.flow_data.shape
        plus = sp.csr_matrix((data_plus, (ii, jj)), shape=shp)
        minus = sp.csr_matrix((data_minus, (ii, jj)), shape=shp)

        pvals = PValues(
            constraint=self.constraint,
            right=plus,
            left=minus,
            N=self.N,
            verbose=self.verbose,
        )

        if self.extend_left:
            log_not_pmat = np.log(1 - self.probabilities_)
            pvals._set_left_data(log_not_pmat, target_vec)

        return pvals

    def multinomial_draw(self, seed=0):
        if (pmat := self.probabilities_) is None:
            raise DataNotSet(
                "the `probabilities_` attribute is unset",
                "this issue is normally resolved by calling the `transform` method",
            )
        rng = np.random.default_rng(seed)
        N = self.N

        out = sp.lil_matrix((N, N), dtype=int)
        if self.constraint in ("production", "doubly"):
            for i in range(N):
                draw = rng.multinomial(self.target_rows_[i], pmat[i])
                (j,) = draw.nonzero()
                out[i, j] = draw[j]
        else:  # self.constraint == "attraction"
            for j in range(N):
                draw = rng.multinomial(self.target_cols_[j], pmat[:, j])
                (i,) = draw.nonzero()
                out[i, j] = draw[i]

        return out.tocsr()


class ProductionConstrained(ConstrainedModel):
    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(constraint="production", **kwargs)

    def transform(self, mat: np.ndarray) -> np.ndarray:
        assert not sp.issparse(mat), "sparse matrices are not supported"
        # We assume that mat is NOT of type  sparse so that sums are flat

        pmat = mat.astype(float)  # to avoid problems with division
        row_sum = mat.sum(axis=1)
        idx = row_sum > 0
        pmat[idx] = pmat[idx] / row_sum[idx, np.newaxis]
        self.probabilities_ = pmat

        return self.target_rows_[:, np.newaxis] * pmat

    #  transform.__doc__ = ConstrainedModel.transform.__doc__


class AttractionConstrained(ConstrainedModel):
    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, **kwargs):
        super().__init__(constraint="attraction", **kwargs)

    def transform(self, mat: np.ndarray) -> np.ndarray:
        assert not sp.issparse(mat), "sparse matrices are not supported"
        # We assume that mat is NOT of type  sparse so that sums are flat

        pmat = mat.astype(float)  # to avoid problems with division
        col_sum = mat.sum(axis=0)
        idx = col_sum > 0
        pmat[:, idx] = pmat[:, idx] / col_sum[np.newaxis, idx]
        self.probabilities_ = pmat

        return pmat * self.target_cols_[np.newaxis, :]

    #  transform.__doc__ = ConstrainedModel.transform.__doc__


class DoublyConstrained(ConstrainedModel):
    """

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(
        self,
        tol: float = 1e-3,
        maxiters: int = 500,
        **kwargs,
    ):
        super().__init__(constraint="doubly", **kwargs)
        self.tol = tol
        self.maxiters = maxiters

    def transform(self, mat: np.ndarray) -> np.ndarray:
        prediction, a, b = simple_ipf(
            mat,
            self.target_rows_,
            self.target_cols_,
            self.tol,
            self.maxiters,
            self.verbose,
        )
        self.balancing_factors_ = (a, b)

        # NB: by default we factor the probabilities as production-constrained
        a = a / self.target_rows_
        self.probabilities_ = a[:, np.newaxis] * mat * b[np.newaxis, :]

        return prediction

    #  transform.__doc__ = ConstrainedModel.transform.__doc__


def simple_ipf(
    mat: np.ndarray,
    target_rows: np.ndarray = None,
    target_cols: np.ndarray = None,
    tol: float = 1e-3,
    maxiters: int = 100,
    verbose: bool = False,
) -> np.ndarray:
    """ Iterative proportional fit procedure."""

    if maxiters < 1:
        raise ValueError("invalid number of maximum iterations")

    N, _ = mat.shape

    if target_rows is None:
        target_rows = np.ones(N)
    if target_cols is None:
        target_cols = np.ones(N)

    niter = 0
    a = np.ones(N)

    while niter < maxiters:
        b = target_cols / (mat.T @ a)
        a = target_rows / (mat @ b)

        out = a[:, np.newaxis] * mat * b[np.newaxis, :]
        # rel_tol instead? this is the problem I address inside CPC, etc.
        bool_row = np.allclose(out.sum(axis=1), target_rows, atol=tol)
        bool_col = np.allclose(out.sum(axis=0), target_cols, atol=tol)

        niter += 1

        if bool_row and bool_col:
            break

    if verbose:
        print(f"Nb iters until convergence: {niter}")

    return out, a, b


# TODO: check below for any useful tests
# def _iterative_proportional_fit(f_mat: Array,
#                                 target_outflow: Array,
#                                 target_inflow: Array,
#                                 rel_tol: float = 1e-3,
#                                 ϵ: float = 1e-6,
#                                 max_iters: int = 1000,
#                                 return_vecs: bool = False,
#                                 safe_mode: bool = False) -> Array:
#     """
#     Apply the IPFP in the case of the doubly constrained model.
#
#     Parameters
#     ----------
#     f_mat : array_like
#         The affinity matrix.
#     target_outflow, target_inflow : array_like
#         Respectively the column-sum and row-sum that the doubly constrained
#         model should have.
#     rel_tol: float, optional
#     ϵ : float, optional
#         The value to use as 'zero' for convergence aspects.
#     max_iters : int, optional
#     return_vecs : bool, optional
#         Return the Ai and Bj normalising vectors (default is False).
#     safe_mode : bool, optional
#         Whether to use assert statements or not (default is False).
#
#     Returns
#     -------
#     np.array
#         The doubly constrained model (or the best approximation to it).
#
#     """
#     N, M = f_mat.shape
#
#     if not isinstance(target_inflow, np.ndarray):
#         target_outflow = np.array(target_outflow)
#     if not isinstance(target_inflow, np.ndarray):
#         target_inflow = np.array(target_inflow)
#
#     if safe_mode:
#         assert N == M, 'matrix is should be square'
#         assert len(target_outflow) == N,\
#             'target_outflow should have same size as matrix'
#         assert len(target_inflow) == N,\
#             'target_inflow should have same size as matrix'
#         assert not np.any(target_outflow < 0),\
#             'target_outflow should be non-negative'
#         assert not np.any(target_inflow < 0),\
#             'target_inflow should be non-negative'
#
#     idx_rows = (target_outflow < ϵ)
#     idx_cols = (target_inflow < ϵ)
#
#     # Verify that the superset of zeros is in target
#     model_outflow = np.asarray(f_mat.sum(axis=1)).flatten()
#     idx_rows_subset = (model_outflow < ϵ)
#     assert np.all(idx_rows[idx_rows_subset]),\
#         'unsatisfiable constraints (zero rows in model and non-zero in target)'
#
#     model_inflow = np.asarray(f_mat.sum(axis=0)).flatten()
#     idx_cols_subset = (model_inflow < ϵ)
#     assert np.all(idx_cols[idx_cols_subset]),\
#         'unsatisfiable constraints (zero cols in model and non-zero in target)'
#
#     # Initialise variables
#     num_iter = 0
#     converged = False
#     outscale_vec_old = np.ones(N)  # vector of A_i (associated to O_i)
#
#     while not converged and num_iter < max_iters:
#         num_iter += 1
#
#         # NB: M @ x is matrix multiplication (no need for x to be a column vec.)
#
#         # Update B_j
#         inscale_vec = np.ones(N)  # vector of B_j (associated to D_j)
#         # Σ_i F_ij * (A_i O_i)
#         denom = f_mat.T @ (outscale_vec_old * target_outflow)
#         np.reciprocal(denom, out=inscale_vec, where=(denom > ϵ))
#         # The where clause takes care of dividing by zero, A_i = 1 in that case
#
#         # Update A_i
#         outscale_vec = np.ones(N)
#         denom = f_mat @ (inscale_vec * target_inflow)  # Σ_j F_ij * (B_j D_j)
#         np.reciprocal(denom, out=outscale_vec, where=(denom > ϵ))
#         # Idem; B_j = 1
#
#         AO_vec = outscale_vec * target_outflow
#         BD_vec = inscale_vec * target_inflow
#         F_rescaled = BD_vec * f_mat * AO_vec[:, np.newaxis]
#
#         # Current error
#         model_outflow = np.asarray(F_rescaled.sum(axis=1)).flatten()
#         model_inflow = np.asarray(F_rescaled.sum(axis=0)).flatten()
#
#         # Note we exclude zero rows or zero columns
#         colsum_error = np.linalg.norm(
#             model_outflow[~idx_rows] / target_outflow[~idx_rows] - 1.0, np.inf)
#         rowsum_error = np.linalg.norm(
#             model_inflow[~idx_cols] / target_inflow[~idx_cols] - 1.0, np.inf)
#         iter_error = np.linalg.norm(outscale_vec - outscale_vec_old)
#
#         # After error, update variable
#         outscale_vec_old = outscale_vec
#
#         if colsum_error < rel_tol and rowsum_error < rel_tol:
#             converged = True
#         elif iter_error < ϵ:
#             warnings.warn(
#                 f'Iterations converged ({num_iter} iters) but constraints cannot be satisfied')
#             # warnings.warn(f'{colsum_error:.3f}, {rowsum_error:.3f}')
#             converged = True
#
#     if not converged:
#         raise FailedToConverge(
#             f'reached maximum number of iterations ({max_iters})')
#
#     return (F_rescaled, outscale_vec, inscale_vec) if return_vecs else F_rescaled


# class FailedToConverge(Exception):
#     """Raised when an iterative method failed to converge."""
#
#     pass
