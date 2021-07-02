import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import pickle

import numpy as np
import scipy.sparse as sp
import graph_tool as gt
from tabulate import tabulate

from spatial_nets.locations import LocationsDataClass
from spatial_nets.utils import sparsity


class DataNotSet(Exception):
    """Raised when a method needs to access the data and it has not been set."""

    pass


class Model(ABC):
    @abstractmethod
    def transform(self, mat: Optional[np.ndarray] = None) -> np.ndarray:
        pass

    def __init__(self, constraint: Optional[str] = None):
        """

        Parameters
        ----------
        constraint  : str
            The type of constraint to use.

        """
        if constraint not in [
            None,
            "production",
            "attraction",
            "doubly",
        ]:
            raise ValueError("invalid constraint")

        self.constraint: Optional[str] = constraint
        self.coef_ = None

    def fit(self, data: LocationsDataClass):
        """
        Fit the model to the observations. This normally involves data copying.

        Parameters
        ----------
        data : LocationsDataClass
            The custom object which we defined to store the data. Note that
            this object needs to have its `flow_data` attribute set.

        Returns
        -------
        self

        """
        if data.flow_data is None:
            raise DataNotSet("the flow data for comparison is needed")

        self.N = len(data)
        self.flow_data = data.flow_data
        self.total_flow_ = self.flow_data.sum()
        self.target_rows_ = data.target_rows
        self.target_cols_ = data.target_cols

        return self

    def fit_transform(
        self, data: LocationsDataClass, mat: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.fit(data).transform(mat)

    # TODO: turn below into the score method
    # def CPC_from_model(self, model: str,
    #                    constraint_type: str,
    #                    params: ParamDict) -> float:
    #     """
    #     Use a particular model and constraint type to calculate the CPC fit.
    #
    #     Parameters
    #     ----------
    #     model : {'gravity', 'radiation', 'io'}
    #     params : dict
    #
    #     Returns
    #     -------
    #     float
    #
    #     """
    #     Data = self.flow_data
    #     return CPC(Data, self.constrained_model(model, constraint_type, params))
    #


class PValues:
    def __init__(
        self,
        right: sp.csr_matrix,
        left: sp.csr_matrix,
        N: int,
        verbose: bool = False,
        model=None,
        constraint=None,
        approx_pvalues=None,
        coef=None,
        balancing_factors=None,
    ):
        mask = sparsity(right)
        if (mask != sparsity(left)).nnz > 0:
            raise ValueError("the sparsity pattern of the matrices does not match")

        self.mask = mask
        self.right = right
        self.left = left
        self.N = N
        self.verbose = verbose

        self._significance = None
        self._significant_right = None
        self._significant_left = None

        self.model = model
        self.constraint = constraint
        self.approx_pvalues = approx_pvalues
        self.coef = coef
        self.balancing_factors = balancing_factors

    def set_significance(self, alpha: float = 0.01):
        self.significance = alpha

        return self

    @property
    def significance(self):
        return self._significance

    @significance.setter
    def significance(self, alpha):
        if 0 < alpha < 1:
            self._significance = alpha
        else:
            raise ValueError("significance should lie inside (0, 1)")

    def save(self, filename: Union[str, os.PathLike]):
        if self.model is None:
            raise DataNotSet("need to provide the name of the model")
        elif self.constraint is None:
            raise DataNotSet("need to provide the name of the constraint")
        elif self.approx_pvalues is None:
            raise DataNotSet("need to specify whether the calculation was approximate")

        if os.path.splitext(filename)[1] not in (".pkl", ".pickle"):
            raise ValueError("unsupported format; use pickle instead")

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def compute_backbone(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        if self.significance is None:
            raise DataNotSet("`set_significance` has not been called")

        # Below we wrote the comparison as XOR
        #  sig_plus: sp.csr_matrix = (self.right < self.significance).multiply(self.mask)
        compare_plus = self.right >= self.significance
        sig_plus = (compare_plus > self.mask) + (compare_plus < self.mask)

        #  sig_minus: sp.csr_matrix = (self.left < self.significance).multiply(self.mask)
        compare_minus = self.left >= self.significance
        sig_minus = (compare_minus > self.mask) + (compare_minus < self.mask)

        if self.verbose:
            n, nplus, nminus = self.mask.nnz, sig_plus.nnz, sig_minus.nnz
            nzero = n - nplus - nminus
            tab = [
                ["Positive (observed larger)", nplus, f"{100*nplus/n:.2f}"],
                ["Negative (model larger)", nminus, f"{100*nminus/n:.2f}"],
                ["Not-significant:", nzero, f"{100*nzero/n:.2f}"],
                ["Total", n, "100.00"],
            ]
            print(tabulate(tab, headers=["", "Nb", "%"]))

        # We "cache" these two matrices
        self._significant_right = sig_plus
        self._significant_left = sig_minus

        return sig_plus, sig_minus

    def compute_not_significant(self) -> sp.csr_matrix:
        if self.significance is None:
            raise DataNotSet("`set_significance` has not been called")

        if self._significant_left is None:
            sig_plus, sig_minus = self.compute_backbone()
        else:
            sig_plus, sig_minus = self._significant_right, self._significant_left

        i, j = self.mask.nonzero()
        not_significant = np.asarray(
            ~np.bitwise_or(sig_plus[i, j], sig_minus[i, j])
        ).flatten()

        return sp.csr_matrix((not_significant, (i, j)), shape=self.mask.shape)

    def compute_graph(self) -> gt.Graph:
        """
        Calculate the significant edges according to a binomial test (or z-test).

        Parameters
        ----------

        Returns
        -------
        gt.Graph

        """
        if self._significant_left is None:
            sig_plus, sig_minus = self.compute_backbone()
        else:
            sig_plus, sig_minus = self._significant_right, self._significant_left

        G = gt.Graph(directed=True)
        G.add_vertex(self.N)
        G.add_edge_list(np.transpose(sig_plus.nonzero()))
        G.add_edge_list(np.transpose(sig_minus.nonzero()))
        weights = np.concatenate(
            (
                np.ones(sig_plus.nnz, dtype=int),
                np.zeros(sig_minus.nnz, dtype=int),
            )
        )
        G.ep["weight"] = G.new_edge_property("short", vals=weights)

        return G