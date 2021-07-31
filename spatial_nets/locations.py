import textwrap
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import pairwise

import graph_tool as gt
import networkx as nx
from spatial_graphs import SpatialGraph, SpatialDiGraph

import spatial_nets.utils as utils

Array = np.ndarray
OptionalFloat = Optional[float]
ParamDict = Dict[str, OptionalFloat]


class LocationsDataClass:
    """
    Locations in 2D space with relative strengths and community attributes.

    % The model represents nodes of a network embedded in 2D space, and provides
    methods for calculating human mobility models.
    The production and attraction attributes measure the relative importance at
    each location (e.g. strenght, population, etc.).

    The community attribute ... TODO

    Attributes
    ----------
    N : int
        The number of locations.
    dmat : np.ndarray
        The NxN matrix of pairwise distances between locations.
    production : np.ndarray
        The vector containing a measure of outgoing strenght at each location.
    attraction : np.ndarray
        The vector containing a measure of incoming strength at each location.

    TODO
    comm_vec : np.ndarray or None, optional
        A vector containing the community (block) assignment of each location
        (default is None). The community methods will be extended in the
        future.

    """

    def __init__(self, data=None, coords=None, **kwargs):
        """
        Initialise the LocationsDataClass object.

        Parameters
        ----------
        data : object to be converted

            Current supported types are:
              a matrix containing OD data
              a tuple containing the production and attraction vectors
              a Graph object

            To be implemented:
              a dict  containing the production and attraction vectors
              a DataFrame containing OD data
              another LocationsDataClass object

        coords : array_like, optional


        """
        self.N: int = None
        self.B: int = None

        self._dmat: np.ndarray = None
        self._production: np.ndarray = None
        self._attraction: np.ndarray = None

        self._flow_data: sp.csr_matrix = None  # TODO: what about other possibilities?
        self.target_rows: np.ndarray = None
        self.target_cols: np.ndarray = None

        #  self._model_params = None
        #  self._affinity_mat: np.ndarray = None
        #  self._balancing_factors: Tuple[np.ndarray, np.ndarray] = None
        #  self._pvalues: Tuple[sp.csr_matrix, sp.csr_matrix] = None
        #  self._significance: float = None

        if isinstance(data, (np.integer, int)):
            if data < 1:
                raise ValueError("invalid nunmber of locations")
            self.N = data

        # For LocationsDataClass or Graph, we can assume the coords are stored
        # inside the object

        elif isinstance(data, LocationsDataClass):
            # TODO: self.copy()
            #  return data.copy()
            pass

        elif isinstance(data, (gt.Graph, gt.GraphView)):
            if (flow := data.ep.get("flow")) is not None:
                self.flow_data = gt.spectral.adjacency(data, weight=flow)
            else:
                raise ValueError("valid inpud needs a `flow` edge property")

            if (pos := data.vp.get("pos")) is not None:
                self.dmat = np.array(list(pos)) * [
                    [1],
                    [-1],
                ]  # due to cairo's coordinate system in the y-direction
            #  elif coords is not None:
            #      self.dmat = coords
            #  TODO: This should be covered by the elif below

        elif isinstance(data, (SpatialGraph, SpatialDiGraph)):
            if data.fmat is not None:
                self.flow_data = data.fmat
            else:
                raise ValueError("SpatialGraph has no flow matrix/")

            if data.dists is not None:
                self.dmat = data.dists

        # In any other case, the coordinates should be passed (the coords
        #  argument takes precedence)

        elif coords is not None:

            if isinstance(data, tuple):
                assert len(data) == 2, "two sets of features are needed"
                prod, attrac = data
                self.production = prod
                self.attraction = attrac

            elif isinstance(data, (np.matrix, np.ndarray)):
                # TODO: test that this does not catch sparse matrix
                self.flow_data = data
            elif sp.issparse(data):
                self.flow_data = data

            self.dmat = coords
            # TODO: this should work for Graph object with no `pos` vp

        else:
            raise ValueError("coordinate matrix is needed")

        # TODO: is this good idea?
        for key, val in kwargs.items():
            setattr(self, key, val)

        # TODO
        #  if comm_vec is None:
        #      self.B = 1
        #      self.comm = np.ones(N, dtype=int)
        #  else:
        #      # self.k = len(np.unique(comm_vec))  #TODO:check below works better
        #      self.B = len(set(comm_vec))
        #      self.comm = np.array(comm_vec)

    def __str__(self) -> str:
        s = f"Locations object: N={self.N}"

        if self.B is not None:
            s += f" and B={self.B}"

        if self.flow_data is not None:
            s += "\n -- Data has been set"

        return textwrap.dedent(s)

    def __len__(self) -> int:
        """Returns the number of locations.

        Returns
        -------
        N : int
            The number of locations.

        """
        return self.N

    def copy(self):
        # TODO
        pass

    @property
    def dmat(self):
        return self._dmat

    @dmat.setter
    def dmat(self, location_mat) -> np.ndarray:
        N, m = location_mat.shape

        if (m != N) and (m != 2):
            raise ValueError("invalid shape for coordinate matrix")

        if self.N is not None:
            if N != self.N:
                raise ValueError("input has invalid length")
        else:
            self.N = N

        # The location mat has the right shape; now do other tests
        if m == 2:
            self._dmat = pairwise.euclidean_distances(location_mat)

        elif np.any(location_mat < 0):
            raise ValueError("matrix should be non_negative")
        else:  # valid NxN matrix
            self._dmat = location_mat

    @property
    def production(self):
        return self._production

    @production.setter
    def production(self, prod: Array):
        """Should be set first."""
        prod = np.array(prod)
        N = len(prod)

        if not np.all(prod > 0):
            import pdb
            pdb.set_trace()
            raise ValueError("input should have strictly positive entries")

        if self.N is not None:
            if N != self.N:
                raise ValueError("input has invalid length")
        else:
            self.N = N

        self._production = prod

    @property
    def attraction(self):
        return self._attraction

    @attraction.setter
    def attraction(self, attrac: Array):
        """Should be set second."""
        attrac = np.array(attrac)

        if not np.all(attrac > 0):
            pdb.set_trace()
            raise ValueError("input should have strictly positive entries")
        if len(attrac) != self.N:
            raise ValueError("input has invalid length")

        self._attraction = attrac

    @property
    def flow_data(self):
        """Retrieve the data stored inside the object."""
        return self._flow_data

    @flow_data.setter
    def flow_data(self, flow_mat: Array):
        """Set a copy of the data inside the object."""
        N, m = flow_mat.shape
        if not sp.issparse(flow_mat):
            flow_mat = sp.csr_matrix(flow_mat)

        if (flow_mat < 0).sum() > 0:  # works for dense and spase matrices
            raise ValueError("data matrix should be non-negative")

        if m != N:
            raise ValueError("invalid shape for matrix")

        if self.N is not None:
            if N != self.N:
                raise ValueError("input has invalid shape")

        else:
            self.N = N

        # NB: We remove the diagonal from the data matrix BEFORE calculating
        # the row and column sums
        self._flow_data = utils.sparsemat_remove_diag(
            flow_mat)  # creates a copy

        # Row and column sums
        self.production = np.asarray(self._flow_data.sum(axis=1)).flatten()
        self.attraction = np.asarray(self._flow_data.sum(axis=0)).flatten()

        self.target_rows = self.production
        self.target_cols = self.attraction

        return None

    # End of the basic initialisation functions
