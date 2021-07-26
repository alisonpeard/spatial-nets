from typing import Optional

import numpy as np
import numpy.ma as ma

from spatial_nets.locations import LocationsDataClass
from spatial_nets.base import Model


class RadiationModel(Model):
    def __init__(
        self,
        constraint: Optional[str] = None,
        threshold: float = np.inf,
        finite_correction: bool = True,
    ):
        if not (threshold > 0):
            raise ValueError("invalid value for threshold")

        super().__init__(constraint=constraint)
        self.threshold = threshold
        self.finite_correction = finite_correction

    def fit(self, data: LocationsDataClass):
        super().fit(data)

        self.production = data.production
        self.attraction = data.attraction
        self.dmat = data.dmat

        return self

    fit.__doc__ = Model.fit.__doc__

    def transform(self, mat=None):
        """
        Compute the radiation model predictions.

        TODO: Mathematical formula goes here

        """
        return self._radiation_matrix(
            threshold=self.threshold,
            finite_correction=self.finite_correction,
        )

    def _io_matrix(self, threshold: float = np.inf) -> np.ndarray:
        """
        Calculate intervening opportunities matrix summing the attraction vector.

        Parameters
        ----------
        threshold : float, optional
            Maximum radius to count locations as intervening opportunities
            (the default is np.inf).

        Returns
        -------
        np.ndarray
            NxN intervening opportunities matrix.

        """
        S = np.zeros((self.N, self.N), dtype=self.attraction.dtype.type)

        # We use masked arrays
        masked_dmat = ma.masked_greater_equal(self.dmat, threshold)
        idx_mat = ma.argsort(masked_dmat, axis=1)
        # sort each row; masked elements are ignored but sorted last

        masked_rel = ma.masked_array(self.attraction)

        for i, idx in enumerate(idx_mat):
            masked_rel.mask = masked_dmat[i].mask
            masked_rel[i] = ma.masked  # origin is not used
            masked_rel[idx[-1]] = ma.masked  # the furthest point is not used

            rolled_idx = np.roll(idx, 1)
            S[i, idx] = masked_rel[rolled_idx].cumsum()

        return S

    def _radiation_matrix(
        self,
        threshold: float = np.inf,
        finite_correction: bool = True,
    ) -> np.ndarray:
        """
        Calculate the radiation matrix.

        Parameters
        ----------
        threshold : float, optional
            Maximum radius to count locations as intervening opportunities
            (the default is np.inf).
        finite_correction :
            Whether to normalise each row using the term $1/(1 - m_i / M)$.

        Returns
        -------
        np.ndarray
            NxN unconstrained radiation matrix.

        """
        S = self._io_matrix(threshold=threshold)
        m = self.attraction[:, np.newaxis]  # column vector
        n = self.attraction[:, np.newaxis]  # column vector

        num = m * n.T
        den = (m + S) * (m + n.T + S)
        with np.errstate(invalid="ignore"):
            # 0/0 inside arrays is invalid floating point, not divide error;
            # to cover the case m_i + S_ij = 0 (can occur when m_i is zero)
            out = num / den
            out[np.isnan(out)] = 0.0

        out[np.diag_indices_from(out)] = 0.0

        if finite_correction:
            M = np.sum(m)
            norm = 1 - m / M  # column vector
            out = out / norm  # each row is scaled

        return out
