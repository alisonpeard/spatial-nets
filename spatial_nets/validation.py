import numpy as np
import graph_tool.all as gt

from typing import Dict

from spatial_nets import *
from spatial_nets import utils


class Experiment(object):
    def __init__(
        self,
        N: int,
        rho: float,
        params: Dict,
        model: str,
        benchmark: str,
        sign: str,
        significance: float = 0.01,
        directed: bool = False,
        verbose: bool = False,
    ) -> None:

        assert model.startswith(("gravity", "radiation"))
        assert model.endswith(("production", "attraction", "doubly"))
        assert benchmark in (("expert", "cerina")
                             ), f"invalid benchmark: {benchmark}"
        assert sign in (("plus", "minus", "weight_covariates"))
        if sign == "weight_covariates":
            raise NotImplementedError

        self.N = N
        self.rho = rho
        self.params = params
        self.model = model
        self.benchmark = getattr(utils, f"benchmark_{benchmark}")
        self.sign = sign
        self.significance = significance
        self.directed = directed
        self.verbose = verbose

        if self.verbose:
            print(self.benchmark)
            print(self.model)
            print(self.sign)

        return None

    def benchmark_graph(self, seed=0, return_backbone=True):

        coords, comm_vec, coo_mat = self.benchmark(
            self.N, self.rho, **self.params, seed=seed, directed=self.directed
        )
        bench = utils.build_weighted_graph(
            coo_mat,
            directed=self.directed,
            coords=coords,
            vertex_properties={"b": comm_vec},
        )
        import pdb
        pdb.set_trace()
        # block_state = gt.BlockState(bench, b=bench.vp.b)

        if return_backbone:
            T_data = coo_mat.tocsr()
            locs = LocationsDataClass(T_data, coords=coords)

            backbone = utils.build_significant_graph(
                locs,
                self.model,
                sign=self.sign,
                coords=coords,
                significance=self.significance,
                verbose=self.verbose,
            )

            if self.verbose:
                print(backbone)

        return (bench, backbone) if return_backbone else bench

    def repeated_runs(
        self,
        nb_repeats: int = 1,
        nb_net_repeats: int = 1,
        start_seed: int = 0,
        **gt_kwargs,
    ):

        # overlap, nmi, vi, B, entropy
        out = np.zeros((nb_repeats * nb_net_repeats, 5))
        out_fix = np.zeros_like(out)

        for k in range(nb_net_repeats):
            bench, backbone = self.benchmark_graph(
                seed=start_seed + k, return_backbone=True
            )
            x = bench.vp.b.a

            for i in range(nb_repeats):
                row = k * nb_repeats + i

                # Varying B
                state = gt.minimize_blockmodel_dl(backbone)
                ov = gt.partition_overlap(x, state.b.a, norm=True)
                vi = gt.variation_information(x, state.b.a, norm=True)
                nmi = gt.mutual_information(x, state.b.a, norm=True) * self.N
                # * N due to  bug in graph-tool's nmi
                out[row] = ov, vi, nmi, state.get_nonempty_B(), state.entropy()

                # Fixed B
                state = gt.minimize_blockmodel_dl(backbone, B_max=2, B_min=2)
                ov = gt.partition_overlap(x, state.b.a, norm=True)
                vi = gt.variation_information(x, state.b.a, norm=True)
                nmi = gt.mutual_information(x, state.b.a, norm=True) * self.N
                # * N due to  bug in graph-tool's nmi
                out_fix[row] = ov, vi, nmi, state.get_nonempty_B(), state.entropy()

        return out, out_fix

    def summarise_results(self, mat):
        mn = mat[:, :-1].mean(axis=0)
        std = mat[:, :-1].std(axis=0)

        k = mat[:, -1].argmin()
        best = mat[k, :-1]

        return mn, std, best
