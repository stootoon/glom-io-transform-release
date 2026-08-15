"""Compute for the explain_models figure: orchestrates the Diag quartic
geometry (top row) and the Free connectivity theory (bottom row)."""

from .compute import Computation, base_context
from . import diag as diag_lib
from . import free as free_lib


class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self,
                selection_metric="ratio",
                seeds=None,
                ref_seed=0,
                ref_train=0,
                diag_seed=0,
                diag_la=None,
                diag_n_train=10,
                diag_gtol=1e-8,
                ):
        print("COMPUTING explain_models.Data.")

        base = base_context()
        split = base.split("trials", "random", "max")

        self.free = free_lib.connectivity_theory(split,
                                                 selection_metric=selection_metric,
                                                 seeds=seeds,
                                                 ref_seed=ref_seed,
                                                 ref_train=ref_train)

        self.diag = diag_lib.quartic_geometry(split,
                                              selection_metric=selection_metric,
                                              seed=diag_seed,
                                              la=diag_la,
                                              train=ref_train,
                                              n_train=diag_n_train,
                                              gtol=diag_gtol)

        self.computed = True
        return self
