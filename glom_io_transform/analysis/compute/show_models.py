"""Compute for the show_models figure: reference Diag/Free fits plus the
generalization metrics dataframe (via compute.generalization)."""

from .compute import Computation, base_context
from .generalization import generalization_df


class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self,
                selection_metric="ratio",
                compute_df=False,
                ):
        print("COMPUTING show_models.Data.")

        base = base_context()

        self.df, self.cache_file = generalization_df(
            base, selection_metric=selection_metric, compute=compute_df)

        split = base.split("trials", "random", "max")
        self.models = {mdl: split.model(mdl).extract(seed=0, train=0, metric=selection_metric)
                       for mdl in ["Diag", "Free"]}

        self.computed = True
        return self
