"""Fig S14: Pearson against cosine correlations between odours, input and output.

Draws what compute.two_correlations computes. Ported from
two-correlations.ipynb (glom-io-transform, 2023). The colour range and scatter
limits are set for the data Fig 5 uses, where nearly every pair of odours is
positively correlated: each glomerulus's mean response is not subtracted, and
most glomeruli respond to most odours.
"""
from matplotlib import pyplot as plt
from matplotlib import cm

from ..compute.two_correlations import off_diag

# One row per population: (key in the data, row label, scatter colour).
ROWS = (("input", "INPUT", cm.Greens(0.6)), ("output", "OUTPUT", cm.Blues(0.6)))

CORR_LIMS    = (0, 1)        # colour range of the correlation matrices
SCATTER_LIMS = (-0.15, 1)    # both scatter axes; the lowest value is about -0.13
SCATTER_TICKS = [0, 0.5, 1]


class Main:
    @classmethod
    def plot(cls, plot_data):
        fig = plt.figure(figsize=(9,5))
        for row, (key, label, colour) in enumerate(ROWS):
            first = 6*row + 1
            ax = plt.subplot(2,6,(first, first+1))
            im = ax.matshow(plot_data.pearson[key], vmin=CORR_LIMS[0], vmax=CORR_LIMS[1], cmap=cm.bwr)
            ax.xaxis.set_ticks_position("bottom"); plt.colorbar(im, ax=ax)
            ax.set_ylabel(label, fontsize=16); ax.set_title("Pearson correlation")

            ax = plt.subplot(2,6,(first+2, first+3))
            im = ax.matshow(plot_data.cosine[key], vmin=CORR_LIMS[0], vmax=CORR_LIMS[1], cmap=cm.bwr)
            ax.xaxis.set_ticks_position("bottom"); plt.colorbar(im, ax=ax)
            ax.set_title("Cosine correlation")

            ax = plt.subplot(2,6,(first+4, first+5))
            ax.scatter(off_diag(plot_data.pearson[key]), off_diag(plot_data.cosine[key]), s=3, color=colour)
            ax.set_xlim(SCATTER_LIMS); ax.set_ylim(SCATTER_LIMS); ax.set_aspect("equal")
            ax.grid(True, linestyle=":")
            ax.set_xticks(SCATTER_TICKS); ax.set_yticks(SCATTER_TICKS)
            for side in ("top", "right"): ax.spines[side].set_visible(False)
            ax.set_xlabel("Pearson correlation", labelpad=-2)
            ax.set_ylabel("Cosine correlation",  labelpad=-4)
            ax.set_title(f"Spearman's ρ = {plot_data.spearman[key]:1.2f}")
        plt.tight_layout()
        return fig
