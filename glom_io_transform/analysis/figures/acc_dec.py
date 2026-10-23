"""Fig 5 and its supplementary panels: decoding accuracy along each transformation.

Draws the results that compute.acc_dec loads. Ported from
proc-transformation-results.ipynb (glom-io-transform, 2023); the plots are
unchanged, apart from the example row's limits (see Main.EXAMPLE_SCALE) and
the accuracy axis extending below 0.5 when the data need it.
"""
import numpy as np
from matplotlib import pyplot as plt

XFORM_COLS = {"ANGLES":"orange",       "LENGTHS":"lime",    "ROTATION":"red",      "COVARIANCE":"cornflowerblue"}
XFORM_LABS = {"ANGLES":"Correlations", "LENGTHS":"Lengths", "ROTATION":"Rotation", "COVARIANCE":"Scatter"}

# The ordering shown in the main figure.
SEQUENCE = ("ANGLES", "LENGTHS", "ROTATION", "COVARIANCE")


def summarize_accuracies(accs):
    """{ordering: seeds x steps}, each accuracy averaged over its splits."""
    return {k: np.mean(np.array(v), axis=-1) for k, v in accs.items() if not isinstance(v, Exception)}


def plot_transformation_accuracies(accs, start = SEQUENCE,
                                   direction = "input-to-output", dx = 0.,
                                   plot_only = None, ax = None, figsize = None, do_legend = True,
                                   marker = "o", grid_on = False, yl = None,
                                   **kwargs):
    """Accuracy after each step: the mean over seeds, with +-3 sd bars.

    With plot_only, that one ordering is drawn onto `ax`. Otherwise every
    ordering is drawn on a 3 x 4 grid in a new figure, starting from `start` and
    then taking each time the remaining ordering whose profile is most
    correlated with the last one drawn.
    """
    accs_ = summarize_accuracies(accs)
    orig_order = list(accs_.keys())
    acc_means = np.array([np.mean(accs_[k], axis=0) for k in orig_order])
    corrs     = np.corrcoef(acc_means)
    used      = [orig_order.index(start)]
    for i in range(len(orig_order)-1):
        iord = np.argsort(-corrs[used[-1]])[1:]
        for j in iord:
            if j not in used:
                used.append(j)
                break
    plot_order = [orig_order[u] for u in used]

    if not hasattr(dx, "__len__"): dx = [dx]*5
    if ax is None:
        plt.figure(figsize=((9,3.5) if plot_only else (9,5)) if figsize is None else figsize)
    for j, k in enumerate(plot_order):
        if plot_only is not None and k != plot_only: continue
        v = np.mean(accs_[k], axis=0)
        s = np.std(accs_[k], axis=0)
        ax = plt.subplot(3,4,j+1) if plot_only is None else (ax if ax is not None else plt.subplot(1,1,1))
        x = np.arange(5)
        if direction == "output-to-input": x = x[::-1]
        x = x + dx
        ax.plot(x, v, marker, color="gray", **kwargs)
        ax.plot([x, x], [v-3*s, v+3*s], color="gray")
        for i in range(4):
            ax.plot([x[i], x[i+1]], [v[i], v[i+1]], color=XFORM_COLS[k[i]], zorder=-1, label=XFORM_LABS[k[i]],
                    linestyle = "--" if direction == "output-to-input" else "-", **kwargs)
        if (j == 0 or plot_only) and do_legend:
            if plot_only: ax.legend()
            else:         ax.legend(fontsize=8, labelspacing=0.1, borderpad=0, frameon=False)
        ax.set_ylim(0.2,1.1)
        ax.set_xticks(np.arange(5), labels=["In","","","","Out"])
        if j % 4 and not plot_only:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Accuracy")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if plot_only or grid_on: ax.grid(True, linestyle=":")
        if plot_only or yl is not None: ax.set_ylim((0.5,1) if yl is None else yl)
    plt.tight_layout()


def plot_transformation(Yseq_, direction, which_odours=(4,5), figsize=(9,2.5),
                        cols=("mediumpurple", "mediumturquoise"), axes=None, titles=None, scale=1.0):
    """Two odours' trials and means at each step, in the plane their means span.

    The axis limits were set for the 2023 responses; `scale` multiplies them,
    so data in other units can be drawn with the same framing.
    """
    which_odours = list(which_odours)
    if not axes: plt.figure(figsize=figsize)
    ttls = titles if titles else [None]*len(Yseq_)
    for i, (Yi_, ttl) in enumerate(zip(Yseq_, ttls)):
        μYi_ = np.mean(Yi_, axis=-1)
        P = μYi_[:, which_odours]
        _, U = np.linalg.eigh(np.dot(P, P.T))
        U = U[:,-2:] @ np.diag([1,-1])
        μyi_ = U.T @ μYi_
        yi = np.einsum('ij,jkl', U.T, Yi_)
        ax = plt.subplot(1,5,i+1) if not axes else axes[i]
        for wo, col in zip(which_odours, cols):
            ax.plot([0, μyi_[0,wo]], [0, μyi_[1,wo]], color=col)
        for j, wo in enumerate(which_odours):
            ax.plot(yi[0,wo,:], yi[1,wo,:], "o", color=cols[j])
        ax.set_xlim(-350*scale, 350*scale)
        ax.set_ylim(-100*scale, 800*scale)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(axis='both', colors='gray')
        ax.set_yticks(np.arange(0,800,250)*scale)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_label_position('bottom')
        ax.yaxis.set_label_position('left')
        if ((i > 0) and direction == "input-to-output") or ((i < len(Yseq_)-1) and direction == "output-to-input"):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if titles: ax.set_title(ttl)


class Main:
    """Fig 5: the example odours along both paths (A), and the accuracies (B)."""
    # The 2023 example panels were framed for time-integrated responses, in z*s
    # over a 5.05 s window. Responses are now means over that window, in z, and
    # so about five times smaller; 1/5 keeps the same framing. Pass 1 to draw
    # the 2023 responses as published.
    EXAMPLE_SCALE = 0.2

    @classmethod
    def plot(cls, plot_data, sequence=SEQUENCE, example_scale=EXAMPLE_SCALE):
        k = sequence
        fig = plt.figure(figsize=(9,6))
        ax = [plt.subplot(4,5,i+1) for i in range(5)]
        plot_transformation(plot_data.Ti2o.Xseq_[k][0], "input-to-output", axes=ax, cols=["mediumvioletred", "purple"],
                            titles=["INPUT"] + [XFORM_LABS[ki] for ki in k], scale=example_scale)
        ax = [plt.subplot(4,5,i+1) for i in range(5,10)]
        plot_transformation(plot_data.To2i.Xseq_[k][0], "output-to-input", axes=ax[::-1], cols=["mediumvioletred", "purple"],
                            titles=["OUTPUT"] + [XFORM_LABS[ki] for ki in k], scale=example_scale)
        ax = plt.subplot(4,5,(11,20))
        plot_transformation_accuracies(plot_data.Ti2o.accs, plot_only=k, direction="input-to-output", linewidth=2, ax=ax,
                                       dx=[0] + [-0.05]*3 + [0])
        plot_transformation_accuracies(plot_data.To2i.accs, plot_only=k, direction="output-to-input", ax=ax, do_legend=False,
                                       dx=[0] + [0.05]*3 + [0], marker="o", linewidth=2)
        # The published floor of 0.5, lowered to the next 0.1 only if an error bar
        # would otherwise be cut off.
        lowest = min(np.min(np.mean(a, axis=0) - 3*np.std(a, axis=0))
                     for a in (summarize_accuracies(T.accs)[k] for T in (plot_data.Ti2o, plot_data.To2i)))
        ax.set_ylim(min(0.5, np.floor(lowest*10)/10), 1.05)
        plt.tight_layout(pad=0, w_pad=1, h_pad=0)
        return fig


class Supp:
    """Every valid ordering of the four transformations, for one direction."""
    @classmethod
    def plot(cls, plot_data, direction="input-to-output"):
        T = plot_data.Ti2o if direction == "input-to-output" else plot_data.To2i
        plot_transformation_accuracies(T.accs, direction=direction, linewidth=1.5, grid_on=True, yl=[0.2,1.05], markersize=5)
        return plt.gcf()
