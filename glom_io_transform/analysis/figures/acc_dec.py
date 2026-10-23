"""Fig 5 and its supplementary panels: decoding accuracy along each transformation.

Draws the results that compute.acc_dec loads. Ported from
proc-transformation-results.ipynb (glom-io-transform, 2023); the plots are
unchanged, apart from the example row's limits (see Main.EXAMPLE_SCALE) and
the accuracy axis extending below 0.5 when the data need it.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

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


# Fig 5 as it appears in the manuscript, styled here rather than in Illustrator.
FONT = "Arial"
# 42 embeds TrueType, so the text stays editable as Arial in Illustrator; the
# default (3) imports as outlines. Read at save time, so pass it to savefig too.
RC = {"font.family": FONT, "pdf.fonttype": 42, "ps.fonttype": 42}
# The manuscript's softer green for Lengths; the supplementary panels keep lime.
MAIN_COLS = {**XFORM_COLS, "LENGTHS": "#5DB13C"}
STEP_FONTSIZE   = 12    # the transformation labels between the example panels
STEP_LABEL_Y    = 0.2   # label height, as a fraction of the panel's height
STEP_ARROW_Y    = 0.33  # where the curved arrow over it starts and ends
AXIS_FONTSIZE   = 14    # Input / Output and Accuracy in B
LEGEND_FONTSIZE = 10
ARROWHEAD_SIZE  = 22    # mutation_scale of the step arrows in B


class _ArrowHandler(HandlerBase):
    """Draws a legend entry as a short arrow, pointing left or right."""
    def __init__(self, leftward):
        super().__init__()
        self.leftward = leftward

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y = height/2 - ydescent
        x0, x1 = -xdescent, width - xdescent
        if self.leftward: x0, x1 = x1, x0
        return [FancyArrowPatch((x0, y), (x1, y), transform=trans, arrowstyle="-|>",
                                mutation_scale=fontsize, shrinkA=0, shrinkB=0,
                                color=orig_handle.get_color(), linewidth=orig_handle.get_linewidth(),
                                linestyle=orig_handle.get_linestyle())]


class Main:
    """Fig 5: the example odours along both paths (A), and the accuracies (B)."""
    # The 2023 example panels were framed for time-integrated responses, in z*s
    # over a 5.05 s window. Responses are now means over that window, in z, and
    # so about five times smaller; 1/5 keeps the same framing. Pass 1 to draw
    # the 2023 responses as published.
    EXAMPLE_SCALE = 0.2

    @classmethod
    def plot(cls, plot_data, sequence=SEQUENCE, example_scale=EXAMPLE_SCALE):
        with plt.rc_context(RC):
            k = sequence
            fig = plt.figure(figsize=(9,6))
            top = [plt.subplot(4,5,i+1) for i in range(5)]
            plot_transformation(plot_data.Ti2o.Xseq_[k][0], "input-to-output", axes=top,
                                cols=["mediumvioletred", "purple"], scale=example_scale)
            bottom = [plt.subplot(4,5,i+1) for i in range(5,10)]
            plot_transformation(plot_data.To2i.Xseq_[k][0], "output-to-input", axes=bottom[::-1],
                                cols=["mediumvioletred", "purple"], scale=example_scale)
            ax = plt.subplot(4,5,(11,20))
            cls.plot_accuracies(ax, plot_data, k)
            plt.tight_layout(pad=0, w_pad=1, h_pad=0)
            # Placed in figure coordinates, so only once the layout is final.
            cls.label_steps(fig, top,    k,       leftward=False)
            cls.label_steps(fig, bottom, k[::-1], leftward=True)
            for text, x, y, size in (("A", top[0].get_position().x0 - 0.06, top[0].get_position().y1 + 0.02, 16),
                                     ("i", top[0].get_position().x0 - 0.05, top[0].get_position().y1 - 0.03, 13),
                                     ("ii", bottom[0].get_position().x0 - 0.05, bottom[0].get_position().y1 - 0.03, 13),
                                     ("B", ax.get_position().x0 - 0.06, ax.get_position().y1 + 0.02, 16)):
                fig.text(x, y, text, fontsize=size, fontweight="bold", va="top")
            fig.canvas.draw()      # create every tick label while Arial is in force
        return fig

    @staticmethod
    def label_steps(fig, axes, steps, leftward):
        """Each transformation's name, under a curved arrow, in the gap it spans."""
        for a, b, step in zip(axes[:-1], axes[1:], steps):
            pa, pb = a.get_position(), b.get_position()
            xm = (pa.x1 + pb.x0)/2
            colour = MAIN_COLS[step]
            fig.text(xm, pa.y0 + STEP_LABEL_Y*pa.height, XFORM_LABS[step], color=colour, fontsize=STEP_FONTSIZE,
                     ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5))
            x0, x1 = xm - 0.025, xm + 0.025
            if leftward: x0, x1 = x1, x0
            fig.add_artist(FancyArrowPatch((x0, pa.y0 + STEP_ARROW_Y*pa.height), (x1, pa.y0 + STEP_ARROW_Y*pa.height),
                                           transform=fig.transFigure, arrowstyle="->", mutation_scale=9,
                                           connectionstyle=f"arc3,rad={0.5 if leftward else -0.5}",
                                           color=colour, linewidth=1.2, linestyle="--" if leftward else "-"))

    @staticmethod
    def plot_accuracies(ax, plot_data, k):
        """Mean accuracy after each step, +-3 sd over seeds; each step drawn as an arrow."""
        lowest = 1.0
        for T, direction, dx, linestyle in ((plot_data.Ti2o, "input-to-output", -0.05, "-"),
                                            (plot_data.To2i, "output-to-input",  0.05, "--")):
            a = summarize_accuracies(T.accs)[k]
            v, s = np.mean(a, axis=0), np.std(a, axis=0)
            lowest = min(lowest, np.min(v - 3*s))
            x = np.arange(5)
            if direction == "output-to-input": x = x[::-1]
            x = x + np.array([0, dx, dx, dx, 0])
            ax.plot([x, x], [v-3*s, v+3*s], color="gray", zorder=2)
            for i in range(4):
                ax.annotate("", xy=(x[i+1], v[i+1]), xytext=(x[i], v[i]), zorder=1,
                            arrowprops=dict(arrowstyle="-|>", mutation_scale=ARROWHEAD_SIZE, shrinkA=0, shrinkB=5,
                                            color=MAIN_COLS[k[i]], linewidth=2, linestyle=linestyle))
            ax.plot(x, v, "o", color="gray", zorder=3)

        # The published floor of 0.5, lowered to the next 0.1 only if an error bar
        # would otherwise be cut off.
        ax.set_ylim(min(0.5, np.floor(lowest*10)/10), 1.05)
        ax.set_xlim(-0.2, 4.2)
        ax.set_xticks(np.arange(5), labels=["Input", "", "", "", "Output"])
        for label in ax.get_xticklabels(): label.set_fontweight("bold"); label.set_fontsize(AXIS_FONTSIZE)
        ax.set_ylabel("Accuracy", fontweight="bold", fontsize=AXIS_FONTSIZE)
        ax.grid(True, linestyle="--", color="0.8", linewidth=0.8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        steps = [Line2D([], [], color=MAIN_COLS[t], linewidth=2, label=XFORM_LABS[t])
                 for t in ("ANGLES", "LENGTHS", "ROTATION", "COVARIANCE")]
        forward  = Line2D([], [], color=MAIN_COLS["ANGLES"], linewidth=1.5, linestyle="-",  label="Input to output")
        backward = Line2D([], [], color=MAIN_COLS["ANGLES"], linewidth=1.5, linestyle="--", label="Output to input")
        blank    = [Line2D([], [], linestyle="none", label="") for _ in range(2)]
        # Two columns, filled column by column: the steps, then the directions.
        ax.legend(handles=steps + [forward, backward] + blank, ncol=2, loc="best",
                  fontsize=LEGEND_FONTSIZE, framealpha=1, edgecolor="0.8",
                  handler_map={forward: _ArrowHandler(leftward=False), backward: _ArrowHandler(leftward=True)})


class Supp:
    """Every valid ordering of the four transformations, for one direction."""
    @classmethod
    def plot(cls, plot_data, direction="input-to-output"):
        T = plot_data.Ti2o if direction == "input-to-output" else plot_data.To2i
        plot_transformation_accuracies(T.accs, direction=direction, linewidth=1.5, grid_on=True, yl=[0.2,1.05], markersize=5)
        return plt.gcf()
