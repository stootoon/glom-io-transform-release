"""Supplementary figures for the matched rois: observed vs predicted.

One figure per metric. Each has two blocks -- the response-loss fits on the
left, the covariance-loss fits on the right -- and each block shows the observed
matrix, the two models' predictions, and a scatter of predicted against
observed with both models overlaid.

    from glom_io_transform.analysis.figures import matched_rois
    matched_rois.Supp.plot(data, metric="cov")

The response panels are also available on their own, so a figure that wants
them in its own grid can draw them onto axes it supplies:

    matched_rois.plot_response_heatmap(ax, M, roi_order=order, im_kwargs=style)
    matched_rois.plot_response_traces(ax, obs, {"Free": pred}, roi=3)
"""
import pandas as pd

from .figures import np, plt, GridSpec, spines_off
from .figures import Figure, rep_style, get_leaf_order_from_covariance
from glom_io_transform.analysis.figures import violin_plots as fig_violin_plots
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from glom_io_transform.model_fitting import proc_fit_models as pfm
from ..compute.matched_rois import (LOSSES, MODELS, METRICS, HALVES, corr_from,
                                    stabilizer_rotation)
from ..compute.generalization import as_labels

TITLES = {"resp": ("Responses", "roi", "odour"),
          "cov":  ("Covariance", "odour", "odour"),
          "corr": ("Correlation", "odour", "odour")}

LOSS_LABELS = {"resp": "fitted on responses", "cov": "fitted on covariances"}
HALF_LABELS = {"train": "training", "vld": "held out"}


def subset_label(plot_data):
    """', odours=<spec>' when a subset was used, so the figure says which."""
    spec = getattr(plot_data, "n_od_train", "max")
    return "" if spec in (None, "max") else f", odours={spec}"


def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def sign_align(pred, obs):
    """Flip each row of pred so it correlates positively with the same row of obs.

    Covariance fitting is quadratic in Z -- Cest = (ZX)' J (ZX) -- so it cannot
    see the sign of a channel's weight, and for the Diag model each gain's sign
    is independently unidentifiable. The signs are therefore arbitrary and are
    fixed here for display; nothing about the fit changes.
    """
    pred, obs = np.asarray(pred), np.asarray(obs)
    signs = np.array([np.sign(pearson(o, p)) or 1.0 for o, p in zip(obs, pred)])
    return pred * signs[:, None], signs


def uniform_ticks(vmin, vmax, nbins=5):
    """Ticks at the positive range's spacing, mirrored into the negative range.

    A TwoSlopeNorm gives each sign its own half of the bar, so a narrow negative
    range gets no ticks at all from the default locator -- the bar then looks
    unlabelled below zero. Where the negative range is too narrow to hold even
    one step, its endpoint is labelled instead, so the reader can still see how
    far down the colours go.
    """
    step = np.diff(MaxNLocator(nbins=nbins).tick_values(0, vmax))[0]
    pos = np.arange(0, vmax + step / 2, step)
    if vmin >= 0:
        return pos
    neg = np.arange(-step, vmin - step / 2, -step)
    neg = neg[neg >= vmin]
    return np.concatenate([[vmin] if len(neg) == 0 else neg[::-1], pos])


# ---------------------------------------------------------------------------
# Response panels, as standalone drawing functions.
#
# Supp draws whole figures of these; the main figure needs the same panels in
# its own grid. Both call the functions below, which draw onto an axes the
# caller supplies and decide nothing about layout.
# ---------------------------------------------------------------------------

# The responses are per-odour z-scored, so they straddle zero -- the matched
# input is 69% negative. A diverging map centred at zero shows that; a
# sequential one has to put "no response" at an end of its range.
RESP_CMAP = "RdBu_r"
# None means take the limits from the data by percentile. A fixed (0, 1) clipped
# every negative value to a single colour, which is most of the input panel, and
# it also disabled the TwoSlopeNorm that response_style and uniform_ticks below
# are written to support.
RESP_VLIM = None
PCTILE    = (1, 99)
FONTSIZE  = 9

# The main figure's left third, as a grid of its own. It holds three groups of
# rows -- the response matrices with one roi under each, the four correlation
# matrices in a 2 x 2, and the violin summary -- and each group needs its own
# row spacing, which a single grid cannot give (hspace is uniform within one).
# So each group is a grid in its own right, over a COMMON column base: six
# columns, spanned three at a time by the correlation panels and two at a time
# by the response panels, so rows of two and rows of three land on the same
# edges. The seventh column is a narrow strip holding the colour bars, which
# keeps them inside the left third rather than overhanging the middle one.
# Every panel spans three columns, and the empty columns either side of each
# pair are what makes the block narrower than the third it sits in. That width
# is what the square correlation panels are sized by, so squeezing here is how
# the correlations are made smaller WITHOUT them drifting out of line with the
# heat maps: narrow every panel by the same amount and they all still share two
# edges. The last column is the colour bar strip.
LEFT_GUTTER  = 0.35
LEFT_WIDTHS  = ((LEFT_GUTTER,) + (1.0,) * 3 + (LEFT_GUTTER,)) * 2
PANEL_COLS   = (1, 6)       # first column of the left and right panel
PANEL_SPAN   = 3
# The correlation group's height is the one that is not free: two square panels
# need twice the panel width plus a gap, and a group shorter than that shrinks
# them, which is what pulls their edges in from the heat maps' above. Erring
# tall is safe -- the slack shows vertically, where it is barely visible, and
# the squares keep the full panel width.
LEFT_GROUPS  = (3.5, 4.0, 1.35)  # responses, correlations, violin
LEFT_WSPACE  = 0.45
LEFT_HSPACE  = 0.16         # between the three groups
RESP_HEIGHTS = (1.0, 1.0, 0.8)  # two rows of heat maps, then the roi traces
RESP_HSPACE  = 0.16         # small: the traces belong to the maps above them
CORR_HSPACE  = 0.22
# The colour bars are inset against the LEFT hand panel of their block, into the
# gap between the two columns. Off the right hand panel they would sit in the
# space between this block and the next, which is space the neighbour wants.
# CBAR_X is in the panel's axes coordinates, so 1.04 is a gap of 4% of a panel
# width.
CBAR_X       = 1.04
CBAR_W       = 0.05
CBAR_NBINS   = 4            # a bar this narrow has room for few labels
TRACE_HEADROOM = 0.55       # of the shared range, for the legend to sit in

# The middle third: the connectivity, a row per piece of Z = R S + 1 b'. Three
# columns throughout -- the two fits' matrices side by side, then what the seeds
# say about them -- except the bottom row, where the orbit is a line plot and
# takes the width of the two matrix columns.
MID_ROWS    = (1.0, 0.89, 0.89, 1.0)
# The matrix rows get the width, since a square panel is only as big as the
# narrower of its two sides and these are all limited by width. The top row has
# its own ratios: b' is a single column of numbers and needs a sliver, and what
# it leaves over goes to the violin beside it. Only the internal boundaries
# differ -- the block's outer edges are shared by every row.
# The third column is empty, and is where the matrices' colour bars go. Without
# it the bar and its labels are drawn over the next panel's y label, since an
# inset colour bar takes no space from the grid.
MID_WIDTHS     = (1.45, 1.45, 0.22, 0.92)
MID_PANEL_COLS = (0, 1, 3)
# The top row has no matrices and so no colour bar column, and its middle panel
# is a single column of numbers. It gets its own widths and its own, smaller,
# spacing: b' belongs beside the violin it is summarised by, and what that
# leaves over goes to the schematic.
MID_TOP_WIDTHS = (2.5, 0.28, 0.72)
MID_TOP_COLS   = (0, 1, 2)
MID_TOP_WSPACE = 0.28
MID_WSPACE  = 0.35          # the matrices carry colour bars in their gaps
MID_HSPACE  = 0.545

# The three blocks do not need equal shares of the figure. The connectivity
# matrices are square and small, so the middle takes width from the left, whose
# own panels have a gutter to give; the outer grid is twelve columns, four per
# block, so a block's share is repeated four times.
THIRDS = (0.80, 1.38, 0.82)
OUTER_WIDTHS = tuple(w for third in THIRDS for w in (third,) * 4)

NUMERALS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi",
            "xii"]

# Colours for model names that may carry a loss suffix ("Free_cov"): hue says
# which model, brightness which loss. Defined next to model_color in pfm, since
# the generalization figures need them too.
variant_color = pfm.variant_color
variant_label = pfm.variant_label

# The r2 panel shows four connectivities: the two fits, and the two hybrids that
# take the rotation from one fit and the stretch from the other. Brightness says
# which fit the ROTATION came from -- light for cov, dark for resp, as elsewhere
# in the figure -- and hue separates the hybrids from the fits they are built
# out of, without moving so far that they stop reading as the same family.
GREEN_SHIFT = 34    # degrees, turquoise (174) toward green (120)


def relative_luminance(color):
    """How light a colour reads, which is not the same as its HSV value."""
    r, g, b = mcolors.to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def greener(color, degrees=GREEN_SHIFT):
    """The same colour with its hue moved toward green, at the same luminance.

    Negative degrees move the other way, toward blue. Rotating hue at a fixed
    HSV value does NOT keep a colour looking equally light -- blue at v=0.95
    reads much darker than turquoise at v=0.95 -- which would break the
    light/dark pairing that carries whether a rotation is present. So the value
    is re-solved to land back on the original luminance.
    """
    hue, sat, _ = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    hue = (hue - degrees / 360) % 1.0
    target = relative_luminance(color)
    values = np.linspace(0.05, 1.0, 256)
    lums = np.array([relative_luminance(mcolors.hsv_to_rgb((hue, sat, v))) for v in values])
    return mcolors.hsv_to_rgb((hue, sat, values[np.argmin(np.abs(lums - target))]))

def lighter(color, factor=0.5):
    """The same colour, but lighter by the given factor (0-1)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def darker(color, factor=0.5):
    """The same colour, but darker by the given factor (0-1)."""
    return tuple(x * (1 - factor) for x in mcolors.to_rgb(color))

OBS_STYLE    = dict(lw=1.1, color="0.2")
MODEL_STYLE  = dict(lw=1.9, alpha=0.95)
TRACE_LEGEND = dict(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.06))


def value_limits(values, vlim):
    """(vmin, vmax) for a whole figure: `vlim` if it is fixed, else percentiles.

    Every panel of a figure shares one scale -- comparing panels is the point,
    and a per-panel autoscale would defeat it -- so pass every value drawn.
    """
    if vlim is not None:
        return tuple(vlim)
    return tuple(np.nanpercentile(np.asarray(values).ravel(), PCTILE))


def response_style(values=None, vlim=RESP_VLIM, cmap=RESP_CMAP, center=0.0):
    """imshow kwargs for the response heat maps.

    Pass every value that will be drawn, so that one scale covers the whole
    figure -- comparing panels is the point, and a per-panel autoscale would
    defeat it. With `vlim` given the values are unused; with vlim=None they set
    the limits by percentile.
    """
    vmin, vmax = value_limits(values, vlim)
    if center is not None and vmin < center < vmax:
        # Data straddling zero needs the map's midpoint pinned there, or each
        # sign gets a share of the range set by the other one's spread.
        return dict(cmap=cmap, norm=TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax))
    return dict(cmap=cmap, vmin=vmin, vmax=vmax)


def roi_cluster_order(M, method="average"):
    """Rois ordered by hierarchically clustering their response profiles.

    M is rois x odours. The clustering runs on the correlation between rois
    ACROSS ODOURS, so rois that answer to the same odours end up adjacent and
    the heat map shows blocks rather than a scatter.

    One order is computed and applied to every panel, so a row means the same
    roi in all of them -- clustering each panel separately would put a
    different roi on each row and the panels could not be read against one
    another.
    """
    return get_leaf_order_from_covariance(np.cov(np.asarray(M)), method)


def roi_order_by_variance(obs):
    """Rois most variable first, by variance across odours of the observed data."""
    return np.argsort(-np.asarray(obs).var(axis=1))


def plot_response_heatmap(ax, M, roi_order=None, im_kwargs=None, fontsize=FONTSIZE,
                          roi_labels=True, ylabel=None, ylabel_color="0.2",
                          xlabel=None, xticklabels=True):
    """One rois x odours response matrix on `ax`. Returns the image.

    `roi_order` reorders the rows; the tick labels keep the ORIGINAL indices,
    so a row can be traced back to the matched pair it came from.
    """
    M = np.asarray(M)
    roi_order = np.arange(M.shape[0]) if roi_order is None else np.asarray(roi_order)
    im = ax.imshow(M[roi_order], aspect="auto", interpolation="nearest",
                   **(response_style() if im_kwargs is None else im_kwargs))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, color=ylabel_color)
    ax.set_yticks(np.arange(len(roi_order)))
    ax.set_yticklabels([str(j) for j in roi_order] if roi_labels else [],
                       fontsize=fontsize * 0.62)
    ax.tick_params(axis="y", length=2, pad=1)
    ax.tick_params(axis="x", labelsize=fontsize * 0.75)
    if not xticklabels:
        ax.set_xticklabels([])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize * 0.9)
    return im


def plot_response_traces(ax, obs, preds=None, roi=None, fontsize=FONTSIZE,
                         ylabel=None, xlabel=None, xticklabels=True,
                         legend=False, legend_kwargs=None):
    """Observed and predicted responses for one roi on `ax`.

    `obs` and each value of `preds` is either the roi's own trace or the whole
    rois x odours matrix, in which case `roi` picks the row. `preds` is keyed by
    model name, which sets each line's colour and its legend entry.
    """
    def row(M):
        M = np.asarray(M)
        if M.ndim == 1:
            return M
        assert roi is not None, "roi is needed to pick a row out of a matrix."
        return M[roi]

    ax.plot(row(obs), label="observed", **OBS_STYLE)
    for name, M in (preds or {}).items():
        ax.plot(row(M), color=variant_color(name), label=variant_label(name),
                **MODEL_STYLE)

    if ylabel is None and roi is not None:
        ylabel = f"roi {roi}"
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    if not xticklabels:
        ax.set_xticklabels([])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize * 0.9)
    if legend:
        ax.legend(**{"fontsize": fontsize * 0.7, **TRACE_LEGEND, **(legend_kwargs or {})})
    spines_off(ax)
    return ax


# ---------------------------------------------------------------------------
# Matrix panels -- covariances and correlations -- as standalone functions,
# for the same reason as the response panels above: Supp draws whole figures of
# them, the main figure wants them in its own grid.
# ---------------------------------------------------------------------------

# Correlations reuse the scale the representation matrices are drawn with, so
# the two figures can be read against each other. Covariances have no natural
# scale and take theirs from the data.
MATRIX_STYLE = {"cov":  {"cmap": "rainbow", "vlim": None},
                "corr": {"cmap": rep_style["cmap"], "vlim": rep_style["vlim"]}}


def matrix_limits(values, metric):
    """(vmin, vmax) shared by every panel of a covariance or correlation figure."""
    return value_limits(values, MATRIX_STYLE[metric]["vlim"])


def matrix_style(values, metric):
    """imshow kwargs for a covariance or correlation matrix."""
    vmin, vmax = matrix_limits(values, metric)
    return dict(cmap=MATRIX_STYLE[metric]["cmap"], vmin=vmin, vmax=vmax)


def cluster_order(C):
    """Odour order from hierarchically clustering an observed covariance matrix.

    C is symmetrised first: a covariance taken between two halves of the data is
    only nearly symmetric, and clustering needs a symmetric similarity. Use the
    order from the COVARIANCE for the correlation figure too, so the two show
    the same odours in the same places.
    """
    C = np.asarray(C)
    return get_leaf_order_from_covariance((C + C.T) / 2)


def plot_matrix(ax, M, order=None, im_kwargs=None, fontsize=FONTSIZE,
                title=None, title_color="0.2", xlabel=None, ylabel=None,
                yticklabels=True, xticklabels=True, aspect="auto"):
    """One odours x odours matrix on `ax`. Returns the image.

    `order` reorders both axes together, so the matrix stays symmetric.

    `aspect` is imshow's: "auto" fills the axes box, "equal" keeps the matrix
    square whatever shape the box is. A figure that draws these in a grid of
    its own wants the second.
    """
    M = np.asarray(M)
    if order is not None:
        order = np.asarray(order)
        M = M[order][:, order]
    im = ax.imshow(M, aspect=aspect, interpolation="nearest",
                   **(matrix_style(M, "corr") if im_kwargs is None else im_kwargs))
    if title is not None:
        ax.set_title(title, fontsize=fontsize, color=title_color)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize * 0.9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize * 0.9)
    if not yticklabels:
        # Panels side by side share the odour axis, so repeating the tick
        # labels only crowds them against the colour bar.
        ax.set_yticklabels([])
    if not xticklabels:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=fontsize * 0.75)
    return im


def add_colorbar(fig, ax, im, fontsize=FONTSIZE, rect=(1.06, 0.0, 0.05, 1.0),
                 orientation="vertical", ticks=None):
    """A slim colour bar inset against `ax`. Returns its axes."""
    cax = ax.inset_axes(list(rect))
    return fill_colorbar(fig, cax, im, fontsize=fontsize,
                         orientation=orientation, ticks=ticks)


def fill_colorbar(fig, cax, im, fontsize=FONTSIZE, orientation="vertical",
                  ticks=None):
    """A colour bar filling an axes the caller made. Returns that axes.

    The axes is usually a cell of the figure's own grid, which is the reliable
    way to line a bar up with panels that have a fixed aspect: an inset is
    positioned against its parent's BOX, and a square panel in a wider box does
    not fill it.
    """
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    if ticks is not None:
        cb.set_ticks(ticks)
    else:
        # A bar a few millimetres wide cannot carry matplotlib's default number
        # of labels without them running together.
        cb.locator = MaxNLocator(nbins=CBAR_NBINS)
        cb.update_ticks()
    # Ticks pointing INTO the bar: outward ones are drawn over the gap between
    # the bar and whatever is beside it, which is space the figure has to leave
    # empty for the sake of a 2 pt line.
    cb.ax.tick_params(direction="in", length=2.5, width=0.5, pad=1.5,
                      labelsize=fontsize * 0.7, color="0.35")
    cb.outline.set_linewidth(0.5)
    return cax


# ---------------------------------------------------------------------------
# The spectrum of the symmetric fit.
#
# Once the transformation is established as symmetric, the eigenvalues are what
# there is to say about it: a symmetric Z has no rotation to describe, only
# gains along an orthogonal set of modes. Reciprocal dendrodendritic inhibition
# gives W = GG' and hence Z = (I + GG')^-1, whose eigenvalues all lie in (0, 1].
# Drawing that interval is what turns a list of eigenvalues into a test.
# ---------------------------------------------------------------------------

# Eigenvalues have no identity across seeds, so they are matched by RANK. The
# band therefore mixes seed-to-seed variability with the spread that sorting
# induces on its own, which is worth a line in the caption.
ADMISSIBLE = (0.0, 1.0)      # what Z = (I + GG')^-1 allows
SEED_LINE  = dict(color="0.6", lw=0.4, alpha=0.5)


def sorted_spectra(Z_by_seed):
    """seeds x modes array of eigenvalues, largest first, one row per seed."""
    spectra = [np.sort(np.linalg.eigvalsh((np.asarray(Z) + np.asarray(Z).T) / 2))[::-1]
               for Z in Z_by_seed]
    return np.array(spectra)


def plot_zsym_spectrum(ax, Z_by_seed, fontsize=FONTSIZE, color=None,
                       show_seeds=True, admissible=ADMISSIBLE, legend=True,
                       quantiles=(25, 75)):
    """Median and inter-quantile band over seeds of the symmetric fit's spectrum.

    Median rather than mean, and quantiles rather than SDs, because the
    eigenvalues are not symmetric about their centre -- at the extreme ranks the
    mean sits up to half an SD away from the median -- so a band drawn
    symmetrically about a mean claims spread on a side the data does not have.
    `quantiles` widens or narrows it; the default is the IQR.

    `Z_by_seed` is any iterable of matrices, one per seed. The shaded strip is
    the interval a reciprocal architecture permits; eigenvalues above it are
    amplified modes and those below zero are inverted, and neither is reachable
    from Z = (I + GG')^-1.
    """
    spectra = sorted_spectra(Z_by_seed)
    median  = np.median(spectra, axis=0)
    lo_q, hi_q = np.percentile(spectra, list(quantiles), axis=0)
    rank    = np.arange(1, spectra.shape[1] + 1)
    color   = pfm.model_color("FreeSym") if color is None else color

    lo, hi = admissible
    ax.axhspan(lo, hi, color="0.90", zorder=0, label="reciprocal admissible")
    ax.axhline(lo, color="0.35", lw=0.9, ls="--", zorder=1)
    ax.axhline(hi, color="0.35", lw=0.9, ls=":", zorder=1)
    if show_seeds:
        # The individual seeds, so the band is not the only evidence that the
        # shape is stable rather than an average over dissimilar spectra. Only
        # the first is labelled, or the legend would carry one entry per seed.
        for i, row in enumerate(spectra):
            ax.plot(rank, row, zorder=2, label="seeds" if i == 0 else None, **SEED_LINE)
    ax.fill_between(rank, lo_q, hi_q, color=color, alpha=0.28, lw=0, zorder=3)
    # One entry for both pink artists: the line is the median, the band around
    # it the quantile range, and splitting them would say the same thing twice.
    band = "IQR" if tuple(quantiles) == (25, 75) else f"{quantiles[0]}-{quantiles[1]}%"
    ax.plot(rank, median, "o-", color=color, ms=3.5, lw=1.5, zorder=4,
            label=f"median, {band}")

    if legend:
        ax.legend(fontsize=fontsize * 0.7, frameon=False, loc="upper right",
                  handlelength=1.4, borderpad=0.2, labelspacing=0.3)
    ax.set_xlabel("mode (rank)", fontsize=fontsize * 0.9)
    ax.set_ylabel("eigenvalue of $Z_\\mathrm{sym}$", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    spines_off(ax)
    return ax


# ---------------------------------------------------------------------------
# The connectivity in the input's own basis.
#
# Z in the eigenbasis of the input covariance is the one description of the fit
# that HAS a closed form -- the Sylvester solution is entrywise there,
# Ztilde_ij = Ctilde_ij / (D_i + D_j) -- so it says what Z does to each input
# mode. It is not the eigenbasis of Z, so the positive/negative split of the
# spectrum panel is not what this shows; the two panels are complementary.
# ---------------------------------------------------------------------------

MODE_CMAP   = "RdBu_r"
MODE_PCTILE = 99          # symmetric limits, robust to a single large entry


def mode_connectivity(Z_by_seed, V_by_seed, reference=0):
    """Z in the input eigenbasis, averaged over seeds.

    Each seed has its own eigenvectors, so two things have to be settled before
    an average means anything. Modes are matched by RANK, most variable first,
    which is what the caller's ordering already provides. And an eigenvector is
    only defined up to sign -- flipping v_i flips row and column i of Ztilde --
    so each seed's basis is sign-aligned to a reference seed's before averaging,
    or the off-diagonal entries would cancel against each other.

    The diagonal is immune to this: flipping v_i leaves Ztilde_ii alone.
    """
    ref = np.asarray(V_by_seed[reference])
    per_seed = []
    for Z, V in zip(Z_by_seed, V_by_seed):
        V = np.asarray(V)
        signs = np.sign(np.sum(V * ref, axis=0))
        signs[signs == 0] = 1.0
        V = V * signs
        per_seed.append(V.T @ np.asarray(Z) @ V)
    return np.mean(per_seed, axis=0), np.array(per_seed)


def plot_mode_connectivity(ax, Z_by_seed, V_by_seed, fontsize=FONTSIZE,
                           reference=0, colorbar=True, cmap=MODE_CMAP):
    """The seed-averaged Ztilde as a heat map, on a symmetric diverging scale.

    Row and column are input modes ordered by variance, so the top left corner
    is what Z does among the modes the input actually varies along, and the
    bottom right what it does in the directions the input barely explores.
    """
    mean_Z, _ = mode_connectivity(Z_by_seed, V_by_seed, reference=reference)
    # Centred at zero, since the sign is the point: a symmetric scale is the
    # only one on which equal positive and negative weights look equal.
    lim = np.nanpercentile(np.abs(mean_Z), MODE_PCTILE)
    im = ax.imshow(mean_Z, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xlabel("input mode (rank)", fontsize=fontsize * 0.9)
    ax.set_ylabel("input mode (rank)", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    if colorbar:
        add_colorbar(ax.get_figure(), ax, im, fontsize=fontsize)
    return im


# ---------------------------------------------------------------------------
# Surrogate calibration panel.
#
# The ladder shows Sym tying or beating Free on the real data, which is only
# evidence of symmetry if the comparison could have come out the other way. The
# surrogate sweep supplies that: data from a KNOWN truth Z = S + alpha A, with
# alpha the size of the antisymmetric part relative to the symmetric one, run
# through the same solvers. This panel puts the real difference on the same axis
# as the calibration, so the reader can see which alpha the observation sits at.
# ---------------------------------------------------------------------------

DIFF_COLUMN   = "Sym - Free"
DIFF_LABEL    = "$R^2$(Sym) $-$ $R^2$(Free)"
ALPHA_LABEL   = "asymmetry of the truth, $\\alpha$"
OBSERVED_LABEL = "observed"
# Light where the truth is symmetric, darkening as it becomes less so, so the
# ramp itself reads as the x axis. The observed violin is left out of the ramp
# and takes the Sym colour it has in the ladder.
SURROGATE_SHADE = "0.85"
SURROGATE_DARKEST = 0.6


def surrogate_violins(surrogate_df, observed, diff_column=DIFF_COLUMN,
                      observed_label=OBSERVED_LABEL):
    """The violins for the surrogate panel: one per alpha, then the real data.

    `surrogate_df` is what compute.matched_rois.surrogate_r2 returns: one row
    per (alpha, seed, train). The differences are medianed over trains within a
    seed and the violins run over seeds, which is how the ladder treats its own
    r2_df -- so the observed violin here and the gap between the Free and Sym
    rungs there are the same numbers.

    `observed` is the real-data differences, one per seed.

    Returns a list of ViolinPlotData, left to right, as violin_data does.
    """
    # surrogate_r2 accepts alpha=None to mean the real data. The ladder is the
    # canonical source for that, so those rows are not used here.
    df = surrogate_df[surrogate_df["alpha"].notna()]
    per_seed = df.groupby(["alpha", "seed"])[diff_column].median().reset_index()

    alphas = sorted(per_seed["alpha"].unique())
    shades = [darker(SURROGATE_SHADE, SURROGATE_DARKEST * i / max(1, len(alphas) - 1))
              for i in range(len(alphas))]

    panel = [fig_violin_plots.ViolinPlotData(
                 vals=list(per_seed.loc[per_seed["alpha"] == alpha, diff_column].values),
                 col=shade, lab=f"{alpha:g}")
             for alpha, shade in zip(alphas, shades)]
    panel.append(fig_violin_plots.ViolinPlotData(
        vals=list(np.asarray(observed).ravel()),
        col=pfm.model_color("FreeSym"), lab=observed_label))
    return panel


def plot_surrogate_alpha(ax, surrogate_df, observed, fontsize=FONTSIZE,
                         diff_column=DIFF_COLUMN, observed_label=OBSERVED_LABEL):
    """Sym minus Free against the asymmetry of the truth, with the real data.

    Above the zero line the symmetric fit is the better one; below it the
    unconstrained fit has found real asymmetry to exploit. Where the violins
    cross zero is the smallest asymmetry this pipeline would have detected.
    """
    violins = surrogate_violins(surrogate_df, observed, diff_column=diff_column,
                                observed_label=observed_label)
    fig_violin_plots.draw_violins(ax, violins)

    # Zero is the decision line, not just a gridline: it is where the two
    # models tie, so it carries the panel's whole claim.
    ax.axhline(0.0, color="0.35", lw=0.9, ls="--", zorder=0)
    # The observed violin is data, not a point on the alpha axis, so it is
    # fenced off rather than left to read as one more alpha.
    ax.axvline(len(violins) - 0.5, color="0.75", lw=0.8, ls=":", zorder=0)

    ax.set_xlabel(ALPHA_LABEL, fontsize=fontsize * 0.9)
    ax.set_ylabel(DIFF_LABEL, fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    ax.yaxis.grid(True, which="major", color="0.8", lw=0.5, zorder=0, ls=":")
    spines_off(ax)
    return ax


# ---------------------------------------------------------------------------
# The connectivity itself: Z = R S + 1 b'.
#
# notes/fit_cov_resp.tex: the covariance loss sees J Z only, and only through
# (J Z)'(J Z) = S'S, so it determines the stretch and nothing else. The response
# loss determines the rotation too, and the mean component 1 b' is invisible to
# one of them and free in the other. The panels below take those three pieces
# one at a time -- what the baseline is worth, whether the two fits agree about
# the stretch, and whether the rotation is the identity.
#
# Everything here reads plot_data.polar, a seed -> {loss: Polar} built in
# compute.matched_rois; the Polar class is where the decomposition's one trap is
# documented (S is singular by construction).
# ---------------------------------------------------------------------------

# One map per factor, so a glance says which is which: a rotation and a stretch
# are different kinds of object and only their zero is shared.
CONN_CMAP    = "RdBu_r"
CONN_CMAPS   = {"R": "RdBu_r", "S": "PuOr_r"}
CONN_PCTILE  = 99          # symmetric limits, robust to a single large entry
CONN_LOSSES  = ("cov", "resp")
NULL_DRAWS   = 50          # rotations per seed for the alignment null
NULL_COLOR   = "0.65"
NULL_SEED    = 0
DEGREES      = 180 / np.pi


def loss_color(loss):
    """The colour the rest of the figure gives a fit, from its loss alone."""
    return variant_color(f"Free_{loss}")


def loss_label(loss):
    return f"Free ({loss})"


def note(ax, text, fontsize=FONTSIZE):
    """An empty panel saying what is missing, for data an old pickle lacks."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=fontsize * 0.8, color="0.5")
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def conn_style(matrices, pctile=CONN_PCTILE, cmap=CONN_CMAP, off_diagonal=True):
    """imshow kwargs on a diverging scale centred at zero, shared by the panels.

    Centred because the sign is the point -- a rotation's entries and a
    stretch's off-diagonals both come in both signs -- and shared because the
    two fits are only worth drawing side by side if one scale covers them.

    `off_diagonal` takes the limits from the off-diagonal entries alone. Both
    matrices here sit near the identity, so their diagonals are an order above
    everything else, and a scale that fits them leaves the structure invisible.
    The diagonal then saturates, which costs nothing: that it is large is the
    one thing about these matrices nobody needs a colour bar to learn.
    """
    def values(M):
        M = np.abs(np.asarray(M))
        # Nothing to exclude unless the matrix has a diagonal to speak of: the
        # baseline strip comes through here too, and it is m x 2.
        if off_diagonal and M.ndim == 2 and M.shape[0] == M.shape[1]:
            return M[~np.eye(len(M), dtype=bool)]
        return M.ravel()
    lim = float(np.nanpercentile(np.concatenate([values(M) for M in matrices]), pctile))
    return dict(cmap=cmap, vmin=-lim, vmax=lim)


def plot_connectivity(ax, M, order=None, im_kwargs=None, fontsize=FONTSIZE,
                      title=None, title_color="0.2", xlabel=None, ylabel=None,
                      ticklabels=True, aspect="equal"):
    """One roi x roi matrix -- a rotation or a stretch -- on `ax`.

    `order` permutes rows and columns together, which for these matrices is a
    change of basis by a permutation and so leaves what they mean alone. The
    tick labels keep the ORIGINAL indices, as the response heat maps' do.
    """
    M = np.asarray(M)
    if order is not None:
        order = np.asarray(order)
        M = M[order][:, order]
    im = ax.imshow(M, aspect=aspect, interpolation="nearest",
                   **(conn_style([M]) if im_kwargs is None else im_kwargs))
    if title is not None:
        ax.set_title(title, fontsize=fontsize, color=title_color)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize * 0.9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize * 0.9)
    # The rois are named down the side only: 16 two-digit labels do not fit
    # along the bottom of a panel this size, and the axis is the same either way.
    labels = list(range(len(M))) if order is None else list(order)
    ax.set_yticks(np.arange(len(M)))
    ax.set_yticklabels([str(j) for j in labels] if ticklabels else [],
                       fontsize=fontsize * 0.62)
    ax.set_xticks(np.arange(len(M)))
    ax.set_xticklabels([])
    ax.tick_params(length=2, pad=1)
    return im


def plot_baseline_strip(ax, polar, losses=CONN_LOSSES, order=None, im_kwargs=None,
                        fontsize=FONTSIZE, ylabel="input roi"):
    """b' for each fit, side by side as one narrow image.

    b' is indexed by INPUT roi, not by output roi: Zbar = 1 b'^T means every
    output roi receives the same weighted combination of inputs. It is a
    weighting, not a per-output offset.
    """
    B = np.column_stack([polar[loss].b for loss in losses])
    if order is not None:
        B = B[np.asarray(order)]
    im = ax.imshow(B, aspect="equal", interpolation="nearest",
                   **(conn_style([B]) if im_kwargs is None else im_kwargs))
    ax.set_xticks(range(len(losses)))
    # Rotated: the strip is two pixels wide, so side by side the two labels
    # would sit on top of each other.
    ax.set_xticklabels(list(losses), fontsize=fontsize * 0.75, rotation=90)
    for tick, loss in zip(ax.get_xticklabels(), losses):
        tick.set_color(loss_color(loss))
    ax.set_ylabel(ylabel, fontsize=fontsize * 0.9)
    # The same original indices the matrices beside it are labelled with, so a
    # row means one roi everywhere in the block.
    ax.set_yticks(np.arange(len(B)))
    ax.set_yticklabels([str(j) for j in (range(len(B)) if order is None else order)],
                       fontsize=fontsize * 0.62)
    ax.tick_params(length=2, pad=1)
    return im


def baseline_fractions(polar_by_seed, loss):
    """|Zbar| / |Z| per seed: how much of the connectivity the baseline is."""
    out = []
    for _, pol in sorted(polar_by_seed.items()):
        p = pol[loss]
        Z = p.R @ p.S + p.Zbar
        out.append(np.linalg.norm(p.Zbar) / np.linalg.norm(Z))
    return np.array(out)


def plot_baseline_fraction(ax, polar_by_seed, losses=CONN_LOSSES,
                           fontsize=FONTSIZE):
    """How large the mean component is, relative to the whole connectivity.

    A norm, not a prediction: what the baseline is worth for GENERALIZATION is
    the ladder's business, in the rungs fitted with and without 1 b'.
    """
    panel = [fig_violin_plots.ViolinPlotData(
                 vals=list(baseline_fractions(polar_by_seed, loss)),
                 col=loss_color(loss), lab=loss_label(loss))
             for loss in losses]
    fig_violin_plots.draw_violins(ax, panel)
    # Where a fit with nothing to say would land. The regularizer pulls Z toward
    # I, and I has a mean component of its own -- 11'/m, a fraction 1/sqrt(m) of
    # I's norm -- so this line is what "the baseline is only the prior" looks
    # like. The covariance fit cannot do anything else: its loss is blind here.
    m = len(next(iter(polar_by_seed.values()))[losses[0]].b)
    ax.axhline(1 / np.sqrt(m), color="0.35", lw=0.9, ls="--", zorder=0,
               label="identity")
    ax.set_ylabel("$\\|\\bar{Z}\\| \\, / \\, \\|Z\\|$", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    ax.yaxis.grid(True, which="major", color="0.8", lw=0.5, zorder=0, ls=":")
    spines_off(ax)
    return ax


def frobenius_cosine(A, B):
    """<A, B> / (|A| |B|): how aligned two matrices are, free of their scale."""
    A, B = np.asarray(A), np.asarray(B)
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B)))


def stretch_cosines(polar_by_seed, n_null=NULL_DRAWS, seed=NULL_SEED,
                    losses=CONN_LOSSES):
    """(observed, null) alignment between the two fits' stretches.

    The statistic is the Frobenius cosine between the DEVIATIONS from the
    identity. Deviations, because both fits are pulled toward I by their
    regularizer and a raw cosine would mostly report that shared pull; and a
    cosine rather than a distance, because the two were selected at different
    lambdas and so are shrunk by different amounts.

    The null re-rotates one fit's stretch: S -> O S O', which keeps its spectrum
    exactly and destroys only its orientation. O comes from the rotations the
    covariance loss cannot see -- those that fix 1, see stabilizer_rotation --
    since those are the alternatives the fit could actually have returned.

    One caveat for the caption: both stretches are singular by construction, so
    both deviations carry a -1 eigenvalue, and the two fits agreeing on WHERE
    that direction lies counts toward the observed cosine. That is agreement
    between the fits rather than an artefact, but it is not agreement about the
    stretch's interesting part.
    """
    rng = np.random.default_rng(seed)
    observed, null = [], []
    for _, pol in sorted(polar_by_seed.items()):
        A, B = pol[losses[0]].S, pol[losses[1]].S
        I = np.eye(len(A))
        observed.append(frobenius_cosine(A - I, B - I))
        for _ in range(n_null):
            O = stabilizer_rotation(len(A), rng)
            null.append(frobenius_cosine(O @ A @ O.T - I, B - I))
    return np.array(observed), np.array(null)


def plot_stretch_alignment(ax, polar_by_seed, n_null=NULL_DRAWS,
                           fontsize=FONTSIZE, losses=CONN_LOSSES):
    """Do the two fits stretch along the same axes, or only by the same amounts?"""
    observed, null = stretch_cosines(polar_by_seed, n_null=n_null, losses=losses)
    panel = [fig_violin_plots.ViolinPlotData(vals=list(observed),
                                             col=loss_color(losses[1]),
                                             lab="the two fits"),
             fig_violin_plots.ViolinPlotData(vals=list(null), col=NULL_COLOR,
                                             lab="re-rotated")]
    fig_violin_plots.draw_violins(ax, panel)
    ax.axhline(0.0, color="0.35", lw=0.9, ls="--", zorder=0)
    ax.set_ylabel("alignment of $S-I$", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    ax.yaxis.grid(True, which="major", color="0.8", lw=0.5, zorder=0, ls=":")
    spines_off(ax)
    return ax


def angle_spectra(polar_by_seed, loss):
    """seeds x m array of R's rotation angles in degrees, largest first."""
    return np.array([pol[loss].angles * DEGREES
                     for _, pol in sorted(polar_by_seed.items())])


def haar_angle_spectra(m, n_draws, seed=NULL_SEED):
    """The same, for rotations drawn at random from those that fix 1.

    What R would look like if the fit had no preference at all -- the reference
    the two fitted rotations are read against.
    """
    from ..compute.matched_rois import Polar
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_draws):
        O = stabilizer_rotation(m, rng)
        # Polar.angles works off R alone; a rotation is its own polar factor.
        out.append(Polar(R=O, S=np.eye(m), b=np.zeros(m), U=O, s=np.ones(m),
                         Vh=np.eye(m)).angles * DEGREES)
    return np.array(out)


def plot_angle_spectra(ax, polar_by_seed, losses=CONN_LOSSES, fontsize=FONTSIZE,
                       quantiles=(25, 75), n_null=NULL_DRAWS, legend=True):
    """Sorted rotation angles, median and inter-quantile band over seeds.

    Sorted rather than binned: the angles have no identity across seeds, so they
    are matched by rank, and the shape of the sorted curve is what says which
    kind of rotation R is. A run at 180 degrees is a REFLECTION -- exactly what a
    symmetric Z with negative eigenvalues has to produce -- and a run at 0 is a
    subspace left alone. A general rotation gives neither, which is what the
    random reference draws.
    """
    band = "IQR" if tuple(quantiles) == (25, 75) else f"{quantiles[0]}-{quantiles[1]}%"
    m = None
    for loss in losses:
        spectra = angle_spectra(polar_by_seed, loss)
        m = spectra.shape[1]
        rank = np.arange(1, m + 1)
        lo, hi = np.percentile(spectra, list(quantiles), axis=0)
        ax.fill_between(rank, lo, hi, color=loss_color(loss), alpha=0.25, lw=0)
        ax.plot(rank, np.median(spectra, axis=0), "o-", ms=3, lw=1.4,
                color=loss_color(loss), label=f"{loss_label(loss)}, {band}")

    null = haar_angle_spectra(m, n_null)
    ax.plot(np.arange(1, m + 1), np.median(null, axis=0), ls="--", lw=1.2,
            color=NULL_COLOR, label="random rotation")

    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_ylim(-8, 188)
    ax.set_xlabel("direction (rank)", fontsize=fontsize * 0.9)
    ax.set_ylabel("rotation angle (deg)", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    if legend:
        ax.legend(fontsize=fontsize * 0.7, frameon=False, loc="upper right",
                  handlelength=1.4, borderpad=0.2, labelspacing=0.3)
    spines_off(ax)
    return ax


# The three quantities the orbit moves: the covariance loss's fit term, which
# cannot move at all, the response loss's, which can, and the regularizer, which
# is what is left to choose the rotation once the covariance loss has stopped
# caring.
ORBIT_SERIES = (("cov_data", "covariance loss"),
                ("resp_data", "response loss"),
                ("reg", "regularizer"))


def orbit_ratios(orbit_df, column):
    """The loss relative to its value at the fit, per (seed, path, angle).

    Relative, because the three quantities have nothing in common but the fit
    they start from: an absolute axis would show one of them and flatten the
    others against it.
    """
    df = orbit_df
    base = df[df["t"] == 0].drop_duplicates("seed").set_index("seed")
    return df[column].values / base.loc[df["seed"], column].values


def plot_orbit(ax, orbit_df, fontsize=FONTSIZE, quantiles=(25, 75), legend=True):
    """Each loss along the rotations the covariance loss cannot see.

    Every point on the x axis is a connectivity R' S + 1 b' -- the fit with its
    rotation turned through that angle -- so the left edge IS the fit. The
    covariance loss's fit term does not move along the whole sweep, which is the
    claim the comparison between the two fits rests on. The response loss's does,
    and so does the regularizer: that is why the covariance fit still comes back
    with some particular rotation, and why the one it comes back with is the
    identity.

    Seeds and paths are pooled, since neither indexes anything the reader is
    being asked to compare -- the spread at each angle is over both.
    """
    df = orbit_df
    colors = {"cov_data": loss_color("cov"), "resp_data": loss_color("resp"),
              "reg": "0.5"}
    degrees = np.degrees(df["t"].values)
    grid = np.unique(degrees)
    for column, label in ORBIT_SERIES:
        ratio = orbit_ratios(df, column)
        by_angle = [ratio[degrees == t] for t in grid]
        lo, hi = (np.array([np.percentile(v, q) for v in by_angle])
                  for q in quantiles)
        ax.fill_between(grid, lo, hi, color=colors[column], alpha=0.25, lw=0)
        ax.plot(grid, [np.median(v) for v in by_angle], lw=1.6,
                color=colors[column], label=label)
    ax.axhline(1.0, color="0.35", lw=0.9, ls="--", zorder=0)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("rotation applied (deg)", fontsize=fontsize * 0.9)
    ax.set_ylabel("loss / loss at the fit", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    if legend:
        ax.legend(fontsize=fontsize * 0.7, frameon=False, loc="upper left",
                  handlelength=1.0, borderpad=0.2, labelspacing=0.3)
    spines_off(ax)
    return ax


# Swapping one factor of the response fit for the covariance fit's, and asking
# what it costs. These are not refits under a constraint -- those are the ladder's
# Rot, Orth and PSD rungs -- but the fitted Z with one piece of it replaced, so
# the answer is about THESE two fits rather than about the model class. The mean
# component stays in both: neither swap has anything to say about it.
# One factor at a time, ending with both: the last rung carries the covariance
# fit's rotation AND stretch with the response fit's mean component, which is
# the ladder's Free cov,bl rung. A bare Z_cov would differ from the others in
# their baseline as well, and would flatter the swaps.
ABLATIONS = (("Z_resp",   "Free\nresp"),
             ("Q=I",      "$R$ from\ncov"),
             ("P=P_cov",  "$S$ from\ncov"),
             ("Z_cov_bl", "$R, S$ from\ncov"))

# The two swaps are the only rungs in the figure that are neither a fit nor a
# refit, and they must not read as either: every hue the ladder uses is taken
# (teal for the fits, green for cov,bl, blue for PSD, purple for the orthogonal
# refits, pink for Sym), so they get a hue of their own. Light and dark carry
# the same thing they carry everywhere else -- which fit the ROTATION came from.
ABLATION_MIX = ("#dfc27d", "#8c510a")


def ablation_colors():
    """Hue says whether a rung is one of the fits or a mixture of the two;
    brightness says which fit its ROTATION came from -- light for the covariance
    fit, dark for the response fit, as everywhere else in the figure.

    The rungs that also appear in the ladder keep the colour they have there, so
    one rung is one colour across the whole figure.
    """
    cov, resp = loss_color("cov"), loss_color("resp")
    light, dark = ABLATION_MIX
    return {"Z_resp": resp, "Z_cov": cov, "Z_cov_bl": greener(cov),
            "Q=I": light, "P=P_cov": dark}


def plot_factor_ablation(ax, r2_df, ablations=ABLATIONS, fontsize=FONTSIZE):
    """Held-out R2 for each fit and for each one-factor swap between them.

    Medianed over trains within a seed and drawn over seeds, which is how the
    ladder treats the same frame -- so the two end violins here are the ladder's
    Free rungs, and what is between them is the price of each factor.
    """
    per_seed = r2_df.groupby("seed").median()
    colors = ablation_colors()
    panel = [fig_violin_plots.ViolinPlotData(vals=list(per_seed[column].values),
                                             col=colors[column], lab=label)
             for column, label in ablations]
    fig_violin_plots.draw_violins(ax, panel)
    ax.set_ylabel("$R^2$ (held out)", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    ax.yaxis.grid(True, which="major", color="0.8", lw=0.5, zorder=0, ls=":")
    spines_off(ax)
    return ax


def stretch_spectra(polar_by_seed, loss):
    """seeds x m array of the stretch's eigenvalues, largest first."""
    return np.array([np.sort(np.linalg.eigvalsh(pol[loss].S))[::-1]
                     for _, pol in sorted(polar_by_seed.items())])


def plot_stretch_spectra(ax, polar_by_seed, losses=CONN_LOSSES, fontsize=FONTSIZE,
                         quantiles=(25, 75), legend=True):
    """The two fits' stretches, mode by mode.

    The companion to the alignment panel: that one says the two stretch along
    the same axes, this one says they stretch by the same amounts, and S = S'
    needs both. The last mode is zero for both, by construction rather than by
    agreement -- J Z has a null direction.
    """
    band = "IQR" if tuple(quantiles) == (25, 75) else f"{quantiles[0]}-{quantiles[1]}%"
    for loss in losses:
        spectra = stretch_spectra(polar_by_seed, loss)
        rank = np.arange(1, spectra.shape[1] + 1)
        lo, hi = np.percentile(spectra, list(quantiles), axis=0)
        ax.fill_between(rank, lo, hi, color=loss_color(loss), alpha=0.25, lw=0)
        ax.plot(rank, np.median(spectra, axis=0), "o-", ms=3, lw=1.4,
                color=loss_color(loss), label=f"{loss_label(loss)}, {band}")
    ax.axhline(1.0, color="0.35", lw=0.9, ls=":", zorder=0)
    ax.set_xlabel("mode (rank)", fontsize=fontsize * 0.9)
    ax.set_ylabel("eigenvalue of $S$", fontsize=fontsize * 0.9)
    ax.tick_params(labelsize=fontsize * 0.75)
    if legend:
        ax.legend(fontsize=fontsize * 0.7, frameon=False, loc="upper right",
                  handlelength=1.4, borderpad=0.2, labelspacing=0.3)
    spines_off(ax)
    return ax


class Supp(Figure):
    """Observed and predicted matrices for one metric; a row per loss mode."""

    # Scatter: a random tenth, drawn larger. All the points make a solid blob at
    # this density; the trend is what the panel is for.
    SCATTER_FRAC = 0.10
    SCATTER_SIZE = 11
    SCATTER_SEED = 0
    # The response figure has far fewer points than a 48x48 matrix and much more
    # room, so it can afford more of them, drawn larger and more opaque.
    SCATTER_FRAC_RESP = 0.20
    SCATTER_SIZE_RESP = 22
    SCATTER_ALPHA_RESP = 0.75

    W_MAP     = {"cov": 1.9, "corr": 1.9}
    W_SCATTER = 2.4
    W_GAP     = 0.5
    H_ROW     = {"cov": 2.4, "corr": 2.4}

    # Response panels: how many rois to draw as traces, chosen by observed variance.
    N_TRACES = 3
    W_HEAT   = 2.6
    W_TRACE  = 3.0
    W_SCATTER_RESP = 4.2
    H_GAP    = 0.60
    H_UNIT   = 1.50

    @classmethod
    def plot(cls, plot_data, metric="cov", **kwargs):
        assert metric in METRICS, f"metric must be one of {METRICS}, got {metric!r}."
        print(f"PLOTTING FIGURE matched_rois ({metric=})")
        if metric == "resp":
            return cls.plot_responses(plot_data, **kwargs)
        return cls.plot_matrices(plot_data, metric=metric, **kwargs)

    @classmethod
    def plot_matrices(cls, plot_data, metric="cov", fig=None, figsize=None, fontsize=None,
                      losses=LOSSES, models=MODELS, half="vld", **kwargs):
        fontsize = FONTSIZE if fontsize is None else fontsize
        panels = {loss: plot_data.matrices[(loss, metric, half)] for loss in losses}
        # The observed matrix is the same data for both losses, so one scale.
        observed   = panels[losses[0]]["obs"]
        vmin, vmax = matrix_limits(observed, metric)
        im_kwargs  = matrix_style(observed, metric)
        order      = cluster_order(plot_data.matrices[(losses[0], "cov", half)]["obs"])

        w_map = cls.W_MAP[metric]
        widths = [w_map] * (1 + len(models)) + [cls.W_SCATTER]
        if figsize is None:
            figsize = (sum(widths) + 2.2, cls.H_ROW[metric] * len(losses) + 0.9)
        fig = plt.figure(figsize=figsize) if fig is None else fig
        gs = GridSpec(len(losses), len(widths), width_ratios=widths, figure=fig,
                      top=0.88, bottom=0.10, left=0.10, right=0.99,
                      wspace=0.45, hspace=0.55)

        axes = {}
        title, ylab, xlab = TITLES[metric]
        for i, loss in enumerate(losses):
            p = panels[loss]
            last = (i == len(losses) - 1)
            for j, key in enumerate(["obs"] + list(models)):
                ax = fig.add_subplot(gs[i, j])
                im = plot_matrix(
                    ax, p[key], order=order, im_kwargs=im_kwargs, fontsize=fontsize,
                    title="observed" if key == "obs" else key,
                    title_color="0.2" if key == "obs" else variant_color(key),
                    xlabel=xlab if last else None,
                    ylabel=ylab if j == 0 else None,
                    yticklabels=(j == 0))
                axes[f"{loss}_{key}"] = ax
                if key == "obs":
                    # Attached to the observed panel, since its data sets the scale.
                    axes[f"{loss}_cbar"] = add_colorbar(fig, ax, im, fontsize=fontsize)

            ax = fig.add_subplot(gs[i, len(widths) - 1])
            obs = np.asarray(p["obs"]).ravel()
            # The same random subset for every model, so the panels compare.
            rng = np.random.default_rng(cls.SCATTER_SEED)
            k = max(1, int(round(cls.SCATTER_FRAC * obs.size)))
            sub = rng.choice(obs.size, size=k, replace=False)
            for name in models:
                pred = np.asarray(p[name]).ravel()
                # r over ALL the points, not just the drawn subset.
                ax.scatter(obs[sub], pred[sub], s=cls.SCATTER_SIZE, alpha=0.45,
                           color=pfm.model_color(name), linewidths=0,
                           label=f"{name}  r = {pearson(obs, pred):+.2f}")
            obs = obs[sub]
            # The same range as the heat maps, so the two are read on one scale.
            # Points outside it are clipped, exactly as the maps saturate.
            lim = (vmin, vmax)
            ax.plot(lim, lim, lw=0.8, color="0.4", zorder=0)
            ax.set_xlim(*lim); ax.set_ylim(*lim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("observed", fontsize=fontsize * 0.9)
            ax.set_ylabel("predicted", fontsize=fontsize * 0.9)
            ax.tick_params(labelsize=fontsize * 0.75)
            ax.legend(fontsize=fontsize * 0.8, frameon=False, markerscale=3, loc="upper left")
            spines_off(ax)
            axes[f"{loss}_scatter"] = ax

            # Row label, naming what the row's models were fitted on.
            b = axes[f"{loss}_obs"].get_position()
            fig.text(0.012, (b.y0 + b.y1) / 2, LOSS_LABELS.get(loss, loss),
                     rotation=90, ha="left", va="center", fontsize=fontsize * 1.15)

        fig.suptitle(f"Matched rois: {title.lower()}, observed vs predicted "
                     f"(seed {plot_data.seed}, train {plot_data.train}{subset_label(plot_data)})",
                     fontsize=fontsize * 1.3, y=0.98)
        return axes

    @classmethod
    def plot_responses(cls, plot_data, fig=None, figsize=None, fontsize=None,
                       losses=LOSSES, models=MODELS, n_traces=None,
                       show_scatter=True, show_train=True, **kwargs):
        """Responses: one block per loss, stacked vertically.

        Each block shows the data the model was FITTED to on the left and the
        held-out data it was scored on next to it, so a model that never fitted
        can be told apart from one that fitted and failed to generalise. Within
        each half: the observed matrix over the two predictions with odours on
        x, then traces for the most variable rois. The scatter at the right is
        the held-out data.
        """
        fontsize = FONTSIZE if fontsize is None else fontsize
        n_traces = cls.N_TRACES if n_traces is None else n_traces
        rows     = ["obs"] + list(models)

        halves = HALVES if show_train else ("vld",)
        panels = {(half, loss): plot_data.matrices[(loss, "resp", half)]
                  for half in halves for loss in losses}

        obs = np.asarray(panels[("vld", losses[0])]["obs"])          # rois x odours
        # One scale over everything drawn, so the halves are comparable.
        every = np.concatenate([np.asarray(m).ravel() for p in panels.values() for m in p.values()])
        vmin, vmax = value_limits(every, RESP_VLIM)
        im_kwargs  = response_style(every)

        # Rois ordered by observed variance on the held-out data, most variable
        # first. The same order and the same traced rois in both halves, so the
        # rows line up across the whole figure.
        roi_order = roi_order_by_variance(obs)
        top = roi_order[:n_traces]

        # Covariance fitting is blind to channel sign, so those predictions carry
        # arbitrary signs. Fix them against the observed data for display.
        shown, flipped = {}, {}
        for key, p in panels.items():
            half, loss = key
            shown[key] = {"obs": np.asarray(p["obs"])}
            for name in models:
                if loss == "resp":
                    shown[key][name] = np.asarray(p[name])
                else:
                    aligned, signs = sign_align(p[name], p["obs"])
                    shown[key][name] = aligned
                    flipped.setdefault((loss, name), int((signs < 0).sum()))
        if flipped:
            print("  sign-aligned rois (covariance fits cannot see channel sign): "
                  + ", ".join(f"{n} {k}/{obs.shape[0]}" for (l, n), k in flipped.items()))

        widths = []
        for _ in halves:
            widths += [cls.W_HEAT, cls.W_TRACE]
        if show_scatter:
            widths.append(cls.W_SCATTER_RESP)
        heights, starts = [], []
        for i, _ in enumerate(losses):
            if i:
                heights.append(cls.H_GAP)
            starts.append(len(heights))
            heights += [1.0] * len(rows)
        if figsize is None:
            figsize = (sum(widths) + 1.6, sum(heights) * cls.H_UNIT + 1.3)
        fig = plt.figure(figsize=figsize) if fig is None else fig
        gs = GridSpec(len(heights), len(widths), width_ratios=widths,
                      height_ratios=heights, figure=fig,
                      top=0.91, bottom=0.09, left=0.07, right=0.99,
                      wspace=0.18, hspace=0.15)

        axes = {}
        for i, loss in enumerate(losses):
            r0 = starts[i]
            last_block = (i == len(losses) - 1)
            for h, half in enumerate(halves):
                p = shown[(half, loss)]
                c_heat, c_trace = 2 * h, 2 * h + 1

                for r, key in enumerate(rows):
                    ax = fig.add_subplot(gs[r0 + r, c_heat])
                    last_row = (r == len(rows) - 1)
                    im = plot_response_heatmap(
                        ax, p[key], roi_order=roi_order, im_kwargs=im_kwargs,
                        fontsize=fontsize, roi_labels=(h == 0),
                        ylabel=("observed" if key == "obs" else key) if h == 0 else None,
                        ylabel_color="0.2" if key == "obs" else pfm.model_color(key),
                        xlabel="odour" if (last_row and last_block) else None,
                        xticklabels=last_row)
                    if r == 0:
                        ax.set_title(f"{LOSS_LABELS.get(loss, loss)} \u2014 {HALF_LABELS[half]}",
                                     fontsize=fontsize * 1.05, pad=6)
                    axes[f"{loss}_{half}_{key}"] = ax

                if last_block and h == len(halves) - 1:
                    # One bar for the figure: everything shares the same scale.
                    axes["cbar"] = add_colorbar(
                        fig, ax, im, fontsize=fontsize, orientation="horizontal",
                        rect=(0.0, -0.85, 1.0, 0.10),
                        ticks=uniform_ticks(vmin, vmax) if "norm" in im_kwargs else None)

                for r, roi in enumerate(top):
                    ax = fig.add_subplot(gs[r0 + r, c_trace])
                    last_row = (r == n_traces - 1)
                    plot_response_traces(
                        ax, p["obs"], {name: p[name] for name in models}, roi=roi,
                        fontsize=fontsize,
                        xlabel="odour" if (last_row and last_block) else None,
                        xticklabels=last_row,
                        legend=(r == 0 and i == 0 and h == 0))
                    axes[f"{loss}_{half}_trace{roi}"] = ax

            if show_scatter:
                ax = fig.add_subplot(gs[r0:r0 + len(rows), len(widths) - 1])
                p = shown[("vld", loss)]
                flat = np.asarray(p["obs"]).ravel()
                rng = np.random.default_rng(cls.SCATTER_SEED)
                k = max(1, int(round(cls.SCATTER_FRAC_RESP * flat.size)))
                sub = rng.choice(flat.size, size=k, replace=False)
                for name in models:
                    pred = np.asarray(p[name]).ravel()
                    ax.scatter(flat[sub], pred[sub], s=cls.SCATTER_SIZE_RESP,
                               alpha=cls.SCATTER_ALPHA_RESP,
                               color=pfm.model_color(name), linewidths=0,
                               label=f"{name}  r = {pearson(flat, pred):+.2f}")
                ax.plot([vmin, vmax], [vmin, vmax], lw=0.8, color="0.4", zorder=0)
                ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("observed (held out)", fontsize=fontsize * 0.9)
                ax.set_ylabel("predicted", fontsize=fontsize * 0.9)
                ax.tick_params(labelsize=fontsize * 0.75)
                ax.legend(fontsize=fontsize * 0.85, frameon=False, markerscale=3,
                          loc="upper left")
                spines_off(ax)
                axes[f"{loss}_scatter"] = ax

        fig.suptitle(f"Matched rois: responses, observed vs predicted "
                     f"(seed {plot_data.seed}, train {plot_data.train}{subset_label(plot_data)})",
                     fontsize=fontsize * 1.3, y=0.99)
        return axes



class Main(Figure):
   @classmethod
   def plot(cls, plot_data, **kwargs):
        print("PLOTTING FIGURE matched_rois")
        gs  = GridSpec(8, 12, width_ratios=OUTER_WIDTHS)
        fig = plt.gcf()

        # Block A, the left third, holds the whole response story in rows that
        # are read against each other: the four response matrices, two rois, the
        # four correlation matrices, and the violin summary. It is three grids
        # over one common column base rather than a single grid, because each
        # group needs its own row spacing and hspace is uniform within a grid;
        # see LEFT_WIDTHS for the columns. The middle third is left empty here,
        # for the connectivity panels.
        left = gs[0:8, 0:4].subgridspec(3, 1, height_ratios=LEFT_GROUPS,
                                        hspace=LEFT_HSPACE)
        def group(spec, nrows, heights, hspace):
            return spec.subgridspec(nrows, len(LEFT_WIDTHS), width_ratios=LEFT_WIDTHS,
                                    height_ratios=heights, wspace=LEFT_WSPACE,
                                    hspace=hspace)
        # The middle third: one row per piece of the connectivity, each row a
        # grid of its own so the top one can have its own column widths.
        mid_gs = gs[0:8, 4:8].subgridspec(4, 1, height_ratios=MID_ROWS,
                                          hspace=MID_HSPACE)
        def mid_row(r):
            top = r == 0
            widths = MID_TOP_WIDTHS if top else MID_WIDTHS
            return mid_gs[r].subgridspec(
                1, len(widths), width_ratios=widths,
                wspace=MID_TOP_WSPACE if top else MID_WSPACE)
        mid_rows = [mid_row(r) for r in range(4)]
        # The rotation first, then the stretch: the rotation is what only one
        # loss determines, so it is the comparison the column exists to make,
        # and this puts the stretch's alignment directly above its spectrum.
        mid_panels = [("conn_schematic", 0, 0), ("baseline_strip", 0, 1),
                      ("baseline_frac", 0, 2),
                      ("R_cov", 1, 0), ("R_resp", 1, 1), ("rot_angles", 1, 2),
                      ("S_cov", 2, 0), ("S_resp", 2, 1), ("stretch_align", 2, 2),
                      ("factor_ablation", 3, 0), ("orbit", 3, 1),
                      ("stretch_spectra", 3, 2)]

        resp_gs = group(left[0], 3, RESP_HEIGHTS, RESP_HSPACE)
        corr_gs = group(left[1], 2, (1.0, 1.0), CORR_HSPACE)
        vln_gs  = group(left[2], 1, (1.0,), 0.0)

        # (name, row, column) for each group, in reading order, where the
        # column is 0 for the left panel and 1 for the right. Two panels to a
        # row throughout, so the responses, the rois and the correlations all
        # sit on the same two edges. The two blocks of matrices carry the same
        # four things -- what went in, what came out, and what each fit predicts
        # -- once as responses and once as correlations.
        resp_panels = [("input_heatmap", 0, 0), ("output_heatmap", 0, 1),
                       ("pred_cov_heatmap", 1, 0), ("pred_resp_heatmap", 1, 1),
                       ("roi_1", 2, 0), ("roi_2", 2, 1)]
        corr_panels = [("corr_input", 0, 0), ("corr_obs", 0, 1),
                       ("corr_pred_free_cov", 1, 0), ("corr_pred_free_resp", 1, 1)]

        def cell(grid, r, c):
            """The (row, panel column) cell of a group's grid."""
            c0 = PANEL_COLS[c]
            return grid[r, c0:c0 + PANEL_SPAN]

        layout = {
            "C":{#x,y,w,h
                "i":  (8,0,4,4, "r2_heldout"),
                "ii": (8,4,2,2, "surrogate_alpha"),
                "iii":(8,6,2,2, "z_spectrum"),
                "iv": (10,4,2,2,"mode_conn"),
                "v":  (10,6,2,2,"schematic"),
                }
            }

        axes = {}
        for name, r, c in resp_panels:
            axes[name] = fig.add_subplot(cell(resp_gs, r, c))
        for name, r, c in corr_panels:
            axes[name] = fig.add_subplot(cell(corr_gs, r, c))
        # The violin runs from the left panel's left edge to the right panel's
        # right edge, so it lines up with the block rather than with the third.
        axes["violin"] = fig.add_subplot(
            vln_gs[0, PANEL_COLS[0]:PANEL_COLS[1] + PANEL_SPAN])
        for k, name in enumerate([n for n, _, _ in resp_panels + corr_panels]
                                 + ["violin"]):
            axes[name].set_title(f"A{NUMERALS[k]}", fontsize=10,
                                 loc="left", pad=0.5)

        for name, r, c in mid_panels:
            cols = MID_TOP_COLS if r == 0 else MID_PANEL_COLS
            axes[name] = fig.add_subplot(mid_rows[r][0, cols[c]])
        for k, name in enumerate([n for n, _, _ in mid_panels]):
            axes[name].set_title(f"B{NUMERALS[k]}", fontsize=10,
                                 loc="left", pad=0.5)

        for block, panels in layout.items():
            for panel, (x,y,w,h, name) in panels.items():
                ax = fig.add_subplot(gs[y:y+h, x:x+w])
                axes[name] = ax
                ax.set_title(f"{block}{panel}", fontsize=10, loc="left", pad=0.5)


        # The responses, as a 2 x 2 in the same arrangement as the correlations
        # below: what went in and what came out on top, what each fit predicts
        # underneath, so a prediction sits under the thing it is predicting.
        inp      = plot_data.fits[("resp", "Free")].data("vld")[0]
        obs      = plot_data.matrices[("resp", "resp", "vld")]["obs"]
        pred     = plot_data.matrices[("resp", "resp", "vld")]["Free"]
        pred_cov = plot_data.matrices[("cov",  "resp", "vld")]["Free"]
        # One roi order for all four, from clustering the observed output: the
        # panels are read against each other, so a row has to mean the same roi
        # in each. Pass `inp` instead to organise the input panel rather than
        # the output one.
        roi_order = roi_cluster_order(obs)
        # One scale over all four panels -- comparing them is the point, and a
        # per-panel autoscale would defeat it.
        heatmaps = [("input_heatmap", inp), ("output_heatmap", obs),
                    ("pred_cov_heatmap", pred_cov), ("pred_resp_heatmap", pred)]
        every = np.concatenate([np.asarray(M).ravel() for _, M in heatmaps])
        im_kwargs = response_style(every, vlim=RESP_VLIM, cmap=RESP_CMAP)
        for i, (name, M) in enumerate(heatmaps):
            ax = axes[name]
            # They share a roi order and a scale, so only the left column names
            # the rows. The odour axis is labelled once, at the bottom of the
            # group, on the roi traces.
            left_column = i % 2 == 0
            im = plot_response_heatmap(ax, M, roi_order=roi_order,
                              im_kwargs=im_kwargs,
                              fontsize=FONTSIZE,
                              roi_labels=left_column,
                              ylabel="roi" if left_column else None,
                              ylabel_color="0.2",
                              xlabel=None, xticklabels=False)
        # One bar for the four, inset against the first panel. An inset takes
        # its space from the figure, so the heat maps keep identical boxes;
        # fig.colorbar(ax=...) would shrink one of them and break the alignment.
        add_colorbar(fig, axes["input_heatmap"], im, fontsize=FONTSIZE,
                     rect=(CBAR_X, 0.0, CBAR_W, 1.0))


        # ROI traces
        which_rois = roi_order_by_variance(obs)[:2]
        pred_keys = [("resp", "Free"), ("cov", "Free")]
        preds = {f"{model}_{loss}": plot_data.matrices[(loss, "resp", "vld")][model]
                 for loss, model in pred_keys}
        for i, roi in enumerate(which_rois):
            ax = axes[f"roi_{i+1}"]
            # WHICH roi each panel shows belongs in the caption; what is on the
            # axis is the same quantity in both, so both say so.
            plot_response_traces(ax, obs, preds, roi=roi,
                                 fontsize=FONTSIZE,
                                 ylabel="response", xlabel="odour", xticklabels=True,
                                 # Stacked, and inside the axes: three entries
                                 # laid out across the top spill into the gap
                                 # above, which belongs to the heat maps.
                                 legend=(i==0),
                                 legend_kwargs={"loc": "upper left", "ncol": 1,
                                                "bbox_to_anchor": None,
                                                "labelspacing": 0.1,
                                                "handlelength": 1.0,
                                                "borderpad": 0.1})
            # This panel and the heat map above it are the same odours in the
            # same order, so their ticks have to land in the same places.
            # imshow puts odour 0 at 0 and pads by half a cell, which is not
            # what a line plot's default margins do -- so take both the ticks
            # and the limits from the heat map rather than setting them twice.
            above = axes[["pred_cov_heatmap", "pred_resp_heatmap"][i]]
            ax.set_xticks(above.get_xticks())
            ax.set_xlim(above.get_xlim())

        # One y axis for the row: two rois drawn at two scales cannot be
        # compared by eye. The headroom is for the legend, which would otherwise
        # sit on top of the traces.
        trace_axes = [axes[f"roi_{i+1}"] for i in range(len(which_rois))]
        lo = min(a.get_ylim()[0] for a in trace_axes)
        hi = max(a.get_ylim()[1] for a in trace_axes)
        for ax in trace_axes:
            ax.set_ylim(lo, hi + TRACE_HEADROOM * (hi - lo))


        ## Correlations, as a 2 x 2: what goes in and what comes out on the top
        # row, what the two Free fits predict for the output on the bottom, so
        # the predictions sit under the thing they are predicting. The scale is
        # fixed (MATRIX_STYLE["corr"]), so all four share it and one colour bar
        # serves the block.
        #
        # The input matrix is not in plot_data.matrices, which holds an observed
        # output and the models' predictions of it. It comes from the same
        # results object as the observed one, and is normalised the same way, so
        # it is the same estimator on the other side of the transformation --
        # not a raw correlation of the input responses.
        r = plot_data.fits[("resp", "Free")].results("vld")
        corr_input = corr_from(r.Cin, r.ref_vars["Cin"], r.eval_vars["Cin"])

        which_corrs = {"corr_obs": ("resp", "resp", "vld"),
                       "corr_pred_free_cov":  ("cov", "Free", "vld"),
                       "corr_pred_free_resp": ("resp", "Free", "vld")}
        panels = [("corr_input", "input", "0.2", corr_input)]
        for name, (loss, model, half) in which_corrs.items():
            p = plot_data.matrices[(loss, "corr", half)]
            observed = model == "resp"
            M = p["obs"] if observed else p[model]
            # Both predictions are Free, so the title has to name the loss as
            # well; the key is the one variant_color and variant_label take,
            # which is what the roi traces' legend is keyed by too.
            key = f"{model}_{loss}"
            panels.append((name,
                           "observed" if observed else variant_label(key),
                           "0.2" if observed else variant_color(key), M))

        for i, (name, title, color, M) in enumerate(panels):
            ax = axes[name]
            left_column, bottom_row = i % 2 == 0, i >= 2
            im = plot_matrix(ax, M, order=None, im_kwargs=matrix_style(M, "corr"),
                        fontsize=FONTSIZE, aspect="equal",
                        title=title, title_color=color,
                        xlabel="odour" if bottom_row else None,
                        ylabel="odour" if left_column else None,
                        xticklabels=bottom_row, yticklabels=left_column)
        add_colorbar(fig, axes["corr_input"], im, fontsize=FONTSIZE,
                     rect=(CBAR_X, 0.0, CBAR_W, 1.0))
           

        df = plot_data.gen_df.copy()
        # Drop the models called FreeSym_resp, FreePSD_resp
        df = df[~df["model"].isin(["FreeSym_resp", "FreePSD_resp", "FreeRot_resp", "FreeOrth_resp"])]
        fig_violin_plots.plot_violins(axes["violin"], df,
                                      sampler="trials",
                                      mode="random",
                                      prefix="corr",
                                      )
            

        ## B: the connectivity, as Z = R S + 1 b'.
        #
        # Rows: what the covariance loss cannot see, then what it determines,
        # then what it leaves free. The matrices are one seed -- the same one the
        # left third draws -- and the third column is what all the seeds say.
        polar = getattr(plot_data, "polar", None)
        mid_names = [n for n, _, _ in mid_panels]
        if polar is None:
            for name in mid_names:
                note(axes[name], "no polar decomposition\n(recompute matched_rois)")
        else:
            example = polar[plot_data.seed if plot_data.seed in polar else min(polar)]
            # One roi order for the whole block, from clustering the response
            # fit's stretch -- the matrix the block is about. Permuting rows and
            # columns together is a change of basis by a permutation, so it
            # leaves every matrix here meaning what it meant; drawn in the rois'
            # own arbitrary order they are all checkerboards.
            conn_order = cluster_order(example["resp"].S)

            # Bii: the mean component, which only the response fit has an
            # opinion about -- the covariance loss cannot see it, so the
            # covariance fit's b' is whatever its regularizer chose.
            plot_baseline_strip(axes["baseline_strip"], example, order=conn_order,
                                fontsize=FONTSIZE)
            plot_baseline_fraction(axes["baseline_frac"], polar, fontsize=FONTSIZE)

            # Biv-Bvi: the rotation, which only the response loss determines --
            # the covariance fit's is whatever its regularizer chose, and the
            # panel's point is that it chose the identity.
            # Bvii-Bix: the stretch, which both losses determine, so the two
            # fits' versions can be compared directly. One scale for each pair.
            for attr, label in [("R", "R"), ("S", "S")]:
                mats = [getattr(example[loss], attr) for loss in CONN_LOSSES]
                im_kwargs = conn_style(mats, cmap=CONN_CMAPS[attr])
                for i, (loss, M) in enumerate(zip(CONN_LOSSES, mats)):
                    im = plot_connectivity(
                        axes[f"{attr}_{loss}"], M, order=conn_order,
                        im_kwargs=im_kwargs, fontsize=FONTSIZE,
                        title=f"${label}$, {loss}",
                        title_color=loss_color(loss),
                        ylabel="roi" if i == 0 else None,
                        xlabel="roi", ticklabels=(i == 0))
                add_colorbar(fig, axes[f"{attr}_{CONN_LOSSES[-1]}"], im,
                             fontsize=FONTSIZE, rect=(CBAR_X, 0.0, CBAR_W, 1.0))

            plot_stretch_alignment(axes["stretch_align"], polar, fontsize=FONTSIZE)
            plot_angle_spectra(axes["rot_angles"], polar, fontsize=FONTSIZE,
                               quantiles=kwargs.get("quantiles", (25, 75)))
            plot_stretch_spectra(axes["stretch_spectra"], polar, fontsize=FONTSIZE,
                                 quantiles=kwargs.get("quantiles", (25, 75)))

            # Bx: what each factor is worth, from the same frame the ladder is
            # drawn from.
            plot_factor_ablation(axes["factor_ablation"], plot_data.r2_df,
                                 fontsize=FONTSIZE)

            orbit_df = getattr(plot_data, "orbit_df", None)
            # "t" is the swept angle. A pickle from before the sweep has the
            # earlier column set, from when the orbit was sampled rather than
            # walked, and there is no way to recover an angle from it.
            if orbit_df is None or "t" not in orbit_df:
                note(axes["orbit"], "no swept orbit\n(recompute matched_rois)")
            else:
                plot_orbit(axes["orbit"], orbit_df, fontsize=FONTSIZE,
                           quantiles=kwargs.get("quantiles", (25, 75)))

        # Bi: drawn by hand, so the panel only reserves the space.
        axes["conn_schematic"].set_xticks([]); axes["conn_schematic"].set_yticks([])
        spines_off(axes["conn_schematic"])


        ## C: R2, Q, sparsity

        # First, compute medians of self.r2_df over trains, and drop that column
        r2_df = plot_data.r2_df.groupby(["seed"]).median().drop(columns=["train"]).reset_index()
        Z_MODELS = r2_df.columns.difference(["seed", "train", "Input", "Output"])
        # Now melt r2_df
        long = pd.DataFrame([
            {"sampler":"trials",
             "mode":"random",
             "outclass":None,
             "model":zmdl,
             "seed": row["seed"],
             "r2_in_out":row["Input"],
             "r2_est_out":row[zmdl],
             "r2_out":row["Output"]}
            for _, row in r2_df.iterrows() for zmdl in Z_MODELS])


        # The two fits and the two recombinations of their factors, then the two
        # constrained refits. Brightness says whether a rotation is present --
        # light for none, dark for one -- and hue says which kind of model it
        # is: the fits themselves, a recombination, or a refit.
        ORDER = {"Z_cov":   "Free\ncov",
                 "Z_cov_bl": "Free\ncov,bl",
                 "Q=I":     "R: Cov\nS: Resp",
                 "P=P_cov": "R: Resp\nS: Cov",
                 "Z_resp":  "Free \nresp",
                 "Z_resp_sym": "Sym\nresp",
                 "Z_psd":   "PSD\nrefit",
                 "Z_rot":   "Rot",
                 "Z_orth":  "Orth",
                 "Z_sym":   "Sym\nrefit",
                 "a X + b": "Lin\nrefit",
                 "1b' only": "1b'\nrefit",
                 "a Z_cov + 1b'": "$aZ_\\text{cov}+1b'$\nrefit",
                 "Z_cov + 1b'": "$Z_\\text{cov}+1b'$\nrefit",
                 }
        
        assert set(ORDER)==set(Z_MODELS), f"ORDER {ORDER} does not match Z_MODELS {Z_MODELS}"
        order = ["Z_cov", "Z_cov_bl", "Z_psd", "Z_rot", "Z_orth", "Z_resp", "Z_sym"]
        Z_MODELS = order
        ORDER = {k: ORDER[k] for k in Z_MODELS}
    
        
        free_cov, free_resp = pfm.variant_color("Free_cov"), pfm.variant_color("Free_resp")
        COLORS = {"Z_cov":   free_cov,                   # the covariance fit, its colour everywhere else
                  # A variant of the covariance fit, so it stays in that family:
                  # the same luminance as Z_cov with the hue shifted to green.
                  # greener() re-solves the value to preserve lightness, so the
                  # two read as a pair rather than one looking heavier.
                  "Z_cov_bl": greener(free_cov),
                  "Z_resp":  free_resp,                  # the response fit, likewise
                  "Q=I":     greener(free_cov),          # Q from cov, S from resp
                  "P=P_cov": greener(free_resp),         # Q from resp, S from cov
                  "Z_resp_sym": lighter(free_resp, 0.25),# the response fit with a symmetry constraint
                  # The refits are models in their own right and appear in the
                  # generalization panels, so they keep their model colour here
                  # rather than being derived from the two fits.
                  # SO(m) is contained in O(m), so Rot is a lighter shade of
                  # Orth rather than an unrelated hue.
                  "Z_rot":  lighter(pfm.model_color("FreeOrth"), 0.45),
                  "Z_orth": pfm.model_color("FreeOrth"),     # #8c6bb1
                  "Z_sym":   pfm.model_color("FreeSym"),
                  "Z_psd":   pfm.model_color("FreePSD"),
                  "a X + b": pfm.model_color("FreeLin"),
                  "1b' only": pfm.model_color("Free1b"),
                  "a Z_cov + 1b'": pfm.model_color("FreeCov1b"),
                  }
        fig_violin_plots.plot_violins(axes["r2_heldout"], long,
                                      sampler="trials", mode="random", prefix="r2",
                                      models=as_labels(ORDER),
                                      colors=COLORS,
                                      reverse=False,
                                      )

        # Add y-axis grid
        axes["r2_heldout"].yaxis.grid(True, which="major", color="0.8", lw=0.5, zorder=0, ls=":")

        # Cii: the surrogate calibration. The observed differences come from the
        # ladder's own r2_df, medianed over trains exactly as the ladder is, so
        # this violin is the gap between its Free and Sym rungs.
        per_seed = plot_data.r2_df.groupby("seed").median()
        observed = (per_seed["Z_sym"] - per_seed["Z_resp"]).values
        surrogate_df = getattr(plot_data, "surrogate_df", None)
        if surrogate_df is None:
            # The sweep is a separate set of runs, one per alpha, so the rest of
            # the figure has to draw without it.
            axes["surrogate_alpha"].text(
                0.5, 0.5, "no surrogate runs loaded\n(set plot_data.surrogate_df)",
                ha="center", va="center", fontsize=FONTSIZE * 0.8, color="0.5",
                transform=axes["surrogate_alpha"].transAxes)
            axes["surrogate_alpha"].set_xticks([]); axes["surrogate_alpha"].set_yticks([])
        else:
            plot_surrogate_alpha(axes["surrogate_alpha"], surrogate_df, observed,
                                 fontsize=FONTSIZE)

        # Ciii: what the winning model actually is. A symmetric Z has no
        # rotation to describe, so its eigenvalues are the whole story.
        plot_zsym_spectrum(axes["z_spectrum"],
                           [Zs["Z_sym"] for Zs in plot_data.Z_vals.values()],
                           quantiles = kwargs.get("quantiles", (25, 75)),
                           fontsize=FONTSIZE)

        # Civ: the same connectivity in the input's own basis. Same seed order
        # for the Z's and the bases, or a seed's Z would be rotated by another
        # seed's eigenvectors.
        seeds = sorted(plot_data.Z_vals)
        plot_mode_connectivity(axes["mode_conn"],
                               [plot_data.Z_vals[s]["Z_sym"] for s in seeds],
                               [plot_data.input_modes[s] for s in seeds],
                               fontsize=FONTSIZE)

        # Cv: drawn by hand, so the panel only reserves the space.
        axes["schematic"].set_xticks([]); axes["schematic"].set_yticks([])
        spines_off(axes["schematic"])
                                      
                                      
                                      


           
        return axes 
                 
