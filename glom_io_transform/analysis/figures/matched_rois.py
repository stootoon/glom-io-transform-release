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
from ..compute.matched_rois import LOSSES, MODELS, METRICS, HALVES
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

RESP_CMAP = "pink"
RESP_VLIM = (0, 1)
PCTILE    = (1, 99)
FONTSIZE  = 9

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
                yticklabels=True):
    """One odours x odours matrix on `ax`. Returns the image.

    `order` reorders both axes together, so the matrix stays symmetric.
    """
    M = np.asarray(M)
    if order is not None:
        order = np.asarray(order)
        M = M[order][:, order]
    im = ax.imshow(M, aspect="auto", interpolation="nearest",
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
    ax.tick_params(labelsize=fontsize * 0.75)
    return im


def add_colorbar(fig, ax, im, fontsize=FONTSIZE, rect=(1.06, 0.0, 0.05, 1.0),
                 orientation="vertical", ticks=None):
    """A slim colour bar inset against `ax`. Returns its axes."""
    cax = ax.inset_axes(list(rect))
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=fontsize * 0.7)
    cb.outline.set_linewidth(0.5)
    return cax


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
        gs  = GridSpec(8, 12)
        fig = plt.gcf()

        layout = {
            "A":{#x,y,w,h
                "i":  (0,0,4,2, "input_heatmap"),
                "ii": (0,2,4,2, "output_heatmap"),
                "iii":(0,4,4,2, "predicted_heatmap"),
                "iv": (0,6,4,1, "roi_1"),
                "v":  (0,7,4,1, "roi_2"),
                },
            "B":{
                "i":  (4,0,2,2, "corr_obs"),
                "ii": (6,0,2,2, "corr_pred_diag_cov"),
                "iii":(4,2,2,2, "corr_pred_free_resp"),
                "iv": (6,2,2,2, "corr_pred_free_cov"),
                "v":  (4,4,4,4, "violin"),
            },
            "C":{
                "i":  (8,0,4,4, "r2_heldout"),
                "ii": (8,4,2,2, "Q_matrix"),
                "iii":(8,6,2,2, "Q_theta"),
                "iv": (10,4,2,4,"sparsity"),
                }
            }

        axes = {} 
        for block, panels in layout.items():
            for panel, (x,y,w,h, name) in panels.items():
                ax = fig.add_subplot(gs[y:y+h, x:x+w])
                axes[name] = ax
                ax.set_title(f"{block}{panel}: {name}", fontsize=10, loc="left", pad=0.5)


        # Observed output heatmap
        inp = plot_data.fits[("resp", "Free")].data("vld")[0]
        obs = plot_data.matrices[("resp", "resp", "vld")]["obs"]
        pred= plot_data.matrices[("resp", "resp", "vld")]["Free"]
        for name, M in zip(["input", "output", "predicted"], [inp, obs, pred]):
            ax = axes[f"{name}_heatmap"]
            im_kwargs = response_style(M, vlim=RESP_VLIM, cmap=RESP_CMAP)
        
            plot_response_heatmap(ax, M, roi_order=None,
                              im_kwargs=im_kwargs,
                              fontsize=FONTSIZE,
                              roi_labels=True,
                              ylabel="roi", ylabel_color="0.2", xlabel="odour", xticklabels=True)
            

        # ROI traces
        which_rois = roi_order_by_variance(obs)[:2]
        pred_keys = [("resp", "Free"), ("cov", "Free")]
        preds = {f"{model}_{loss}": plot_data.matrices[(loss, "resp", "vld")][model]
                 for loss, model in pred_keys}
        for i, roi in enumerate(which_rois):
            ax = axes[f"roi_{i+1}"]
            plot_response_traces(ax, obs, preds, roi=roi,
                                 fontsize=FONTSIZE,
                                 ylabel=f"roi {roi}", xlabel="odour", xticklabels=True,
                                 legend=(i==0), legend_kwargs={"loc": "upper left"})


        ## B: Correlations
        which_corrs = {"corr_obs": ("resp", "resp", "vld"),
                       "corr_pred_diag_cov": ("cov", "Diag", "vld"),
                       "corr_pred_free_resp": ("resp", "Free", "vld"),
                       "corr_pred_free_cov":  ("cov", "Free", "vld")}

        for name, (loss, model, half) in which_corrs.items():
            ax = axes[name]
            p  = plot_data.matrices[(loss, "corr", half)]
            M = p["obs"] if model == "resp" else p[model]
            im_kwargs = matrix_style(M, "corr")
            
            plot_matrix(ax, M, order=None, im_kwargs=im_kwargs, fontsize=FONTSIZE,
                        title="observed" if model == "obs" else model,
                        title_color="0.2" if model == "obs" else variant_color(model),
                        xlabel="odour", ylabel="odour", yticklabels=True)
           

        df = plot_data.gen_df.copy()
        # Drop the models called FreeSym_resp, FreePSD_resp
        df = df[~df["model"].isin(["FreeSym_resp", "FreePSD_resp"])]
        fig_violin_plots.plot_violins(axes["violin"], df,
                                      sampler="trials",
                                      mode="random",
                                      prefix="corr",
                                      )
            

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
                 "a Z_cov + 1b'": "aZ_cov+1b'\nrefit",
                 }
        free_cov, free_resp = pfm.variant_color("Free_cov"), pfm.variant_color("Free_resp")
        COLORS = {"Z_cov":   free_cov,                   # the covariance fit, its colour everywhere else
                  "Z_resp":  free_resp,                  # the response fit, likewise
                  "Q=I":     greener(free_cov),          # Q from cov, S from resp
                  "P=P_cov": greener(free_resp),         # Q from resp, S from cov
                  "Z_resp_sym": lighter(free_resp, 0.25),# the response fit with a symmetry constraint
                  # The refits are models in their own right and appear in the
                  # generalization panels, so they keep their model colour here
                  # rather than being derived from the two fits.
                  "Z_rot":  pfm.model_color("FreeRot"),      # #d62728
                  "Z_orth": pfm.model_color("FreeOrth"),     # #8c6bb1
                  "Z_sym":   pfm.model_color("FreeSym"),
                  "Z_psd":   pfm.model_color("FreePSD"),
                  "a X + b": pfm.model_color("FreeLin"),
                  "1b' only": pfm.model_color("Free1b"),
                  "a Z_cov + 1b'": pfm.model_color("FreeCov1b"),
                  }
        assert set(ORDER)==set(Z_MODELS), f"ORDER {ORDER} does not match Z_MODELS {Z_MODELS}"
        fig_violin_plots.plot_violins(axes["r2_heldout"], long,
                                      sampler="trials", mode="random", prefix="r2",
                                      models=as_labels(ORDER),
                                      colors=COLORS,
                                      reverse=False,
                                      )

        # Add y-axis grid
        axes["r2_heldout"].yaxis.grid(True, which="major", color="0.8", lw=0.5, zorder=0, ls=":")
                                      
                                      
                                      


           
        return axes 
                 
