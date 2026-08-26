import os, sys, logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg 
#import compute

try:
    from label_axes import label_axes
except ImportError:
    # Define a dummy function if label_axes is not available
    print("label_axes module not found. Using dummy function.")
    class label_axes:
        @staticmethod
        def label_axes(*args, **kwargs):
            pass

from glom_io_transform import paths

project_path = paths.proj_path
sys.path.append(project_path)

art_path     = os.path.join(project_path, "art")

plt.rcParams['figure.figsize']    = [8, 3]
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.style.use("default")

from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering, leaves_list
from mpl_toolkits.axes_grid1 import make_axes_locatable

def spines_off(ax = None, which=["top", "right"]):
    if ax is None:
        ax = plt.gca()
        
    for w in which:
        ax.spines[w].set_visible(False)
    return ax

def reduce_vertical_gap(ax_top, ax_bottom, reduction):
    """
    Reduce the vertical gap between two vertically stacked axes
    by `reduction` (in figure coordinates).

    Keeps the top of ax_top fixed and the bottom of ax_bottom fixed.
    """
    pos1 = ax_top.get_position()
    pos2 = ax_bottom.get_position()

    # split the reduction evenly between the two axes
    shift = reduction / 2.0

    new_pos1 = [pos1.x0, pos1.y0 - shift, pos1.width, pos1.height + shift]
    new_pos2 = [pos2.x0, pos2.y0, pos2.width, pos2.height + shift]
    # ax_top: extend downward
    ax_top.set_position(new_pos1)    # ax_bottom: extend upward
    ax_bottom.set_position(new_pos2)
 
def _shrink_gaps(starts, sizes, reduction):
    """New (starts, sizes) for spans on an INCREASING axis, gaps shrunk equally.

    Every gap loses the same amount, the space that frees up is shared between
    the spans in proportion to their current size -- so the ratios a GridSpec
    set are preserved -- and the two outer edges do not move.

    Doing this in one pass is the point. Applying a pairwise "close this gap by
    growing both of its axes" down the list instead makes every INTERIOR span
    grow twice, once for the gap on each side, so the middle of three ends up
    twice as large as its neighbours.
    """
    starts, sizes = np.asarray(starts, float), np.asarray(sizes, float)
    n = len(starts)
    per_gap  = reduction / (n - 1)                       # each gap loses this
    new_sizes = sizes * (1 + reduction / sizes.sum())    # each span gains its share

    new_starts = [starts[0]]
    for i in range(n - 1):
        old_gap = starts[i + 1] - (starts[i] + sizes[i])
        new_starts.append(new_starts[i] + new_sizes[i] + old_gap - per_gap)
    return np.array(new_starts), new_sizes


def reduce_vertical_gaps(ax_list, reduction):
    """
    Reduce vertical gaps between a list of vertically stacked axes by `reduction` (in figure coordinates).

    Keeps the top of the first axis fixed and the bottom of the last axis fixed.
    """
    if len(ax_list) < 2:
        return  # nothing to reduce

    pos = [ax.get_position() for ax in ax_list]
    # The list runs top to bottom but y increases upward, so the coordinate that
    # increases along the list is the NEGATED top edge. Laying the spans out in
    # -y lets the same one-pass calculation serve both directions.
    tops, heights = _shrink_gaps([-p.y1 for p in pos], [p.height for p in pos], reduction)
    for ax, p, u0, h in zip(ax_list, pos, tops, heights):
        ax.set_position([p.x0, -u0 - h, p.width, h])


def reduce_horizontal_gap(ax_left, ax_right, reduction):
    """
    Reduce the horizontal gap between two horizontally stacked axes
    by `reduction` (in figure coordinates).

    Keeps the left of ax_left fixed and the right of ax_right fixed.
    """
    pos1 = ax_left.get_position()
    pos2 = ax_right.get_position()

    # split the reduction evenly between the two axes
    shift = reduction / 2.0

    new_pos1 = [pos1.x0, pos1.y0, pos1.width + shift, pos1.height]
    new_pos2 = [pos2.x0 - shift, pos2.y0, pos2.width + shift, pos2.height]

    # ax_left: extend rightward
    ax_left.set_position(new_pos1)
    # ax_right: extend leftward
    ax_right.set_position(new_pos2)

def reduce_horizontal_gaps(ax_list, reduction):
    """
    Reduce horizontal gaps between a list of horizontally stacked axes by `reduction` (in figure coordinates).

    Keeps the left of the first axis fixed and the right of the last axis fixed.
    """
    if len(ax_list) < 2:
        return  # nothing to reduce

    pos = [ax.get_position() for ax in ax_list]
    x0s, widths = _shrink_gaps([p.x0 for p in pos], [p.width for p in pos], reduction)
    for ax, p, x0, w in zip(ax_list, pos, x0s, widths):
        ax.set_position([x0, p.y0, w, p.height])
    
# ---------------------------------------------------------------------------
# Figure tidying: uniform type sizes, label margins, and packing panels by what
# is actually DRAWN rather than by their gridspec slots.
# ---------------------------------------------------------------------------

def _axes_of(fig, axes):
    """The axes to act on: everything in the figure, or the list given."""
    return list(fig.axes) if axes is None else list(axes)


def _managed(ax):
    """True if something else decides where this axes goes.

    Colorbars made with inset_axes or an axes divider carry a locator that is
    consulted at every draw, so setting their position is silently undone --
    and freezing one would break the link that keeps it beside its parent.
    """
    return ax.get_axes_locator() is not None


def set_label_sizes(fig, size, axes=None, which="both"):
    """One font size for the x and/or y axis LABELS across a figure."""
    for ax in _axes_of(fig, axes):
        if which in ("x", "both"): ax.xaxis.label.set_size(size)
        if which in ("y", "both"): ax.yaxis.label.set_size(size)
    return _axes_of(fig, axes)


def set_tick_sizes(fig, size, axes=None, which="both"):
    """One font size for the x and/or y TICK labels across a figure.

    `fig.axes` includes colorbars and insets, so the default reaches those too,
    which is usually what uniformity means. Pass an explicit list to restrict it.
    """
    axis = {"x": "x", "y": "y", "both": "both"}[which]
    for ax in _axes_of(fig, axes):
        ax.tick_params(axis=axis, labelsize=size)
    return _axes_of(fig, axes)


def set_label_pads(fig, pad, axes=None, which="both"):
    """Put every axis label `pad` POINTS from its tick labels.

    labelpad is already measured from the axis bbox including the tick labels,
    so this is the margin asked for. Axes whose label was placed by hand with
    set_label_coords ignore labelpad, so automatic positioning is restored
    first -- via a private attribute, for want of a public way to undo it.
    """
    for ax in _axes_of(fig, axes):
        for name in (("xaxis", "yaxis") if which == "both" else (which + "axis",)):
            axis = getattr(ax, name)
            axis._autolabelpos = True
            axis.labelpad = pad
    return _axes_of(fig, axes)


def fit_to_drawn(fig, axes=None):
    """Set each axes' position to the box its contents are actually drawn in.

    An axes with a fixed aspect (imshow and matshow set one) shrinks inside its
    slot to keep pixels square, and get_position() keeps reporting the SLOT. So
    the whitespace around such a panel lives inside its own rectangle, and no
    amount of moving positions will close it. Freezing the position onto the
    drawn box makes what is reported match what is seen -- and is stable,
    because a box that already has the right aspect is left alone at the next
    draw.

    The axes stop tracking their gridspec afterwards, which is the point.
    """
    fig.canvas.draw()
    for ax in _axes_of(fig, axes):
        if not _managed(ax):
            ax.set_position(ax.get_position(original=False))
    return _axes_of(fig, axes)


def _pack(starts, sizes, lead, trail, gaps, grow):
    """New starts/sizes so the INKED spans are separated by `gaps`.

    Spans run along an increasing axis. `lead`/`trail` are how far the ink
    overhangs each span at its low and high end -- tick labels and axis labels.
    The outer edges of the ink stay put, every gap becomes exactly what was
    asked for, and whatever space that frees goes to the spans named in `grow`,
    in proportion to their current size.
    """
    starts, sizes = np.asarray(starts, float), np.asarray(sizes, float)
    lead, trail   = np.asarray(lead, float), np.asarray(trail, float)
    n = len(starts)

    ink_start = starts[0] - lead[0]
    ink_end   = starts[-1] + sizes[-1] + trail[-1]
    # What is left for the boxes once the ink overhangs and the gaps are paid for.
    room  = (ink_end - ink_start) - (lead + trail).sum() - np.sum(gaps)
    slack = room - sizes.sum()

    new_sizes = sizes.copy()
    if slack and len(grow):
        share = sizes[grow] / sizes[grow].sum()
        new_sizes[grow] = sizes[grow] + slack * share

    new_starts = np.empty(n)
    cursor = ink_start
    for i in range(n):
        new_starts[i] = cursor + lead[i]
        cursor += lead[i] + new_sizes[i] + trail[i] + (gaps[i] if i < n - 1 else 0.0)
    return new_starts, new_sizes


def _pack_axes(ax_list, gap, grow, vertical, measure, use_drawn,
               passes=6, tol=1e-4):
    """Apply _pack_once until the gaps stop moving.

    One pass is not enough when any panel is resized: its tick labels are
    re-laid-out at the new width, so the ink overhangs measured beforehand are
    stale and the gaps next to it come out wrong. Each pass measures the
    figure as it now stands, so repeating converges. Panels that only move
    settle on the first pass.
    """
    fig = ax_list[0].get_figure()
    assert not any(_managed(ax) for ax in ax_list), (
        "Pass panel axes only: colorbars and insets are placed by a locator, "
        "so setting their position does nothing.")
    if use_drawn:
        fit_to_drawn(fig, ax_list)
    for _ in range(passes):
        moved = _pack_once(ax_list, gap, grow, vertical, measure)
        if moved < tol:
            break
    return ax_list


def _pack_once(ax_list, gap, grow, vertical, measure):
    """One measure-and-place pass. Returns how far the largest edge moved."""
    fig = ax_list[0].get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    boxes = [ax.get_position(original=False) for ax in ax_list]
    if measure == "tight":
        inks = [ax.get_tightbbox(renderer).transformed(inv) for ax in ax_list]
    elif measure == "axes":
        inks = boxes
    else:
        raise ValueError(f"measure must be 'tight' or 'axes', got {measure!r}.")

    n = len(ax_list)
    gaps = np.full(n - 1, float(gap)) if np.isscalar(gap) else np.asarray(gap, float)
    assert len(gaps) == n - 1, f"Need {n-1} gaps for {n} axes, got {len(gaps)}."

    grow = np.arange(n) if grow is None else np.asarray(grow, int)
    stuck = [i for i in grow
             if ax_list[i].get_aspect() != "auto" and ax_list[i].get_adjustable() == "box"]
    assert not stuck, (
        f"Axes {stuck} have a fixed aspect with adjustable='box', so growing them "
        f"in one direction only is undone at the next draw. Leave them out of "
        f"`grow`, or give them set_adjustable('datalim').")

    if vertical:
        # The list runs top to bottom while y increases upward, so lay the spans
        # out along -y: the coordinate that increases along the list.
        starts = [-b.y1 for b in boxes]
        sizes  = [b.height for b in boxes]
        lead   = [i.y1 - b.y1 for b, i in zip(boxes, inks)]
        trail  = [b.y0 - i.y0 for b, i in zip(boxes, inks)]
    else:
        starts = [b.x0 for b in boxes]
        sizes  = [b.width for b in boxes]
        lead   = [b.x0 - i.x0 for b, i in zip(boxes, inks)]
        trail  = [i.x1 - b.x1 for b, i in zip(boxes, inks)]

    new_starts, new_sizes = _pack(starts, sizes, lead, trail, gaps, grow)

    for ax, b, start, size in zip(ax_list, boxes, new_starts, new_sizes):
        if vertical:
            ax.set_position([b.x0, -start - size, b.width, size])
        else:
            ax.set_position([start, b.y0, size, b.height])

    return max(np.max(np.abs(new_starts - np.asarray(starts))),
               np.max(np.abs(new_sizes - np.asarray(sizes))))


def pack_horizontally(ax_list, gap, grow=None, measure="tight", use_drawn=True):
    """Set the gaps between a row of panels to exactly `gap`, in figure coords.

    gap      : one value, or one per gap (n-1 of them).
    grow     : indices of the panels that absorb the freed space, in proportion
               to their current width. None means all of them. Panels left out
               keep their size and are only moved.
    measure  : "tight" spaces the panels by their INK -- tick labels and axis
               labels included, which is the whitespace the eye sees. "axes"
               spaces the plot boxes instead.
    use_drawn: freeze fixed-aspect panels onto their drawn box first, so the
               gaps are between what is visible rather than between gridspec
               slots. See fit_to_drawn.

    The outer two edges do not move.
    """
    return _pack_axes(ax_list, gap, grow, vertical=False,
                      measure=measure, use_drawn=use_drawn)


def pack_vertically(ax_list, gap, grow=None, measure="tight", use_drawn=True):
    """pack_horizontally for a column of panels, ordered top to bottom."""
    return _pack_axes(ax_list, gap, grow, vertical=True,
                      measure=measure, use_drawn=use_drawn)


def get_leaf_order_from_covariance(C, method='ward'):
    """
    Given a covariance matrix C, return the optimal leaf order of indices
    for clustering visualization.

    Parameters:
    - C: (n x n) covariance matrix (symmetric, positive semi-definite)
    - method: linkage method for clustering (default: 'ward')

    Returns:
    - ordered_idx: array of reordered indices (length n)
    """
    # Convert to correlation matrix if needed
    std = np.sqrt(np.diag(C))
    corr = C / np.outer(std, std)

    # Convert to a distance matrix
    dist = 1 - corr
    assert np.allclose(dist, dist.T), "Distance matrix must be symmetric"
    dist = (dist + dist.T) / 2  # ensure symmetry
    np.fill_diagonal(dist, 0)  # ensure diagonals are zero
    # Cross-correlation inputs (e.g. ref vs held-out trials) have diagonals < 1,
    # so renormalized off-diagonals can exceed 1, giving small negative
    # distances that optimal_leaf_ordering rejects. Clip them to zero.
    dist = np.clip(dist, 0, None)

    # Condense distance and perform clustering
    lnk = linkage(squareform(dist), method=method)
    optimal_linkage = optimal_leaf_ordering(lnk, squareform(dist))
    ordered_idx = leaves_list(optimal_linkage)

    return ordered_idx

from sklearn.linear_model import LinearRegression

def get_leaves_for_connectivity(Z):
    Zod = Z * (1 - np.eye(Z.shape[0]))
    leaves = leaves_list(linkage(Zod, method='average'))
    return leaves

def best_fit(u, v):
    """Find the best fit line for u and v."""
    model = LinearRegression()
    model.fit(u.reshape(-1, 1), v)
    return model.predict(u.reshape(-1, 1))


class Panels:
    @classmethod
    def plot(cls, data, axes, *args, **kwargs):
        raise NotImplementedError("plot() method not implemented")
        

class Figure:
    def __init__(self, plot_data):
        assert hasattr(plot_data, "computed"), "plot_data must have a 'computed' attribute"
        self.plot_data = plot_data

    def compute_and_plot(self, args):
        if not self.plot_data.computed:
            self.plot_data.compute()
        self.plot(args, self.plot_data)
        
    @classmethod
    def plot(cls, args, plot_data):
        raise NotImplementedError("plot() method not implemented")

class Schem(Panels):
    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print("PLOTTING PANELS Schem")
        assert len(axes) == 1, "Schem should only have one axis"
        assert "art_file" in kwargs, "art_file must be provided in kwargs"
        assert os.path.exists(kwargs["art_file"]), f"Art file not found at {kwargs['art_file']}"
        ax_schem = axes[0]
        if ax_schem is None:
            print("No ax_schem provided, skipping plotting")
        else:
            plt.sca(ax_schem)
            img = mpimg.imread(kwargs["art_file"])
            img_artist = ax_schem.imshow(img, **(kwargs["imshow"] if "imshow" in kwargs else {}))
            # Keep the axes box filling its slot and pad the data limits
            # symmetrically instead of shrinking the box around the image;
            # this centers the image in the slot.
            ax_schem.set_adjustable('datalim')
            ax_schem.axis('off')
            img_artist.set_clip_on(False)
    
# House style for representation (odour x odour) matrices. The single place
# to change the paper-wide colormap / range for representation plots.
rep_style = {"cmap": "Spectral_r", "vlim": (0, 1)}


class Reps(Panels):
    """Plotting of representation (odour x odour) matrices and observed-vs-
    predicted scatters. ALL representation plotting goes through here so that
    odour orderings and color schemes are defined in one place."""

    @staticmethod
    def odour_order(method=None, C=None, n=None, linkage_method="average", from_order="X0Y0"):
        """Positions that reorder an odour axis into the requested order.

        The named orderings are defined in data.odours; this only converts them
        into positions, which is what plain (unlabelled) matrices need.

        method:
          None or "natural"        : leave as-is (needs n or C for the length);
          "cluster"                : leaf order from clustering the matrix C;
          a name in odours.ORDERS  : "X0Y0", "tbet", "chemical_class",
                                     "input", "output" -- delegated;
          a sequence of odour names: used directly;
          a sequence of integers   : an explicit permutation, passed through.

        from_order names the order the data's odour axis is currently in
        (default "X0Y0", the order the responses are stored in). Positions are
        only meaningful relative to it.
        """
        def positions_for(names):
            from glom_io_transform.data.odours import odours
            current = odours.get_order(from_order)
            missing = [nm for nm in names if nm not in current]
            assert not missing, f"Odours not present in the {from_order} order: {missing}"
            assert n is None or len(names) == n, \
                f"Ordering has {len(names)} odours but {n} were expected."
            return np.array([current.index(nm) for nm in names])

        if method is None or (isinstance(method, str) and method == "natural"):
            assert n is not None or C is not None, "Need n or C for the natural ordering."
            return np.arange(n if n is not None else C.shape[0])
        if isinstance(method, str):
            if method == "cluster":
                assert C is not None, "Need a matrix C for the cluster ordering."
                return get_leaf_order_from_covariance(C + C.T, linkage_method)
            from glom_io_transform.data.odours import odours, ORDERS
            if method in ORDERS:
                return positions_for(odours.get_order(method))
            raise ValueError(f"Unknown odour ordering '{method}'. "
                             f"Use 'natural', 'cluster', one of {ORDERS}, or a sequence.")
        seq = list(method)
        if seq and all(isinstance(s, str) for s in seq):
            return positions_for(seq)
        order = np.asarray(seq)
        assert np.array_equal(np.sort(order), np.arange(len(order))), \
            "Explicit ordering must be a permutation of 0..n-1."
        return order

    @classmethod
    def matrix(cls, C, ax, order=None, cbar=False, cbar_tag="ρ",
               cmap=None, vlim=None, fontsize=12, **kwargs):
        """One representation matrix in house style: ticks off, Odour labels,
        optional inset colorbar with a tag above it. Returns the image."""
        if ax is None:
            print("No axis provided, skipping matrix plot")
            return None
        cmap = rep_style["cmap"] if cmap is None else cmap
        vmin, vmax = rep_style["vlim"] if vlim is None else vlim
        order = cls.odour_order(order, C=C) if not isinstance(order, np.ndarray) else order
        im = ax.matshow(C[order][:, order], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Odour", fontsize=fontsize)
        ax.set_ylabel("Odour", fontsize=fontsize)
        if cbar:
            cbar_ax = ax.inset_axes([1.025, 0, 0.05, 0.9])  # [x0, y0, width, height]
            cb = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
            cb.ax.tick_params(labelsize=10)
            if cbar_tag:
                ax.text(1.05, 0.925, cbar_tag, transform=ax.transAxes,
                        fontsize=14, va='bottom', ha='center')
        return im

    @classmethod
    def scatter(cls, C_obs, preds, ax, colors=None, subsample=None, rng=0,
                s=10, alpha=0.5, lims=(-0.1, 1), tick_step=0.2,
                id_line={"ls": ":", "color": "gray", "lw": 1},
                legend_kw={"labelspacing": 0, "fontsize": 10, "borderpad": 0,
                           "frameon": False, "loc": "upper left"},
                fontsize=12):
        """Observed-vs-predicted scatter, overlaying one or more predictions.

        preds: {label: matrix} (or a single matrix). All matrix elements are
        used: these are cross-correlations, so every element is informative.
        subsample: fraction of points to show per series (seeded via rng)."""
        if ax is None:
            print("No axis provided, skipping scatter plot")
            return
        if not isinstance(preds, dict):
            preds = {"": preds}
        rng = np.random.default_rng(rng)
        x = np.asarray(C_obs).flatten()
        for name, Cp in preds.items():
            y = np.asarray(Cp).flatten()
            rho = np.corrcoef(x, y)[0, 1]
            mask = np.ones(x.size, bool) if subsample is None else rng.random(x.size) < subsample
            color = colors[name] if isinstance(colors, dict) else colors
            label = (f"{name} " if name else "") + f"$\\rho$={rho:.2f}"
            ax.scatter(x[mask], y[mask], s=s, alpha=alpha, edgecolor=None,
                       color=color, label=label)
        if id_line:
            ax.plot(list(lims), list(lims), **id_line)
        ax.set_xlim(*lims); ax.set_ylim(*lims)
        ax.set_aspect('equal', adjustable='box')
        if tick_step is not None:
            tt = np.arange(0, lims[1] + tick_step/2, tick_step)
            ax.set_xticks(tt); ax.set_yticks(tt)
        ax.legend(**legend_kw)
        ax.set_xlabel("Observed", fontsize=fontsize)
        ax.set_ylabel("Predicted", fontsize=fontsize)
        spines_off(ax)
