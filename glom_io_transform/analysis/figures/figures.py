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
 
def reduce_vertical_gaps(ax_list, reduction):
    """
    Reduce vertical gaps between a list of vertically stacked axes by `reduction` (in figure coordinates).

    Keeps the top of the first axis fixed and the bottom of the last axis fixed.
    """

    n = len(ax_list)
    if n < 2:
        return  # nothing to reduce

    # Calculate the total reduction for each gap
    total_reduction = reduction / (n - 1)

    # Apply the reduction to each pair of adjacent axes
    for i in range(n - 1):
        reduce_vertical_gap(ax_list[i], ax_list[i + 1], total_reduction)
    
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
