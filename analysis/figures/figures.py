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

import paths

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
            ax_schem.axis('off')
            img_artist.set_clip_on(False)
    
class Reps(Panels):
    @classmethod
    def plot(cls, plot_data, axes, *args, cmap="Spectral_r", vlim = None, show_corr = {}, id_line = {}, include_diag=True,  **kwargs):
        print("PLOTTING PANELS FreeReps")
        print(f"{include_diag=}")
        assert len(axes) == 3, "Expected 3 axes for FreeReps"
        ax_true, ax_fit, ax_fit_vs = axes

        Cvld = plot_data.Rep_out
        Cpred_vld = plot_data.Rep_est
        lo = get_leaf_order_from_covariance(Cvld + Cvld.T, "average")
        vmin, vmax = np.percentile(Cvld, [1, 99]) if vlim is None else vlim

        ims = {"true": None, "fit": None}
        
        if ax_true is None:
            print("No ax_true provided, skipping plotting")
        else:
            ims["true"] = ax_true.matshow(Cvld[lo][:, lo], vmin=vmin, vmax=vmax, cmap=cmap)
            ax_true.set_xticks([]); ax_true.set_yticks([])
            ax_true.set_xlabel("Odour"); ax_true.set_ylabel("Odour")
            
        if ax_fit is None:
            print("No ax_fit provided, skipping plotting")
        else:
            ims["fit"] = ax_fit.matshow(Cpred_vld[lo][:, lo], vmin=vmin, vmax=vmax, cmap=cmap)
            ax_fit.set_xticks([]); ax_fit.set_yticks([])
            ax_fit.set_xlabel("Odour"); ax_fit.set_ylabel("Odour")

        if ax_fit_vs is None:
            print("No ax_fit_vs provided, skipping plotting")
        else:
            # Get the elements on and above the diagonal
            inds = np.triu_indices_from(Cvld, k=0 if include_diag else 1)
            C_obs = Cvld[inds]
            C_pred = Cpred_vld[inds]
            corr = np.corrcoef(C_obs, C_pred)[0, 1]
    
            ax_fit_vs.scatter(C_obs, C_pred, s=5, alpha=0.2, label=f"$\\rho$={corr:.2f}")
            ax_fit_vs.axis("square")
            ax_fit_vs.set_ylim(-0.01, 1.01)
            ax_fit_vs.set_xlim(-0.01, 1.01)
            ax_fit_vs.set_xlabel("Observed")
            ax_fit_vs.set_ylabel("Predicted")
            show_corr and ax_fit_vs.legend(**show_corr)
            xl = ax_fit_vs.get_xlim()
            id_line and ax_fit_vs.plot(xl, xl, **id_line)
            spines_off(ax_fit_vs)

        return ims
