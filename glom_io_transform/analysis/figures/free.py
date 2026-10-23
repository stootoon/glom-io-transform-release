"""Free-model panels. Each panel takes the namespace produced by
compute.free (e.g. connectivity_theory) as its data argument; nothing here
knows which figure is calling."""
import numpy as np

from .figures import plt, spines_off
from .figures import Panels

from scipy.cluster.hierarchy import linkage, leaves_list

from matplotlib import cm

mode_colors = [cm.RdYlBu(i) for i in [0.2, 0.3, 0.8, 0.8]]
set_alpha = lambda c, a: (c[0], c[1], c[2], a)


class FreeConn(Panels):
    @classmethod
    def plot(cls, free, axes, *args, **kwargs):
        print("PLOTTING PANELS FreeConn")
        assert len(axes) == 2, "Expected 2 axes for FreeConn"
        ax_conn, ax_conn_ = axes

        W  = free.Z - np.eye(free.Z.shape[0])
        Uw, sw, Vhw = np.linalg.svd(W)

        W_ = free.Z_ - np.eye(free.Z_.shape[0])

        sk = sw.copy(); sk[3:] = 0
        Wk = Uw @ np.diag(sk) @ Vhw

        lo = leaves_list(linkage(W, method='complete'))
        vmin, vmax = np.percentile(W, [2, 98])
        cmap = "RdYlBu_r"

        if ax_conn is None:
            print("No ax_conn provided, skipping plotting")
        else:
            im = ax_conn.matshow(W[lo][:, lo], vmin=vmin, vmax=vmax, cmap=cmap)
            ax_conn.set_xticks([]); ax_conn.set_yticks([])
            # Put the xlabel "Input Glomerulus" on top of the ax_conn
            ax_conn.set_xlabel("Input Glomerulus")
            ax_conn.xaxis.set_label_position('top')

            ax_conn.set_ylabel("Output Glomerulus")

            # Add a colorbar BELOW the ax_conn, with a fraction of the width of ax_conn and a small padding
            cbar_ax = ax_conn.inset_axes([0, -0.075, 1, 0.05])  # [x0, y0, width, height]
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Connection Strength', labelpad=-0.5, fontsize=10)
            # Set the font size of the colorbar ticks to 8
            cbar.ax.tick_params(labelsize=10)
            # Set the ticks to be at [-0.01, 0, 0.01]
            #cbar.set_ticks([-0.01, 0, 0.01])


        if ax_conn_ is None:
            print("No ax_conn_ provided, skipping plotting")
        else:
            #vmin, vmax = np.percentile(Wk, [2, 98])
            im = ax_conn_.matshow(Wk[lo][:, lo], vmin=vmin, vmax=vmax, cmap=cmap)
            ax_conn_.set_xticks([]); ax_conn_.set_yticks([])
            ax_conn_.set_xlabel("Input Glomerulus", fontsize=10);
            ax_conn_.xaxis.set_label_position('top')
            ax_conn_.set_ylabel("Output Glomerulus", fontsize=10)

            cbar_ax = ax_conn_.inset_axes([0, -0.075, 1, 0.05])  # [x0, y0, width, height]
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Connection Strength', labelpad=-0.5, fontsize=10)
            # Set the font size of the colorbar ticks to 8
            cbar.ax.tick_params(labelsize=10)
            # Set the ticks to be at [-0.01, 0, 0.01]
            #cbar.set_ticks([-0.01, 0, 0.01])


class FreeConnModes(Panels):
    @classmethod
    def plot(cls, free, axes, *args, **kwargs):
        print("PLOTTING PANELS FreeConnModes")
        assert len(axes) == 3, "Expected 3 axes for FreeConnModes"

        W  = free.Z - np.eye(free.Z.shape[0])
        U, s, Vh = np.linalg.svd(W)

        W_ = free.Z_ - np.eye(free.Z_.shape[0])
        U_,s_,Vh_ = np.linalg.svd(W_)

        # For each of the top 3 modes of U, find the best matching modes in U_
        ind_best = [np.argmax(np.abs(U_.T @ U[:, i])) for i in range(3)]
        sgn_best = [np.sign(U_[:,ib].T @ U[:, i]) for i,ib  in enumerate(ind_best)]

        for i, ax in enumerate(axes):
            iord = np.argsort(U[:, i])
            ax.plot(U[iord, i], label=f"Conn.", color = set_alpha([0,0,0,0], 0.125), lw=4)
            ax.plot(sgn_best[i] * U_[iord, ind_best[i]], label=f"Rep. Diff.", color = mode_colors[i], lw=1)
            i < 2 and ax.set_xticks([])  # Only show x-ticks on the last plot
            i == 2 and ax.set_xlabel("Glomerulus (sorted)")
            # Get the yticks, round them to 1 decimal place, and set them as the new yticks
            ax.set_yticks([-0.2,0,0.2] if i in [0, 2] else [-0.2, 0, 0.2, 0.4])
            # Set the fontisize of the yticks to 8
            ax.tick_params(axis='y', labelsize=8)
            # Turn the top and right spines off
            spines_off(ax)
            ax.set_ylabel(f"Mode {i+1}", labelpad=-0.1)
            ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=0.5, handletextpad=0.5, labelspacing=0, borderpad=0.1)


# Distance between mode groups, in x units, and the width of one bar. A group
# occupies bar_width either side of its centre, so a spacing much beyond
# 2*bar_width strands the groups in whitespace and forces the panel wider than
# the data needs. Pass group_spacing= to try another.
GROUP_SPACING = 0.6
BAR_WIDTH     = 0.25


class FreeConnModesHist(Panels):
    @classmethod
    def plot(cls, free, axes, group_spacing=GROUP_SPACING, bar_width=BAR_WIDTH,
             *args, **kwargs):
        print("PLOTTING PANELS FreeConnModesHist")
        assert len(axes) == 1, "Expected 1 axis for FreeConnModesHist"
        ax = axes[0]
        pv = free.props_vals

        best_w = np.array([pvi.val_max_w for pvi in pv])
        best_x = np.array([pvi.val_max_x for pvi in pv])

        # The columns of best_w and best_x contain data for each mode
        # Make a mean+/-std bar chart, one for each mode, with the values for w and x side by side to compare them
        modes = best_w.shape[1]
        width = bar_width
        # Groups sit at i*group_spacing rather than at i, so narrowing the
        # spacing pulls them together instead of leaving gaps between them.
        centres = [i * group_spacing for i in range(modes)]
        for i in range(modes):
            centre = centres[i]
            mean_w = np.mean(best_w[:, i])
            std_w = np.std(best_w[:, i])
            mean_x = np.mean(best_x[:, i])
            std_x = np.std(best_x[:, i])

            col = set_alpha(mode_colors[i], 1.)
            # Make the edges and std lines the same color as the bars
            ax.bar(centre - width/2, mean_w, width, yerr=None, label=f"Rep. Diff." if i == 0 else None,
                   color = col,
                   ecolor=col,
                   edgecolor = col)
            # Draw the error bar manually
            ax.plot([centre - width/2, centre - width/2], [mean_w, mean_w + std_w], color=col, linewidth=2)

            col = set_alpha(mode_colors[i], 0.4)
            ax.bar(centre + width/2, mean_x, width, yerr=None, label=f"Input" if i == 0 else None,
                   color = col,
                   ecolor = col,
                   edgecolor = col)
            ax.plot([centre + width/2, centre + width/2], [mean_x, mean_x + std_x], color=col, linewidth=2)
        ax.set_xticks(centres)
        ax.set_xticklabels([f"Mode {i+1}" for i in range(modes)])
        ax.set_ylabel("Max Corr. w/ Conn. Mode")
        ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=0.5, handletextpad=0.5, labelspacing=0, borderpad=0.1)
        spines_off(ax)
