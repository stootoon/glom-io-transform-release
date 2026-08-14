import os
import numpy as np

from .figures import plt, GridSpec, mpimg, spines_off 
from .figures import Panels, Figure, Reps, Schem
from .figures import paths

from .figures import best_fit
from .figures import reduce_vertical_gaps
from .figures import get_leaf_order_from_covariance, get_leaves_for_connectivity

from scipy.cluster.hierarchy import linkage, leaves_list

from matplotlib import cm

mode_colors = [cm.RdYlBu(i) for i in [0.2, 0.3, 0.8, 0.8]]
set_alpha = lambda c, a: (c[0], c[1], c[2], a)

class FreeConn(Panels):
    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print("PLOTTING PANELS FreeConn")
        assert len(axes) == 2, "Expected 2 axes for FreeConn"
        ax_conn, ax_conn_ = axes

        W  = plot_data.Z - np.eye(plot_data.Z.shape[0])
        Uw, sw, Vhw = np.linalg.svd(W)

        W_ = plot_data.Z_ - np.eye(plot_data.Z_.shape[0])
        
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
            cbar.set_ticks([-0.01, 0, 0.01])
            

        if ax_conn_ is None:
            print("No ax_conn_ provided, skipping plotting")
        else:
            #vmin, vmax = np.percentile(Wk, [2, 98])
            im = ax_conn_.matshow(Wk[lo][:, lo], vmin=vmin, vmax=vmax, cmap=cmap)
            ax_conn_.set_xticks([]); ax_conn_.set_yticks([])
            ax_conn_.set_xlabel("Input Glomerulus", fontsize=10);
            ax_conn_.xaxis.set_label_position('top')
            ax_conn_.set_ylabel("Output Glomerulus", fontsize=10)

            # Add a colorbar BELOW the ax_conn, with a fraction of the width of ax_conn and a small padding
            #cbar_ax = ax_conn_.inset_axes([0, -0.075, 1, 0.05])  # [x0, y0, width, height]
            #cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            #cbar.set_label('Connection Strength', labelpad=0.5)
            cbar_ax = ax_conn_.inset_axes([0, -0.075, 1, 0.05])  # [x0, y0, width, height]
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Connection Strength', labelpad=-0.5, fontsize=10)
            # Set the font size of the colorbar ticks to 8
            cbar.ax.tick_params(labelsize=10)
            # Set the ticks to be at [-0.01, 0, 0.01]
            cbar.set_ticks([-0.01, 0, 0.01])

class FreeConnModes(Panels):
    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print("PLOTTING PANELS FreeConnModes")
        assert len(axes) == 3, "Expected 3 axes for FreeConnModes"
        ax_11, ax_21, ax_12 = axes

        W  = plot_data.Z - np.eye(plot_data.Z.shape[0])
        U, s, Vh = np.linalg.svd(W)

        W_ = plot_data.Z_ - np.eye(plot_data.Z_.shape[0])
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


class FreeConnModesHist(Panels):
    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print("PLOTTING PANELS FreeConnModesHist")
        assert len(axes) == 1, "Expected 1 axis for FreeConnModesHist"
        ax = axes[0]
        pv = plot_data.props_vals

        best_w = np.array([pvi.val_max_w for pvi in pv])
        best_x = np.array([pvi.val_max_x for pvi in pv])

        ind_best_w = np.array([pvi.ind_max_w for pvi in pv])
        ind_best_x = np.array([pvi.ind_max_x for pvi in pv])

        # The columns of best_w and best_x contain data for each mode
        # Make a mean+/-std bar chart, one for each mode, with the values for w and x side by side to compare them
        modes = best_w.shape[1]
        width = 0.25
        for i in range(modes):
            mean_w = np.mean(best_w[:, i])
            std_w = np.std(best_w[:, i])
            mean_x = np.mean(best_x[:, i])
            std_x = np.std(best_x[:, i])

            col = set_alpha(mode_colors[i], 1.)
            # Make the edges and std lines the same color as the bars
            ax.bar(i - width/2, mean_w, width, yerr=None, label=f"Rep. Diff." if i == 0 else None,
                   color = col,
                   ecolor=col,
                   edgecolor = col)
            # Draw the error bar manually
            ax.plot([i - width/2, i - width/2], [mean_w, mean_w + std_w], color=col, linewidth=2)
 
            col = set_alpha(mode_colors[i], 0.4)
            ax.bar(i + width/2, mean_x, width, yerr=None, label=f"Input" if i == 0 else None,
                   color = col,
                   ecolor = col,
                   edgecolor = col)
            ax.plot([i + width/2, i + width/2], [mean_x, mean_x + std_x], color=col, linewidth=2)
        ax.set_xticks(range(modes))
        ax.set_xticklabels([f"Mode {i+1}" for i in range(modes)])
        ax.set_ylabel("Max Corr. w/ Conn. Mode")
        ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=0.5, handletextpad=0.5, labelspacing=0, borderpad=0.1)
        spines_off(ax)
          
       
class Main(Figure):
    @classmethod
    def plot(cls, plot_data):
        print("PLOTTING FIGURE FitFree")
        art_path = os.path.join(paths.proj_path, "art")

        gs = GridSpec(6, 4)
        fig = plt.gcf()

        top_half = slice(0,3)
        bot_half = slice(3,6)

        ax_free_schem  = fig.add_subplot(gs[top_half,0])
        Schem.plot(plot_data, [ax_free_schem], art_file=os.path.join(art_path, "free_schem_no_diag.png"))
        
        ax_true, ax_fit, ax_fit_vs = [fig.add_subplot(gs[w]) for w in [(top_half, 1), (top_half, 2), (top_half, 3)]]
        ims = Reps.plot(plot_data, [ax_true, ax_fit, ax_fit_vs], cmap="bwr", vlim=[-0.2, 1], include_diag = False, show_corr = {"fontsize":8}, id_line = {"lw":1, "ls":":", "color":"black", "alpha":0.5})
        # Move the xlabels for ax_true and ax_fit to the top
        for ax in [ax_true, ax_fit]:
            ax.xaxis.set_label_position('top')
            ax.set_xlabel(ax.get_xlabel())

        for ax,key in zip([ax_true, ax_fit], ["true", "fit"]):
            cbar_ax = ax.inset_axes([1.025, 0, 0.05, 0.9])  # [x0, y0, width, height]
            cbar = plt.colorbar(ims[key], cax=cbar_ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=10)
            # Write the text "rho" above the colorbar
            ax.text(1.05, 0.925, "ρ", transform=ax.transAxes, fontsize=14, va='bottom', ha='center')

        
        
        ax_conn, ax_conn_ = [fig.add_subplot(gs[w]) for w in [(bot_half, 0), (bot_half, 1)]]
        FreeConn.plot(plot_data, [ax_conn, ax_conn_])

        ax_modes = [fig.add_subplot(gs[(3+i, 2)]) for i in range(3)]
        FreeConnModes.plot(plot_data, ax_modes)

        ax_hist = fig.add_subplot(gs[(bot_half, 3)])
        FreeConnModesHist.plot(plot_data, [ax_hist])

        # Return these in label order                
        return {"circ":ax_free_schem,
                "true": ax_true, "fit": ax_fit, "fit_vs": ax_fit_vs,
                "conn": ax_conn, "conn_": ax_conn_, #"conn_vs": ax_conn_vs,
                "modes": ax_modes,
                "hist": ax_hist,
                }

