import os
import numpy as np

from .figures import plt, GridSpec, mpimg, spines_off
from .figures import Panels, Figure, Schem
from .figures import paths

from .figures import best_fit
from .figures import reduce_vertical_gaps
from .figures import get_leaf_order_from_covariance, get_leaves_for_connectivity

from scipy.cluster.hierarchy import linkage, leaves_list

from matplotlib import cm

mode_colors = [cm.RdYlBu(i) for i in [0.2, 0.3, 0.8, 0.8]]
set_alpha = lambda c, a: (c[0], c[1], c[2], a)

# Shared styling for the Diag phase-plane / approximation panels.
# Region backgrounds: grayscale ramp, darker where the point colors are lighter.
region_shades = {"U": "0.77", "J": "0.90", "W": "0.97"}
LABEL_COLOR   = "0.25"
INSET_COLOR   = "0.2"
DIAG_COLOR_BY = {"vals": "zz", "vmin": -1, "vmax": 1, "cmap": "RdYlBu_r"}


class DiagPhase(Panels):
    """Phase plane of the per-unit quartic: x = tilt (-|x|^3 h), y = redundancy
    (|x|^2 g, plotted decreasing upward), with the cusp, region shading, and
    floating loss-quartic insets for one exemplar per region per tilt side."""

    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print("PLOTTING PANELS DiagPhase")
        assert len(axes) == 1, "DiagPhase should only have one axis"
        ax1 = axes[0]
        if ax1 is None:
            print("No axis provided, skipping plotting")
            return

        d = plot_data.diag
        Q, Xn = d.Q, d.Xn
        vals = getattr(d, DIAG_COLOR_BY["vals"])
        vmin, vmax, cmap = DIAG_COLOR_BY["vmin"], DIAG_COLOR_BY["vmax"], DIAG_COLOR_BY["cmap"]

        xx_pl = d.hh   # x-axis: -|x_i|^3 h_i
        yy_pl = d.gg   # y-axis: |x_i|^2 g_i, plotted decreasing upward (inverted below)
        pts = ax1.scatter(xx_pl, yy_pl, c=vals, alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.axhline(0, zorder=-1, lw=1, color="k", alpha=0.5, ls="-")
        ax1.axvline(0, zorder=-1, lw=1, color="k", alpha=0.5, ls="-")

        # One exemplar per region per side of the tilt axis (L/R), picked
        # at extreme points so the region shapes and tilts are clearly visible.
        TILT_WEIGHT = 4   # |x| units are ~4x smaller than g units; balances the scores
        extreme_score = {
            "W": lambda i: -yy_pl[i] + TILT_WEIGHT * np.abs(xx_pl[i]),
            "J": lambda i: np.abs(xx_pl[i]),
            "U": lambda i: yy_pl[i] + TILT_WEIGHT * np.abs(xx_pl[i]),
        }
        exemplars = {}
        for name, mask, which_ord in [("U", d.in_U, {"L": -1, "R": -1}),
                                      ("J", d.in_J, {"L": -2, "R": -1}),
                                      ("W", d.in_W, {"L": -2, "R": -1})]:
            for side, smask in [("L", mask & (xx_pl < 0)), ("R", mask & (xx_pl > 0))]:
                idx = np.where(smask)[0]
                if len(idx):
                    order = np.argsort(extreme_score[name](idx))
                    exemplars[(name, side)] = idx[order[which_ord[side]]]
        for (name, side), wi in exemplars.items():
            ax1.plot(xx_pl[wi], yy_pl[wi], "o", color="gray", markerfacecolor="none", markersize=12)

        ax1.set_xlabel("Tilt", fontsize=14)        # -|x_i|^3 h_i
        ax1.set_ylabel("Redundancy", fontsize=14)  # |x_i|^2 g_i

        # Colorbar in an inset axes so it doesn't steal width from the panel
        # (keeps the phase plot aligned with the panel below it).
        cbar_ax = ax1.inset_axes([1.02, 0, 0.04, 1])
        cb = plt.colorbar(pts, cax=cbar_ax)
        cb.set_ticks(np.arange(-1, 1.1, 0.2))
        cb.ax.set_xlabel("Output", fontsize=14, labelpad=6)

        ax1.set_xlim(-0.25, 0.25)
        ax1.invert_yaxis()   # negative g (double-well side) on top

        # Cusp curve: 27 h^2 = 4 (-g)^3, i.e. x = +/- sqrt(4/27 (-y)^3) for y <= 0
        yvals = np.linspace(min(ax1.get_ylim()), 0, 100)
        cusp_x = np.sqrt(4/27 * (-yvals)**3)
        ax1.plot(cusp_x, yvals, "k--", lw=1, label="Cusp curve")
        ax1.plot(-cusp_x, yvals, "k--", lw=1)

        # Shade the U, J, W regions (data coords; axis inversion handles orientation)
        x0, x1 = ax1.get_xlim()
        y0, y1 = ax1.get_ylim()   # inverted axis: y0 (bottom edge) > y1 (top edge)
        ylo, yhi = min([y0, y1]), max([y0, y1])
        yv = np.linspace(ylo, 0, 200)          # negative-g side
        cx = np.sqrt(4/27 * (-yv)**3)
        ax1.axhspan(0, yhi, color=region_shades["U"], lw=0, zorder=-2)             # U: g > 0
        ax1.fill_betweenx(yv, cx, x1, color=region_shades["J"], lw=0, zorder=-2)   # J: right of cusp
        ax1.fill_betweenx(yv, x0, -cx, color=region_shades["J"], lw=0, zorder=-2)  # J: left of cusp
        ax1.fill_betweenx(yv, -cx, cx, color=region_shades["W"], lw=0, zorder=-2)  # W: inside cusp

        lab_x = 0.065
        label_pos = {"W": (lab_x+0.01, 0.91), "J": (lab_x+0.01, 0.475), "U": (lab_x+0.0125, 0.16)}
        for name, (tx, ty) in label_pos.items():
            ax1.text(tx, ty, name, color=LABEL_COLOR, transform=ax1.transAxes,
                     ha="center", fontsize=16, fontweight="bold")
        ax1.set_xlim(x0, x1); ax1.set_ylim(y0, y1)

        # Insets: floating loss quartics, two exemplars per region (one per tilt side)
        def add_loss_inset(i, xc, yc, color, sc=1.0, mincol=None, zmin=None, zmax=None):
            axi = ax1.inset_axes([xc - 0.09, yc, 0.18 * sc, 0.14 * sc])
            zv = np.linspace(Q.z.min() if zmin is None else zmin/Xn[i],
                             Q.z.max() if zmax is None else zmax/Xn[i], 100)
            axi.set_ymargin(0.25)
            axi.plot(zv*Xn[i], Q.LiFUN(i, zv), lw=1, color="k")
            mincol = "r" if mincol is None else mincol
            axi.plot(Q.z[i]*Xn[i], Q.LiFUN(i, Q.z[i]), "o", color=mincol, markersize=4)
            axi.patch.set_alpha(0)   # transparent: just a floating quartic
            [axi.spines[s].set_visible(False) for s in ["top", "right", "left"]]
            axi.set_yticks([]); axi.set_xticks(np.arange(-2, 3, 1))
            axi.set_xlabel("Output", fontsize=7, labelpad=0)
            axi.spines["bottom"].set_position(("outward", 2))
            axi.tick_params(labelsize=6, length=2, pad=1)
            return axi

        # Left insets share one x position, right insets another; rows per region:
        # W in the top corners, J just above the g = 0 line, U in the bottom corners.
        inset_xc = {"L": 0.12, "R": 0.96}
        f0 = (0 - y0) / (y1 - y0)   # axes-fraction height of the g = 0 line
        inset_yc = {"W": 0.82, "J": f0 + 0.05, "U": 0.075}
        for (name, side), wi in exemplars.items():
            mincol = pts.cmap(pts.norm(vals[wi]))
            axi = add_loss_inset(wi, inset_xc[side], inset_yc[name], INSET_COLOR,
                                 sc=0.5, mincol=mincol, zmin=-1.5, zmax=1.5)
            axi.axvline(0, color="k", lw=0.5, alpha=0.5, ls=":")
            if name != "U":
                axi.set_xlabel(None)

        return pts


class DiagApprox(Panels):
    """Per-region approximation quality: approximate gain prediction vs the
    fitted gain, one panel per region (W, J, U), against the identity line."""

    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print("PLOTTING PANELS DiagApprox")
        assert len(axes) == 3, "Expected 3 axes for DiagApprox (W, J, U)"

        d = plot_data.diag
        vals = getattr(d, DIAG_COLOR_BY["vals"])
        vmin, vmax, cmap = DIAG_COLOR_BY["vmin"], DIAG_COLOR_BY["vmax"], DIAG_COLOR_BY["cmap"]
        masks = {"W": d.in_W, "J": d.in_J, "U": d.in_U}

        for name, ax in zip("WJU", axes):
            if ax is None:
                print(f"No axis provided for {name}, skipping plotting")
                continue
            mask = masks[name]
            zz_sub = d.zz[mask]
            order = np.argsort(zz_sub)
            ax.scatter(zz_sub[order], d.fits[name]["approx"][mask][order],
                       c=vals[mask][order],
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       alpha=0.75, marker="o", s=20, label="approx")
            uu = np.linspace(-1.5, 1.5, 100)
            xl = ax.get_xlim(); yl = ax.get_ylim()
            ax.plot(uu, uu, "-", color="gray", zorder=-1, lw=1)
            ax.set_xlim(xl); ax.set_ylim(yl)
            ax.grid(True, linestyle=":", lw=1)
            [f(0, color="k", lw=0.5, alpha=0.5, ls="-") for f in [ax.axvline, ax.axhline]]
            # Match the panel background to the region's gray in the phase plane
            ax.set_facecolor(region_shades[name])
            spines_off(ax)
            ax.set_xlabel("Output", fontsize=14)
            ax.set_ylabel(d.fits[name]["approx_tex"], fontsize=10)
            ax.text(0.925, 0.055, name, color=LABEL_COLOR, transform=ax.transAxes,
                    ha="right", fontsize=16, fontweight="bold")


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
        print("PLOTTING FIGURE ExplainModels")
        art_path = os.path.join(paths.proj_path, "art")

        # 2 x 12 layout (6 rows so the modes panel can stack 3 axes):
        #   top:    | geom (3) | phase (3) | W (2) | J (2) | U (2) |
        #   bottom: | conn (3) | conn_ (3) | modes (3) | hist (3)  |
        gs = GridSpec(6, 12)
        fig = plt.gcf()

        top_half = slice(0,3)
        bot_half = slice(3,6)

        # --- Top row: Diag model logic ---
        ax_geom = fig.add_subplot(gs[top_half, 0:3])
        geom_art = os.path.join(art_path, "diag_geometry.png")
        if os.path.exists(geom_art):
            Schem.plot(plot_data, [ax_geom], art_file=geom_art)
        else:
            print(f"Geometry schematic not found at {geom_art}; leaving panel blank.")
            ax_geom.axis("off")

        ax_phase = fig.add_subplot(gs[top_half, 3:6])
        DiagPhase.plot(plot_data, [ax_phase])

        ax_approx = [fig.add_subplot(gs[top_half, sl]) for sl in
                     [slice(6, 8), slice(8, 10), slice(10, 12)]]
        DiagApprox.plot(plot_data, ax_approx)

        # --- Bottom row: Free model logic ---
        ax_conn, ax_conn_ = [fig.add_subplot(gs[w]) for w in
                             [(bot_half, slice(0, 3)), (bot_half, slice(3, 6))]]
        FreeConn.plot(plot_data, [ax_conn, ax_conn_])

        ax_modes = [fig.add_subplot(gs[3+i, 6:9]) for i in range(3)]
        FreeConnModes.plot(plot_data, ax_modes)

        ax_hist = fig.add_subplot(gs[bot_half, 9:12])
        FreeConnModesHist.plot(plot_data, [ax_hist])

        # Return these in label order
        return {"geom": ax_geom,
                "phase": ax_phase,
                "approx_W": ax_approx[0], "approx_J": ax_approx[1], "approx_U": ax_approx[2],
                "conn": ax_conn, "conn_": ax_conn_,
                "modes": ax_modes,
                "hist": ax_hist,
                }
