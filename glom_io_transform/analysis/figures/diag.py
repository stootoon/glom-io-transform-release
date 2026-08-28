"""Diag-model panels. Each panel takes the namespace produced by
compute.diag (e.g. quartic_geometry) as its data argument; nothing here
knows which figure is calling."""
import numpy as np

from .figures import plt, spines_off
from .figures import Panels

# Region backgrounds: grayscale ramp, darker where the point colors are lighter.
region_shades = {"U": "0.77", "J": "0.90", "W": "0.97"}
LABEL_COLOR   = "0.25"
INSET_COLOR   = "0.2"
DIAG_COLOR_BY = {"vals": "zz", "vmin": -1, "vmax": 1, "cmap": "RdYlBu_r"}


class DiagPhase(Panels):
    """Phase plane of the per-unit quartic: x = tilt (-|x|^3 h), y = alignment
    (-|x|^2 g), with the cusp, region shading, and floating loss-quartic insets
    for one exemplar per region per tilt side.

    Alignment is minus the redundancy, so the double-well side is positive and
    the axis runs the natural way up. Plotting -g on an upright axis puts every
    point exactly where plotting g on an inverted one did, which is why the
    inset and label positions below are unchanged."""

    @classmethod
    def plot(cls, diag, axes, *args, **kwargs):
        print("PLOTTING PANELS DiagPhase")
        assert len(axes) == 1, "DiagPhase should only have one axis"
        ax1 = axes[0]
        if ax1 is None:
            print("No axis provided, skipping plotting")
            return

        d = diag
        Q, Xn = d.Q, d.Xn
        vals = getattr(d, DIAG_COLOR_BY["vals"])
        vmin, vmax, cmap = DIAG_COLOR_BY["vmin"], DIAG_COLOR_BY["vmax"], DIAG_COLOR_BY["cmap"]

        xx_pl = d.hh    # x-axis: -|x_i|^3 h_i
        yy_pl = -d.gg   # y-axis: alignment, -|x_i|^2 g_i
        pts = ax1.scatter(xx_pl, yy_pl, c=vals, alpha=0.75, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.axhline(0, zorder=-1, lw=1, color="k", alpha=0.5, ls="-")
        ax1.axvline(0, zorder=-1, lw=1, color="k", alpha=0.5, ls="-")

        # One exemplar per region per side of the tilt axis (L/R), picked
        # at extreme points so the region shapes and tilts are clearly visible.
        TILT_WEIGHT = 4   # |x| units are ~4x smaller than g units; balances the scores
        extreme_score = {
            "W": lambda i: yy_pl[i] + TILT_WEIGHT * np.abs(xx_pl[i]),
            "J": lambda i: np.abs(xx_pl[i]),
            "U": lambda i: -yy_pl[i] + TILT_WEIGHT * np.abs(xx_pl[i]),
        }
        exemplars = {}
        for name, mask, which_ord in [("U", d.in_U, {"L": -1, "R": -1}),
                                      ("J", d.in_J, {"L": -1, "R": -1}),
                                      ("W", d.in_W, {"L": -1, "R": -1})]:
            for side, smask in [("L", mask & (xx_pl < 0)), ("R", mask & (xx_pl > 0))]:
                idx = np.where(smask)[0]
                if len(idx):
                    order = np.argsort(extreme_score[name](idx))
                    exemplars[(name, side)] = idx[order[which_ord[side]]]
        for (name, side), wi in exemplars.items():
            ax1.plot(xx_pl[wi], yy_pl[wi], "o", color="gray", markerfacecolor="none", markersize=12)

        ax1.set_xlabel("Tilt", fontsize=14)        # -|x_i|^3 h_i
        ax1.set_ylabel("Alignment", fontsize=14)   # -|x_i|^2 g_i

        # Colorbar in an inset axes so it doesn't steal width from the panel
        # (keeps the phase plot aligned with the panel below it).
        cbar_ax = ax1.inset_axes([1.02, 0, 0.04, 1])
        cb = plt.colorbar(pts, cax=cbar_ax)
        cb.set_ticks(np.arange(-1, 1.1, 0.2))
        cb.ax.set_xlabel("Output", fontsize=14, labelpad=6)

        ax1.set_xlim(-0.25, 0.25)

        # Cusp curve: 27 h^2 = 4 (-g)^3 = 4 a^3, so x = +/- sqrt(4/27 y^3) for y >= 0
        yvals = np.linspace(0, max(ax1.get_ylim()), 100)
        cusp_x = np.sqrt(4/27 * yvals**3)
        ax1.plot(cusp_x, yvals, "k--", lw=1, label="Cusp curve")
        ax1.plot(-cusp_x, yvals, "k--", lw=1)

        # Shade the U, J, W regions (data coords; axis inversion handles orientation)
        x0, x1 = ax1.get_xlim()
        y0, y1 = ax1.get_ylim()   # inverted axis: y0 (bottom edge) > y1 (top edge)
        ylo, yhi = min([y0, y1]), max([y0, y1])
        yv = np.linspace(0, yhi, 200)          # positive-alignment side
        cx = np.sqrt(4/27 * yv**3)
        ax1.axhspan(ylo, 0, color=region_shades["U"], lw=0, zorder=-2)             # U: a < 0
        ax1.fill_betweenx(yv, cx, x1, color=region_shades["J"], lw=0, zorder=-2)   # J: right of cusp
        ax1.fill_betweenx(yv, x0, -cx, color=region_shades["J"], lw=0, zorder=-2)  # J: left of cusp
        ax1.fill_betweenx(yv, -cx, cx, color=region_shades["W"], lw=0, zorder=-2)  # W: inside cusp

        lab_x = 0.065
        label_pos = {"W": (lab_x+0.02, 0.891), "J": (lab_x+0.03, 0.475), "U": (lab_x+0.0205, 0.16)}
        for name, (tx, ty) in label_pos.items():
            ax1.text(tx, ty, name, color=LABEL_COLOR, transform=ax1.transAxes,
                     ha="center", fontsize=16, fontweight="bold")
        ax1.set_xlim(x0, x1); ax1.set_ylim(y0, y1)

        # Insets: floating loss quartics, two exemplars per region (one per tilt side)
        def add_loss_inset(i, xc, yc, color, sc=1.0, mincol=None, zmin=None, zmax=None):
            axi = ax1.inset_axes([xc - 0.09, yc, 0.25 * sc, 0.14 * sc])
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
        inset_xc = {"L": 0.12, "R": 0.94}
        f0 = (0 - y0) / (y1 - y0)   # axes-fraction height of the g = 0 line
        inset_yc = {"W": 0.82, "J": f0 + 0.075, "U": 0.125}
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
    def plot(cls, diag, axes, *args, **kwargs):
        print("PLOTTING PANELS DiagApprox")
        assert len(axes) == 3, "Expected 3 axes for DiagApprox (W, J, U)"

        d = diag
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
