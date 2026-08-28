"""Odour-labelling check for Tobias.

Top half:    input correlation heatmap, odours in the stored 'input' order,
             both axes labelled with odour names.
Bottom half: 6 x 8 grid, one panel per odour in the same order, showing the
             trial-averaged z-scored time course of the first two glomeruli of
             the first input experiment.

Needs $DATA set (get_data_dir() reads $DATA/tobias/allExp for the raw .mat),
plus the usual $GLOM_IO and $GLOM_IO_DATA.

    python odour_order_check.py [out.png]
"""
import sys
import io
import contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from glom_io_transform.model_fitting import driver
from glom_io_transform.data.odours import odours
from glom_io_transform.data.responses import (load_glomerular_experiments,
                                              z_score_experiment)
from glom_io_transform.analysis.figures.figures import rep_style

OUT = sys.argv[1] if len(sys.argv) > 1 else "odour_order_check.png"
N_ROIS = 2          # how many glomeruli of the first experiment to show
NAME_CHARS = 20     # odour names are long; truncate for the labels

order = odours.get_order("chemical_class")
# --- top panel: the correlation matrix the manuscript figure shows ---------
# get_data(full=True) is the reduced (roi, odour, repetition) form, which is
# what the correlations are computed from; selection is BY NAME so nothing
# here depends on a positional assumption about the odour axis.
with contextlib.redirect_stdout(io.StringIO()):
    X, _ = driver.get_data(full=True)
pooled = np.vstack([a.mean("repetition").sel(odour=order).values for a in X])
R = np.corrcoef(pooled.T)

# --- bottom panels: z-scored time courses ---------------------------------
# First output of z_score_experiment is ca2t_z: (roi, odour, repetition, time),
# per-trial. Average over repetition for the trial-averaged trace.
exps = load_glomerular_experiments("OMP")
g = exps[0]
ca2t_z = z_score_experiment(g)[0]
traces = ca2t_z.isel(roi=slice(0, N_ROIS)).sel(odour=order).mean("repetition")
t = traces.coords["time"].values
roi_ids = g.ca2.coords["roi_id"].values[:N_ROIS]
vals = traces.values                       # (n_rois, 48, n_time)
lo, hi = np.nanmin(vals), np.nanmax(vals)
pad = 0.06 * (hi - lo)

fig = plt.figure(figsize=(19, 27))
gs = GridSpec(2, 1, figure=fig, height_ratios=[13, 12], hspace=0.10,
              left=0.10, right=0.97, top=0.965, bottom=0.03)

axh = fig.add_subplot(gs[0])
im = axh.imshow(R, cmap=rep_style["cmap"], vmin=0, vmax=1)
short = [n[:NAME_CHARS] for n in order]
axh.set_xticks(range(len(order))); axh.set_xticklabels(short, rotation=90, fontsize=7.5)
axh.set_yticks(range(len(order))); axh.set_yticklabels(short, fontsize=7.5)
axh.tick_params(length=2, pad=1)
axh.set_title("Input correlations (trial-averaged, all glomeruli), odours in the "
              "stored 'input' order", fontsize=13, fontweight="bold", pad=12)
cb = fig.colorbar(im, ax=axh, fraction=0.028, pad=0.015)
cb.set_label("Pearson correlation", fontsize=10)

inner = gs[1].subgridspec(6, 8, hspace=0.42, wspace=0.18)
colors = ["#c1272d", "#0b6fa4"]
odour_on = g.odour_start
for k, name in enumerate(order):
    ax = fig.add_subplot(inner[k // 8, k % 8])
    ax.axvspan(odour_on, t[-1], color="0.92", zorder=-2)   # odour on
    ax.axhline(0, color="0.5", lw=0.6, zorder=-1)
    for j in range(N_ROIS):
        ax.plot(t, vals[j, k], lw=1.1, color=colors[j],
                label=f"roi {roi_ids[j]}" if k == 0 else None)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(lo - pad, hi + pad)
    ax.text(0.03, 0.97, f"{k + 1}. {name[:NAME_CHARS]}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.6, fontweight="bold")
    ax.tick_params(labelsize=6)
    if k % 8:
        ax.set_yticklabels([])
    if k // 8 < 5:
        ax.set_xticklabels([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if k == 0:
        ax.legend(fontsize=6, frameon=False, loc="upper right", handlelength=0.8)

fig.text(0.10, 0.485,
         f"Trial-averaged z-scored time courses, experiment "
         f"{g.name}, first {N_ROIS} glomeruli. Shading = odour on "
         f"({odour_on:.2f}s). Panels in the same 'input' order as above; "
         f"x = time (s), y = z-score.",
         fontsize=12, fontweight="bold", ha="left", va="bottom")

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"n odours {len(order)}, experiment {g.name}, rois {list(roi_ids)}, "
      f"{len(t)} time samples")
print(f"wrote {OUT}")
