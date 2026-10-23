"""Correlation heat map of one side's responses, odours in a chosen order.

For checking that an ordering means what we think it means: both axes are
labelled with odour names, so the matrix can be compared against someone
else's figure odour by odour.

    python glom_io_transform/data/odour_order_check.py input in
    python glom_io_transform/data/odour_order_check.py chemical_class out

Writes <prefix>_glom_<in|out>_ord<order>.png.

Odours are selected BY NAME throughout, so nothing here depends on a
positional assumption about the data's odour axis -- which is the assumption
under suspicion whenever this script is being run.
"""
import argparse
import io
import contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glom_io_transform.model_fitting import driver
from glom_io_transform.data.odours import odours, ORDERS
from glom_io_transform.analysis.figures.figures import rep_style

NAME_CHARS = 22     # odour names are long; truncate for the tick labels


def correlations(side, order):
    """Trial-averaged Pearson correlation between odours, across glomeruli."""
    with contextlib.redirect_stdout(io.StringIO()):
        X, Y = driver.get_data(full=True)
    arrays = X if side == "in" else Y
    # One matrix of glomeruli x odours, pooled over experiments.
    pooled = np.vstack([a.mean("repetition").sel(odour=order).values for a in arrays])
    return np.corrcoef(pooled.T), pooled.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("order", choices=list(ORDERS),
                        help="Which odour ordering to arrange the matrix by.")
    parser.add_argument("side", choices=["in", "out"],
                        help="Which side's responses to correlate.")
    parser.add_argument("--prefix", default="corr", help="Leading part of the file name.")
    parser.add_argument("--outdir", default=".", help="Where to write it.")
    args = parser.parse_args()

    order = odours.get_order(args.order)
    R, n_glom = correlations(args.side, order)

    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(R, cmap=rep_style["cmap"], vmin=0, vmax=1)
    short = [n[:NAME_CHARS] for n in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels(short, rotation=90, fontsize=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(short, fontsize=8)
    ax.tick_params(length=2, pad=1)
    side_name = "input" if args.side == "in" else "output"
    ax.set_title(f"{side_name.capitalize()} correlations, trial-averaged over "
                 f"{n_glom} glomeruli\nodours in '{args.order}' order",
                 fontsize=14, fontweight="bold", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("Pearson correlation", fontsize=11)

    out = f"{args.outdir}/{args.prefix}_glom_{args.side}_ord{args.order}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    off = R[~np.eye(len(R), dtype=bool)]
    print(f"{n_glom} glomeruli, {len(order)} odours, mean off-diagonal "
          f"correlation {off.mean():+.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
