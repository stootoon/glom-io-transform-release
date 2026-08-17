"""Compare the exported matched-ROI responses against our own, odour by odour.

For each of the matched pairs, both sides are plotted over the 48 odours in the
order the exported CSV's header gives them, with our own response for the same
ROI -- selected by odour name, so the two traces claim to be the same odour at
the same x position -- min-max normalised on top. Where they disagree, they
disagree about which odour a value belongs to.

Run:
    python glom_io_transform/data/check_matched_csv.py
    python -m glom_io_transform.data.check_matched_csv --out /tmp/matched.pdf
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from glom_io_transform.data.matched import load_matched

CSV_STYLE  = dict(color="0.15", lw=1.1, marker="o", ms=2.6, label="exported csv")
OURS_STYLE = dict(color="#d1495b", lw=1.1, marker="o", ms=2.6, label="ours (min-max)")


def unit(v):
    """Min-max normalise to [0, 1], so an arbitrary scale doesn't hide the shape."""
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else v * 0


def plot_matched_comparison(X, Y, out, m=None):
    """One row per matched pair, input on the left and output on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    m = load_matched() if m is None else m
    names = [str(c) for c in m.csv_column.values]
    x = np.arange(len(names))

    # Our arrays are labelled by odour, so ask for the header order by name
    # rather than assuming the two are stored the same way round.
    ours = {"input": X.sel(odour=names).values, "output": Y.sel(odour=names).values}

    n = m.sizes["match"]
    # The default top/bottom margins are a fixed fraction of the figure, which
    # on a figure this tall is inches of white space; set them explicitly.
    fig, axs = plt.subplots(n, 2, figsize=(17, 2.0 * n), sharex=True,
                            gridspec_kw=dict(hspace=0.55, wspace=0.08,
                                             top=0.965, bottom=0.05, left=0.04, right=0.995))

    for i, match in enumerate(m.match.values):
        p = m.sel(match=match)
        for j, side in enumerate(("input", "output")):
            ax   = axs[i, j]
            row  = int(p[f"{side}_row"])
            csv  = p[side].values
            mine = unit(ours[side][row])
            ax.plot(x, csv,  **CSV_STYLE)
            ax.plot(x, mine, **OURS_STYLE)
            r = np.corrcoef(csv, mine)[0, 1]
            ax.set_title(f"match {match} · {side} {str(p[f'{side}_exp'].values)} "
                         f"roi {int(p[f'{side}_local_roi'])} (row {row}) · r = {r:+.2f}",
                         fontsize=9, pad=3)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False, ncol=2,
                          loc="lower left", bbox_to_anchor=(0, 1.35))

    for ax in axs[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=6.5)
        ax.set_xlabel("odour, in the order of the exported csv header", fontsize=9)

    fig.suptitle("Matched ROIs: exported responses vs ours, by odour name", fontsize=12, y=0.99)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Wrote {out}")
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--xy",  default=os.path.join(here, "XY.p"),
                   help="pickle holding the trial-averaged X and Y (default: data/XY.p)")
    p.add_argument("--out", default="matched_csv_comparison.pdf")
    args = p.parse_args(argv)

    with open(args.xy, "rb") as f:
        d = pickle.load(f)
    plot_matched_comparison(d["X"], d["Y"], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
