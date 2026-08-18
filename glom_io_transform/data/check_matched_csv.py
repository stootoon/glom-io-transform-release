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
from glom_io_transform.data.odours import get_data_file

CSV_STYLE  = dict(color="0.15", lw=1.1, marker="o", ms=2.6, label="exported csv")
RAW_LABELS = ("exported csv / scale", "ours (raw)")
OURS_STYLE = dict(color="#d1495b", lw=1.1, marker="o", ms=2.6, label="ours (min-max)")


def unit(v):
    """Min-max normalise to [0, 1], so an arbitrary scale doesn't hide the shape."""
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else v * 0



def raw_responses(m):
    """The un-normalised exports, relabelled and reordered to match m.csv_column.

    Their header carries the OLD, mislabelled odour order while the values are
    the same as the corrected normalised export, so the labels cannot be taken
    at face value. Every un-normalised column min-max maps exactly (<1e-9) onto
    a column of the normalised one, which identifies what each really is.
    """
    import numpy as np
    import pandas as pd

    meta_columns = set(pd.read_csv(get_data_file("matched_roi_pairs_metadata.csv")).columns)
    # m is ordered by match_id; put the raw frame in the same order, keyed on the
    # pair of row indices rather than on match_id, which has not always agreed.
    order = {(int(a), int(b)): i for i, (a, b) in
             enumerate(zip(m.input_row.values, m.output_row.values))}
    out = {}
    for side in ("input", "output"):
        un = pd.read_csv(get_data_file(f"matched_roi_{side}_trial_averaged.csv"))
        pos = [order[(int(a), int(b))] for a, b in zip(un["input_row"], un["output_row"])]
        assert sorted(pos) == list(range(len(order))), \
            f"The raw {side} export covers different pairs than the metadata."
        un = un.iloc[np.argsort(pos)]
        cu = [c for c in un.columns if c not in meta_columns]
        U  = un[cu].values
        N  = m[side].values                       # already sorted by match
        unit_rows = (U - U.min(1, keepdims=True)) / (U.max(1, keepdims=True) - U.min(1, keepdims=True))
        col_of = {}
        for j in range(len(cu)):
            d = np.abs(N - unit_rows[:, [j]]).max(0)
            k = int(d.argmin())
            assert d[k] < 1e-9, (f"{side} column {cu[j]!r} does not min-max onto any normalised "
                                 f"column (closest is {d[k]:.2g}); the two exports disagree.")
            col_of[k] = j
        assert len(col_of) == len(cu), f"{side}: the column matching is not one-to-one."
        perm = [col_of[k] for k in range(len(cu))]
        moved = sum(1 for k, j in enumerate(perm) if cu[j] != str(m.csv_column.values[k]))
        print(f"{side}: raw export labels "
              + ("agree with where the values land; no relabelling needed."
                 if moved == 0 else
                 f"are wrong for {moved}/{len(cu)} columns; relabelled from the normalised export."))
        out[side] = U[:, perm]
    return out

def plot_matched_comparison(X, Y, out, m=None, raw=False):
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

    # In raw mode both sides are plotted in their own units: ours as they are,
    # his divided by one constant per side (the median least-squares slope), so
    # that any REMAINING per-roi scale difference stays visible rather than
    # being fitted away roi by roi.
    his = raw_responses(m) if raw else {s: m[s].values for s in ("input", "output")}
    scale = {}
    for side in ("input", "output"):
        if raw:
            o = np.stack([ours[side][int(m.sel(match=k)[f"{side}_row"])] for k in m.match.values])
            sl = (his[side] * o).sum(1) / (o * o).sum(1)
            scale[side] = float(np.median(sl))
            print(f"{side}: his/ours slope median {scale[side]:.2f}  "
                  f"[{sl.min():.2f}, {sl.max():.2f}]")
        else:
            scale[side] = 1.0

    n = m.sizes["match"]
    # The default top/bottom margins are a fixed fraction of the figure, which
    # on a figure this tall is inches of white space; set them explicitly.
    fig, axs = plt.subplots(n, 2, figsize=(17, 2.0 * n), sharex=True,
                            gridspec_kw=dict(hspace=0.55, wspace=0.30,
                                             top=0.965, bottom=0.05, left=0.04, right=0.995))

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    for i, match in enumerate(m.match.values):
        p = m.sel(match=match)

        # The input-output correlation of a pair is invariant to permuting the
        # odours of both, so ours and his are comparable whatever the column
        # order is -- any difference is preprocessing, not ordering.
        ours_r = pearson(X.values[int(p.input_row)], Y.values[int(p.output_row)])
        his_r  = float(p.correlation)
        for j, side in enumerate(("input", "output")):
            ax   = axs[i, j]
            row  = int(p[f"{side}_row"])
            csv  = his[side][i] / scale[side]
            mine = ours[side][row] if raw else unit(ours[side][row])
            ax.plot(x, csv,  **CSV_STYLE)
            ax.plot(x, mine, **OURS_STYLE)
            r = np.corrcoef(csv, mine)[0, 1]
            # 'pos' is the positional index within the experiment, which is what
            # the export's local_roi is; roi_id is the label from the source file.
            # They are not the same number wherever ROIs were dropped.
            src = X if side == "input" else Y
            ax.set_title(f"match {match} · {side} {str(p[f'{side}_exp'].values)} "
                         f"pos {int(p[f'{side}_local_roi'])} / roi_id {int(src.roi_id.values[row])} "
                         f"(row {row}) · r = {r:+.2f}",
                         fontsize=9, pad=3)
            if not raw:
                ax.set_ylim(-0.05, 1.05)
                ax.set_yticks([0, 1])
            ax.tick_params(labelsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0 and j == 0:
                if raw:
                    for line, lab in zip(ax.lines, RAW_LABELS):
                        line.set_label(lab)
                ax.legend(fontsize=8, frameon=False, ncol=2,
                          loc="lower left", bbox_to_anchor=(0, 1.35))

        # Between the two panels, in the gutter the wspace above leaves.
        bl, br = axs[i, 0].get_position(), axs[i, 1].get_position()
        fig.text((bl.x1 + br.x0) / 2, (bl.y0 + bl.y1) / 2,
                 f"in\u2013out r\n\nours    {ours_r:+.2f}\nTobias  {his_r:+.2f}\n"
                 f"\u0394 {his_r - ours_r:+.2f}",
                 ha="center", va="center", fontsize=8, family="monospace",
                 color="0.15" if abs(his_r - ours_r) < 0.1 else "#d1495b")

    for ax in axs[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=6.5)
        ax.set_xlabel("odour, in the order of the exported csv header", fontsize=9)

    title = ("Matched ROIs: UN-NORMALISED exported responses vs ours "
             f"(his divided by {scale['input']:.2f} on the input side, {scale['output']:.2f} on the output)"
             if raw else "Matched ROIs: exported responses vs ours, by odour name")
    fig.suptitle(title, fontsize=12, y=0.99)
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
    p.add_argument("--raw", action="store_true",
                   help="use the un-normalised exports (matched_roi_*_trial_averaged.csv)")
    args = p.parse_args(argv)

    with open(args.xy, "rb") as f:
        d = pickle.load(f)
    plot_matched_comparison(d["X"], d["Y"], args.out, raw=args.raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
