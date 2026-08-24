"""Violin panels for generalization dataframes.

One panel is one split (sampler, mode), one metric family (prefix) and
optionally one outclass: an Input violin, then one per model, and for corr_en an
Output violin last. plot_violins draws a panel onto an axis the caller supplies,
runs whatever comparisons it is given and brackets them, so any figure can use
these panels wherever it likes.

    from glom_io_transform.analysis.figures.violin_plots import plot_violins
    plot_violins(ax, df, "trials", "random", prefix="corr",
                 comparisons=["Input>Model"])
"""
import matplotlib
import numpy as np

from collections import OrderedDict
from dataclasses import dataclass

from .figures import np, plt, spines_off
from .figures import Panels

import glom_io_transform.model_fitting.proc_fit_models as pfm

from ..compute.generalization import (MODEL_LABELS, METRIC_COLUMNS, as_labels,
                                      models_in, compare_panel)


# Full axis labels, not stems: cov and corr are distances to the output, so they
# are mismatches, but corr_en is a property of each matrix on its own and there
# is nothing it is a mismatch FROM. See notes/generalization_statistics.md.
METRIC_LABELS = {"cov":     "Covariance Mismatch",
                 "corr":    "Correlation Mismatch",
                 "corr_en": "Correlation Energy",
                 "r2":      "R\u00b2"}

# The label here and the columns there are two halves of one registration, so a
# metric added to only one of them is caught at import rather than at draw time.
assert set(METRIC_LABELS) == set(METRIC_COLUMNS), (
    f"METRIC_LABELS and METRIC_COLUMNS must cover the same metrics; "
    f"{sorted(set(METRIC_LABELS) ^ set(METRIC_COLUMNS))} is in only one.")

def group_order(models, prefix):
    """The violin labels of a panel, left to right.

    Used both to draw the violins and to place the significance brackets, so the
    two cannot disagree about which x position a group sits at. Whether there is
    an Output group comes from the metric's registry entry.
    """
    cols = METRIC_COLUMNS[prefix]
    return ["Input"] + list(models.values()) + (["Output"] if "Output" in cols else [])


# Headroom above the data, as a fraction of the largest value drawn: YPAD so a
# violin's own kernel tail is not clipped, then one BRACKET_ROW per stacked row
# of significance brackets.
YPAD        = 1.08
BRACKET_ROW = 0.135


def data_span(data):
    """(top, floor) of the violins drawn, where floor is 0 unless the data goes
    below it.

    Mismatches and energies start at zero, and an axis that starts anywhere else
    misreads them; R2 can be negative, and clipping it at zero would hide the
    conditions that fail.
    """
    top   = max(np.nanmax(d.vals) for d in data)
    below = min(np.nanmin(d.vals) for d in data)
    return float(top), float(min(0.0, below))


def assign_bracket_rows(spans):
    """Stack overlapping brackets: narrowest first, each into the lowest free row."""
    rows, occupied = [0] * len(spans), []
    for i in sorted(range(len(spans)), key=lambda k: abs(spans[k][1] - spans[k][0])):
        lo, hi = sorted(spans[i])
        r = 0
        while any(not (hi < a or lo > b) for (a, b, rr) in occupied if rr == r):
            r += 1
        rows[i] = r
        occupied.append((lo, hi, r))
    return rows


@dataclass
class ViolinPlotData:
    vals: list
    col: str
    lab: str

def draw_violins(axi:matplotlib.axes.Axes, data:OrderedDict[str, ViolinPlotData]) -> matplotlib.axes.Axes:
    # Create a violin plot of the three distributions
    qs = [[0.25, 0.75]] * len(data)
    vals = [d.vals for d in data]
    cols = [d.col for d in data]
    labs = [d.lab for d in data]
    parts = axi.violinplot(vals, showmeans=False, showmedians=True, showextrema=False, quantiles=qs)
    for pc, col in zip(parts['bodies'], cols):
        pc.set_facecolor(col)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for partname in ('cmedians','cquantiles'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
    for pos, d in enumerate(vals, start=1):
      q1, q3 = np.percentile(d, [25, 75])
      axi.vlines(pos, q1, q3, color='black', linewidth=1)
    axi.set_xticks(np.arange(1, len(data)+1))
    axi.set_xticklabels(labs)
    return axi


def violin_data(df, sampler, mode, prefix="corr", outclass=None, models=None,
                colors=None):
    """The violins of one panel, left to right: Input, each model, and an Output
    if the metric has one.

    Which column each group reads comes from METRIC_COLUMNS, so a metric is
    registered in one place and both the violins and the statistics follow.

    `colors` maps model name -> colour, overriding pfm.variant_color. Needed for
    condition names that are not "<Model>_<loss>".

    Split out from the drawing so that a caller can see what a panel will
    contain -- which models survived the filtering, in what order -- without
    having to draw it.
    """
    assert prefix in METRIC_LABELS, f"prefix must be one of {list(METRIC_LABELS)}"
    models = as_labels(models, df)
    colors = {} if colors is None else colors
    cols   = METRIC_COLUMNS[prefix]

    rows = (df["sampler"] == sampler) & (df["mode"] == mode)
    if outclass is not None:
        rows &= (df["outclass"] == outclass)

    # Only the models actually fitted for THIS panel: a run may cover fewer
    # models than the frame as a whole (the matched runs do), and an empty
    # violin is a ValueError inside matplotlib rather than a blank.
    fitted = set(df[rows]["model"].unique())
    names  = [m for m in models if m in fitted]
    assert names, (f"No models for sampler={sampler}, mode={mode}, outclass={outclass}; "
                   f"the dataframe has {sorted(df['model'].unique())}.")

    def values(model, column):
        return df[rows & (df["model"] == model)][column].values

    # Input and Output do not depend on the model, so read them off whichever
    # model is present rather than assuming Diag was fitted.
    first = names[0]
    panel = [ViolinPlotData(values(first, cols["Input"]), "LightGray", "Input")]
    def colour_for(name):
        # Not colors.get(name, variant_color(name)): the default would be
        # evaluated even for an overridden name, and variant_color rejects
        # names it cannot parse.
        if name in colors:
            return colors[name]
        try:
            return pfm.variant_color(name)
        except AssertionError:
            raise KeyError(
                f"No colour for {name!r}: it is not in `colors`, and it is not a "
                f"'<Model>_<loss>' name, so there is no default to fall back on. "
                f"Either add it to `colors` or rename it. Drawing: {names}.") from None

    panel += [ViolinPlotData(values(m, cols["est"]), colour_for(m), models[m])
              for m in names]
    if "Output" in cols:
        panel.append(ViolinPlotData(values(first, cols["Output"]), "Gray", "Output"))
    return panel


def panel_brackets(df, prefix, sampler, mode, comparisons, outclass=None,
                   models=None, correction=None, verbose=False):
    """The significance brackets of one panel, and how many rows they stack into.

    Returns ([(low label, high label, mark), ...], n_rows). The row count comes
    back alongside the brackets because a figure needs it to reserve headroom
    BEFORE any panel is drawn.
    """
    models = as_labels(models, df)
    res = compare_panel(df, prefix, sampler, mode, comparisons,
                        outclass=outclass, correction=correction, models=models)
    res = res[res["requested"].isin(comparisons)]
    if not len(res):
        return [], 0

    if verbose:
        where = f"{sampler} {mode}" + (f" / {outclass}" if outclass else "")
        print(f"  {prefix}  {where}   (n = {int(res['n'].iloc[0])})")
        for r in res.itertuples():
            sided = "2-sided" if r.alternative == "two-sided" else "1-sided"
            print(f"    {r.comparison:<22} {sided}  median diff {r.median_diff:+.4g} "
                  f"[{r.iqr_lo:+.4g}, {r.iqr_hi:+.4g}]  p = {r.p_adj:.3g}  {r.mark}")

    order = group_order(models, prefix)
    spans = [(order.index(r.lo) + 1, order.index(r.hi) + 1) for r in res.itertuples()]
    return [(r.lo, r.hi, r.mark) for r in res.itertuples()], max(assign_bracket_rows(spans)) + 1


def plot_violins(ax, df, sampler, mode, prefix="corr", outclass=None, models=None,
                 colors=None, comparisons=None, correction=None, verbose=False,
                 ylabel=True, ylim=None, fontsize=10,
                 brackets=None, bracket_base=None, bracket_step=None, reverse=False):
    """One generalization violin panel onto `ax`, with its statistics. Returns the axis.

    Everything it needs about the data is in `df`, a generalization dataframe,
    so any figure can call it for any panel it wants:

        plot_violins(ax, df, "trials", "random", comparisons=["Input>Model"])

    `comparisons` are the strings parse_comparison understands. The panel runs
    its own tests, draws its own brackets, and sizes the axis to fit them.

    A figure drawing several panels that must share one y scale should call
    panel_brackets itself, take the largest row count over its panels, and pass
    `brackets`, `bracket_base` and `bracket_step` explicitly -- which is what
    Supp does. Passing `brackets` skips the tests.
    """
    data = violin_data(df, sampler, mode, prefix=prefix, outclass=outclass,
                       models=models, colors=colors)
    if reverse:
        data = list(reversed(data))
    draw_violins(ax, data)

    n_rows = 0
    if brackets is None and comparisons:
        brackets, n_rows = panel_brackets(df, prefix, sampler, mode, comparisons,
                                          outclass=outclass, models=models,
                                          correction=correction, verbose=verbose)
    # A panel drawn on its own sizes itself: the data, plus room for its brackets.
    if brackets and (bracket_base is None or bracket_step is None or ylim is None):
        top, floor = data_span(data)
        span = top - floor
        bracket_base = top + (YPAD - 1) * span         if bracket_base is None else bracket_base
        bracket_step = BRACKET_ROW * span              if bracket_step is None else bracket_step
        if ylim is None:
            ylim = (floor, top + (YPAD - 1 + n_rows * BRACKET_ROW) * span)

    # Font sizes are in points, so they do not scale with the figure: text that
    # reads well on a 24-inch figure is unreadable on a 6-inch one. Size them
    # relative to the caller's base size instead.
    if ylabel:
        ax.set_ylabel(METRIC_LABELS[prefix], fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize * 0.9)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if brackets:
        # Positions come from the violins just drawn, so a bracket can never
        # point at the wrong group.
        at = {d.lab: i + 1 for i, d in enumerate(data)}
        spans = [(at[lo], at[hi]) for lo, hi, _ in brackets]
        for (lo, hi, mark), row in zip(brackets, assign_bracket_rows(spans)):
            y = bracket_base + row * bracket_step
            x1, x2 = at[lo], at[hi]
            ax.plot([x1, x1, x2, x2], [y - bracket_step*0.12, y, y, y - bracket_step*0.12],
                    lw=0.8, color="0.2", clip_on=False)
            ax.text((x1 + x2)/2, y + bracket_step*0.08, mark, ha="center",
                    va="bottom", fontsize=fontsize * (0.9 if mark == "n.s." else 1.1),
                    color="0.2")
    spines_off(ax)
    return ax


class GenViolin(Panels):
    """plot_violins in the Panels protocol, for figures that lay panels out by
    class rather than calling the function directly."""

    @classmethod
    def plot(cls, df, axes, *args, sampler=None, mode=None, **kwargs):
        assert len(axes) == 1, "GenViolin should only have one axis"
        assert sampler is not None and mode is not None, "sampler and mode must be given"
        if axes[0] is None:
            print("No axis provided, skipping plotting")
            return
        return plot_violins(axes[0], df, sampler, mode, **kwargs)
