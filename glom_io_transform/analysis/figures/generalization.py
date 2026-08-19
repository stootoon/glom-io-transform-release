"""Generalization violin panels.

GenViolin plots a single panel (one split, one metric family, optionally one
outclass) into a given axis, so figures can place them anywhere. Supp lays out
the full supplementary figure for one metric: violins per split type plus the
per-outclass breakdown.
"""
import matplotlib
import numpy as np

from collections import OrderedDict
from dataclasses import dataclass

from .figures import np, plt, GridSpec, spines_off
from .figures import Panels, Figure

import glom_io_transform.model_fitting.proc_fit_models as pfm

# Full axis labels, not stems: cov and corr are distances to the output, so they
# are mismatches, but corr_en is a property of each matrix on its own and there
# is nothing it is a mismatch FROM. See notes/generalization_statistics.md.
METRIC_LABELS = {"cov":     "Covariance Mismatch",
                 "corr":    "Correlation Mismatch",
                 "corr_en": "Correlation Energy"}
MODEL_LABELS  = {"Diag": "Diag", "DiagOnlyInh": "DiagInh", "Free": "Free", "FreeLat": "FreeLat"}


@dataclass
class ViolinPlotData:
    vals: list
    col: str
    lab: str

def violin_plots(axi:matplotlib.axes.Axes, data:OrderedDict[str, ViolinPlotData]) -> matplotlib.axes.Axes:
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


class GenViolin(Panels):
    """One generalization violin panel: model performance on one split
    (sampler, mode) for one metric family (prefix), optionally restricted to
    one outclass. Input violin first; for corr_en, an Output violin last."""

    @classmethod
    def plot(cls, df, axes, *args, sampler=None, mode=None, prefix="corr",
             outclass=None, models=None, ylabel=True, ylim=None, fontsize=10, **kwargs):
        assert len(axes) == 1, "GenViolin should only have one axis"
        assert sampler is not None and mode is not None, "sampler and mode must be given"
        assert prefix in METRIC_LABELS, f"prefix must be one of {list(METRIC_LABELS)}"
        ax = axes[0]
        if ax is None:
            print("No axis provided, skipping plotting")
            return

        models = MODEL_LABELS if models is None else models

        mask = (df["sampler"] == sampler) & (df["mode"] == mode)
        if outclass is not None:
            mask &= (df["outclass"] == outclass)

        # Only the models actually fitted for THIS panel: a run may cover fewer
        # models than MODEL_LABELS knows about (the matched runs do), and an
        # empty violin is a ValueError inside matplotlib rather than a blank.
        fitted = set(df[mask]["model"].unique())
        model_names = [m for m in models if m in fitted]
        assert model_names, (f"No models for sampler={sampler}, mode={mode}, outclass={outclass}; "
                             f"the dataframe has {sorted(df['model'].unique())}.")

        # Cin and Cstar do not depend on the model, so read them off whichever
        # model is present rather than assuming Diag was fitted.
        ref = mask & (df["model"] == model_names[0])

        if prefix in ["cov", "corr"]:
            in_out = df[ref][f"{prefix}_in_out"].values
            est    = [df[mask & (df["model"] == m)][f"{prefix}_est_out"].values for m in model_names]
            data = ([ViolinPlotData(in_out, "LightGray", "Input")] +
                    [ViolinPlotData(e, pfm.model_color(m), models[m]) for e, m in zip(est, model_names)])
        else:   # corr_en: separate in / out / est columns
            en_in  = df[ref][f"{prefix}_in"].values
            en_out = df[ref][f"{prefix}_out"].values
            est    = [df[mask & (df["model"] == m)][f"{prefix}_est"].values for m in model_names]
            data = ([ViolinPlotData(en_in, "LightGray", "Input")] +
                    [ViolinPlotData(e, pfm.model_color(m), models[m]) for e, m in zip(est, model_names)] +
                    [ViolinPlotData(en_out, "Gray", "Output")])

        violin_plots(ax, data)
        # Font sizes are in points, so they do not scale with the figure: text
        # that reads well on a 24-inch figure is unreadable on a 6-inch one.
        # Size them relative to the caller's base size instead.
        if ylabel:
            ax.set_ylabel(METRIC_LABELS[prefix], fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize * 0.9)
        if ylim is not None:
            ax.set_ylim(*ylim)
        spines_off(ax)
        return ax


class Supp(Figure):
    """Supplementary generalization figure for one metric family: violins per
    split type on top, the per-outclass breakdown below."""

    # Every split this figure knows how to draw, in the order it draws them.
    # Only those actually present in the dataframe get a panel.
    ALL_SPLITS = [("trials", "random"), ("odours", "random"),
                  ("odours", "inclass"), ("odours", "outclass")]

    # Per-prefix y limits (from the demo notebook); None = autoscale.
    # Columns each metric family draws, used to set the scale from the data.
    YCOLS = {"cov":     ("cov_in_out", "cov_est_out"),
             "corr":    ("corr_in_out", "corr_est_out"),
             "corr_en": ("corr_en_in", "corr_en_out", "corr_en_est")}

    # How much room to leave above the largest value, so the top of a violin is
    # not clipped by its own kernel tail.
    YPAD = 1.08

    # Figure geometry, in inches. The width follows the number of violins the
    # figure actually draws rather than being passed in per metric: corr_en has
    # an extra Output violin, and a run with fewer models has fewer.
    W_PER_VIOLIN = 0.62
    W_PANEL_PAD  = 0.9     # axis labels, ticks and the gap between panels
    W_YLABEL     = 0.7     # the leftmost panel carries the y label
    H_PER_ROW    = 3.6
    FONTSIZE     = 10

    @classmethod
    def data_ylim(cls, df, prefix):
        """(0, max) over everything this figure will draw.

        From the data rather than a constant, so it follows a refit instead of
        silently going stale -- and one limit for the whole figure, since the
        split and outclass rows are meant to be read against each other.
        """
        vals = np.concatenate([df[c].values for c in cls.YCOLS[prefix] if c in df])
        top = np.nanmax(vals)
        assert np.isfinite(top), f"No finite {prefix} values to set the y scale from."
        return (0, float(top) * cls.YPAD)

    @classmethod
    def plot(cls, plot_data, prefix="corr", fig=None, figsize=None, ylim=None,
             fontsize=None, **kwargs):
        print(f"PLOTTING FIGURE Generalization ({prefix=})")
        df = plot_data.df

        # Which panels to draw is a property of the dataframe, not of this
        # function: a run that only fitted trials/random (the matched runs, say)
        # has no odours splits and no outclasses, and asking for them would
        # either raise or draw empty axes.
        present = set(map(tuple, df[["sampler", "mode"]].drop_duplicates().values))
        splits = [sm for sm in cls.ALL_SPLITS if sm in present]
        assert splits, f"No known splits in the dataframe; found {sorted(present)}."
        outclasses = (sorted(df[df["outclass"].notnull()]["outclass"].unique())
                      if "outclass" in df else [])

        n_rows = 2 if outclasses else 1

        if figsize is None:
            # One violin per model that was actually fitted, plus Input, plus
            # Output for corr_en.
            n_models  = len(set(df["model"].unique()) & set(MODEL_LABELS))
            n_violins = n_models + 1 + (1 if prefix == "corr_en" else 0)
            n_cols    = max(len(splits), len(outclasses) or 1)
            figsize   = (cls.W_YLABEL + n_cols * (n_violins * cls.W_PER_VIOLIN + cls.W_PANEL_PAD),
                         n_rows * cls.H_PER_ROW)
        fontsize = cls.FONTSIZE if fontsize is None else fontsize
        gs = GridSpec(n_rows, 12)
        # This figure is typically plotted several times (once per metric), so
        # unlike the single-shot Mains we don't draw on the current figure:
        # make a fresh one unless the caller supplies theirs.
        fig = plt.figure(figsize=figsize) if fig is None else fig
        axes = {}

        panel_ylim = cls.data_ylim(df, prefix) if ylim is None else ylim

        w_top = 12 // len(splits)
        for i, (sampler, mode) in enumerate(splits):
            ax = fig.add_subplot(gs[0, w_top*i:w_top*(i+1)])
            GenViolin.plot(df, [ax], sampler=sampler, mode=mode, prefix=prefix,
                           ylabel=(i == 0), ylim=panel_ylim, fontsize=fontsize)
            ax.set_title(f"{sampler} {mode}", fontsize=fontsize)
            axes[f"{sampler}_{mode}"] = ax

        w = 12 // len(outclasses) if outclasses else 0
        for i, outclass in enumerate(outclasses):
            ax = fig.add_subplot(gs[1, w*i:w*(i+1)])
            GenViolin.plot(df, [ax], sampler="odours", mode="outclass", prefix=prefix,
                           outclass=outclass, ylabel=(i == 0), ylim=panel_ylim, fontsize=fontsize)
            ax.set_title(f"Outclass: {outclass}", fontsize=fontsize)
            axes[f"outclass_{outclass}"] = ax

        return axes
