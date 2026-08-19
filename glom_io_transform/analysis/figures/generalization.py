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

METRIC_LABELS = {"cov": "Covariance", "corr": "Correlation", "corr_en": "Correlation Energy"}
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
             outclass=None, models=None, ylabel=True, ylim=None, **kwargs):
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
        if ylabel:
            ax.set_ylabel(f"{METRIC_LABELS[prefix]} Mismatch")
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
    ylim_splits   = {"cov": (0, 0.6), "corr": (0, 0.5), "corr_en": (0, 0.5)}
    ylim_outclass = {"cov": (0, 0.6), "corr": (0, 0.5), "corr_en": (0, 0.5)}

    @classmethod
    def plot(cls, plot_data, prefix="corr", fig=None, figsize=(16, 8), ylim_splits=None, ylim_outclass=None, **kwargs):
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
        if fig is None and n_rows == 1:
            figsize = (figsize[0], figsize[1] / 2)
        gs = GridSpec(n_rows, 12)
        # This figure is typically plotted several times (once per metric), so
        # unlike the single-shot Mains we don't draw on the current figure:
        # make a fresh one unless the caller supplies theirs.
        fig = plt.figure(figsize=figsize) if fig is None else fig
        axes = {}

        w_top = 12 // len(splits)
        for i, (sampler, mode) in enumerate(splits):
            ax = fig.add_subplot(gs[0, w_top*i:w_top*(i+1)])
            GenViolin.plot(df, [ax], sampler=sampler, mode=mode, prefix=prefix,
                           ylabel=(i == 0), ylim=ylim_splits if ylim_splits is not None else cls.ylim_splits[prefix])
            ax.set_title(f"{sampler} {mode}")
            axes[f"{sampler}_{mode}"] = ax

        w = 12 // len(outclasses) if outclasses else 0
        for i, outclass in enumerate(outclasses):
            ax = fig.add_subplot(gs[1, w*i:w*(i+1)])
            GenViolin.plot(df, [ax], sampler="odours", mode="outclass", prefix=prefix,
                           outclass=outclass, ylabel=(i == 0), ylim=ylim_outclass if ylim_outclass is not None else cls.ylim_outclass[prefix])
            ax.set_title(f"Outclass: {outclass}")
            axes[f"outclass_{outclass}"] = ax

        return axes
