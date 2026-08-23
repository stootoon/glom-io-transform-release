"""The supplementary generalization figure.

Supp lays out one metric family: violins per split type on top, the per-outclass
breakdown below. The panels themselves live in violin_plots, which is where to
look to draw one somewhere else.
"""
import numpy as np

from .figures import np, plt, GridSpec
from .figures import Figure

from ..compute.generalization import as_labels
from .violin_plots import (YPAD, BRACKET_ROW, group_order,
                           panel_brackets, plot_violins)


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

    # Figure geometry, in inches. The width follows the number of violins the
    # figure actually draws rather than being passed in per metric: corr_en has
    # an extra Output violin, and a run with fewer models has fewer.
    W_PER_VIOLIN = 0.62
    W_PANEL_PAD  = 0.9     # axis labels, ticks and the gap between panels
    W_YLABEL     = 0.7     # the leftmost panel carries the y label
    H_PER_ROW    = 3.6
    FONTSIZE     = 10

    @classmethod
    def data_span(cls, df, prefix):
        """(top, floor) over everything this figure will draw.

        The floor is 0 unless the data goes below it, so a mismatch still starts
        at zero while a metric that can be negative is not clipped.
        """
        vals = np.concatenate([df[c].values for c in cls.YCOLS[prefix] if c in df])
        top = np.nanmax(vals)
        assert np.isfinite(top), f"No finite {prefix} values to set the y scale from."
        return float(top), float(min(0.0, np.nanmin(vals)))

    @classmethod
    def data_ylim(cls, df, prefix):
        """The y limits for the whole figure, brackets aside.

        From the data rather than a constant, so it follows a refit instead of
        silently going stale -- and one limit for the whole figure, since the
        split and outclass rows are meant to be read against each other.
        """
        top, floor = cls.data_span(df, prefix)
        return (floor, top + (YPAD - 1) * (top - floor))

    @classmethod
    def plot(cls, plot_data, prefix="corr", fig=None, figsize=None, ylim=None,
             fontsize=None, comparisons=None, correction=None, verbose_stats=True,
             models=None, **kwargs):
        print(f"PLOTTING FIGURE Generalization ({prefix=})")
        df = plot_data.df
        # One mapping for the whole figure, so every panel and every bracket
        # agrees about which models are drawn and in what order.
        models = as_labels(models, df)

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
            n_violins = len(group_order(models, prefix))
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

        # Comparisons first: they decide how much headroom the axis needs, and
        # the correction family is the set drawn in a panel.
        panels = [(s_, m_, None) for s_, m_ in splits] + \
                 [("odours", "outclass", oc) for oc in outclasses]
        # Every panel shares one y scale, so the brackets are worked out up front:
        # the tallest stack over all panels sets the headroom they all get.
        brackets, n_bracket_rows = {}, 0
        for key in (panels if comparisons else []):
            brackets[key], rows = panel_brackets(
                df, prefix, key[0], key[1], comparisons, outclass=key[2],
                models=models, correction=correction, verbose=verbose_stats)
            n_bracket_rows = max(n_bracket_rows, rows)

        top, floor = cls.data_span(df, prefix)
        span = top - floor
        panel_ylim = ylim if ylim is not None else \
            (floor, top + (YPAD - 1 + n_bracket_rows * BRACKET_ROW) * span)
        bracket_base = top + (YPAD - 1) * span
        bracket_step = BRACKET_ROW * span

        def bracket_args(key):
            if not brackets.get(key):
                return {}
            return dict(brackets=brackets[key],
                        bracket_base=bracket_base, bracket_step=bracket_step)

        w_top = 12 // len(splits)
        for i, (sampler, mode) in enumerate(splits):
            ax = fig.add_subplot(gs[0, w_top*i:w_top*(i+1)])
            plot_violins(ax, df, sampler, mode, prefix=prefix, models=models,
                         ylabel=(i == 0), ylim=panel_ylim, fontsize=fontsize,
                         **bracket_args((sampler, mode, None)))
            ax.set_title(f"{sampler} {mode}", fontsize=fontsize)
            axes[f"{sampler}_{mode}"] = ax

        w = 12 // len(outclasses) if outclasses else 0
        for i, outclass in enumerate(outclasses):
            ax = fig.add_subplot(gs[1, w*i:w*(i+1)])
            plot_violins(ax, df, "odours", "outclass", prefix=prefix, models=models,
                         outclass=outclass, ylabel=(i == 0), ylim=panel_ylim,
                         fontsize=fontsize, **bracket_args(("odours", "outclass", outclass)))
            ax.set_title(f"Outclass: {outclass}", fontsize=fontsize)
            axes[f"outclass_{outclass}"] = ax

        return axes
