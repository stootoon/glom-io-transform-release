"""Supplementary figures for the matched rois: observed vs predicted.

One figure per metric. Each has two blocks -- the response-loss fits on the
left, the covariance-loss fits on the right -- and each block shows the observed
matrix, the two models' predictions, and a scatter of predicted against
observed with both models overlaid.

    from glom_io_transform.analysis.figures import matched_rois
    matched_rois.Supp.plot(data, metric="cov")
"""
from .figures import np, plt, GridSpec, spines_off
from .figures import Figure, rep_style
from glom_io_transform.model_fitting import proc_fit_models as pfm
from ..compute.matched_rois import LOSSES, MODELS, METRICS

TITLES = {"resp": ("Responses", "odour", "roi"),
          "cov":  ("Covariance", "odour", "odour"),
          "corr": ("Correlation", "odour", "odour")}

LOSS_LABELS = {"resp": "fitted on responses", "cov": "fitted on covariances"}


class Supp(Figure):
    """Observed and predicted matrices for one metric; a row per loss mode."""

    # Correlations reuse the style the representation matrices are drawn with, so
    # the two are read on the same scale. The others take their limits from the
    # OBSERVED data, and every panel in the figure then shares them -- comparing
    # panels is the whole point, so a per-panel autoscale would defeat it.
    STYLE = {"resp": {"cmap": "RdYlBu",    "vlim": None},
             "cov":  {"cmap": "rainbow",   "vlim": None},
             "corr": {"cmap": rep_style["cmap"], "vlim": rep_style["vlim"]}}
    PCTILE = (1, 99)

    W_MAP     = {"resp": 0.85, "cov": 1.9, "corr": 1.9}
    W_SCATTER = 2.4
    H_ROW     = {"resp": 3.0, "cov": 2.4, "corr": 2.4}
    FONTSIZE  = 9

    @classmethod
    def limits(cls, observed, metric):
        vlim = cls.STYLE[metric]["vlim"]
        if vlim is not None:
            return vlim
        return tuple(np.nanpercentile(np.asarray(observed).ravel(), cls.PCTILE))

    @classmethod
    def plot(cls, plot_data, metric="cov", fig=None, figsize=None, fontsize=None,
             losses=LOSSES, models=MODELS, **kwargs):
        assert metric in METRICS, f"metric must be one of {METRICS}, got {metric!r}."
        print(f"PLOTTING FIGURE matched_rois ({metric=})")
        fontsize = cls.FONTSIZE if fontsize is None else fontsize
        panels = {loss: plot_data.panels[(loss, metric)] for loss in losses}
        cmap = cls.STYLE[metric]["cmap"]
        # The observed matrix is the same data for both losses, so one scale.
        vmin, vmax = cls.limits(panels[losses[0]]["obs"], metric)

        w_map = cls.W_MAP[metric]
        widths = [w_map] * (1 + len(models)) + [cls.W_SCATTER]
        if figsize is None:
            figsize = (sum(widths) + 2.2, cls.H_ROW[metric] * len(losses) + 0.9)
        fig = plt.figure(figsize=figsize) if fig is None else fig
        gs = GridSpec(len(losses), len(widths), width_ratios=widths, figure=fig,
                      top=0.88, bottom=0.10, left=0.10, right=0.99,
                      wspace=0.45, hspace=0.55)

        axes = {}
        title, ylab, xlab = TITLES[metric]
        for i, loss in enumerate(losses):
            p = panels[loss]
            last = (i == len(losses) - 1)
            for j, key in enumerate(["obs"] + list(models)):
                ax = fig.add_subplot(gs[i, j])
                im = ax.imshow(np.asarray(p[key]), cmap=cmap, vmin=vmin, vmax=vmax,
                               aspect="auto", interpolation="nearest")
                name = "observed" if key == "obs" else key
                ax.set_title(name, fontsize=fontsize,
                             color="0.2" if key == "obs" else pfm.model_color(key))
                if last:
                    ax.set_xlabel(xlab, fontsize=fontsize * 0.9)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=fontsize * 0.9)
                ax.tick_params(labelsize=fontsize * 0.75)
                axes[f"{loss}_{key}"] = ax

                if key == "obs":
                    # Attached to the observed panel, since its data sets the scale.
                    cax = ax.inset_axes([1.06, 0.0, 0.05, 1.0])
                    cb = fig.colorbar(im, cax=cax)
                    cb.ax.tick_params(labelsize=fontsize * 0.7)
                    cb.outline.set_linewidth(0.5)
                    axes[f"{loss}_cbar"] = cax

            ax = fig.add_subplot(gs[i, len(widths) - 1])
            obs = np.asarray(p["obs"]).ravel()
            for name in models:
                ax.scatter(obs, np.asarray(p[name]).ravel(), s=2, alpha=0.35,
                           color=pfm.model_color(name), label=name, linewidths=0)
            lim = (min(obs.min(), vmin), max(obs.max(), vmax))
            ax.plot(lim, lim, lw=0.8, color="0.4", zorder=0)
            ax.set_xlim(*lim); ax.set_ylim(*lim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("observed", fontsize=fontsize * 0.9)
            ax.set_ylabel("predicted", fontsize=fontsize * 0.9)
            ax.tick_params(labelsize=fontsize * 0.75)
            ax.legend(fontsize=fontsize * 0.8, frameon=False, markerscale=3, loc="upper left")
            spines_off(ax)
            axes[f"{loss}_scatter"] = ax

            # Row label, naming what the row's models were fitted on.
            b = axes[f"{loss}_obs"].get_position()
            fig.text(0.012, (b.y0 + b.y1) / 2, LOSS_LABELS.get(loss, loss),
                     rotation=90, ha="left", va="center", fontsize=fontsize * 1.15)

        fig.suptitle(f"Matched rois: {title.lower()}, observed vs predicted "
                     f"(seed {plot_data.seed}, train {plot_data.train})",
                     fontsize=fontsize * 1.3, y=0.98)
        return axes
