"""Supplementary figures for the matched rois: observed vs predicted.

One figure per metric. Each has two blocks -- the response-loss fits on the
left, the covariance-loss fits on the right -- and each block shows the observed
matrix, the two models' predictions, and a scatter of predicted against
observed with both models overlaid.

    from glom_io_transform.analysis.figures import matched_rois
    matched_rois.Supp.plot(data, metric="cov")
"""
from .figures import np, plt, GridSpec, spines_off
from .figures import Figure
from glom_io_transform.model_fitting import proc_fit_models as pfm
from ..compute.matched_rois import LOSSES, MODELS, METRICS

TITLES = {"resp": ("Responses", "odour", "roi"),
          "cov":  ("Covariance", "odour", "odour"),
          "corr": ("Correlation", "odour", "odour")}

LOSS_LABELS = {"resp": "fitted on responses", "cov": "fitted on covariances"}


class Supp(Figure):
    """Observed and predicted matrices for one metric, both loss modes."""

    CMAP = {"resp": "RdBu_r", "cov": "RdBu_r", "corr": "RdBu_r"}
    # Widths in INCHES, so a 48x48 matrix and a 48x16 one both come out square-ish
    # without the figure ballooning.
    W_MAP     = {"resp": 0.85, "cov": 1.9, "corr": 1.9}
    W_SCATTER = 2.4
    W_GAP     = 0.5                                      # between the two blocks
    H_ROW     = 3.4
    FONTSIZE  = 9

    @classmethod
    def limits(cls, panels, metric):
        """One symmetric colour scale for every heat map in the figure."""
        if metric == "corr":
            return -1.0, 1.0
        vals = np.concatenate([np.asarray(m).ravel() for p in panels.values() for m in p.values()])
        v = float(np.nanpercentile(np.abs(vals), 99.5))
        return -v, v

    @classmethod
    def plot(cls, plot_data, metric="cov", fig=None, figsize=None, fontsize=None,
             losses=LOSSES, models=MODELS, **kwargs):
        assert metric in METRICS, f"metric must be one of {METRICS}, got {metric!r}."
        print(f"PLOTTING FIGURE matched_rois ({metric=})")
        fontsize = cls.FONTSIZE if fontsize is None else fontsize
        panels = {loss: plot_data.panels[(loss, metric)] for loss in losses}
        vmin, vmax = cls.limits(panels, metric)

        w_map = cls.W_MAP[metric]
        block = [w_map] * (1 + len(models)) + [cls.W_SCATTER]
        widths, n_block = [], len(block)
        for i, _ in enumerate(losses):
            if i:
                widths.append(cls.W_GAP)
            widths += block
        if figsize is None:
            figsize = (sum(widths) + 1.6, cls.H_ROW)     # + margins for the labels
        fig = plt.figure(figsize=figsize) if fig is None else fig
        gs = GridSpec(1, len(widths), width_ratios=widths, figure=fig,
                      top=0.78, bottom=0.17, left=0.05, right=0.99, wspace=0.40)

        axes, col = {}, 0
        title, ylab, xlab = TITLES[metric]
        for i, loss in enumerate(losses):
            if i:
                col += 1                       # the gap column
            p = panels[loss]
            for j, key in enumerate(["obs"] + list(models)):
                ax = fig.add_subplot(gs[0, col]); col += 1
                M = np.asarray(p[key])
                ax.imshow(M, cmap=cls.CMAP[metric], vmin=vmin, vmax=vmax,
                          aspect="auto", interpolation="nearest")
                name = "observed" if key == "obs" else key
                ax.set_title(name, fontsize=fontsize,
                             color="0.2" if key == "obs" else pfm.model_color(key))
                ax.set_xlabel(xlab, fontsize=fontsize * 0.9)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=fontsize * 0.9)
                ax.tick_params(labelsize=fontsize * 0.75)
                axes[f"{loss}_{key}"] = ax

            ax = fig.add_subplot(gs[0, col]); col += 1
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
            ax.legend(fontsize=fontsize * 0.8, frameon=False, markerscale=3,
                      loc="upper left")
            spines_off(ax)
            axes[f"{loss}_scatter"] = ax

            # One heading per block, centred over its panels.
            b0 = axes[f"{loss}_obs"].get_position(); b1 = ax.get_position()
            fig.text((b0.x0 + b1.x1) / 2, 0.87, LOSS_LABELS.get(loss, loss),
                     ha="center", va="bottom", fontsize=fontsize * 1.2)

        fig.suptitle(f"Matched rois: {title.lower()}, observed vs predicted "
                     f"(seed {plot_data.seed}, train {plot_data.train})",
                     fontsize=fontsize * 1.3, y=1.03)
        return axes
