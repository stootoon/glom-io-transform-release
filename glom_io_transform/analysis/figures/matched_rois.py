"""Supplementary figures for the matched rois: observed vs predicted.

One figure per metric. Each has two blocks -- the response-loss fits on the
left, the covariance-loss fits on the right -- and each block shows the observed
matrix, the two models' predictions, and a scatter of predicted against
observed with both models overlaid.

    from glom_io_transform.analysis.figures import matched_rois
    matched_rois.Supp.plot(data, metric="cov")
"""
from .figures import np, plt, GridSpec, spines_off
from .figures import Figure, rep_style, get_leaf_order_from_covariance
from matplotlib.colors import TwoSlopeNorm
from glom_io_transform.model_fitting import proc_fit_models as pfm
from ..compute.matched_rois import LOSSES, MODELS, METRICS

TITLES = {"resp": ("Responses", "roi", "odour"),
          "cov":  ("Covariance", "odour", "odour"),
          "corr": ("Correlation", "odour", "odour")}

LOSS_LABELS = {"resp": "fitted on responses", "cov": "fitted on covariances"}


class Supp(Figure):
    """Observed and predicted matrices for one metric; a row per loss mode."""

    # Correlations reuse the style the representation matrices are drawn with, so
    # the two are read on the same scale. The others take their limits from the
    # OBSERVED data, and every panel in the figure then shares them -- comparing
    # panels is the whole point, so a per-panel autoscale would defeat it.
    STYLE = {"resp": {"cmap": "RdYlBu_r",  "vlim": None, "center": 0.0},
             "cov":  {"cmap": "rainbow",   "vlim": None, "center": None},
             "corr": {"cmap": rep_style["cmap"], "vlim": rep_style["vlim"], "center": None}}
    PCTILE = (1, 99)

    # Scatter: a random tenth, drawn larger. All the points make a solid blob at
    # this density; the trend is what the panel is for.
    SCATTER_FRAC = 0.10
    SCATTER_SIZE = 11
    SCATTER_SEED = 0

    # Metrics whose axes are reordered by clustering the observed matrix.
    CLUSTERED = ("cov", "corr")

    W_MAP     = {"resp": 0.85, "cov": 1.9, "corr": 1.9}
    W_SCATTER = 2.4
    W_GAP     = 0.5
    H_ROW     = {"resp": 3.0, "cov": 2.4, "corr": 2.4}
    FONTSIZE  = 9

    @classmethod
    def limits(cls, observed, metric):
        vlim = cls.STYLE[metric]["vlim"]
        if vlim is not None:
            return vlim
        return tuple(np.nanpercentile(np.asarray(observed).ravel(), cls.PCTILE))

    # Response panels: how many rois to draw as traces, chosen by observed variance.
    N_TRACES = 3
    W_HEAT   = 3.2
    W_TRACE  = 2.4

    @classmethod
    def plot(cls, plot_data, metric="cov", **kwargs):
        assert metric in METRICS, f"metric must be one of {METRICS}, got {metric!r}."
        print(f"PLOTTING FIGURE matched_rois ({metric=})")
        if metric == "resp":
            return cls.plot_responses(plot_data, **kwargs)
        return cls.plot_matrices(plot_data, metric=metric, **kwargs)

    @classmethod
    def plot_matrices(cls, plot_data, metric="cov", fig=None, figsize=None, fontsize=None,
                      losses=LOSSES, models=MODELS, **kwargs):
        fontsize = cls.FONTSIZE if fontsize is None else fontsize
        panels = {loss: plot_data.panels[(loss, metric)] for loss in losses}
        cmap = cls.STYLE[metric]["cmap"]
        # The observed matrix is the same data for both losses, so one scale.
        vmin, vmax = cls.limits(panels[losses[0]]["obs"], metric)

        # Responses straddle zero with different ranges either side, so a plain
        # linear scale would put the colour-map's midpoint somewhere other than
        # zero. TwoSlopeNorm gives each sign its own half of the map.
        centre = cls.STYLE[metric]["center"]
        norm = None
        if centre is not None and vmin < centre < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=centre, vmax=vmax)
        im_kwargs = dict(cmap=cmap, norm=norm) if norm is not None else \
                    dict(cmap=cmap, vmin=vmin, vmax=vmax)

        # Order both axes by clustering the OBSERVED covariance, and use that one
        # order for the covariance and correlation figures alike so they can be
        # read against each other. The cross-covariance (train vs vld) is only
        # nearly symmetric, and the ordering needs a symmetric similarity.
        order = None
        if metric in cls.CLUSTERED:
            ref = np.asarray(plot_data.panels[(losses[0], "cov")]["obs"])
            order = get_leaf_order_from_covariance((ref + ref.T) / 2)

        def arrange(M):
            M = np.asarray(M)
            return M if order is None else M[order][:, order]

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
                im = ax.imshow(arrange(p[key]), aspect="auto",
                               interpolation="nearest", **im_kwargs)
                name = "observed" if key == "obs" else key
                ax.set_title(name, fontsize=fontsize,
                             color="0.2" if key == "obs" else pfm.model_color(key))
                if last:
                    ax.set_xlabel(xlab, fontsize=fontsize * 0.9)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=fontsize * 0.9)
                else:
                    # Every panel shares the odour axis, so repeating the tick
                    # labels only crowds them against the colour bar.
                    ax.set_yticklabels([])
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
            # The same random subset for every model, so the panels compare.
            rng = np.random.default_rng(cls.SCATTER_SEED)
            k = max(1, int(round(cls.SCATTER_FRAC * obs.size)))
            sub = rng.choice(obs.size, size=k, replace=False)
            for name in models:
                ax.scatter(obs[sub], np.asarray(p[name]).ravel()[sub],
                           s=cls.SCATTER_SIZE, alpha=0.45,
                           color=pfm.model_color(name), label=name, linewidths=0)
            obs = obs[sub]
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

    @classmethod
    def plot_responses(cls, plot_data, fig=None, figsize=None, fontsize=None,
                       losses=LOSSES, models=MODELS, n_traces=None,
                       show_scatter=True, **kwargs):
        """Responses: stacked heat maps, plus traces for the most variable rois.

        The heat maps put odours on x and stack observed over the two
        predictions, so the same odour lines up vertically across all three. The
        second column shows the rois with the largest observed variance across
        odours -- the ones with something to predict -- as overlaid traces.
        """
        fontsize = cls.FONTSIZE if fontsize is None else fontsize
        n_traces = cls.N_TRACES if n_traces is None else n_traces
        panels = {loss: plot_data.panels[(loss, "resp")] for loss in losses}
        rows   = ["obs"] + list(models)

        obs = np.asarray(panels[losses[0]]["obs"])          # rois x odours
        vmin, vmax = cls.limits(obs, "resp")
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0 < vmax else None
        im_kwargs = dict(cmap=cls.STYLE["resp"]["cmap"],
                         **(dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)))

        # Same rois for every block: the observed data is the same, and the
        # point is to compare what each fit does with them.
        top = np.argsort(-obs.var(axis=1))[:n_traces]

        block = [cls.W_HEAT, cls.W_TRACE] + ([cls.W_TRACE] if show_scatter else [])
        widths = []
        for i, _ in enumerate(losses):
            if i:
                widths.append(cls.W_GAP if hasattr(cls, "W_GAP") else 0.5)
            widths += block
        if figsize is None:
            figsize = (sum(widths) + 1.4, 1.35 * len(rows) + 1.2)
        fig = plt.figure(figsize=figsize) if fig is None else fig
        gs = GridSpec(len(rows), len(widths), width_ratios=widths, figure=fig,
                      top=0.86, bottom=0.16, left=0.07, right=0.99,
                      wspace=0.40, hspace=0.45)

        axes, col = {}, 0
        for i, loss in enumerate(losses):
            if i:
                col += 1
            p = panels[loss]

            for r, key in enumerate(rows):
                ax = fig.add_subplot(gs[r, col])
                im = ax.imshow(np.asarray(p[key]), aspect="auto",
                               interpolation="nearest", **im_kwargs)
                name = "observed" if key == "obs" else key
                ax.set_ylabel(name, fontsize=fontsize,
                              color="0.2" if key == "obs" else pfm.model_color(key))
                ax.tick_params(labelsize=fontsize * 0.75)
                if r < len(rows) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("odour", fontsize=fontsize * 0.9)
                axes[f"{loss}_{key}"] = ax
                if r == 0:
                    ax.set_title(LOSS_LABELS.get(loss, loss), fontsize=fontsize * 1.15, pad=6)

            # Horizontal colour bar under the stack, spanning the heat maps.
            cax = ax.inset_axes([0.0, -0.75, 1.0, 0.09])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=fontsize * 0.7)
            cb.outline.set_linewidth(0.5)
            axes[f"{loss}_cbar"] = cax
            col += 1

            for r, roi in enumerate(top):
                ax = fig.add_subplot(gs[r, col])
                ax.plot(np.asarray(p["obs"])[roi], lw=1.0, color="0.2", label="observed")
                for name in models:
                    ax.plot(np.asarray(p[name])[roi], lw=1.0, color=pfm.model_color(name),
                            label=name)
                ax.set_ylabel(f"roi {roi}", fontsize=fontsize * 0.9)
                ax.tick_params(labelsize=fontsize * 0.75)
                if r < n_traces - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("odour", fontsize=fontsize * 0.9)
                if r == 0:
                    ax.legend(fontsize=fontsize * 0.7, frameon=False, ncol=3,
                              loc="lower left", bbox_to_anchor=(0, 1.0))
                spines_off(ax)
                axes[f"{loss}_trace{roi}"] = ax
            col += 1

            if show_scatter:
                ax = fig.add_subplot(gs[:, col]); col += 1
                flat = np.asarray(p["obs"]).ravel()
                rng = np.random.default_rng(cls.SCATTER_SEED)
                k = max(1, int(round(cls.SCATTER_FRAC * flat.size)))
                sub = rng.choice(flat.size, size=k, replace=False)
                for name in models:
                    ax.scatter(flat[sub], np.asarray(p[name]).ravel()[sub],
                               s=cls.SCATTER_SIZE, alpha=0.45,
                               color=pfm.model_color(name), label=name, linewidths=0)
                lim = (min(flat.min(), vmin), max(flat.max(), vmax))
                ax.plot(lim, lim, lw=0.8, color="0.4", zorder=0)
                ax.set_xlim(*lim); ax.set_ylim(*lim)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("observed", fontsize=fontsize * 0.9)
                ax.set_ylabel("predicted", fontsize=fontsize * 0.9)
                ax.tick_params(labelsize=fontsize * 0.75)
                ax.legend(fontsize=fontsize * 0.8, frameon=False, markerscale=3, loc="upper left")
                spines_off(ax)
                axes[f"{loss}_scatter"] = ax

        fig.suptitle(f"Matched rois: responses, observed vs predicted "
                     f"(seed {plot_data.seed}, train {plot_data.train})",
                     fontsize=fontsize * 1.3, y=0.98)
        return axes
