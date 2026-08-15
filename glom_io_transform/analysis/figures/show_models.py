import os
import matplotlib
from .figures import np, plt, GridSpec, spines_off
from .figures import Figure, Schem
from .figures import paths

import glom_io_transform.model_fitting.proc_fit_models as pfm

from collections import OrderedDict
from dataclasses import dataclass

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


class Main(Figure):
    @classmethod
    def plot(cls, plot_data, **kwargs):
        print("PLOTTING FIGURE ShowModels")
        art_path = os.path.join(paths.proj_path, "art")

        gs = GridSpec(2, 4)
        fig = plt.gcf()

        ax_diag_schem  = fig.add_subplot(gs[0,0])
        Schem.plot(plot_data, [ax_diag_schem], art_file=os.path.join(art_path, "diag_schem.png"))

        ax_free_schem = fig.add_subplot(gs[1,0])
        Schem.plot(plot_data, [ax_free_schem], art_file=os.path.join(art_path, "free_schem.png"))

        corr_diag = plot_data.models["Diag"].vld_corrs["Cest"]
        corr_free = plot_data.models["Free"].vld_corrs["Cest"]
        corr_star = plot_data.models["Diag"].vld_corrs["Cstar"]
        assert np.allclose(plot_data.models["Diag"].vld_corrs["Cstar"],
                           plot_data.models["Free"].vld_corrs["Cstar"]), "Cstar should be the same for Diag and Free models"

        ax_diag_rep = fig.add_subplot(gs[0,1])
        im=  ax_diag_rep.matshow(corr_diag, vmin=0, vmax=1, cmap="Spectral_r")

        ax = ax_diag_rep
        cbar_ax = ax.inset_axes([1.025, 0, 0.05, 0.9])  # [x0, y0, width, height]
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=10)
            # Write the text "rho" above the colorbar
        ax.text(1.05, 0.925, "ρ", transform=ax.transAxes, fontsize=14, va='bottom', ha='center')

        ax_free_rep = fig.add_subplot(gs[1,1])
        ax_free_rep.matshow(corr_free, vmin=0, vmax=1, cmap="Spectral_r")

        ax_star_rep = fig.add_subplot(gs[0,2])
        ax_star_rep.matshow(corr_star, vmin=0, vmax=1, cmap="Spectral_r")

        rep_axes = [ax_diag_rep, ax_free_rep, ax_star_rep]
        [[ax.set_xticks([]), ax.set_yticks([])] for ax in rep_axes]
        [ax.set_xlabel("Odour", fontsize=12) and ax.set_ylabel("Odour", fontsize=12) for ax in rep_axes]

        ax_scat = fig.add_subplot(gs[1,2])
        x = corr_star.flatten()
        y_diag = corr_diag.flatten()
        y_free = corr_free.flatten()
        rho_diag = np.corrcoef(x, y_diag)[0,1]
        rho_free = np.corrcoef(x, y_free)[0,1]
        # Show a random subset of the data to avoid overplotting
        mask_diag = np.random.choice([True, False], size=x.shape, p=[0.1, 0.9])
        mask_free = np.random.choice([True, False], size=x.shape, p=[0.1, 0.9])
        ax_scat.scatter(x[mask_diag], y_diag[mask_diag], s=10, alpha=0.5, edgecolor=None, color=pfm.model_color("diag"), label=f"Diag $\\rho$={rho_diag:.2f}")
        ax_scat.scatter(x[mask_free], y_free[mask_free], s=10, alpha=0.5, edgecolor=None, color=pfm.model_color("free"), label=f"Free $\\rho$={rho_free:.2f}")
        ax_scat.plot([-0.1, 1], [-0.1,1], ":", color="gray", lw=1)
        tt = np.arange(0,1.1,0.2)
        ax_scat.set_xlim(-0.1,1)
        ax_scat.set_ylim(-0.1,1)
        ax_scat.set_aspect('equal', adjustable='box')
        ax_scat.set_xticks(tt); ax_scat.set_yticks(tt)
        ax_scat.legend(labelspacing=0, fontsize=10, borderpad=0, frameon=False, loc = "upper left")
        ax_scat.set_xlabel("Observed", fontsize=12)
        ax_scat.set_ylabel("Predicted", fontsize=12)

        spines_off(ax_scat)


        ax_gen_trials = fig.add_subplot(gs[0,3])
        ax_gen_outclass= fig.add_subplot(gs[1,3])
        df = plot_data.df
        split_descr = {"trials": ("trials", "random"),
               "odours_random,": ("odours", "random"),
               "odours_inclass": ("odours", "inclass"),
               "odours_outclass": ("odours", "outclass")}
        order = ["trials", "odours_outclass"]
        model_labs = {"Diag": "Diag",
                      "DiagOnlyInh": "DiagInh",
                      "Free": "Free",
                      "FreeLat": "FreeLat"}
        model_names = list(model_labs.keys())
        which_models= list(model_labs.keys())
        cols = ["Gray"] + [pfm.model_color(m) for m in model_names]
        prefix = "corr"
        for i, (axi, split_name) in enumerate(zip([ax_gen_trials, ax_gen_outclass], order)):
            sampler, mode = split_descr[split_name]
            split_mask   = (df["sampler"] == sampler) & (df["mode"] == mode)
            in_out       = df[split_mask & (df['model']=="Diag")][f"{prefix}_in_out"].values
            models_est_out = [df[split_mask & (df['model']==m)][f"{prefix}_est_out"].values for m in model_names]
            axi = violin_plots(axi, [ViolinPlotData(in_out, "LightGray", "Input")] + [ViolinPlotData(models_est_out[i], pfm.model_color(model_names[i]), model_labs[model_names[i]]) for i in range(len(model_names))])
            axi.set_ylim(0.1, 0.51)
            axi.set_yticks([0.1, 0.2,0.3,0.4,0.5])
            axi.set_xlabel("Model", fontsize=12)
            axi.set_ylabel("Correlation Mismatch", fontsize=12)
            spines_off(axi)

        return {"circ_diag": ax_diag_schem,
                "circ_free": ax_free_schem,
                "rep_diag": ax_diag_rep,
                "rep_free": ax_free_rep,
                "rep_star": ax_star_rep,
                "scatter": ax_scat,
                "gen_trials": ax_gen_trials,
                "gen_outclass": ax_gen_outclass
                }
