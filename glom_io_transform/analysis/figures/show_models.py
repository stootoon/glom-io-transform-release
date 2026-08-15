import os
import matplotlib
from .figures import np, plt, GridSpec, spines_off
from .figures import Figure, Schem, Reps
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

        # Representation matrices and observed-vs-predicted scatter, all in
        # house style via Reps (shared odour ordering and color scheme).
        order = Reps.odour_order(n=corr_star.shape[0])   # natural order

        ax_diag_rep = fig.add_subplot(gs[0,1])
        Reps.matrix(corr_diag, ax_diag_rep, order, cbar=True)

        ax_free_rep = fig.add_subplot(gs[1,1])
        Reps.matrix(corr_free, ax_free_rep, order)

        ax_star_rep = fig.add_subplot(gs[0,2])
        Reps.matrix(corr_star, ax_star_rep, order)

        ax_scat = fig.add_subplot(gs[1,2])
        Reps.scatter(corr_star, {"Diag": corr_diag, "Free": corr_free}, ax_scat,
                     colors={"Diag": pfm.model_color("diag"), "Free": pfm.model_color("free")},
                     subsample=0.1)


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
