import os
from .figures import np, plt, GridSpec, spines_off
from .figures import Figure, Schem, Reps
from .figures import paths

from .violin_plots import GenViolin, report_comparisons

import glom_io_transform.model_fitting.proc_fit_models as pfm


class Main(Figure):
    @classmethod
    def plot(cls, plot_data, **kwargs):
        print("PLOTTING FIGURE ShowModels")
        art_path = os.path.join(paths.proj_path, "art")

        gs = GridSpec(2, 4)
        fig = plt.gcf()

        ax_diag_schem  = fig.add_subplot(gs[0,1])
        Schem.plot(plot_data, [ax_diag_schem], art_file=os.path.join(art_path, "diag_schem.png"))

        ax_free_schem = fig.add_subplot(gs[0,2])
        Schem.plot(plot_data, [ax_free_schem], art_file=os.path.join(art_path, "free_schem.png"))

        corr_diag = plot_data.models["Diag"].vld_corrs["Cest"]
        corr_free = plot_data.models["Free"].vld_corrs["Cest"]
        corr_star = plot_data.models["Diag"].vld_corrs["Cstar"]
        corr_in   = plot_data.models["Diag"].vld_corrs["Cin"]
        assert np.allclose(plot_data.models["Diag"].vld_corrs["Cstar"],
                           plot_data.models["Free"].vld_corrs["Cstar"]), "Cstar should be the same for Diag and Free models"

        # Representation matrices and observed-vs-predicted scatter, all in
        # house style via Reps (shared odour ordering and color scheme).
        order = Reps.odour_order("input") #n=corr_star.shape[0])   # natural order

        ax_diag_rep = fig.add_subplot(gs[1,1])
        Reps.matrix(corr_diag, ax_diag_rep, order)

        ax_free_rep = fig.add_subplot(gs[1,2])
        Reps.matrix(corr_free, ax_free_rep, order)

        ax_in_rep = fig.add_subplot(gs[0,0])
        Reps.matrix(corr_in, ax_in_rep, order, cbar=True)

        ax_star_rep = fig.add_subplot(gs[1,0])
        Reps.matrix(corr_star, ax_star_rep, order)

        # ax_scat = fig.add_subplot(gs[1,2])
        # Reps.scatter(corr_star, {"Diag": corr_diag, "Free": corr_free}, ax_scat,
        #              colors={"Diag": pfm.model_color("diag"), "Free": pfm.model_color("free")},
        #              subsample=0.1)

        # Generalization violins for the two headline splits (correlation metric)
        ax_gen_trials  = fig.add_subplot(gs[0,3])
        ax_gen_outclass = fig.add_subplot(gs[1,3])
        # Every pairwise test for both panels, printed rather than drawn: the
        # brackets would crowd a panel this size, and reporting all pairs is
        # what a reader asking "but what about X vs Y?" actually wants.
        # stats=False silences them; the panels are unchanged either way.
        stats = kwargs.get("stats", True)
        ylim = {"trials":(-0.01, 0.51), "odours":(-0.01, 0.51)} 
        for axi, (sampler, mode) in zip([ax_gen_trials, ax_gen_outclass],
                                        [("trials", "random"), ("odours", "outclass")]):
            GenViolin.plot(plot_data.df, [axi], sampler=sampler, mode=mode, prefix="corr",
                           ylim=ylim[sampler],
                           )
            y0, y1 = ylim[sampler]
            axi.set_yticks([t for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5] if y0 <= t <= y1])
            axi.set_xlabel("Model", fontsize=12)
            axi.set_ylabel("Correlation Mismatch", fontsize=12)
            if stats:
                report_comparisons(plot_data.df, "corr", sampler, mode)

        return {"circ_diag": ax_diag_schem,
                "circ_free": ax_free_schem,
                "rep_diag": ax_diag_rep,
                "rep_free": ax_free_rep,
                "rep_star": ax_star_rep,
                "rep_in":   ax_in_rep,
                "gen_trials": ax_gen_trials,
                "gen_outclass": ax_gen_outclass
                }
