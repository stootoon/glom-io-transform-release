import os, sys

from .figures import np, plt, GridSpec, spines_off
from .figures import Panels, Figure, Reps, Schem
from .figures import paths

import proc_fit_models as pfm
            
def make_bar_chart(plot_data, stat='ratio', bar_width=0.25, keep = [], shift_labs = False):
    # Plot a bar chart showing the best_means for the vld set for each model
    # Add error bars showing the best_stds for the vld set for each model
    # Use the model names as the x-ticks
    # Order the models by the best_means for the vld set
    # Give each bar a different color
    param_strs = {}
    models = plot_data.models
    all_models = sorted(models, key = lambda x: models[x]["best_means"]["vld"]["pearson"])
    ii = 0
    which_models = []
    for fld,col in zip(["in-in", "out-out", "in-out"], ["lightgray", "darkgray", "wheat"]):
        which_models.append(fld)
        m = np.mean(plot_data.corrs[stat][fld])
        s = np.std(plot_data.corrs[stat][fld])
        plt.bar(ii, m, yerr=s, width=bar_width, label=which_models[-1], color=col, error_kw={"ecolor":col})
        ii += 1

    for i, name in enumerate(all_models):
        mdl = models[name]
        best_means = mdl["best_means"]["vld"][stat]
        best_stds  = mdl["best_stds"]["vld"][stat]
        print(f"{name:20s} {best_means:.3f} +/- {best_stds:.3f}")
        # if name not in ["Diag", "Free", "FreeSym", "FreeAsym", "DiagPosBg", "DiagPosBgRank1Sym", "IdPosBgSqrtCov"]: continue
        if len(keep)>0 and (name not in keep): continue
        which_models.append(name)
        col = pfm.model_color(name)
        plt.bar(ii, best_means, yerr=best_stds, width=bar_width, label=name, color=col, error_kw={"ecolor":col})
        ii += 1
        if len(mdl["params"]) == 1:
            param_strs[name] = f'{mdl["params"][0]}={mdl["best_params"]:.2g}'
        elif len(mdl["params"]) == 2:
            param_strs[name] = ", ".join([f'{p}={v:.2g}' for p, v in zip(mdl["params"], mdl["best_params"])])

    # Set the x-ticks to be the model names and the parameters that gave the best_means for the vld set
    xt = list(range(ii))
    xt_labs = [("out-" if "-" not in name else "") + f'{name.lower()}' for name in which_models]

    plt.xticks(xt, xt_labs, rotation=0, fontsize=9)
    # Move some of the x-tick labels vertically down
    labs = plt.gca().get_xticklabels()
    # If the label has one of the following names, move it down
    if shift_labs:
        for lab in labs:
            if lab.get_text() in ["Free", "FreeAsym", "DiagPosBg"]:
                lab.set_y(-0.05)


    #plt.grid(True, which='both', axis='y', linestyle=':')
    plt.ylim(0.0,0.7)
    ylab = {"r2":"$R^2$",
            "ratio":"Ratio",
            "pearson":"Representational Correlation",
            "spearman":"$\\rho_\\text{sp}$"}[stat]
    
    plt.ylabel(ylab, fontsize=12)
    # Turn off the top and right spines
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    return ii

class ModelComparison(Panels):
    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print("PLOTTING PANELS ModelComparison")
        assert len(axes) == 1, "ModelComparison should only have one axis"
        ax = axes[0]
        if ax is None:
            print("No axes provided, skipping plotting")
        else:
            plt.sca(ax)
            ii = make_bar_chart(plot_data, stat='pearson', bar_width=0.45, keep=["Diag", "Free"])
            ax.set_ylim(0,1.05)
            ax.set_xlim(-0.5, ii-0.5)
            # Make grids on the yticks
            # plt.grid(True, which='both', axis='y', linestyle=':')

class CorrelationEnergy(Panels):
    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print("PLOTTING PANELS CorrelationEnergy")
        assert len(axes) == 1, "CorrelationEnergy should only have one axis"
        assert "free_corr_energy" in kwargs, "CorrelationEnergy requires free_corr_energy in kwargs"

        diag_corr_energy = plot_data.corr_energy
        free_corr_energy = kwargs["free_corr_energy"]
        
        data={
            "in":   np.vstack([diag_corr_energy[fld][:,0] for fld in ["train", "vld", "test"]]),
            "out":  np.vstack([diag_corr_energy[fld][:,1] for fld in ["train", "vld", "test"]]),
            "diag": np.vstack([diag_corr_energy[fld][:,2] for fld in ["vld", "test"]]),
            "free": np.vstack([free_corr_energy[fld][:,2] for fld in ["vld", "test"]]),
            }

        
        ax = axes[0]

        # Make a barchart with the mean and std of the correlation energy for each of the four categories
        # Use the order "in", "diag", "free", "out"
        keys = ["in", "diag", "free", "out"]
        cols = {"in": "lightgray", "out":"gray", "diag":pfm.model_color("diag"), "free": pfm.model_color("free")}

        n_od = 48
        
        for i, key in enumerate(keys):
            m = np.mean((data[key]/(n_od**2 - n_od)))
            s = np.std((data[key]/(n_od**2 - n_od)))
            ax.bar(i, m, yerr=s, color=cols[key], width=0.5, error_kw={"ecolor":cols[key]})

        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys)
        ax.set_ylabel("Mean Correlation Energy", fontsize=12)
        spines_off(ax)
             
class Linearization(Panels):
    @classmethod
    def plot(cls, plot_data, axes, *args, **kwargs):
        print(f"PLOTTING PANELS Linearization")
        assert len(axes) == 2, "Linearization should have two axes"

        ax0, ax1 = axes
        plt.sca(ax0)
        for i, pv in enumerate(plot_data.props_vals):
            ax0.scatter(pv.z,
                        pv.delta_z + 1, s = 10, alpha=0.5, color=f"C{i}")

        xl = ax0.get_xlim()
        ax0.plot(xl, xl, color="gray", linestyle=":", zorder=-1, lw=1)

        ax0.set_xlabel("True Gain", fontsize=12)
        ax0.set_ylabel("Linearized Gain", fontsize=12)
        ax0.set_xticks(1 + np.array([-1, -0.5, 0, 0.5, 1]))
        ax0.set_yticks(1 + np.array([-1, -0.5, 0, 0.5, 1]))
        ax0.set_aspect('equal', adjustable='box')
        # Turn the top and right spines off
        [ax0.spines[spine].set_visible(False) for spine in ['top', 'right']]

        sc = 1e6
        for i, pv in enumerate(plot_data.props_vals):
            ax1.scatter(pv.delta_z+1, pv.delta_z_est+1, s = 10, alpha=0.5, color=f"C{i}")
        xl = ax1.get_xlim()
        xv = np.linspace(xl[0], xl[1], 100)
        ax1.plot(xv, xv, color="gray", linestyle=":", zorder=-1, lw=1)
        [ax1.spines[spine].set_visible(False) for spine in ['top', 'right']]
        ax1.set_xlabel("Linearized Gain", fontsize=12)
        ax1.set_ylabel("Approximation", fontsize=12)
            
               
class Main(Figure):
    @classmethod
    def plot(cls, plot_data, **kwargs):
        print("PLOTTING FIGURE FitDiag")
        art_path = os.path.join(paths.proj_path, "art")
        
        gs = GridSpec(2, 4)
        fig = plt.gcf()

        ax_diag_schem  = fig.add_subplot(gs[0,0])
        Schem.plot(plot_data, [ax_diag_schem], art_file=os.path.join(art_path, "diag_schem.png"))

        ax_free_schem = fig.add_subplot(gs[1,0])
        Schem.plot(plot_data, [ax_free_schem], art_file=os.path.join(art_path, "free_schem.png"))
        
        # ax_true, ax_fit, ax_fit_vs = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
        # ims = Reps.plot(plot_data, [ax_true, ax_fit, ax_fit_vs], cmap="bwr", vlim=[-0.2, 1], include_diag = False, show_corr = {"fontsize":8}, id_line = {"lw":1, "ls":":", "color":"black", "alpha":0.5})

        # for ax in [ax_true, ax_fit]:
        #     ax.xaxis.set_label_position('top')
        #     # Set the fontsize of the x-axis label to 12
        #     ax.set_xlabel(ax.get_xlabel())

        # for ax,key in zip([ax_true, ax_fit], ["true", "fit"]):
        #     cbar_ax = ax.inset_axes([1.025, 0, 0.05, 0.9])  # [x0, y0, width, height]
        #     cbar = plt.colorbar(ims[key], cax=cbar_ax, orientation='vertical')
        #     cbar.ax.tick_params(labelsize=10)
        #     # Write the text "rho" above the colorbar
        #     ax.text(1.05, 0.925, "ρ", transform=ax.transAxes, fontsize=14, va='bottom', ha='center')

        # ax_mdl_cmp = fig.add_subplot(gs[1, 0])
        # ModelComparison.plot(plot_data, [ax_mdl_cmp])

        # ax_lin = fig.add_subplot(gs[1, 2])
        # ax_lin_b = fig.add_subplot(gs[1, 3])
        # Linearization.plot(plot_data, [ax_lin, ax_lin_b])

        # ax_corr_energy = fig.add_subplot(gs[1, 1])
        # CorrelationEnergy.plot(plot_data, [ax_corr_energy], **kwargs) 
        
        return {"circ": ax_diag_schem,
                }
    
                # "true": ax_true,
                # "fit": ax_fit,
                # "fit_vs": ax_fit_vs,
                # "mdl_cmp": ax_mdl_cmp,
                # "lin": ax_lin,
                # "lin_b": ax_lin_b,
                # "corr_energy": ax_corr_energy
                # }
        
