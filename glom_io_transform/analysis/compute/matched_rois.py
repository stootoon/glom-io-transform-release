"""Observed vs predicted responses, covariances and correlations, matched rois.

For one seed and one training split, refits each model at the lambda the
generalization analysis would have selected, and keeps what the supplementary
figures draw: the observed matrix, each model's prediction, and the flattened
pairs behind the scatter.

Refitting rather than reading the stored results is deliberate: only the
covariances are saved by pack_split_results, so the response panels have no
other source. The fit is cheap on 16 rois, and doing all three metrics from one
refit keeps them consistent with each other.
"""
import numpy as np

from glom_io_transform.model_fitting import driver
from .compute import Computation, base_context, seed_config, seed_data

LOSSES  = ("resp", "cov")
MODELS  = ("Diag", "Free")
SPLIT   = ("trials", "random")      # (sampler, mode); n_od_train is separate
METRICS = ("resp", "cov", "corr")


def corr_from(C, ref_vars, eval_vars):
    """The correlation matrix as results.Extraction.vld_corrs defines it.

    Each matrix is normalised by its OWN variances, so the correlation view is
    invariant to a per-odour rescaling of the estimate -- which is why it
    isolates structure rather than scale.
    """
    return C / np.sqrt(np.outer(ref_vars, eval_vars))


def refit(loss, model_name, seed=0, train=0, sampler=SPLIT, matched=True,
          n_od_train="max"):
    """Refit one model at the lambda the generalization analysis selects.

    n_od_train is the odour spec the fits were generated with ("max",
    "18_rand_0", "18_var_output", ...). It picks the fit directory, and travels
    in the config so the regenerated data is subset the same way.
    """
    base  = base_context(loss=loss, matched=matched)
    split = base.split(*sampler, n_od_train)
    model = split.model(model_name)

    # Same lambda rule as generalization_df: the smallest for the Diag family,
    # the best-by-metric for the others.
    ext = model.extract(seed=seed, train=train, metric="ratio",
                        la="min" if model_name.startswith("Diag") else None)
    config = seed_config(model, seed, ext.la, expect_model=model_name)
    XX, YY = seed_data(config)
    results, mdl = driver.run(config, X=XX, Y=YY, return_model=True)

    p_final = results["p_final"]
    Z = mdl.get("Z", p_final)
    vld = results["split"].vld[train]
    return {"la": ext.la, "Z": Z, "XX": XX, "YY": YY, "vld": vld,
            "n_rois": YY.vld.shape[0], "n_odours": YY.vld.shape[1],
            "n_od_train": n_od_train, "center": mdl.center}


def panels_for(fits, metric):
    """{'obs': M, 'Diag': M, 'Free': M} for one loss and one metric."""
    if metric == "resp":
        # rois x odours, the natural orientation of the data: the response
        # figure puts odours on the x axis and stacks the rois.
        any_fit = next(iter(fits.values()))
        out = {"obs": np.asarray(any_fit["YY"].vld)}
        for name, f in fits.items():
            out[name] = f["Z"] @ np.asarray(f["XX"].vld)
        return out

    any_fit = next(iter(fits.values()))
    v = any_fit["vld"]
    if metric == "cov":
        out = {"obs": v.Cstar}
        for name, f in fits.items():
            out[name] = f["vld"].Cest
    else:
        out = {"obs": corr_from(v.Cstar, v.ref_vars["Cstar"], v.eval_vars["Cstar"])}
        for name, f in fits.items():
            w = f["vld"]
            out[name] = corr_from(w.Cest, w.ref_vars["Cest"], w.eval_vars["Cest"])
    return out


class Data(Computation):
    """Refits for the matched-roi supplementary figures."""

    def compute(self, seed=0, train=0, losses=LOSSES, models=MODELS, sampler=SPLIT,
                matched=True, n_od_train="max"):
        print("COMPUTING matched_rois.Data.")
        self.seed, self.train, self.n_od_train = seed, train, n_od_train
        self.losses, self.models = tuple(losses), tuple(models)
        self.fits = {}
        for loss in self.losses:
            for name in self.models:
                print(f"  refitting {name} at loss={loss} ...")
                self.fits[(loss, name)] = refit(loss, name, seed=seed, train=train,
                                                sampler=sampler, matched=matched,
                                                n_od_train=n_od_train)
                f = self.fits[(loss, name)]
                print(f"    lambda = {f['la']:.3g}, {f['n_rois']} rois x {f['n_odours']} odours")
        self.panels = {(loss, metric): panels_for({n: self.fits[(loss, n)] for n in self.models},
                                                  metric)
                       for loss in self.losses for metric in METRICS}
        self.computed = True
        return self

    def lambdas(self):
        return {k: f["la"] for k, f in self.fits.items()}
