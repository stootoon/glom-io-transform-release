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
from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np

from glom_io_transform.model_fitting import driver
from .generalization import generalization_df
from .compute import Computation, base_context, seed_config, seed_data

LOSSES  = ("resp", "cov")
MODELS  = ("Diag", "Free")
SPLIT   = ("trials", "random")      # (sampler, mode); n_od_train is separate
METRICS = ("resp", "cov", "corr")
HALVES  = ("train", "vld")          # fitted to, and scored on


def corr_from(C, ref_vars, eval_vars):
    """The correlation matrix as results.Extraction.vld_corrs defines it.

    Each matrix is normalised by its OWN variances, so the correlation view is
    invariant to a per-odour rescaling of the estimate -- which is why it
    isolates structure rather than scale.
    """
    return C / np.sqrt(np.outer(ref_vars, eval_vars))


@dataclass
class Fit:
    """One model refitted at one lambda, with the data it was fitted to.

    The two halves -- "train", what the model saw, and "vld", what it was
    scored on -- are reached through data() and results() rather than by
    knowing which attribute holds which, since every caller wants one or the
    other by name.
    """
    name:       str
    loss:       str
    la:         float
    Z:          np.ndarray
    XX:         Any          # SplitSamples of inputs
    YY:         Any          # SplitSamples of outputs
    train_idx:  int
    trn:        Any          # RunResults for the half the model was fitted to
    vld:        Any          # RunResults for the held-out half
    n_od_train: Any
    center:     bool

    @property
    def n_rois(self):
        return self.YY.vld.shape[0]

    @property
    def n_odours(self):
        return self.YY.vld.shape[1]

    def data(self, half):
        """(X, Y) -- the responses -- for one half."""
        assert half in HALVES, f"half must be one of {HALVES}, got {half!r}."
        if half == "train":
            return self.XX.trains[self.train_idx], self.YY.trains[self.train_idx]
        return self.XX.vld, self.YY.vld

    def results(self, half):
        """The RunResults for one half: covariances, and the variances behind them."""
        assert half in HALVES, f"half must be one of {HALVES}, got {half!r}."
        return self.trn if half == "train" else self.vld


def refit(loss, model_name, seed=0, train=0, sampler=SPLIT, matched=True,
          n_od_train="max", la=None):
    """Refit one model at a given lambda.

    n_od_train is the odour spec the fits were generated with ("max",
    "18_rand_0", "18_var_output", ...). It picks the fit directory, and travels
    in the config so the regenerated data is subset the same way.

    la selects the regularization:
        None    the rule the generalization analysis uses -- the smallest
                lambda for the Diag family, the best-by-metric for the others
        "min"   the smallest lambda that was fitted
        "max"   the largest
        float   the fitted lambda nearest to this value (log spacing)

    The lambdas available are those the sweep produced, so an explicit value is
    snapped to the nearest one rather than fitted afresh.
    """
    base  = base_context(loss=loss, matched=matched)
    split = base.split(*sampler, n_od_train)
    model = split.model(model_name)

    la_spec = ("min" if model_name.startswith("Diag") else None) if la is None else la
    ext = model.extract(seed=seed, train=train, metric="ratio", la=la_spec)
    config = seed_config(model, seed, ext.la, expect_model=model_name)
    XX, YY = seed_data(config)
    results, mdl = driver.run(config, X=XX, Y=YY, return_model=True)

    return Fit(name=model_name, loss=loss, la=ext.la,
               Z=mdl.get("Z", results["p_final"]),
               XX=XX, YY=YY, train_idx=train,
               trn=results["split"].trains[train],
               vld=results["split"].vld[train],
               n_od_train=n_od_train, center=mdl.center)


# One extractor per metric, so the caller below does no dispatching of its own.
# Responses come out rois x odours, the natural orientation of the data; the
# other two are odours x odours.

def _observed(metric, fit, half):
    if metric == "resp":
        return np.asarray(fit.data(half)[1])
    r = fit.results(half)
    if metric == "cov":
        return r.Cstar
    return corr_from(r.Cstar, r.ref_vars["Cstar"], r.eval_vars["Cstar"])


def _predicted(metric, fit, half):
    if metric == "resp":
        return fit.Z @ np.asarray(fit.data(half)[0])
    r = fit.results(half)
    if metric == "cov":
        return r.Cest
    return corr_from(r.Cest, r.ref_vars["Cest"], r.eval_vars["Cest"])


def observed_and_predicted(fits, metric, half):
    """The matrices to draw: {'obs': M, 'Diag': M, 'Free': M}.

    One loss, one metric, one half -- the observed matrix, and beside it what
    each model predicted for the same data.

    `fits` is keyed by model name alone -- every fit in it came from the same
    split, so they share their observed data and any one of them can supply it.

    half selects the data the models were scored on ("vld") or fitted to
    ("train"); the figures show both, so a model failing on held-out data can
    be told apart from one that never fitted in the first place.
    """
    assert metric in METRICS, f"metric must be one of {METRICS}, got {metric!r}."
    assert fits, "No fits to compare."
    shared = next(iter(fits.values()))
    return {"obs": _observed(metric, shared, half),
            **{name: _predicted(metric, fit, half) for name, fit in fits.items()}}


class Data(Computation):
    """Refits for the matched-roi supplementary figures."""

    def compute(self, seed=0, train=0, losses=LOSSES, models=MODELS, sampler=SPLIT,
                matched=True, n_od_train="max", la=None):
        """la is passed to refit: None for the usual rule, "min"/"max", a float,
        or a {model_name: value} dict to set it per model."""
        print("COMPUTING matched_rois.Data.")
        self.seed, self.train, self.n_od_train, self.la = seed, train, n_od_train, la
        self.losses, self.models = tuple(losses), tuple(models)

        base_cov   = base_context(loss="cov", matched=matched)
        df_cov, _  = generalization_df(base_cov, splits=[(sampler[0], sampler[1], n_od_train)], which_models=self.models)
        base_resp  = base_context(loss="resp", matched=matched)
        df_resp, _ = generalization_df(base_resp, splits=[(sampler[0], sampler[1], n_od_train)], which_models=self.models)
        df_cov["model"] = df_cov["model"] + "_cov"
        df_resp["model"] = df_resp["model"] + "_resp"
        self.gen_df = pd.concat([df_cov, df_resp], axis=0)
 
        self.fits = {}
        for loss in self.losses:
            for name in self.models:
                print(f"  refitting {name} at loss={loss} ...")
                la_for = la.get(name) if isinstance(la, dict) else la
                fit = refit(loss, name, seed=seed, train=train, sampler=sampler,
                            matched=matched, n_od_train=n_od_train, la=la_for)
                self.fits[(loss, name)] = fit
                print(f"    lambda = {fit.la:.3g}, {fit.n_rois} rois x {fit.n_odours} odours")

        # One keying for everything the figures read: (loss, metric, half).
        # `fits` is keyed (loss, name); observed_and_predicted wants just the
        # names, so the inner comprehension drops the loss it selected on.
        self.matrices = {
            (loss, metric, half): observed_and_predicted(
                {n: self.fits[(loss, n)] for n in self.models}, metric, half)
            for loss in self.losses for metric in METRICS for half in HALVES}

       
        self.computed = True
        return self

    def lambdas(self):
        return {k: fit.la for k, fit in self.fits.items()}
