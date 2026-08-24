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
from tqdm import tqdm

from sklearn.linear_model import LinearRegression 
from glom_io_transform.model_fitting.conn_models.common import compute_r2
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
    # No metric: extract picks one matching the loss this model was fitted against.
    ext = model.extract(seed=seed, train=train, la=la_spec)
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

def polar_decomp(Z):
    """The polar decomposition of a matrix: Z = U @ P, with U unitary and P positive semidefinite.

    The decomposition is unique if Z is full rank, which it is for the models
    here. The unitary part is the closest rotation to Z in Frobenius norm, and
    the positive semidefinite part is the closest positive semidefinite matrix.
    """
    U, s, Vh = np.linalg.svd(Z, full_matrices=False)
    P = (Vh.T * s) @ Vh
    return U @ Vh, P

from glom_io_transform.model_fitting.conn_models.free import Model    as Free
from glom_io_transform.model_fitting.conn_models.free import SymModel as FreeSym
from glom_io_transform.model_fitting.conn_models.free import PSDModel as FreePSD
from glom_io_transform.model_fitting.conn_models.free import RotModel as FreeRot
class Data(Computation):
    """Refits for the matched-roi supplementary figures."""

    def build_r2_fits(self, max_seed = np.inf, max_train = np.inf):
        df = self.df_resp[self.df_resp["model"] == "Free_resp"]
        confs = df[["sampler", "mode", "n_od_train"]].drop_duplicates().values
        assert len(confs) == 1, f"Expected one sampler/mode/n_od_train, got {len(confs)}."
        sampler, mode, n_od_train = confs[0]

        # FreeRot and FreeOrth are both RotModel: FreeRot pins the SO(m)
        # component, FreeOrth sweeps `reflect` as a hyperparameter and so needs
        # the chosen value carried back to rebuild Z (see ext.hyper below).
        free_models = {"Free": Free, "FreeSym": FreeSym, "FreePSD": FreePSD,
                       "FreeRot": FreeRot, "FreeOrth": FreeRot}
        Z_from_p    = {name: mdl.Z_from_p for name, mdl in free_models.items()}
        
        model_cov   = (base_context(loss="cov", matched=True)
                      .split(sampler, mode, n_od_train)
                      .model("Free"))
        model_resps = {name: (base_context(loss="resp", matched=True)
                           .split(sampler, mode, n_od_train)
                           .model(name)) for name in free_models}

        seed_train = df[["seed", "train"]].drop_duplicates().values

        # Sort seed_train by seed, then train, so the first time a seed is seen it is train=0.
        seed_train = sorted(seed_train, key=lambda x: (x[0], x[1]))
        self.Q_resp = {}
        r2 = []
        current_seed = None
        Z_vals = {}
        for seed, train in tqdm(seed_train):
            if seed > max_seed or train > max_train: continue

            ext_cov   = model_cov.extract( seed=seed, train=train, with_params=True)
            ext_resps = {k:mr.extract(seed=seed, train=train, with_params=True)
                         for k, mr in model_resps.items()}
            if seed != current_seed:
                current_seed = seed
                # The data is the same for all three Free models -- same seed,
                # same sampler, same preprocessing -- so one config is enough,
                # and it is only needed when the seed changes.
                X, Y = seed_data(seed_config(model_resps["Free"], seed,
                                             ext_resps["Free"].la, expect_model="Free"))

                Xvld, Yvld = X.vld, Y.vld
                n_roi = Xvld.shape[0]

                Z_cov   = ext_cov.params["p_final"].reshape(n_roi, n_roi, order="C")
                # Any hyperparameter other than lambda changes what a parameter
                # vector means -- FreeOrth's p is a different matrix in each
                # component of O(m) -- so pass the selection through.
                Z_resps = {k: Z_from_p[k](
                                ext_resps[k].params["p_final"], n_roi,
                                **{h: v for h, v in ext_resps[k].hyper.items() if h != "λ"})
                           for k in ext_resps}

                Q_resp, P_resp = polar_decomp(Z_resps["Free"])
                if train == 0:
                    self.Q_resp[(seed, train)] = Q_resp

                Q_cov,  P_cov  = polar_decomp(Z_cov)

                Z_vals[seed] = {
                        # The two fits, and the two recombinations of their factors.
                        "Z_resp":  Z_resps["Free"],
                        "Z_resp_sym": (Z_resps["Free"] + Z_resps["Free"].T)/2,
                        "Z_cov":   Z_cov.copy(),
                        "Q=I":     P_resp.copy(),
                        "P=P_cov": Q_resp @ P_cov,
                        # The constrained refits: symmetric, and symmetric PSD. Unlike
                        # "Q=I" these are fitted under the constraint rather than being
                        # a fitted Z with its rotation deleted afterwards.
                        "Z_sym":   Z_resps["FreeSym"],
                        "Z_psd":   Z_resps["FreePSD"],
                        # Scaled orthogonal: rotations only, and either component.
                        "Z_rot":   Z_resps["FreeRot"],
                        "Z_orth":  Z_resps["FreeOrth"],
                }



            Xtrn, Ytrn = X.trains[train], Y.trains[train]
            
            

            Yhat = {"Input": Xvld, "Output": Ytrn}
            for name, Z in Z_vals[seed].items():
                Yhat[name] = Z @ Xvld

            Xtrn_vec = np.array(X.trains).reshape(-1,1)
            Ytrn_vec = np.array(Y.trains).reshape(-1,1)
            Yhat["a X + b"] = LinearRegression().fit(Xtrn_vec, Ytrn_vec).predict(Xvld.reshape(-1,1)).reshape(n_roi, -1)

            # Z_cov
            Z_cov = Z_vals[seed]["Z_cov"]
            ZXtrn = [Z_cov @ Xi for Xi in X.trains]

            r2_vals = {name: compute_r2(Yvld, Yhat[name], is_cross=True) for name in Yhat}
            r2_vals["seed"] = seed
            r2_vals["train"] = train
            r2.append(r2_vals)

        self.r2_df = pd.DataFrame(r2)
        self.Z_vals = Z_vals
            
    
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
        self.df_resp = df_resp
        df_cov["model"] = df_cov["model"] + "_cov"
        df_resp["model"] = df_resp["model"] + "_resp"
        self.gen_df = pd.concat([df_cov, df_resp], axis=0)

        self.build_r2_fits()
        
        
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
