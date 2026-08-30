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

from scipy.linalg import expm, schur
from sklearn.linear_model import LinearRegression 
from glom_io_transform.model_fitting.conn_models.common import compute_r2, get_IJN
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


# ---------------------------------------------------------------------------
# Z = R S + 1 b', the decomposition the two losses are compared through.
#
# notes/fit_cov_resp.tex: the response and covariance losses split into a mean
# part and a mean-subtracted part, the regularizer sends the covariance fit's
# mean part to that of the identity, and so the only thing worth comparing
# between the two fits is J Z. Polar-decomposing THAT -- not Z -- gives a
# rotation R and a stretch S, and since (J Z)' (J Z) = S' S, the covariance loss
# determines S alone while the response loss determines both.
# ---------------------------------------------------------------------------

def mean_component(Z):
    """b', the row for which Zbar = 1 b'^T: the column means of Z.

    This is the component the covariance loss cannot see, because J 1 = 0 kills
    it before the loss is taken.
    """
    return np.asarray(Z).mean(axis=0)


def centered(Z):
    """J Z, the part of the connectivity both losses see."""
    Z = np.asarray(Z)
    return Z - mean_component(Z)


RANK_TOL = 1e-10        # relative to the largest singular value
ORBIT_PATHS = 4         # rotation paths per seed for rotation_orbit
ORBIT_STEPS = 19        # angles along each, 0 to pi


@dataclass(frozen=True)
class Polar:
    """J Z = R S, alongside the b' that was subtracted to get there.

    S IS SINGULAR BY CONSTRUCTION. Every column of J Z sums to zero, so J Z has
    rank at most m - 1 and the smallest singular value is exactly zero. Both of
    the matching singular directions are still determined -- the left one is
    1/sqrt(m), the right one is Z^-1 1 -- but their SIGN PAIRING is not: R and
    R(I - 2vv') give the same J Z, and they differ in det. So det(R), and any
    statistic that turns on a single reflection, is not identified by the fit.
    `from_Z` resolves it by taking the R closer to the identity, which can only
    understate how far R is from I -- the conservative direction for a figure
    whose point is that R is NOT the identity.
    """
    R:  np.ndarray
    S:  np.ndarray
    b:  np.ndarray          # Zbar = 1 b'^T
    U:  np.ndarray
    s:  np.ndarray          # singular values of J Z, descending; the last is 0
    Vh: np.ndarray

    @classmethod
    def from_Z(cls, Z):
        b = mean_component(Z)
        U, s, Vh = np.linalg.svd(centered(Z), full_matrices=False)
        # The free sign, resolved toward the identity: see the class docstring.
        flip = U.copy()
        flip[:, -1] *= -1
        if np.trace(flip @ Vh) > np.trace(U @ Vh):
            U = flip
        return cls(R=U @ Vh, S=(Vh.T * s) @ Vh, b=b, U=U, s=s, Vh=Vh)

    @property
    def Zbar(self):
        """The rank-one mean component as a matrix, ready to add back to R S."""
        return np.ones((len(self.b), 1)) @ self.b[None, :]

    @property
    def rank(self):
        """How many singular values are not the structural zero."""
        return int(np.sum(self.s > RANK_TOL * self.s[0]))

    @property
    def angles(self):
        """R's rotation angles, in radians, largest first.

        An orthogonal matrix turns each of a set of two-dimensional planes
        through some angle and leaves the rest of the space alone or reflects
        it. The real Schur form lays that out directly -- a 2 x 2 block per
        plane, a 1 x 1 block per real eigenvalue -- so each plane is counted
        once, a reflected direction reads as pi, and an untouched one as 0.

        Taking the angles from the eigenvalues instead would undercount: a
        reflection of multiplicity k comes back from `eigvals` as k values with
        numerical noise for imaginary parts, and any rule for pairing conjugates
        then drops some of them.

        One angle per DIMENSION, so the array is always m long and seeds can be
        compared rank by rank: a plane turned through theta contributes theta
        twice, because it accounts for two of the m directions.
        """
        T = schur(self.R, output="real")[0]
        angles, i, m = [], 0, len(self.R)
        while i < m:
            if i + 1 < m and abs(T[i + 1, i]) > RANK_TOL:
                angles += [abs(np.arctan2(T[i + 1, i], T[i, i]))] * 2
                i += 2
            else:
                angles.append(0.0 if T[i, i] > 0 else np.pi)
                i += 1
        return np.sort(angles)[::-1]


def one_perp(m):
    """An orthonormal basis for the vectors orthogonal to 1, as m x (m-1)."""
    U, _, _ = np.linalg.svd(np.eye(m) - np.ones((m, m)) / m)
    return U[:, :m - 1]


def stabilizer_rotation(m, rng):
    """A random rotation that leaves 1 alone.

    Not a general rotation, and the difference matters. Replacing R by R' keeps
    the covariance loss only if R' J Z is itself J of something -- its columns
    must still sum to zero -- and 1' R' J Z = (R'' 1)' J Z vanishes for every Z
    only when R' fixes 1. The rotations the covariance loss cannot see are
    therefore O(m-1) acting on 1-perp, not O(m), and drawing the null from O(m)
    would put it further away than the fit could ever have gone.
    """
    B = one_perp(m)
    A = rng.standard_normal((m - 1, m - 1))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))          # Haar, not just "some QR factor"
    return B @ Q @ B.T + np.ones((m, m)) / m


def rotation_generator(m, rng):
    """A random antisymmetric matrix acting only on 1-perp, scaled to unit rate.

    exp(t A) is then a rotation that fixes 1 -- one of the rotations the
    covariance loss cannot see -- and turns its fastest plane through exactly t
    radians, so a sweep in t is a sweep in a quantity the reader can name.

    Sweeping is what a rotation orbit needs and random draws cannot give. Two
    independent rotations of a 15-dimensional space are near enough the same
    distance apart every time, so a sample of them lands in a clump and says
    nothing about what happens between there and the identity. A path from the
    identity outward does. And exp(t A) is always in SO(m): the path exists,
    where a path to a REFLECTION would not -- see Polar for why that matters
    here.
    """
    B = one_perp(m)
    G = rng.standard_normal((m - 1, m - 1))
    A = B @ ((G - G.T) / 2) @ B.T
    # An antisymmetric matrix's eigenvalues are the +-i theta of its planes.
    return A / np.abs(np.linalg.eigvals(A).imag).max()


def rotation_orbit(Z, Xs, Ys, center, n_paths=ORBIT_PATHS, n_steps=ORBIT_STEPS,
                   t_max=np.pi, seed=0):
    """Both losses along the rotations the covariance loss cannot see.

    Walks paths through the orbit {R' S + 1 b'} of connectivities that share the
    fitted stretch, and reports what each loss makes of them. The covariance
    loss's data term is flat along the whole orbit by construction -- that is
    the claim the comparison between the two fits rests on, and this is what
    shows it holding numerically rather than only on paper. The response loss's
    data term is not flat, and neither is the regularizer, which is why the
    covariance fit still comes back with a particular rotation: the prior
    chooses where the data cannot.

    `n_paths` generators are drawn and each is swept over `n_steps` angles from
    0 to `t_max`, so every path passes through the fit itself at t = 0 and the
    sweep is over an angle rather than over a set of unrelated rotations.

    Losses are the models' own: a mean over trains of a mean over elements,
    halved. The regularizer is returned WITHOUT a lambda, since the two fits
    were selected at different ones.
    """
    Z = np.asarray(Z)
    m = Z.shape[0]
    I, J, _ = get_IJN(m)
    A = J if center else I
    JZ, Zbar = centered(Z), np.ones((m, 1)) @ mean_component(Z)[None, :]
    norm = np.linalg.norm(Z)

    def losses(Zq):
        resp = np.mean([np.mean((Yk - Zq @ Xk) ** 2) for Xk, Yk in zip(Xs, Ys)]) / 2
        cov  = np.mean([np.mean((Yk.T @ A @ Yk - Xk.T @ Zq.T @ A @ Zq @ Xk) ** 2)
                        for Xk, Yk in zip(Xs, Ys)]) / 2
        return resp, cov, np.mean((Zq - I) ** 2) / 2

    rng = np.random.default_rng(seed)
    rows = []
    for path in range(n_paths):
        G = rotation_generator(m, rng)
        for t in np.linspace(0.0, t_max, n_steps):
            Zq = expm(t * G) @ JZ + Zbar
            resp, cov, reg = losses(Zq)
            rows.append({"path": path, "t": t,
                         "moved": np.linalg.norm(Zq - Z) / norm,
                         "resp_data": resp, "cov_data": cov, "reg": reg})
    return pd.DataFrame(rows)

from glom_io_transform.model_fitting.conn_models.free import Model    as Free
from glom_io_transform.model_fitting.conn_models.free import SymModel as FreeSym
from glom_io_transform.model_fitting.conn_models.free import PSDModel as FreePSD
from glom_io_transform.model_fitting.conn_models.free import RotModel as FreeRot

# Rebuilding Z from a stored parameter vector needs the class that packed it.
MODEL_CLASSES = {"Free": Free, "FreeSym": FreeSym, "FreePSD": FreePSD,
                 "FreeRot": FreeRot, "FreeOrth": FreeRot}
SURROGATE_MODELS = ("Free", "FreeSym")


def surrogate_r2(alphas, n_od_train="18_rand_0", sampler=SPLIT,
                 models=SURROGATE_MODELS, max_seed=np.inf, max_train=np.inf):
    """Held-out R2 for Free and FreeSym on the surrogate data, over alpha.

    The surrogate's ground truth is Z = S + alpha A, with S symmetric and
    ||A|| = ||S||, so alpha is the size of the antisymmetric part relative to
    the symmetric one, and alpha = 0 is a truth that is symmetric exactly. The
    point of the sweep is calibration: it says how much asymmetry there would
    have to be before Free beats FreeSym, which is what makes the tie between
    them on the real data evidence rather than an absence of it.

    Pass alpha=None among the alphas to get the same numbers on the REAL data,
    so the observation lands on the same axis as the calibration.

    Each alpha is a separate set of fits, in its own alpha=<a> fit directory, so
    they all have to have been run and --loadmodels'd first.

    Returns one row per (alpha, seed, train), with a column per model, their
    paired difference, and the R2 the true connectivity itself reaches on the
    validation odours -- the ceiling for that alpha, and NaN on the real data.
    """
    rows = []
    for alpha in alphas:
        base = base_context(loss="resp", matched=True, alpha=alpha)
        df, _ = generalization_df(base, splits=[(sampler[0], sampler[1], n_od_train)],
                                  which_models=list(models))
        fitted = {name: base.split(sampler[0], sampler[1], n_od_train).model(name)
                  for name in models}

        # Sorted by seed so the data is regenerated only when the seed changes;
        # it is the same for every model and every train within a seed.
        seed_train = sorted(df[["seed", "train"]].drop_duplicates().values.tolist())
        current_seed, X, Y, truth_r2 = None, None, None, np.nan
        for seed, train in tqdm(seed_train, desc=f"alpha={alpha}"):
            if seed > max_seed or train > max_train:
                continue
            exts = {name: mdl.extract(seed=seed, train=train, with_params=True)
                    for name, mdl in fitted.items()}
            if seed != current_seed:
                current_seed = seed
                X, Y = seed_data(seed_config(fitted[models[0]], seed,
                                             exts[models[0]].la,
                                             expect_model=models[0]))
                # get_data attaches the ground truth to the outputs it made.
                truth = getattr(Y, "surrogate", None)
                truth_r2 = truth["truth_r2_vld"] if truth is not None else np.nan

            n_roi = X.vld.shape[0]
            r2 = {}
            for name, ext in exts.items():
                # Any hyperparameter other than lambda changes what a parameter
                # vector means, so pass the winning choices through.
                Z = MODEL_CLASSES[name].Z_from_p(
                        ext.params["p_final"], n_roi,
                        **{h: v for h, v in ext.hyper.items() if h != "λ"})
                r2[name] = compute_r2(Y.vld, Z @ X.vld, is_cross=True)

            rows.append({"alpha": alpha, "seed": seed, "train": train,
                         **r2,
                         "Sym - Free": r2["FreeSym"] - r2["Free"],
                         "truth": truth_r2})
    return pd.DataFrame(rows)


def fit_gains(V, Xs, Ys):
    """Gains for a connectivity that is diagonal in a FIXED basis, plus a row.

        Ztilde = diag(g_1 ... g_{m-1}, 0) + e_m r',      Z = V Ztilde V'

    In the basis V the loss separates: each gain is a one-parameter regression
    of one output mode on the same input mode, and the last row is one more
    regression, of the uniform output component on the patterned inputs. About
    2m parameters against Free's m^2, and closed form -- no optimizer.

    The basis is an argument rather than something fitted here, because the
    point of the model is that it is FIXED: estimated once from the whole
    dataset and shared by every seed. A basis re-estimated per fit brings its
    own noise, and then a departure from diagonality cannot be told from an
    error in the axes it is diagonal with respect to.

    The last mode is the uniform direction, whose input component is zero (per
    odour normalization gives 1'X = 0). So its gain multiplies nothing and is
    pinned at zero, and the last COLUMN of Ztilde is unidentifiable for the same
    reason and is left out. The last ROW is not: it is how the fit reproduces
    the mean output, the r block of notes/fit_cov_resp.tex.
    """
    Xt = [V.T @ np.asarray(Xk) for Xk in Xs]
    Yt = [V.T @ np.asarray(Yk) for Yk in Ys]
    num = sum((Yk * Xk).sum(axis=1) for Xk, Yk in zip(Xt, Yt))
    den = sum((Xk * Xk).sum(axis=1) for Xk in Xt)
    m = len(V)
    assert den[-1] < 1e-8 * den[0], (
        f"The last mode should carry no input power, but it has {den[-1]:.3g} "
        f"against {den[0]:.3g} in the first. The basis is not sorted by "
        f"variance, or the inputs are not normalized per odour.")
    g = np.zeros(m)
    g[:-1] = num[:-1] / den[:-1]

    A = np.hstack([Xk[:-1] for Xk in Xt]).T          # patterned inputs
    b = np.concatenate([Yk[-1] for Yk in Yt])        # the uniform output
    r = np.linalg.lstsq(A, b, rcond=None)[0]

    Ztilde = np.diag(g)
    Ztilde[-1, :-1] = r
    return V @ Ztilde @ V.T, g


def mode_powers(V, X, Y):
    """Power along each mode of `V`: in the input, and in the output.

    The diagonals of V' X X' V and V' Y Y' V. What the whitening panel compares:
    a gain g_i takes D_i to g_i^2 D_i, and whitening is that being flat.
    """
    Xv, Yv = np.asarray(X), np.asarray(Y)
    return np.diag(V.T @ (Xv @ Xv.T) @ V), np.diag(V.T @ (Yv @ Yv.T) @ V)


class Data(Computation):
    """Refits for the matched-roi supplementary figures."""

    @staticmethod
    def fit_aZcov_b(Z, X, Y, a = None):
        """a and b' for  Y ~ a (Z X) + 1 b'^T X,  by least squares on the trains.

        b' lives in R^n_roi, one weight per INPUT roi, because the component the
        covariance loss cannot see is Zbar = (11'/m) Z = 1 b'^T: constant across
        rois and varying by odour. So the design column for b'_k is X[k] tiled
        over the roi axis -- not a per-roi indicator, which would fit the
        transpose of what is missing.
        """
        n_roi = X.trains[0].shape[0]
        cols, ys = [], []
        for Xi, Yi in zip(X.trains, Y.trains):
            cols.append(np.column_stack([(Z @ Xi).ravel()]
                                        + [np.tile(Xi[k], n_roi) for k in range(n_roi)]))
            ys.append(Yi.ravel())
        A = np.vstack(cols)
        y = np.concatenate(ys)
        if a is None:
            ab = np.linalg.lstsq(A, y, rcond=None)[0]
            return ab[0], ab[1:]
        else:
            y = y - a * A[:, 0]
            A = A[:, 1:]  # drop the first column, which is a (Z X)
            b = np.linalg.lstsq(A, y, rcond=None)[0]            
            return a, b
        
    @staticmethod
    def apply_aZcov_b(Z, a, b, Xq):
        """a (Z Xq) + 1 b'^T Xq.

        The second term is an outer product, NOT `+ b`: b' is per input roi and
        reaches the prediction through Xq, giving one value per odour shared by
        every roi. Adding b directly would broadcast along the odour axis.
        """
        return a * (Z @ Xq) + np.ones((Xq.shape[0], 1)) @ (b @ Xq)[None, :]
    
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
        self.polar = {}         # seed -> {loss: Polar}, the J Z = R S + 1b' split
        # The gain model needs a basis fixed across seeds, so it cannot be
        # fitted until every seed's input covariance has been seen. The arrays
        # are small -- rois x odours -- so they are kept for a second pass
        # rather than the data being regenerated.
        self.input_cov = {}     # seed -> the input covariance it saw
        split_arrays = {}       # seed -> (X.trains, Y.trains, X.vld, Y.vld)
        self.input_modes = {}   # seed -> eigenvectors of the input covariance
        self.input_vars  = {}   # seed -> the matching eigenvalues
        orbit = []              # both losses over the rotations cov cannot see
        r2 = []
        current_seed = None
        Z_vals = {}
        affine  = {}          # per-seed predictions that are not of the form Z @ X
        self.a_cov = {}       # the fitted weight on Z_cov, worth checking: see above
        self.lin   = {}
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
                config = seed_config(model_resps["Free"], seed,
                                     ext_resps["Free"].la, expect_model="Free")
                X, Y = seed_data(config)
                center = config.get("init_args", {}).get("center", True)

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

                # The eigenbasis of the input covariance, most variable mode
                # first. Z expressed here is what the figure's mode_conn panel
                # draws; eigh returns ascending, so the order is reversed.
                Cxx  = np.mean([Xk @ Xk.T for Xk in X.trains], axis=0)
                D, V = np.linalg.eigh(Cxx)
                order = np.argsort(D)[::-1]
                self.input_modes[seed] = V[:, order]
                self.input_vars[seed]  = D[order]
                self.input_cov[seed]   = Cxx
                split_arrays[seed] = (list(X.trains), list(Y.trains), Xvld, Yvld)

                # J Z = R S, per notes/fit_cov_resp.tex -- NOT Z. The mean
                # component is not comparable between the two fits (the
                # covariance loss cannot see it, so its regularizer sets it),
                # which is exactly why it is split off rather than decomposed.
                pol = {"cov": Polar.from_Z(Z_cov),
                       "resp": Polar.from_Z(Z_resps["Free"])}
                self.polar[seed] = pol

                orbit.append(rotation_orbit(Z_resps["Free"], X.trains, Y.trains,
                                            center, seed=seed).assign(seed=seed))

                Z_vals[seed] = {
                        # The two fits, and the two recombinations of their factors.
                        "Z_resp":  Z_resps["Free"],
                        "Z_resp_sym": (Z_resps["Free"] + Z_resps["Free"].T)/2,
                        "Z_cov":   Z_cov.copy(),
                        # The covariance fit with the response fit's mean
                        # component in place of its own -- REPLACED, not added.
                        # Z_cov already carries a mean component, the one its
                        # regularizer chose for it, so adding a second would
                        # leave this rung with two and make it differ from the
                        # fit in more than the one thing it is meant to test.
                        "Z_cov_bl": centered(Z_cov) + pol["resp"].Zbar,
                        # The rotation deleted, and the stretch swapped for the
                        # covariance fit's -- both keeping the mean component,
                        # which neither swap has anything to say about.
                        "Q=I":     pol["resp"].S + pol["resp"].Zbar,
                        "P=P_cov": pol["resp"].R @ pol["cov"].S + pol["resp"].Zbar,
                        # The constrained refits: symmetric, and symmetric PSD. Unlike
                        # "Q=I" these are fitted under the constraint rather than being
                        # a fitted Z with its rotation deleted afterwards.
                        "Z_sym":   Z_resps["FreeSym"],
                        "Z_psd":   Z_resps["FreePSD"],
                        # Scaled orthogonal: rotations only, and either component.
                        "Z_rot":   Z_resps["FreeRot"],
                        "Z_orth":  Z_resps["FreeOrth"],
                }

                # The affine rungs depend on the seed only -- they are fitted on
                # all of X.trains -- so they belong here rather than in the loop
                # over trains below.
                Xtrn_vec = np.array(X.trains).reshape(-1, 1)
                Ytrn_vec = np.array(Y.trains).reshape(-1, 1)
                fitted = LinearRegression().fit(Xtrn_vec, Ytrn_vec)
                self.lin[seed] = fitted.coef_[0, 0], fitted.intercept_[0]
                a_cov, b_cov = self.fit_aZcov_b(Z_cov, X, Y)
                # b'-only: the same fit with the covariance model removed, so the
                # rung above it can be read as what Z_cov adds over the mean
                # component alone. a_cov comes out NEGATIVE at most lambdas, so
                # without this baseline that rung is easy to misread.
                _, b_only = self.fit_aZcov_b(np.zeros((n_roi, n_roi)), X, Y)
                _, b_a1   = self.fit_aZcov_b(Z_cov, X, Y, a=1.0)
                affine[seed] = {
                    "a X + b":       fitted.predict(Xvld.reshape(-1, 1)).reshape(n_roi, -1),
                    "1b' only":      self.apply_aZcov_b(np.zeros((n_roi, n_roi)), 0.0, b_only, Xvld),
                    "a Z_cov + 1b'": self.apply_aZcov_b(Z_cov, a_cov, b_cov, Xvld),
                    "Z_cov + 1b'":   self.apply_aZcov_b(Z_cov, 1.0, b_a1, Xvld),
                }
                self.a_cov[seed] = a_cov

            Xtrn, Ytrn = X.trains[train], Y.trains[train]

            Yhat = {"Input": Xvld, "Output": np.array(Y.trains).mean(axis=0)}
            for name, Z in Z_vals[seed].items():
                Yhat[name] = Z @ Xvld
            Yhat.update(affine[seed])
            
            r2_vals = {name: compute_r2(Yvld, Yhat[name], is_cross=True) for name in Yhat}
            r2_vals["seed"] = seed
            r2_vals["train"] = train
            r2.append(r2_vals)

        self.r2_df = pd.DataFrame(r2)
        self.orbit_df = pd.concat(orbit, ignore_index=True)
        self.Z_vals = Z_vals

        # One basis for every seed, from every seed's input covariance: the
        # model is a claim about a fixed set of axes, so the axes must not move
        # between fits. Most variable mode first, as input_modes is.
        Cxx_all = np.mean(list(self.input_cov.values()), axis=0)
        D, V = np.linalg.eigh(Cxx_all)
        order = np.argsort(D)[::-1]
        self.basis, self.basis_vars = V[:, order], D[order]

        self.gains, self.mode_power = {}, {}
        gain_r2 = {}
        for seed, (Xs, Ys, Xv, Yv) in sorted(split_arrays.items()):
            Z_gain, g = fit_gains(self.basis, Xs, Ys)
            self.gains[seed] = g
            # Measured on the held-out half, like the R2 beside it: the panel
            # asks what the transformation does to data it did not see.
            d_in, d_out = mode_powers(self.basis, Xv, Yv)
            self.mode_power[seed] = {"input": d_in, "output": d_out, "gain": g}
            gain_r2[seed] = compute_r2(Yv, Z_gain @ Xv, is_cross=True)
            Z_vals[seed]["Z_gain"] = Z_gain
        self.r2_df["Z_gain"] = self.r2_df["seed"].map(gain_r2)
            
    
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
        # Print the Unique models in the gen_df
        print(f"Unique models in gen_df: {self.gen_df['model'].unique()}")

        self.build_r2_fits()

        self.surrogate_df = surrogate_r2(alphas=[0.0, 0.2, 0.4, 0.6,0.8, 1.0], 
                                       n_od_train=n_od_train, sampler=sampler)
        
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
