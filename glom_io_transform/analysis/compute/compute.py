import os, sys, json, pickle, logging
import numpy as np
import pickle
# Import simplenamespace
from types import SimpleNamespace 
from pathlib import Path

import glom_io_transform.paths as paths


from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

import glom_io_transform.model_fitting.proc_fit_models as pfm
import glom_io_transform.model_fitting.driver as driver
import glom_io_transform.model_fitting.split as split
import glom_io_transform.model_fitting.results as results

from glom_io_transform.model_fitting.conn_models.common import get_Cstar
from glom_io_transform.model_fitting.conn_models.diag import Model as Diag
from glom_io_transform.model_fitting.conn_models.free import Model as Free

print("Loading ", __file__)

standardization = "separate"
normalization   = ["odour", "std"]
center          = True


def base_context(models_dir=None, standardization="separate",
                 normalization="odour_std", center=True, loss="cov", matched=False,
                 alpha=None):
    """The results.BaseContext shared by the paper figures."""
    if models_dir is None:
        models_dir = os.path.join(paths.proj_path, "model_fitting")
    return results.BaseContext(fits_root=os.path.join(models_dir, "fits"),
                               models_dir=models_dir,
                               standardization=standardization,
                               normalization=normalization,
                               center=center,
                               loss=loss,
                               matched=matched,
                               alpha=alpha)


def seed_config(model, seed, la, expect_model):
    """Load the in.N.p run config for the given model, seed and lambda."""
    sel = (model.df["seed"] == seed) & (model.df["λ"] == la)
    files = model.df[sel]["file"].unique()
    assert len(files) == 1, f"Expected exactly one input file for {seed=}, λ={la}, found {len(files)}."
    with open(os.path.join(model.base_dir, files[0]), "rb") as f:
        config = pickle.load(f)
    assert config["seed"] == seed, f"Seed mismatch: {config['seed']} vs {seed}"
    assert config["model"] == expect_model, f"Model mismatch: {config['model']} vs {expect_model}"
    # The configs carry absolute paths resolved wherever they were generated, so
    # they are wrong on any other machine. data_file has a default to fall back
    # to; match_file does not, so look for it by name in $GLOM_IO_DATA and fail
    # loudly if it is not there -- silently dropping it would fit the full
    # population while everything else still said "matched".
    data_file = config.get("data_file")
    if data_file is not None and not os.path.exists(data_file):
        print(f"Data file {data_file} not found; falling back to $GLOM_IO_DATA default.")
        config.pop("data_file")

    match_file = config.get("match_file")
    if match_file is not None and not os.path.exists(match_file):
        local = os.path.join(os.environ["GLOM_IO_DATA"], os.path.basename(match_file))
        assert os.path.exists(local), (
            f"Match file {match_file} not found, and neither is {local}. "
            f"This is a matched run; refusing to fall back to the full population.")
        print(f"Match file {match_file} not found; using {local}.")
        config["match_file"] = local
    return config


# Regenerating the splits is the expensive part of any loop over seeds or
# trains, and what comes back depends on the seed, the sampler and the
# preprocessing -- not on lambda, and not on which train the caller goes on to
# use. So a loop over the trains of one seed asks for the same arrays every
# time. Cleared with seed_data.cache.clear() if the data on disk changes.
_SPLIT_CACHE = {}


def seed_data(config, cache=True):
    """Regenerate the (X,Y) splits used for a run from its config.

    match_file has to travel with the rest: without it a matched run silently
    regenerates the full population, which does not error anywhere -- it just
    quietly answers a different question.

    The result is SHARED between callers, so treat the arrays as read-only --
    the models do, and are checked to. Pass cache=False for a private copy.
    """
    kwargs = dict(normalization=config["normalization"],
                  standardization=config["standardization"],
                  data_file=config.get("data_file"),
                  match_file=config.get("match_file"),
                  seed=config["seed"],
                  sampler=config["sampler"],
                  # Which odours the run used, for the same reason as
                  # match_file: without it the data comes back with all
                  # 48 and quietly answers a different question.
                  odour_spec=config["sampler"].get("split", {}).get("n_od_train", "max"),
                  # Surrogate runs must regenerate the same surrogate, not the
                  # real data, or the refit would be scored against the wrong Y.
                  alpha=config.get("alpha"),
                  target_r2=config.get("target_r2"))
    if not cache:
        return driver.get_data(**kwargs)
    # The sampler is a nested dict, so serialise rather than hash the values.
    key = json.dumps(kwargs, sort_keys=True, default=str)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = driver.get_data(**kwargs)
    return _SPLIT_CACHE[key]


seed_data.cache = _SPLIT_CACHE

def _deal_trials(df, n_slots, seed):
    """Assign each (glomerulus, odour)'s trials to `n_slots` disjoint draws.

    Returns (list of index arrays, n_cells_short). A cell with fewer trials
    than slots cannot fill them disjointly, so its trials wrap around and the
    count of such cells is reported rather than hidden -- the input has exactly
    3 trials per cell and the output 3 to 5, so this bites only where 4 are
    asked for.
    """
    rng = np.random.default_rng(seed)
    slots = [[] for _ in range(n_slots)]
    short = 0
    for _, group in df.groupby(["glob_id", "odour"], sort=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        if len(idx) < n_slots:
            short += 1
            idx = np.resize(idx, n_slots)
        for slot, chosen in zip(slots, idx[:n_slots]):
            slot.append(chosen)
    return [np.array(slot) for slot in slots], short


def output_noise_floor(config, seed, floor_seed=20260828):
    """How far apart two independent measurements of C*(train, vld) sit.

    The models are scored as RMS(Cest(train, vld) - C*(train, vld)), where
    every term is a single trial draw. This forms the same quantity twice from
    TRIAL-DISJOINT draws,

        C*(Y_train_a, Y_vld_a)   against   C*(Y_train_b, Y_vld_b),

    so it has the shape and the single-trial character of what the models are
    judged on, and no trial is shared between the two estimates.

    Slots needed: 2 when the train and vld ODOURS are disjoint (the odours
    sampler -- a cell belongs to one side or the other, never both), 4 when
    they coincide (the trials sampler, which needs a train and a vld draw per
    estimate out of the same cell).

    NOTE, for the caption: this measures OUTPUT trial noise only. The models'
    Cest is built from single INPUT trials through Z, so a model with the true
    Z would still be charged for input noise that this floor does not include
    -- the floor therefore sits below what any model could actually reach. The
    opposing error is that a difference of two independent estimates has twice
    the noise variance of one, inflating this by sqrt(2). The two partly
    offset, by an amount that depends on the relative noise on each side, and
    neither is corrected here: the panel compares models, and a reference line
    that is off by a constant factor does not change that comparison.

    Returns {"cov": ..., "corr": ...}, or None if the frames cannot be drawn.
    """
    kwargs = dict(normalization=config["normalization"],
                  standardization=config["standardization"],
                  data_file=config.get("data_file"),
                  match_file=config.get("match_file"),
                  seed=config["seed"],
                  sampler=config["sampler"],
                  odour_spec=config["sampler"].get("split", {}).get("n_od_train", "max"))
    _, _, _, Ydf = driver.get_data(return_dfs=True, **kwargs)

    split_cfg = config["sampler"].get("split", {})
    train_ods = split_cfg.get("train_odours")
    vld_ods   = split_cfg.get("vld_odours")
    if train_ods is None or vld_ods is None:
        # A trials run names no odour sets: both sides are every odour present.
        train_ods = vld_ods = sorted(Ydf["odour"].unique())
    same_odours = set(train_ods) == set(vld_ods)

    n_slots = 4 if same_odours else 2
    slots, short = _deal_trials(Ydf, n_slots, floor_seed + int(seed))
    if short:
        print(f"  noise floor: {short} cells had fewer than {n_slots} trials; "
              f"their draws reuse trials and are not fully disjoint.")

    def matrix(slot, odour_names):
        rows = Ydf.loc[slot]
        return split.df2mat(rows[rows["odour"].isin(odour_names)])

    if same_odours:
        train_a, vld_a, train_b, vld_b = (matrix(slots[0], train_ods), matrix(slots[1], vld_ods),
                                          matrix(slots[2], train_ods), matrix(slots[3], vld_ods))
    else:
        train_a, vld_a = matrix(slots[0], train_ods), matrix(slots[0], vld_ods)
        train_b, vld_b = matrix(slots[1], train_ods), matrix(slots[1], vld_ods)

    # Normalised the way the pipeline normalises these roles: the two train
    # draws together, each vld draw on its own. (The real run stacks ten train
    # draws rather than two, so the train scale differs marginally.)
    center = config.get("init_args", {}).get("center", True)
    normalization = config["normalization"]
    norm_y = normalization[1] if isinstance(normalization, list) else normalization
    pp_a = driver.preproc(split.SplitSamples(trains=[train_a, train_b], test=vld_a, vld=vld_b),
                          config["standardization"], norm_y)

    def cstar_and_corr(ref, ev):
        C  = get_Cstar(ref, center, X2=ev)
        rv = np.diag(get_Cstar(ref, center))
        ev_= np.diag(get_Cstar(ev,  center))
        return C, C / np.sqrt(np.outer(rv, ev_))

    C_a, R_a = cstar_and_corr(pp_a.trains[0], pp_a.test)
    C_b, R_b = cstar_and_corr(pp_a.trains[1], pp_a.vld)
    rms = lambda M: float(np.mean(np.asarray(M) ** 2) ** 0.5)
    return {"cov": rms(C_a - C_b), "corr": rms(R_a - R_b)}


def compute_correlation(X):
    C = np.cov(X.T, bias=True) * X.shape[0]
    v = np.diag(C)
    R = C /np.sqrt(v[:,None] * v[None,:])
    return R

def compute_pearson_energy(R):
    return np.sum(R**2) - R.shape[0]

def compute_corr_energy(X):
    return compute_pearson_energy(compute_correlation(X))

class Computation:
    def __init__(self, *args, **kwargs):
        self.computed = False
        
    def compute(self, *args, **kwargs):
        raise NotImplementedError("compute() method not implemented")

           
            
           
