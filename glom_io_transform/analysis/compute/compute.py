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

           
            
           
