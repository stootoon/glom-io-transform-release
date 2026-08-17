import os, sys, pickle, logging
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
                 normalization="odour_std", center=True):
    """The results.BaseContext shared by the paper figures."""
    if models_dir is None:
        models_dir = os.path.join(paths.proj_path, "model_fitting")
    return results.BaseContext(fits_root=os.path.join(models_dir, "fits"),
                               models_dir=models_dir,
                               standardization=standardization,
                               normalization=normalization,
                               center=center)


def seed_config(model, seed, la, expect_model):
    """Load the in.N.p run config for the given model, seed and lambda."""
    sel = (model.df["seed"] == seed) & (model.df["λ"] == la)
    files = model.df[sel]["file"].unique()
    assert len(files) == 1, f"Expected exactly one input file for {seed=}, λ={la}, found {len(files)}."
    with open(os.path.join(model.base_dir, files[0]), "rb") as f:
        config = pickle.load(f)
    assert config["seed"] == seed, f"Seed mismatch: {config['seed']} vs {seed}"
    assert config["model"] == expect_model, f"Model mismatch: {config['model']} vs {expect_model}"
    # driver.run reads data_file from the config and asserts it exists;
    # drop stale paths so it falls back to the $GLOM_IO_DATA default.
    data_file = config.get("data_file")
    if data_file is not None and not os.path.exists(data_file):
        print(f"Data file {data_file} not found; falling back to $GLOM_IO_DATA default.")
        config.pop("data_file")
    return config


def seed_data(config):
    """Regenerate the (X,Y) splits used for a run from its config."""
    return driver.get_data(normalization=config["normalization"],
                           standardization=config["standardization"],
                           data_file=config.get("data_file"),
                           seed=config["seed"],
                           sampler=config["sampler"])

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

           
            
           
