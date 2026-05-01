import os, sys, pickle, logging
import numpy as np
import pickle
# Import simplenamespace
from types import SimpleNamespace 
from pathlib import Path
import paths

sys.path.append(paths.proj_path)
sys.path.append(paths.fits_root)
print("Adding parent")
sys.path.append(str(Path(paths.conn_models_path).parent))

from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

import ob_io_conn_models
from ob_io_conn_models.models.common import get_Cstar
from ob_io_conn_models.models import diag as Diag

import proc_fit_models as pfm

import driver as conn2_driver

print("Loading ", __file__)

standardization = "separate"
normalization   = ["odour", "std"]
center          = True

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

           
            
           
