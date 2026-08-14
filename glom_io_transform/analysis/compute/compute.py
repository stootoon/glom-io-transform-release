import os, sys, pickle, logging
import numpy as np
import pickle
# Import simplenamespace
from types import SimpleNamespace 
from pathlib import Path

import glom_io_transform.analysis.paths as paths


from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

import glom_io_transform.model_fitting.proc_fit_models as pfm
import glom_io_transform.model_fitting.driver as driver

from glom_io_transform.model_fitting.conn_models.common import get_Cstar
from glom_io_transform.model_fitting.conn_models.diag import Model as Diag
from glom_io_transform.model_fitting.conn_models.free import Model as Free

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

           
            
           
