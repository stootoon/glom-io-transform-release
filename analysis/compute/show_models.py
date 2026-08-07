import os, sys
import pickle
import numpy as np
import pandas as pd

from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from .compute import Computation
from .compute import paths

from model_fitting.driver import RunResults
import model_fitting.results as results

def vld_fun_ratio(vld):
    in_out = np.mean((vld.Cin - vld.Cstar)**2)**0.5
    est_out=np.mean((vld.Cest - vld.Cstar)**2)**0.5
    return in_out, est_out 

def vld_fun_corr(corrs):
    in_out = np.mean((corrs["Cin"] - corrs["Cstar"])**2)**0.5
    est_out=np.mean((corrs["Cest"] - corrs["Cstar"])**2)**0.5
    return in_out, est_out

def compute_corr_energ_(C):
    # If C is a square matrix, use the off-diagonal elements to compute the energy
    if C.shape[0] == C.shape[1]:
        total_energy = np.sum(C**2) - np.sum(np.diag(C)**2)
        n_elements = C.shape[0] * (C.shape[0] - 1)
        return total_energy / n_elements
    else:
        # If C is not square, compute the energy using all elements
        return np.mean(C**2)
        
def vld_fun_corr_energy(corrs):
    [in_, out_, est_] = [compute_corr_energ_(corrs[key]) for key in ["Cin", "Cstar", "Cest"]]
    return in_, out_, est_

class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute(self, 
                selection_metric = "ratio",
                compute_df = False,                
                ):
        
        print("COMPUTING show_models.Data.")
        # What data do we need?
        # Cstar for a given seed
        # Diag output for that seed
        # Free output for that seed
        # generalization results
        
        splits = [
            ("trials", "random", "max"),
            ("odours", "random", "max"),
            ("odours", "inclass", "max"),
            ("odours", "outclass", "max")
        ] 
        base = results.BaseContext(fits_root = paths.proj_path + "/model_fitting/fits",
                                   models_dir= paths.proj_path + "/model_fitting",
                                   standardization="separate",
                                   normalization="odour_std",
                                   center=True)
        
        which_models = ["Diag", "DiagOnlyInh", "Free", "FreeLat"]
        # Import cartesiaon product

        if compute_df:
            from itertools import product
            selection_metric = "ratio"
            records = []
            for split_name in splits:
                split = base.split(*split_name)
                for model_name in which_models:
                    model   = split.model(model_name)
                    n_train = len(model.df["ref"].unique())
                    #report  = model.report(metric=selection_metric, extra_fields = ["outclass"] if "outclass" in split_name else [])     
                    n_seeds = len(model.df["seed"].unique())
                    # I wanted to use report to get the best parameter values, but extract already does that.
                    # All in need to know is whether to include the outclass
                    outclasses = model.df["outclass"].unique()
                    iterable = product(range(n_seeds), range(n_train), outclasses)
                    for seed, train, outclass in tqdm(iterable, total=n_seeds*n_train*len(outclasses)):
                        extra_fields = {"outclass":outclass} if "outclass" in split_name else {}
                        #print(f"{split_name}, {model_name}, seed={seed}, train={train}, outclass={outclass}")
                        res = model.extract(seed=seed, train=train, la="min" if model_name.startswith("Diag") else None, metric=selection_metric, **extra_fields)
                        cov_in_out, cov_est_out = vld_fun_ratio(res.vld)
                        corr_in_out, corr_est_out = vld_fun_corr(res.vld_corrs)
                        corr_en_in, corr_en_out, corr_en_est = vld_fun_corr_energy(res.vld_corrs)
                        records.append({
                            "sampler": split_name[0],
                            "mode": split_name[1],
                            "n_od_train": split_name[2],
                            "model": model_name,
                            "seed": seed,
                            "train": train,
                            "outclass": outclass,
                            "cov_in_out": cov_in_out,
                            "cov_est_out": cov_est_out,
                            "corr_in_out": corr_in_out,
                            "corr_est_out": corr_est_out,
                            "corr_en_in": corr_en_in,
                            "corr_en_out": corr_en_out,
                            "corr_en_est": corr_en_est
                        })
            self.df = pd.DataFrame(records)
            

        self.computed = True
        return self
    
