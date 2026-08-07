import os, sys
import pickle
import numpy as np

from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr

from .compute import Computation
from .compute import paths

import model_fitting.results as results


class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute(self, 
                selection_metric = "ratio",
                
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
        base = results.BaseContext(fits_root = proj_path + "/model-fitting/fits",
                                   models_dir=proj_path + "/model-fitting",
                                   standardization="separate",
                                   normalization="odour_std",
                                   center=True)
        
        which_models = ["Diag", "DiagOnlyInh", "Free", "FreeLat"]
        compute_df = False
        if compute_df:
        # Import cartesiaon product
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
    
