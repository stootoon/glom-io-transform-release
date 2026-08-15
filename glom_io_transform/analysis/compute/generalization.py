"""Generalization performance of the models across split types.

Builds (or loads from cache) the dataframe of validation metrics per
(split, model, seed, train, outclass), with three metric families:
  cov_*     : covariance mismatch (in_out, est_out)
  corr_*    : correlation mismatch (in_out, est_out)
  corr_en_* : correlation energy (in, out, est)
"""
import os
import pickle
import numpy as np
import pandas as pd

from itertools import product
from tqdm import tqdm

from .compute import Computation, base_context

SPLITS = [
    ("trials", "random", "max"),
    ("odours", "random", "max"),
    ("odours", "inclass", "max"),
    ("odours", "outclass", "max"),
]

WHICH_MODELS = ["Diag", "DiagOnlyInh", "Free", "FreeLat"]


def vld_fun_ratio(vld):
    in_out = np.mean((vld.Cin - vld.Cstar)**2)**0.5
    est_out = np.mean((vld.Cest - vld.Cstar)**2)**0.5
    return in_out, est_out

def vld_fun_corr(corrs):
    in_out = np.mean((corrs["Cin"] - corrs["Cstar"])**2)**0.5
    est_out = np.mean((corrs["Cest"] - corrs["Cstar"])**2)**0.5
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


def generalization_df(base, splits=SPLITS, which_models=WHICH_MODELS,
                      selection_metric="ratio", compute=False, cache_file=None):
    """Load (or compute and cache) the generalization metrics dataframe."""
    if cache_file is None:
        cache_file = os.path.join(base.models_dir, "generalization_results.pkl")

    if not compute:
        assert os.path.exists(cache_file), f"Could not find {cache_file}."
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
        print(f"Loaded generalization results from {cache_file}.")
        return df

    records = []
    for split_name in splits:
        split = base.split(*split_name)
        for model_name in which_models:
            model = split.model(model_name)
            n_train = len(model.df["ref"].unique())
            n_seeds = len(model.df["seed"].unique())
            outclasses = model.df["outclass"].unique()
            iterable = product(range(n_seeds), range(n_train), outclasses)
            for seed, train, outclass in tqdm(iterable, total=n_seeds*n_train*len(outclasses)):
                extra_fields = {"outclass": outclass} if "outclass" in split_name else {}
                res = model.extract(seed=seed, train=train,
                                    la="min" if model_name.startswith("Diag") else None,
                                    metric=selection_metric, **extra_fields)
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
    df = pd.DataFrame(records)
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)
    print(f"Wrote {cache_file}.")
    return df


class Data(Computation):
    """Compute for the supplementary generalization figures."""
    def compute(self, selection_metric="ratio", compute_df=False):
        print("COMPUTING generalization.Data.")
        base = base_context()
        self.df = generalization_df(base, selection_metric=selection_metric, compute=compute_df)
        self.computed = True
        return self
