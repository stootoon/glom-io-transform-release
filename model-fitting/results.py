import os,sys
import pickle
import numpy as np
from dataclasses import dataclass, field

from layout import build_fit_dir
from proc_fit_models import subdirs as MODEL_STRS

@dataclass(frozen=True)
class Extraction:
    seed: int
    train: int
    la: float
    vld: object
    params: object = None

    @property
    def vld_corrs(self):
        v = self.vld
        return {fld: getattr(v, fld)/np.sqrt(np.outer(v.ref_vars[fld], v.eval_vars[fld])) for fld in ["Cin", "Cstar", "Cest"]} 


@dataclass
class ModelResults:
    name: str
    df: object
    base_dir: str
    _reports: dict = field(default_factory=dict, init=False, repr=False)
    _file_cache: dict = field(default_factory=dict, init=False, repr=False)

    def report(self, metric="ratio", extra_fields = []):
        key = tuple([metric] + extra_fields)
        if key in self._reports:
            return self._reports[key]
        df = self.df
        test = df[df["split"] == "test"]
        fields = ["seed", "λ"] + extra_fields 
        per = test.groupby(fields, as_index=False)[metric].mean() # Averages over test vs train[0..N]
        # Find the index of the best λ
        loc = per.groupby(["seed"]+extra_fields)[metric].idxmin() if metric == "ratio" else per.groupby(["seed"]+extra_fields)[metric].idxmax()
        best = per.loc[loc] # The best records
        vld = df[df["split"] == "vld"]
        vld_per = vld.groupby(fields, as_index=False)[metric].mean() # Averge over vld vs train[0...N]
        self._reports[key] = vld_per.merge(best[fields], on=fields) # Fore each seed + extra_fields, report the validation data on the λ that gave the best results
        return self._reports[key]

    def _results_for(self, seed, la, train, **kwargs):
        # one out.N.p (seed, la) holds all splits/refs - find it, load once, cache by file
        train_str = f"train[{train}]"
        selector = (self.df["seed"] == seed) & (self.df["λ"] == la) & (self.df["split"] == "trains") & (self.df["ref"] == train_str)
        for fld,value in kwargs.items():
            selector = selector & (self.df[fld] == value)
        files = self.df[selector]["file"]
        assert len(files) == 1, f"Expected exactly one file for seed={seed}, λ={la}, train={train}, but found {len(files)} files."
        fname = files.values[0].replace("in.", "out.")
        if fname not in self._file_cache:
            with open(os.path.join(self.base_dir, fname), "rb") as f:
                self._file_cache[fname] = pickle.load(f)
        return self._file_cache[fname]["results"]

    def extract(self, seed=0, train=0, metric="ratio", with_params =False, **kwargs):
        rep = self.report(metric, extra_fields = list(kwargs))
        sel = rep["seed"] == seed
        for fld, val in kwargs.items():
            sel &= rep[fld] == val
        la = rep[sel]["λ"].values[0]
        results = self._results_for(seed, la, train, **kwargs)
        split = results["split"]
        params = {k: v for k, v in results.items() if k != "split"} if with_params else None
        return Extraction(seed=seed, train=train, la=la, vld=split.vld[train], params=params)


@dataclass
class BaseContext:
    fits_root: str
    models_dir: str
    standardization: str
    normalization: str
    center: bool
    loaded_models: dict = field(default_factory=dict, init=False, repr=False)
    def load_from_file(self, models_file, force_reload=False):
        in_cache = models_file in self.loaded_models
        do_load  = (not in_cache) or force_reload
        if do_load:
            if os.path.exists(models_file):
                with open(models_file, "rb") as f:
                    self.loaded_models[models_file] = pickle.load(f)
                print(f"Loaded pre-saved model from {models_file}.")
                sys.stdout.flush()
            else:
                raise FileNotFoundError(f"No pre-saved models found at {models_file}.")
        else:
            print(f"Using cached models from {models_file}.")
        return self.loaded_models[models_file]
        
    
    def split(self, sampler, mode, n_od_train, load_if_available=True, force_reload = False): 
        models_file = os.path.join(self.models_dir, f"{sampler}_{mode}_{n_od_train}.p")
        loaded_models = None
        if load_if_available:
            if os.path.exists(models_file):
                loaded_models = self.load_from_file(models_file, force_reload)
        if loaded_models is None:
            raise NotImplementedError(
                "Loading results directly from the target fit directory is not yet implemented.; "
                f"expected a pre-saved models pickle at {models_file} ")
        return SplitContext(self, sampler, mode, n_od_train, loaded_models)


@dataclass
class SplitContext:
    base: BaseContext
    sampler: str
    mode: str
    n_od_train: int
    loaded_models: dict

    def model(self, name):
        b = self.base
        base_dir = build_fit_dir(
            root=b.fits_root,
            center=b.center,
            standardization=b.standardization,
            normalization=b.normalization,
            sampler_type=self.sampler,
            split_mode=self.mode,
            n_od_train=self.n_od_train,
            name=MODEL_STRS[name])
        assert os.path.exists(base_dir), f"Directory does not exist: {base_dir}"
        return ModelResults(name=name, df=self.loaded_models[name]["df"], base_dir=base_dir)

