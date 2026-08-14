import os,sys
import pickle
import numpy as np
from dataclasses import dataclass, field

from .layout import build_fit_dir
from .proc_fit_models import subdirs as MODEL_STRS


class _CompatUnpickler(pickle.Unpickler):
    """Load pickles written before the package refactor, when model_fitting
    modules (split, driver, common, ...) were importable as top-level names,
    or written by driver.py running as a script (classes stamped __main__)."""

    # Where classes pickled from __main__ (or moved modules) may live now.
    _legacy_homes = ("driver", "split", "common")

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            candidates = (self._legacy_homes if module == "__main__"
                          else (module,))
            for cand in candidates:
                try:
                    return super().find_class("glom_io_transform.model_fitting." + cand, name)
                except (ModuleNotFoundError, AttributeError):
                    continue
            raise


def load_pickle(path):
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()

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

    def report(self, metric="ratio", extra_fields = ()):
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

    def _resolve_la(self, la):
        las = np.sort(self.df["λ"].unique())
        if la == "min": 
            return las[0]
        if la == "max":
            return las[-1]
        la = float(la)
        return las[np.argmin(np.abs(np.log(las) - np.log(la)))]
    
    
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
            self._file_cache[fname] = load_pickle(os.path.join(self.base_dir, fname))
            self._file_cache[fname]["results"]["file"] = fname
        return self._file_cache[fname]["results"]

    def extract(self, seed=0, train=0, metric="ratio", la = None, with_params =False, **kwargs):
        if la is None:
            rep = self.report(metric, extra_fields = list(kwargs))
            sel = rep["seed"] == seed
            for fld, val in kwargs.items():
                sel &= rep[fld] == val
            la = rep[sel]["λ"].values[0]
        else:
            la = self._resolve_la(la)
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
    def split(self, sampler, mode, n_od_train): 
       return SplitContext(self, sampler, mode, n_od_train)


@dataclass
class SplitContext:
    base: BaseContext
    sampler: str
    mode: str
    n_od_train: int
    loaded_models: dict = field(init=False, repr=False)
    split_dir: str = field(init=False, repr=False)
    
    def __post_init__(self):
        b = self.base
        self.split_dir = os.path.dirname(
            build_fit_dir(
                root=b.fits_root,
                center=b.center,
                standardization=b.standardization,
                normalization=b.normalization,
                sampler_type=self.sampler,
                split_mode=self.mode,
                n_od_train=self.n_od_train,
                name = "_"))
        models_file = os.path.join(self.split_dir, "loaded_models.p")
        assert os.path.exists(models_file), f"Expected loaded models file at {models_file} but it does not exist."
        self.loaded_models = load_pickle(models_file)
        print(f"Loaded split models from {models_file}.")
        sys.stdout.flush()
    
    def model(self, name):
        base_dir = os.path.join(self.split_dir, MODEL_STRS[name])
        assert os.path.exists(base_dir), f"Directory does not exist: {base_dir}"
        return ModelResults(name=name, df=self.loaded_models[name]["df"], base_dir=base_dir)

