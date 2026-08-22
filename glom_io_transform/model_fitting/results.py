import os,sys
import pickle
import numpy as np
from dataclasses import dataclass, field

from .layout import build_fit_dir
from .proc_fit_models import subdirs as MODEL_STRS

# Metrics where a SMALLER value is the better fit. The rest -- r2, pearson,
# spearman, and their _resp counterparts -- are correlations or R2, where
# bigger is better. A response fit adds a parallel set of _resp metrics, so the
# direction cannot be decided by the exact name "ratio" alone.
LOWER_IS_BETTER = ("ratio", "ratio_resp")


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
        # Accept any sequence: the default is a tuple, and the body concatenates
        # extra_fields onto lists of column names.
        extra_fields = list(extra_fields)
        key = tuple([metric] + extra_fields)
        if key in self._reports:
            return self._reports[key]
        df = self.df
        test = df[df["split"] == "test"]
        fields = ["seed", "λ"] + extra_fields 
        per = test.groupby(fields, as_index=False)[metric].mean() # Averages over test vs train[0..N]
        # Find the index of the best λ
        by_seed = per.groupby(["seed"] + extra_fields)[metric]
        loc = by_seed.idxmin() if metric in LOWER_IS_BETTER else by_seed.idxmax()
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
    # Which tree to read: the loss the models were fitted against, and whether
    # they were fitted on the matched subset. Defaults reproduce the paths that
    # existed before either was part of the layout.
    loss: str = "cov"
    matched: bool = False
    def split(self, sampler, mode, n_od_train, load=True, check_fresh=True):
       return SplitContext(self, sampler, mode, n_od_train, load=load, check_fresh=check_fresh)


def newest_out_file(split_dir):
    """(path, mtime) of the most recently written out.*.p under a split, or None.

    The fits live one level down, in a directory per model, so this is a couple
    of directory listings rather than a walk.
    """
    newest = None
    for model_dir in os.scandir(split_dir):
        if not model_dir.is_dir():
            continue
        for entry in os.scandir(model_dir.path):
            if entry.name.startswith("out.") and entry.name.endswith(".p"):
                mtime = entry.stat().st_mtime
                if newest is None or mtime > newest[1]:
                    newest = (entry.path, mtime)
    return newest


@dataclass
class SplitContext:
    base: BaseContext
    sampler: str
    mode: str
    n_od_train: int
    # Loading the models is the expensive part; callers that only want to know
    # where the split lives (a path, a timestamp) can skip it with load=False.
    load: bool = True
    # Whether to insist the loaded models are newer than the fits they came
    # from. Turn it off to look at a split while its fits are still running.
    check_fresh: bool = True
    loaded_models: dict = field(init=False, repr=False, default=None)
    split_dir: str = field(init=False, repr=False)
    models_file: str = field(init=False, repr=False)
    
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
                loss=b.loss,
                matched=b.matched,
                name = "_"))
        self.models_file = os.path.join(self.split_dir, "loaded_models.p")
        if not self.load:
            return
        assert os.path.exists(self.models_file), f"Expected loaded models file at {self.models_file} but it does not exist."
        if self.check_fresh:
            newest = newest_out_file(self.split_dir)
            if newest is not None:
                path, mtime = newest
                assert os.path.getmtime(self.models_file) >= mtime, (
                    f"{self.models_file} is older than {path}: the fits have been "
                    f"rerun since the models were loaded. Rerun --loadmodels, or pass "
                    f"check_fresh=False to look at them anyway.")
        self.loaded_models = load_pickle(self.models_file)
        print(f"Loaded split models from {self.models_file}.")
        sys.stdout.flush()
    
    def model(self, name):
        assert self.loaded_models is not None, \
            "This split was built with load=False, so it has no models; rebuild it with load=True."
        base_dir = os.path.join(self.split_dir, MODEL_STRS[name])
        assert os.path.exists(base_dir), f"Directory does not exist: {base_dir}"
        return ModelResults(name=name, df=self.loaded_models[name]["df"], base_dir=base_dir)

