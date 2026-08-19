import argparse
from tqdm import tqdm
import yaml
import hashlib
import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
import itertools, copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import NamedTuple, List

from glom_io_transform.model_fitting import common, split
from glom_io_transform.model_fitting.layout import build_fit_dir, get_split_mode
from glom_io_transform.data.odours import odours

def add_path_env_var(name):
    assert name in os.environ, f"Did not find environment variable {name}."
    path = os.environ[name]
    assert len(path), "Path to {name} is empty."
    assert os.path.exists(path), f"{path} does not exist"
    sys.path.append(path)

add_path_env_var("GLOM_IO_DATA")
    
from glom_io_transform.model_fitting.conn_models.common   import get_Cstar
from glom_io_transform.model_fitting.conn_models.diag     import Model as Diag
from glom_io_transform.model_fitting.conn_models.free     import Model as Free
from glom_io_transform.model_fitting.conn_models.free_lat import Model as FreeLat

from glom_io_transform.model_fitting import proc_fit_models as pfm

class OverallStdScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.overall_std_ = None

    def fit(self, X, y=None):
        # Compute the overall standard deviation
        self.overall_std_ = np.std(X)
        return self

    def transform(self, X, y=None):
        # Check if the scaler was fitted
        if self.overall_std_ is None:
            raise RuntimeError("The scaler has not been fitted yet")

        # Scale the data
        return X / self.overall_std_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)


def array_fingerprint(x, *, decimals=6, dtype=np.float32, algo="blake2b"):
    x = np.asarray(x)
    x = x.astype(dtype, copy=False)

    # quantize deterministically
    xq = np.round(x, decimals=decimals)

    # canonical bytes (C-order)
    xb = np.ascontiguousarray(xq).view(np.uint8)

    h = hashlib.blake2b(digest_size=16) if algo == "blake2b" else hashlib.sha256()
    h.update(str(xq.shape).encode())
    h.update(str(xq.dtype).encode())
    h.update(xb.tobytes())
    return h.hexdigest()

known_models = {"Diag"                :Diag,
                "Free"                :Free,
                "FreeLat"             :FreeLat,
                }

def verify_odour_order(arrays, name):
    """Assert every array's odour axis is in the stored X0Y0 order.

    Only checks arrays that carry an odour coordinate; plain arrays (from an
    X0Y0 file written before the data was labelled) are passed over, since
    there is nothing to check them against.
    """
    expected = odours.get_order("X0Y0")
    for i, Xi in enumerate(arrays):
        got = getattr(Xi, "coords", {}).get("odour") if hasattr(Xi, "coords") else None
        if got is None:
            continue
        got = [str(o) for o in got.values]
        assert got == expected, (
            f"{name}[{i}] is not in the stored X0Y0 odour order: "
            f"first difference at position {next(j for j, (a, b) in enumerate(zip(got, expected)) if a != b)} "
            f"({got[:3]}... vs {expected[:3]}...)"
            if sorted(got) == sorted(expected) else
            f"{name}[{i}] has different odours from the X0Y0 order: "
            f"only in data {sorted(set(got) - set(expected))}, "
            f"only in order {sorted(set(expected) - set(got))}")

def get_matched_data(X, Y, match_file):
    def _get_roi(X, which_exp, which_roi, match=None):
        """One ROI's responses, tagged with which one it is.

        experiment / roi_id / roi_label already ride along as scalar coordinates.
        What is missing is local_roi -- the POSITION within the experiment, which is
        what the matched-pair metadata indexes by, and which is not the same number
        as roi_id -- and the match the roi belongs to. Attaching them here means
        they survive a concat, so nothing downstream has to track list order.
        """
        for Xi in X:
            if which_exp in Xi.experiment.values:
                assert (Xi.experiment.values == which_exp).all(), \
                    f"{which_exp} shares an array with other experiments; positional indexing is unsafe."
                assert 0 <= which_roi < Xi.sizes["roi"], \
                    f"roi {which_roi} out of range for {which_exp}, which has {Xi.sizes['roi']} rois"
                # isel with a LIST keeps the roi dimension (size 1), which
                # data_to_df needs -- Xi[which_roi] would drop it and come back
                # 2-D. The per-roi coords stay attached either way.
                Xret = Xi.isel(roi=[which_roi]).assign_coords(local_roi=which_roi)
                return Xret if match is None else Xret.assign_coords(match=match)
        raise ValueError(f"Experiment {which_exp} not found in X")

    print(f"Loading matched data from {match_file}")
    match_info = pd.read_csv(match_file)

    # Iterate each row
    Xm, Ym = [], []
    for index, row in match_info.iterrows():
        exp_input  = row["input_exp"]
        roi_input  = row["input_local_roi"]
        exp_output = row["output_exp"]
        roi_output = row["output_local_roi"]
        match      = row["match_id"] if "match_id" in row else index
        Xm.append(_get_roi(X, exp_input,  roi_input,  match=match))
        Ym.append(_get_roi(Y, exp_output, roi_output, match=match))

    # The response loss pairs row i of X with row i of Y, so the two lists have
    # to be the same length and in the same match order. Both hold by
    # construction here; assert it anyway, because a truncated or reordered
    # match file would otherwise give a confidently wrong fit.
    assert len(Xm) == len(Ym), f"{len(Xm)} matched inputs but {len(Ym)} matched outputs."
    mismatched = [(int(x.match), int(y.match)) for x, y in zip(Xm, Ym)
                  if int(x.match) != int(y.match)]
    assert not mismatched, f"Input and output rows are not in the same match order: {mismatched[:5]}"
    print(f"Loaded {len(Xm)} matched roi pairs.")

    return Xm, Ym

def get_data(full=False, normalization="roi", standardization="train",
             seed = 0, data_file = None, sampler="trials",
             return_inds=False, match_file = None,
             ):
    # Use the directory of this file to find the data.
    if data_file is None:
        data_dir = os.environ["GLOM_IO_DATA"] 
        data_file = os.path.join(data_dir, "X0Y0_new.p")
        assert os.path.exists(data_file), f"{data_file} does not exist"

    print(f"Loading data from {data_file}")
    # Load X0 and Y0 from X0Y0.p
    with open(data_file, "rb") as f:
        data = pickle.load(f)
        X0 = data["X0"]
        Y0 = data["Y0"]       

    X, Y = (X0, Y0) if match_file is None else get_matched_data(X0, Y0, match_file)

    if full: return X, Y

    # data_to_df indexes odours positionally, so the labels stop here. Check
    # them first: everything downstream -- gen_split's class lookup above all --
    # assumes this axis is in the stored X0Y0 order, and that assumption has
    # been wrong before. Arrays without labels are older files, and pass.
    verify_odour_order(X, "X0")
    verify_odour_order(Y, "Y0")

    Xdf = split.data_to_df(X, split.IoType.INPUT)
    Ydf = split.data_to_df(Y, split.IoType.OUTPUT)

    if type(normalization) is str: normalization = [normalization] * 2 # same normalization for X and Y
    
    assert type(normalization) is list and len(normalization) == 2 and all(type(n) is str for n in normalization), "Normalization should be a string or a list of two strings."

    if isinstance(sampler, dict):
        sampler = split.make_sampler(sampler)
    elif isinstance(sampler, str):
        sampler = split.make_sampler({"type":sampler})
    else:
        raise ValueError(f"Don't know what to do for sampler {sampler}.")

    # Make sure sampler has a generate method. This is a requirement of the SplitSampler class.
    assert hasattr(sampler, "generate") and callable(sampler.generate), "Sampler should have a generate method."

    print("Assembling INPUTS.")
    Xinds = sampler.generate(Xdf, seed=seed)
    Xss    = Xinds.materialize(Xdf, split.df2mat) # Returns SplitSample
    Xss_pp = preproc(Xss, standardization, normalization[0])

    print("Assembling OUTPUTS.") 
    Yinds  = sampler.generate(Ydf, seed=seed)
    Yss    = Yinds.materialize(Ydf, split.df2mat) # Comes back as a SplitSample
    Yss_pp = preproc(Yss, standardization, normalization[1])

    return (Xss_pp, Yss_pp, Xinds, Yinds) if return_inds else (Xss_pp, Yss_pp)

def scale_by_cells(Xss):
    # The loss compares X.T @ J @ X between input and output populations, 
    # which have different cell counts. Without this scalling, these are 
    # scatter matrices, and not comparable. Scaling by 1/sqrt(n_cells) makes
    # them covariances. We do it here rather than folding 1/n into J, because
    # we want J to remain idempotent.
    def scale(A):
        return A / np.sqrt(A.shape[0])
    return split.SplitSamples(trains = [scale(X) for X in Xss.trains],
                              test   = scale(Xss.test),
                                vld  = scale(Xss.vld))

def preproc(Xss, standardization, normalization):

    if normalization == "none":
        print("No normalization, just scaling by number of cells.")
        return scale_by_cells(Xss)
    
    Xtrains, Xtest, Xvld = Xss.trains, Xss.test, Xss.vld

    if standardization == "train":
        print("Standardizing training, test and validation sets based on training parameters.")
        XSS = [Xtrains, Xtrains, Xtrains]
    elif standardization == "separate":
        print("Standardizing training, test and validation sets separately.")
        XSS = [Xtrains, [Xtest], [Xvld]]
    else:
        raise ValueError(f"Don't know what to do for standardization {standardization}.")

    #pdb.set_trace()
    XX = [Xss.trains, [Xss.test], [Xss.vld]] # What we'll actually standardize

    if normalization == "roi":
        print(f"Normalizing by ROI.")
        SS = [StandardScaler().fit(np.hstack(X).T) for X in XSS]
        XXpp = [[SSi.transform(Xij.T).T for Xij in Xi] for SSi, Xi in zip(SS, XX)]
    elif normalization == "odour":
        print(f"Normalizing by odour.")
        SS = [StandardScaler().fit(np.vstack(X)) for X in XSS]
        XXpp = [[SSi.transform(Xij) for Xij in Xi] for SSi, Xi in zip(SS, XX)]
    elif normalization == "std":
        print(f"Normalizing by overall standard deviation.")
        SS = [OverallStdScaler().fit(np.vstack(X)) for X in XSS]
        XXpp = [[SSi.transform(Xij) for Xij in Xi] for SSi, Xi in zip(SS, XX)]
    else:
        raise ValueError(f"Don't know to do normalization '{normalization}'.")

    XXpp = split.SplitSamples(trains = XXpp[0], test = XXpp[1][0], vld = XXpp[2][0])

    def is_zscored(X):
        return np.allclose(X.mean(axis=0), 0) and np.allclose(X.std(axis=0), 1)

    def assert_normalization(Xs, normalization):        
        if normalization == "roi":
            assert is_zscored(np.hstack(Xs).T), "Training data ROIs are not standardized."
        elif normalization == "odour":
            assert is_zscored(np.vstack(Xs)), "Training data odours are not standardized."
        elif normalization == "std":
            assert np.allclose(np.vstack(Xs).std(), 1), "Training data std is not 1."
        elif normalization == "none":
            assert True
        else:
            raise ValueError(f"Don't know what to do for {normalization=}.")
        return True

    assert_normalization(XXpp.trains, normalization)
    if standardization == "separate":
        for XX in [XXpp.test, XXpp.vld]:
            assert_normalization([XX], normalization)
    
    return scale_by_cells(XXpp) 


class RunResults(NamedTuple):
    Cstar: np.ndarray
    Cest: np.ndarray
    Cin: np.ndarray
    ref_vars: dict
    eval_vars: dict
    is_cross: bool
    hashes: dict

SplitResults = split.Split[List[RunResults]]

def pack_split_results(XX, YY, Z, center):
    def one(Xref, Yref, Xeval, Yeval, is_cross):
        Yref_est  = Z @ Xref
        Yeval_est = Z @ Xeval
        ref_arrs = {"Cin":Xref, "Cstar":Yref, "Cest":Yref_est}
        eval_arrs= {"Cin":Xeval, "Cstar":Yeval, "Cest":Yeval_est}
        return RunResults(Cstar=get_Cstar(Yref,    center, X2=Yeval),
                          Cest=get_Cstar(Yref_est, center, X2=Yeval_est),
                          Cin =get_Cstar(Xref,     center, X2=Xeval),
                          is_cross=is_cross,
                          ref_vars = {k: np.diag(get_Cstar(v, center)) for k,v in ref_arrs.items()}, 
                          eval_vars= {k: np.diag(get_Cstar(v, center)) for k,v in eval_arrs.items()},
                          hashes   = {
                              k:array_fingerprint(arr)
                              for k, arr in
                              zip(["Xref", "Yref", "Xeval", "Yeval", "Yref_est", "Yeval_est"],
                                  [Xref, Yref, Xeval, Yeval, Yref_est, Yeval_est])
                          }
                          )
    
    XYpairs = list(zip(XX.trains, YY.trains))
    return SplitResults(
        trains = [one(Xref, Yref, Xref, Yref, is_cross=False) for Xref, Yref in XYpairs],
        test   = [one(Xref, Yref, XX.test, YY.test, is_cross=True) for Xref, Yref in XYpairs],
        vld    = [one(Xref, Yref, XX.vld,  YY.vld,  is_cross=True) for Xref, Yref in XYpairs])

def gen_split(seed, sampler):
    if sampler["type"] not in ["trials", "odours"]:
        raise ValueError(f"Don't know how to generate split odours for sampler {sampler}.")
    
    if sampler["type"] == "trials":
        if "split" not in sampler:
            sampler["split"] = {"n_od_train":"max"}
        for (fld,val) in [("n_od_test",0), ("n_od_vld",0), ("mode", "random")]:
            if fld not in sampler["split"]:
                sampler["split"][fld] = val

    assert "split" in sampler, "Split configuration must be specified in sampler for odours sampler."
    split_config = sampler["split"]
    required_fields = ["mode", "n_od_train", "n_od_test", "n_od_vld"]
    for field in required_fields:
        assert field in split_config, f"Field '{field}' must be specified in split configuration for odours sampler."            
    n_od_train = split_config["n_od_train"]
    n_od_test  = split_config["n_od_test"]
    n_od_vld   = split_config["n_od_vld"]
    # The indices produced here are positions along the data's odour axis, which
    # is in the stored ("X0Y0") order -- NOT the order odour_labels.mat is in.
    # Map classes through the odour names so the two frames agree; indexing
    # odours.classes positionally would group the wrong odours.
    storage_names = odours.get_order("X0Y0")
    class_of      = dict(zip(odours.names, odours.classes))
    missing       = [n for n in storage_names if n not in class_of]
    assert not missing, f"No chemical class for odours: {missing}"
    classes       = [class_of[n] for n in storage_names]
    n_od          = len(storage_names)

    if n_od_train != "max":
        n_od_train = int(n_od_train)
        assert n_od_train + n_od_test + n_od_vld <= n_od, f"{n_od_train=} + {n_od_test=} + {n_od_vld=} > {n_od=}. Not enough odours to split."

    unique_classes  = sorted(set(classes))
    odours_in_class = {c:[] for c in unique_classes}
    for i, c in enumerate(classes):
        odours_in_class[c].append(i)

    np.random.seed(seed)
    mode = split_config["mode"]
    if mode == "random":
        odour_inds   = np.random.permutation(n_od)
        test_odours  = odour_inds[:n_od_test]
        vld_odours   = odour_inds[n_od_test:n_od_test+n_od_vld]
        if n_od_train == "max":
            n_od_train   = n_od - n_od_test - n_od_vld
        train_odours = odour_inds[n_od_test+n_od_vld:][:n_od_train]
    elif mode in ["inclass", "outclass"]:   
        if mode == "inclass":
            # Leave one odour out from each class for testing and validation
            test_vld = [np.random.choice(odours_in_class[c], size=1, replace=False)[0] for c in unique_classes]
        elif mode == "outclass":
            # Leave one class out for testing and validation
            assert "outclass" in split_config, "Field 'outclass' must be specified in split configuration for outclass mode."
            outclass = split_config["outclass"]
            assert outclass in unique_classes, f"Outclass '{outclass}' is not a valid class. Valid classes are {unique_classes}."
            test_vld = odours_in_class[outclass]
        else:
            raise ValueError(f"Unknown mode '{mode}' for odours sampler.")

        assert n_od_test + n_od_vld <= len(test_vld), "Not enough odours to leave out for test and validation sets."
        test_vld    = np.random.permutation(test_vld)
        test_odours = test_vld[:n_od_test]
        vld_odours  = test_vld[n_od_test:n_od_test+n_od_vld]
        train_avail = sorted(set(range(n_od)) - set(test_vld))
        if n_od_train == "max":
            n_od_train = len(train_avail)
        assert len(train_avail) >= n_od_train, f"{len(train_avail)=} < {n_od_train=}. Not enough odours left for training set."
        train_odours = np.random.choice(list(train_avail), size=n_od_train, replace=False)

    else:
        raise ValueError(f"Unknown mode '{mode}' for odours sampler.")

    return {"train_odours":train_odours.tolist(), "test_odours":test_odours.tolist(), "vld_odours":vld_odours.tolist()}

def run(config, X=None, Y=None, return_dataset = False, return_model = False):
    if (X is None and Y is not None) or (X is not None and Y is None):
        raise ValueError("Either both X and Y should be provided, or neither should be provided.")

    # Get the seed, λ and trial from the config.
    seed  = config["seed"]
    λ     = config["λ"] if "λ" in config else None
    normalization = config["normalization"]
    standardization = config["standardization"]
    data_file = config["data_file"] if "data_file" in config else None
    if data_file is not None:
        assert os.path.exists(data_file), f"Data file {data_file} does not exist."
    match_file = config["match_file"] if "match_file" in config else None
    if match_file is not None:
        assert os.path.exists(match_file), f"Match file {match_file} does not exist."
    
    # Set the seed.
    np.random.seed(seed)
    assert "sampler" in config, "'Sampler' should be specified in the config."
    
    # Get the data.
    if X is None and Y is None:
        XX, YY = get_data(normalization=normalization,
                          standardization=standardization,
                          data_file=data_file,
                          seed = seed,
                          sampler = config["sampler"],
                          match_file = match_file,
                          )
    else:
        XX, YY = X, Y  

    dataset = {"XX": XX, "YY": YY}
        
    n_cells = XX.trains[0].shape[0]

    if config["model"] not in known_models:
        raise ValueError(f"Don't know what to do for model {config['model']}.")

    Model = known_models[config["model"]]

    context = {"np":np,
               "n_cells":n_cells}

    init_args = common.eval_fields(config["init_args"], context=context) if "init_args" in config else {}
    #init_args, min_args = [common.eval_fields(config[name], context=context) if name in config else {} for name in ["init_args", "min_args"]]
    print(f"{init_args=:}")

    if λ is not None: init_args["λ"] = λ

    mdl = Model(XX.trains, YY.trains, **init_args)
    
    context["mdl"] = mdl
    min_args = common.eval_fields(config["min_args"], context=context) if "min_args" in config else {}
    p_init_specs = min_args.pop("p_init", None)

    p0 = None if p_init_specs is None else [mdl.init_from(*spec) for spec in p_init_specs]
    
    print(f"Running model...")
    mdl.minimize(p0=p0, **min_args)
    print("Model fitting complete.")
    
    if hasattr(mdl.results, "x"):
        p_final = mdl.results.x
    elif hasattr(mdl, "p_final"):
        p_final = mdl.p_final
    else:
        raise ValueError("Couldn't find final parameters in mdl.results.x or mdl.p_final.") 

    results = {"p_init": mdl.p0, "p_final": p_final, "mdl.results": mdl.results,
               "history": mdl.history, "restart_changes": mdl.restart_changes, "all_runs": mdl.all_runs}
 
    # For the training, test and validation data, compute the Cstar values.

    Z = mdl.get("Z", p_final)
    results["split"] = pack_split_results(XX, YY, Z, mdl.center)
    
    if not (return_dataset or return_model):
        return results
    else:
        ret_val = [results]
        if return_dataset:
            ret_val.append(dataset)
        if return_model:
            ret_val.append(mdl)

        return tuple(ret_val)


def load_models(base_dir):
    # Find all collected.p files in folders one level down from base_dir
    # If any of these are newer than loaded_models.p in the base_dir,
    # adds them to the left_to_load list. These are then loaded and update the loaded_models dictionary.
    

    preloaded_file = os.path.join(base_dir, "loaded_models.p")
    preloaded_exists = os.path.exists(preloaded_file)
    if preloaded_exists:
        print(f"Preloaded file {preloaded_file} exists")
        with open(preloaded_file, "rb") as f:
            loaded_models = pickle.load(f)
        print(f"Loaded preloaded models from {preloaded_file}")
        last_update = os.path.getmtime(preloaded_file)
        print(f"Preloaded file {preloaded_file} last updated at {last_update}")
    else:
        print(f"Preloaded file {preloaded_file} does not exist")
        last_update = 0
        loaded_models = {}

    # Find all subfolders
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    has_collected = [os.path.basename(s) for s in subfolders if os.path.exists(os.path.join(s, "collected.p"))]
    print(f"Found {len(has_collected)} subfolders with collected.p: "+ ", ".join([s for s in has_collected]))

    dirs_to_names = {v:k for k,v in pfm.subdirs.items()}
    is_model =[s for s in has_collected if s in dirs_to_names]
    print(f"Found {len(is_model)} subfolders that are known models: "+ ", ".join([f"{s}({dirs_to_names[s]})" for s in is_model]))

    # Find the collected.p that are more recent than loaded_models.p
    more_recent = [s for s in is_model if os.path.getmtime(os.path.join(base_dir, s, "collected.p")) > last_update]
    print(f"Found {len(more_recent)} subfolders with collected.p more recent than loaded_models.p: "+ ", ".join([s for s in more_recent]))

    left_to_load = [dirs_to_names[s] for s in more_recent]
    if left_to_load:
        print(f"Loading models: {left_to_load}")
        loaded_fields = pfm.load_models(base_dir, load_config_from_input=True, load_only=left_to_load)
        loaded_models.update(loaded_fields)

        with open(preloaded_file, "wb") as f:
            pickle.dump(loaded_models, f)

        print(f"Preloaded models saved to {preloaded_file}.")
    else:
        print("No new models to load.")

if __name__ == "__main__":
    # Load ArgumentParser and get arguments
    # The arguments should be:
    # A --gen flag and the name of a YAML file to use to generate run configurations.
    # A --run flag and the name of an input file to use to run a single configuration.
    # A --collect flag and the name of a directory to collect results from.
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen",          help="Generate run configurations from YAML file.",              type=str)
    parser.add_argument("--run",          help="Run a single configuration from an input pickle file.",    type=str, nargs="+")
    parser.add_argument("--collect",      help="Collect results from a directory.",                        type=str)
    parser.add_argument("--loadmodels",   help="Load models from  MODEL/collected.p found in the given base dir.", type=str)
    parser.add_argument("--inputfields",  help="Fields to include from the input pickle file.", nargs="+", type=str)
    parser.add_argument("--outputfields", help="Fields to include in the output pickle file.",  nargs="+", type=str)
    parser.add_argument("--min_method", help="Minimization method to override the one in the config file.", type=str)
    parser.add_argument("--match-file", dest="match_file", type=str,
                        help="Fit only the matched roi pairs in this csv. Overrides the yaml, "
                             "and puts the fits under matched=True.")
    parser.add_argument("--loss", type=str, choices=["cov", "resp"],
                        help="Loss to fit against. Overrides the yaml, and puts the fits under loss=<loss>.")
    args = parser.parse_args()
    
    # If the inputfields argument is not None, then set the inputfields to be an empty list.
    if args.inputfields is None:
        args.inputfields = []
    
    if args.outputfields is None:
        args.outputfields = []    
    
    # If the --gen flag is used, generate run configurations from a YAML file.
    if args.gen is not None:
        # Load the YAML file.
        with open(args.gen, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # The variant lives on the command line rather than in the yaml: it would
        # otherwise mean a near-duplicate yaml per (loss x matched) combination,
        # and the choice is already recorded on disk -- baked into every in.N.p
        # and visible in the fit path as loss=.../matched=...
        if args.match_file is not None:
            config["match_file"] = args.match_file
            print(f"Fitting the matched pairs in {args.match_file}.")
        if args.loss is not None:
            config.setdefault("init_args", {})["loss"] = args.loss
            print(f"Fitting against the {args.loss} loss.")
        # Paths are resolved here, at generation time, and the absolute result is
        # what gets pickled into every in.N.p -- run() only checks it still exists.
        for fld in ("data_file", "match_file"):
            if fld in config:
                config[fld] = config[fld].replace("$GLOM_IO_DATA", os.environ["GLOM_IO_DATA"])
                assert os.path.exists(config[fld]), f"{fld} {config[fld]} does not exist."

        required_fields = ["model", "normalization", "standardization", "sampler", "seeds", "init_args"]
        for field in required_fields:
            assert field in config, f"The '{field}' field must be in the YAML file."
            
        # Generate the run configurations.
        init_flds_expand = {}
        config["init_args"] = common.eval_fields(config["init_args"], context={"np":np})
        init_flds_expand = {fld[:-2]:val for fld,val in config["init_args"].items() if fld.endswith("__")}
        for fld in init_flds_expand:
            # Remove the expanded field.
            del config["init_args"][fld+"__"]
            print(f"Init field {fld}: {len(init_flds_expand[fld])} values from {init_flds_expand[fld][0]} to {init_flds_expand[fld][-1]}.")

        name       = os.path.splitext(args.gen)[0] if "name" not in config else config["name"] 
        new_dir    =  build_fit_dir(config=config, name=name)
        split_mode = get_split_mode(config)
        os.makedirs(new_dir, exist_ok=True)
        print(f"Created directory {new_dir}.")

        all_init_args = [None]            
        if len(init_flds_expand) > 0:
            all_init_args = []
            # For each combination of the values of the expanded fields,
            # create a new init_args dictionary by copying conifg["init_args"]
            # and adding the expanded fields and their values.
            flds = list(init_flds_expand.keys())
            vals = list(init_flds_expand.values())
            for fld_vals in itertools.product(*vals):
                init_args = copy.deepcopy(config["init_args"])
                for fld, val in zip(flds, fld_vals):
                    init_args[fld] = val
                all_init_args.append(init_args)                        
        else:
            all_init_args = [config["init_args"]]

        all_min_args = [config["min_args"]] if "min_args" in config else [None]

        base_config = {
            "sampler": config["sampler"],
            "model": config["model"],
            "normalization": config["normalization"],
            "standardization": config["standardization"]
        }

        for fld in ("data_file", "match_file"):
            if fld in config:
                base_config[fld] = config[fld]

        if split_mode == "outclass":
            classes = sorted(set(odours.classes))
            if 'control' in classes:
                classes.remove('control')
            variants = []
            for c in classes:
                v = copy.deepcopy(base_config)
                v["sampler"]["split"]["outclass"]=c
                variants.append(v)
        else:
            variants = [base_config]
            
        run_id = 0
        for variant in variants:
            for seed in range(config["seeds"]):
                # Create the run configuration.
                variant["seed"] = seed
                if variant["sampler"]["type"] in ["trials", "odours"]:
                    split_ods = gen_split(seed, variant["sampler"])
                    # Update base_config with split_ods
                    variant["sampler"]["split"].update(split_ods) # Fills in train_inds, test_inds, vld_inds if they are in split_ods

                for init_args in all_init_args:
                    for min_args in all_min_args:
                        new_config = copy.deepcopy(variant)

                        if init_args is not None: new_config["init_args"] = init_args
                        if min_args  is not None: new_config["min_args"]  = min_args

                        # Create a filename for the run configuration.
                        filename = f"in.{run_id}.p"
                        # Save the run configuration to the filename.
                        output_file = os.path.join(new_dir, filename)
                        with open(output_file, "wb") as f:
                            pickle.dump(new_config, f)
                        print(f"new_config: {new_config}")
                        print(f"Saved run configuration to {output_file}.")
                        run_id += 1

    elif args.run is not None:
        input_files = args.run
        print(f"Running {len(input_files)} inputs.")
        for in_file in input_files:
            # Load the input pickle file.
            print(f"Loading input file {in_file}.")
            with open(in_file, "rb") as f:
                config = pickle.load(f)

            if args.min_method is not None:
                config["min_args"] = {} if "min_args" not in config else config["min_args"]
                config["min_args"]["method"] = args.min_method
                print(f"Overriding minimization method to {args.min_method}.")
            # Run the configuration.
            print("Running configuration", config)
            results = run(config)
        
            # Save the results to a file with the same name as the input file, but with 'in' replaced by 'out'.
            output_file = in_file.replace("in.", "out.")
            assert output_file != in_file, "Input file name must contain 'in.'."
            with open(output_file, "wb") as f:
                pickle.dump({"config":config, "results":results}, f)
        
            print(f"Saved results to {output_file}.")
            print(f"ALLDONE")

    elif args.loadmodels is not None:
        loaddir = Path(args.loadmodels)
        assert loaddir.exists() and loaddir.is_dir(), f"{loaddir} does not exist or is not a directory."
        load_models(str(loaddir))
        print(f"ALLDONE") 
    elif args.collect is not None:
        # Iterate over all the pickle files in the directory that start with 'out'.
        # Load each one. 
        # Save the results to 'collected.p' in the directory.
        records = []
        # Only list the files in the directory that start witih "out"
        collect_dir = Path(args.collect) 
        filenames = [f.name for f in collect_dir.iterdir() if f.is_file() and f.name.startswith("out")]
        print(f"Collecting results from {len(filenames)} out files in {args.collect}.")
        for filename in tqdm(filenames):
            full_filename = os.path.join(args.collect, filename)
            with open(full_filename, "rb") as f:
                new_record = {"file":filename}
                data = pickle.load(f)                
                config = data["config"]
                # From the 'config' field, extract seed and any fields specified by --inputfields.
                new_record.update({k:config[k] for k in args.inputfields + ["seed"]})

                results = data["results"]
                new_record["split"] = results["split"]
                # Also extract any other fields specified by the 'outputfields' field of the args.
                new_record.update({k:results[k] for k in args.outputfields})
                records.append(new_record)
    # Save the results to 'collected.p' in the directory.
        with open(os.path.join(args.collect, "collected.p"), "wb") as f:
            pickle.dump(records, f)
        print(f"Saved results to {os.path.join(args.collect, 'collected.p')}.")
        print(f"ALLDONE")
    else:
        print("No arguments provided. Use --help for help.")
        
            
            
        
    
    
    
    
