import argparse
import yaml
import hashlib
import pickle
import os, sys
import numpy as np
import itertools, copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pdb

import common

def add_path_env_var(name):
    assert name in os.environ, f"Did not find environment variable {name}."
    path = os.environ[name]
    assert len(path), "Path to {name} is empty."
    assert os.path.exists(path), f"{path} does not exist"
    sys.path.append(path)

add_path_env_var("OB_IO_CONN_MODELS")
add_path_env_var("GLOM_IO_DATA")
    
from models.common import get_Cstar

from models.diag                    import Model as Diag
from models.diag_bg                 import Model as DiagBg
from models.diag_bg_cov             import Model as DiagBgCov
from models.diag_bg_cov2            import Model as DiagBgCov2
from models.diag_bg_rank1           import Model as DiagBgRank1
from models.diag_bg_rank1_uncon     import Model as DiagBgRank1Uncon
from models.diag_bg_rank1_sym_uncon import Model as DiagBgRank1SymUncon
from models.id_bg_rank1_uncon       import Model as IdBgRank1Uncon
from models.id_bg_rank1_sym_uncon   import Model as IdBgRank1SymUncon
from models.id_bg_cov               import Model as IdBgCov
from models.free                    import Model as Free
from models.free_sym                import Model as FreeSym
from models.free_asym               import Model as FreeAsym
from models.id_free_sym             import Model as IdFreeSym
from models.z_diag_rank_sym         import Model as ZDiagRankSym
from models.decorr                  import Model as Decorr
from models.inference               import Model as Inference

import models.free_gen as free_gen
from models.free_gen import Model as FreeGen

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
                "DiagBg"              :DiagBg,
                "DiagBgCov"           :DiagBgCov,
                "DiagBgCov2"          :DiagBgCov2,
                "DiagBgRank1"         :DiagBgRank1,
                "DiagBgRank1Uncon"    :DiagBgRank1Uncon,
                "DiagBgRank1SymUncon" :DiagBgRank1SymUncon,                    
                "IdBgRank1Uncon"      :IdBgRank1Uncon,
                "IdBgRank1SymUncon"   :IdBgRank1SymUncon,
                "IdBgCov"             :IdBgCov,
                "Free"                :Free,
                "FreeSym"             :FreeSym,
                "FreeAsym"            :FreeAsym,
                "IdFreeSym"           :IdFreeSym,
                "ZDiagRankSym"        :ZDiagRankSym,
                "Decorr"              :Decorr,
                "Inference"           :Inference,
                "FreeGen"             :FreeGen,
                }

def get_data(full=False, normalization="roi", standardization="train", average = False, data_file = None, by_odour=False, frac_test=0.2, frac_vld=0.2):
    # Use the directory of this file to find the data.
    if data_file is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(data_dir, "X0Y0_new.p")
        assert os.path.exists(data_file), f"{data_file} does not exist"

    print(f"Loading data from {data_file}")
    # Load X0 and Y0 from X0Y0.p
    with open(data_file, "rb") as f:
        data = pickle.load(f)
        X0 = data["X0"]
        Y0 = data["Y0"]       

    if full:
        return X0, Y0

    if average:
        X = np.concatenate([X0i.mean(axis=-1) for X0i in X0], axis=0)
        Y = np.concatenate([Y0i.mean(axis=-1) for Y0i in Y0], axis=0)
        Xtrain, Xtest, Xvld = [X, X, X]
        Ytrain, Ytest, Yvld = [Y, Y, Y]
    elif by_odour:
        print("Splitting data by odour.")
        Xtrain, Xtest, Xvld, train_inds, test_inds, vld_inds = common.split_by_odour(X0, frac_test=frac_test, frac_vld=frac_vld, return_inds = True)
        print(f"Split data into {len(train_inds)} training, {len(test_inds)} test and {len(vld_inds)} validation odours.")
        print(f"\tTraining   odours: {train_inds}")
        print(f"\tTest       odours: {test_inds}")
        print(f"\tValidation odours: {vld_inds}")
        Ytrain, Ytest, Yvld = common.split_by_odour(Y0, train_inds = train_inds, test_inds = test_inds, vld_inds = vld_inds, return_inds = False)
    else:
        Xtrain, Xtest, Xvld, *_ = common.split(X0)
        Ytrain, Ytest, Yvld, *_ = common.split(Y0)

    if standardization == "train":
        print("Standardizing training, test and validation sets based on training parameters.")
        XSS = [Xtrain] * 3
        YSS = [Ytrain] * 3
    elif standardization == "separate":
        print("Standardizing training, test and validation sets separately.")
        XSS = [Xtrain, Xtest, Xvld]
        YSS = [Ytrain, Ytest, Yvld]
    else:
        raise ValueError(f"Don't know what to do for standardization {standardization}.")

    if type(normalization) is str:
        normalization = [normalization] * 2

    assert type(normalization) is list and len(normalization) == 2, "normalization must be a list of length 2."

    XY_ss = []
    for norm_type, d_type, XYSS, XY in zip(normalization, ["input", "output"], [XSS, YSS], [[Xtrain, Xtest, Xvld], [Ytrain, Ytest, Yvld]]):
        if norm_type == "roi":
            print(f"Normalizing {d_type} by ROI.")
            SS = [StandardScaler().fit(XYi.T) for XYi in XYSS] 
            XY_ss.append([SSi.transform(Xi.T).T for SSi, Xi in zip(SS, XY)])
            #Xtrain_ss, Xtest_ss, Xvld_ss = [SSXi.transform(Xi.T).T for SSXi, Xi in zip(SSX, [Xtrain, Xtest, Xvld])]
            #SSY = [StandardScaler().fit(Yi.T) for Yi in YSS]
            #Ytrain_ss, Ytest_ss, Yvld_ss = [SSYi.transform(Yi.T).T for SSYi, Yi in zip(SSY, [Ytrain, Ytest, Yvld])]
        elif norm_type == "odour":
            print(f"Normalizing {d_type} by odour.")
            SS = [StandardScaler().fit(XYi) for XYi in XYSS]
            XY_ss.append([SSi.transform(Xi) for SSi, Xi in zip(SS, XY)])
            #SSX = [StandardScaler().fit(Xi) for Xi in XSS]
            #Xtrain_ss, Xtest_ss, Xvld_ss = [SSXi.transform(Xi) for SSXi, Xi in zip(SSX, [Xtrain, Xtest, Xvld])]
            #SSY = [StandardScaler().fit(Yi) for Yi in YSS]
            #Ytrain_ss, Ytest_ss, Yvld_ss = [SSYi.transform(Yi) for SSYi, Yi in zip(SSY, [Ytrain, Ytest, Yvld])]
        elif norm_type == "std":
            print(f"Normalizing {d_type} by overall standard deviation.")
            SS = [OverallStdScaler().fit(XYi) for XYi in XYSS]
            XY_ss.append([SSi.transform(Xi) for SSi, Xi in zip(SS, XY)])
            #SSX = [OverallStdScaler().fit(Xi) for Xi in XSS]
            #Xtrain_ss, Xtest_ss, Xvld_ss = [SSXi.transform(Xi) for SSXi, Xi in zip(SSX, [Xtrain, Xtest, Xvld])]
            #SSY = [OverallStdScaler().fit(Yi) for Yi in YSS]
            #Ytrain_ss, Ytest_ss, Yvld_ss = [SSYi.transform(Yi) for SSYi, Yi in zip(SSY, [Ytrain, Ytest, Yvld])]        
        elif norm_type == "none":
            print(f"Not normalizing {d_type}.")
            XY_ss.append(XY)
            #Xtrain_ss, Xtest_ss, Xvld_ss = XSS
            #Ytrain_ss, Ytest_ss, Yvld_ss = YSS
        else:
            raise ValueError(f"Don't know to do normalization {normalization} for {d_type}.")

    Xtrain_ss, Xtest_ss, Xvld_ss = XY_ss[0]
    Ytrain_ss, Ytest_ss, Yvld_ss = XY_ss[1]
        
    return (Xtrain_ss, Ytrain_ss) if average else (Xtrain_ss, Xtest_ss, Xvld_ss, Ytrain_ss, Ytest_ss, Yvld_ss)                                                      
    
def run(config, X=None, Y=None, return_dataset = False, return_model = False):
    if (X is None and Y is not None) or (X is not None and Y is None):
        raise ValueError("Either both X and Y should be provided, or neither should be provided.")
 
    # Get the seed, λ and trial from the config.
    seed  = config["seed"]
    λ     = config["λ"] if "λ" in config else None
    trial = config["trial"]
    normalization = config["normalization"]
    standardization = config["standardization"]
    data_file = config["data_file"] if "data_file" in config else None
    if data_file is not None:
        data_file.replace("$GLOM_IO_DATA", os.environ["GLOM_IO_DATA"])
        assert os.path.exists(data_file), f"Data file {data_file} does not exist."
    
    # Set the seed.
    np.random.seed(seed)

    # Get the data.
    if X is None and Y is None:
        Xtrain, Xtest, Xvld, Ytrain, Ytest, Yvld = get_data(normalization=normalization, standardization=standardization, data_file=data_file)
        dataset = [("train", Xtrain, Ytrain), ("test", Xtest, Ytest), ("vld", Xvld, Yvld)]
    else:
        Xtrain, Ytrain = X, Y
        dataset = [("train", Xtrain, Ytrain)]
        
    n_cells = Xtrain.shape[0]

    if config["model"] not in known_models:
        raise ValueError(f"Don't know what to do for model {config['model']}.")

    Model               = known_models[config["model"]]

    context = {"np":np,
               "n_cells":n_cells}

    init_args = common.eval_fields(config["init_args"], context=context) if "init_args" in config else {}
    #init_args, min_args = [common.eval_fields(config[name], context=context) if name in config else {} for name in ["init_args", "min_args"]]
    print(f"{init_args=:}")

    if λ is not None: init_args["λ"] = λ

    if "parameterization" in config:
        assert config["model"] == "FreeGen", "Parameterization is only supported for FreeGen model."
        param_class = config["parameterization"]["class"]
        param_args  = common.eval_fields(config["parameterization"].copy(), context=context)
        del param_args["class"]
        if param_class == "Diag":
            P = free_gen.Diag(n_cells, **param_args)
        elif param_class == "DiagRankKSym":
            P = free_gen.DiagRankKSym(n_cells, **param_args)
        else:
            raise ValueError(f"Don't know what to do for parameterization class {param_class}.")
        init_args["ZFUN"] = P.ZFUN
        init_args["p0_fun"] = P.p0
        
    mdl = Model(Xtrain, Ytrain, **init_args)

    context["mdl"] = mdl
    min_args = common.eval_fields(config["min_args"], context=context) if "min_args" in config else {}

    mdl.minimize(**min_args)
    if hasattr(mdl.results, "x"):
        p_final = mdl.results.x
    elif hasattr(mdl, "p_final"):
        p_final = mdl.p_final
    else:
        raise ValueError("Couldn't find final parameters in mdl.results.x or mdl.p_final.") 

    Cstar_fun = lambda Y: mdl.get_Cstar(Y) if config["model"].endswith("Gen") else get_Cstar(Y, mdl.center)
    
    results = {"p_init": mdl.p0, "p_final": p_final, "mdl.results": mdl.results}
   
    # For the training, test and validation data, compute the Cstar values.
    for name, X, Y in dataset: 
        Cin   = Cstar_fun(X)
        Cstar = Cstar_fun(Y)
        # Compute the estimated Cstar.
        Y_est = mdl.ZFUN(p_final) @ X
        Cest  = Cstar_fun(Y_est)
        results[name]={"Cstar": Cstar, "Cest": Cest, "Cin": Cin,
                       "X_hash": array_fingerprint(X),
                       "Y_hash": array_fingerprint(Y),
                       "Y_est_hash": array_fingerprint(Y_est),
                       }

    if not (return_dataset or return_model):
        return results
    else:
        ret_val = [results]
        if return_dataset:
            ret_val.append(dataset)
        if return_model:
            ret_val.append(mdl)

        return tuple(ret_val)

        

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
    parser.add_argument("--inputfields",  help="Fields to include from the input pickle file.", nargs="+", type=str)
    parser.add_argument("--outputfields", help="Fields to include in the output pickle file.",  nargs="+", type=str)
    parser.add_argument("--min_method", help="Minimization method to override the one in the config file.", type=str)
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
        if "data_file" in config:
            config["data_file"] = config["data_file"].replace("$GLOM_IO_DATA", os.environ["GLOM_IO_DATA"])
            assert os.path.exists(config["data_file"]), f"Data file {config['data_file']} does not exist."
    
        # Generate the run configurations.
        init_flds_expand = {}
        assert "init_args" in config, "The 'init_args' field must be in the YAML file."        
        config["init_args"] = common.eval_fields(config["init_args"], context={"np":np})
        init_flds_expand = {fld[:-2]:val for fld,val in config["init_args"].items() if fld.endswith("__")}
        for fld in init_flds_expand:
            # Remove the expanded field.
            del config["init_args"][fld+"__"]
            print(f"Init field {fld}: {len(init_flds_expand[fld])} values from {init_flds_expand[fld][0]} to {init_flds_expand[fld][-1]}.")

        center = config["init_args"]["center"] if "center" in config["init_args"] else None
            
        # Create a directory with the same name as the YAML file, but without the extension.
        norm_val = config["normalization"]
        # norm_val can be a list so convert it to a string if needed.
        if isinstance(norm_val, list):
            norm_val = "_".join([str(n) for n in norm_val])
        new_dir = f"fits/center={center}/standardization={config['standardization']}/normalization={norm_val}/{os.path.splitext(args.gen)[0]}"
        os.makedirs(new_dir, exist_ok=True)
        print(f"Created directory {new_dir}.")


        run_id = 0
        for trial in range(config["n_trials"]):
            seed = trial
            # Create the run configuration.
            base_config = {"trial": trial, "seed": seed, "model": config["model"], "normalization": config["normalization"], "standardization": config["standardization"]}
            if "data_file" in config:
                base_config["data_file"] = config["data_file"]
            
            all_init_args = [None]            
            if "init_args" in config:
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

            all_min_args = [None]
            if "min_args"  in config:
                all_min_args = [config["min_args"]]

            for init_args in all_init_args:
                for min_args in all_min_args:
                    new_config = copy.deepcopy(base_config)
                    if init_args is not None: new_config["init_args"] = init_args
                    if min_args  is not None: new_config["min_args"]  = min_args
                    if "parameterization" in config:
                        new_config["parameterization"] = config["parameterization"]
                        
                    # Create a filename for the run configuration.
                    filename = f"in.{run_id}.p"
                    # Save the run configuration to the filename.
                    output_file = os.path.join(new_dir, filename)
                    with open(output_file, "wb") as f:
                        pickle.dump(new_config, f)
                    print(f"new_config: {new_config}")
                    print(f"Saved run configuration to {output_file}.")
                    # Increment the run_id.
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
    elif args.collect is not None:
        # Iterate over all the pickle files in the directory that start with 'out'.
        # Load each one. 
        # Save the results to 'collected.p' in the directory.
        records = []
        for filename in os.listdir(args.collect):
            if filename.startswith("out"):
                full_filename = os.path.join(args.collect, filename)
                with open(full_filename, "rb") as f:
                    new_record = {"file":filename}
                    data = pickle.load(f)                
                    config = data["config"]
                    # From the 'config' field, extract the fields "seed", "λ", "trial" and any others specified by the 'inputfields' field of the args.
                    new_record.update({k:config[k] for k in args.inputfields + ["seed", "trial"]})
                    
                    results = data["results"]
                    # For each of the fields 'train', 'test' and 'vld', extract the fields "Cstar" and "Cest".
                    for name in ["train", "test", "vld"]:
                        new_record.update({f"{name}_{k}":results[name][k] for k in ["Cstar", "Cest", "Cin"]})                
                    # Also extract any other fields specified by the 'outputfields' field of the args.
                    new_record.update({k:results[k] for k in args.outputfields})
                    records.append(new_record)
        # Save the results to 'collected.p' in the directory.
        with open(os.path.join(args.collect, "collected.p"), "wb") as f:
            pickle.dump(records, f)
        print(f"Saved results to {os.path.join(args.collect, 'collected.p')}.")
    else:
        print("No arguments provided. Use --help for help.")
        
            
            
        
    
    
    
    
