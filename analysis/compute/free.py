import os, sys
import pickle
import numpy as np

from types import SimpleNamespace

from .compute import Computation
from .compute import paths
from .compute import get_Cstar
from .compute import pfm
from .compute import center, standardization, normalization
from .compute import conn2_driver
from .compute import compute_corr_energy

from ob_io_conn_models.models import free as Free 

def compute_props(Xtrn, Ytrn, Z, best_la):
    n_input, n_odours = Xtrn.shape
    assert np.allclose(Xtrn.mean(axis=0), 0), "Input data should be mean subtracted per odour."

    Z = Z.reshape(n_input, n_input)
    
    CY = np.cov(Ytrn.T, bias=True) * Ytrn.shape[0]
    VY, s, _ = np.linalg.svd(CY)
    S = np.diag(s)

    X = Xtrn @ VY
    Ux, sx, Vxt = np.linalg.svd(X)

    E0 = S - X.T @ X
    la = best_la * n_odours**2/n_input**2

    W  = Z - np.eye(Z.shape[0])
    Uw, sw, Vwt = np.linalg.svd(W)

    W_ = 2 * X @ E0 @ X.T / la
    Uw_, sw_, Vwt_ = np.linalg.svd(W_)

    UwtUw_ = Uw[:,:3].T @ Uw_
    UwtUx  = Uw[:,:3].T @ Ux
    ind_max_w  = np.argmax(np.abs(UwtUw_), axis=1)
    ind_max_x  = np.argmax(np.abs(UwtUx),  axis=1)
    val_max_w  = np.max(np.abs(UwtUw_), axis=1)
    val_max_x  = np.max(np.abs(UwtUx),  axis=1)
    return SimpleNamespace(
        Z=Z, # The gains applied
        Xtrain = Xtrn,
        Ytrain = Ytrn, # Training data
        X = X,
        best_la = best_la, # The best lambda found for this model
        W_ = W_, 
        la = la, # The lambda used for the Z_ values
        ind_max_w = ind_max_w, # The index of the maximum overlap with each mode of W
        ind_max_x = ind_max_x, # The index of the maximum overlap with each mode of X 
        val_max_w = val_max_w, # The value of the maximum overlap 
        val_max_x = val_max_x, # The value of the maximum overlap
    )

class Data(Computation):
    def __init__(self, *args, k=2, λ=[3.2e6], center=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.props_vals = []
        self.corr_energy = {"train":[], "test":[], "vld":[]}
        self.corrs = {fld:{"model-out": []} for fld in ["r2", "pearson", "spearman"]}

    def compute(self,
                best_stat="r2",
                best_stat_includes_diag=True,
                corr_stats_include_diag={"model-out":True},
                ):

        corrs_fields = set(self.corrs["r2"].keys())
        include_diag_fields = set(corr_stats_include_diag.keys())
        assert corrs_fields == include_diag_fields, f"Fields in corrs and corr_stats_include_diag must match. Found {corrs_fields} and {include_diag_fields}."
         
        print(f"COMPUTING free.Data.")

        norm_str = "_".join(normalization) if isinstance(normalization, (list, tuple)) else normalization
        self.base_dir = os.path.join(paths.fits_root, "fits", f"{center=}", f"standardization={standardization}", f"normalization={norm_str}")
        print(f"Loading models from {self.base_dir=}") 
        self.models = pfm.load_models(self.base_dir, load_config_from_input=True, load_only = ["Free"], stats_include_diag = best_stat_includes_diag)  

        pfm.compute_best_params(self.models, best_stat=best_stat)
        best_la = self.models["Free"]["best_params"]
        self.best_la = best_la
        df = self.models["Free"]["df"]

        fits_dir = os.path.join(self.base_dir, pfm.subdirs["Free"])
        assert os.path.exists(fits_dir), f"Fits dir {fits_dir} does not exist."

        corr_funs = {"r2":pfm.r2_fun, "pearson":pfm.pearson_fun, "spearman":pfm.spearman_fun}

        n_trials = self.models["Free"]["config"]["n_trials"]

        for trial in range(n_trials):
            # Input file is the field called "file" that matches the trial and la=best_la
            in_file = os.path.join(fits_dir, df[(df["trial"]==trial) & (df["λ"]==best_la)]["file"].values[0])
            with open(in_file, "rb") as f:
                in_data = pickle.load(f)
 
            out_file = in_file.replace("in.", "out.")
            with open(out_file, "rb") as f:
                out_data = pickle.load(f)
                print(f"Loaded outputs from {out_file=}")

            assert in_data["normalization"] == normalization, f"Normalization mismatch: {in_data['normalization']} vs {normalization}"
            assert in_data["standardization"] == standardization, f"Standardization mismatch: {in_data['standardization']} vs {standardization}"
            assert in_data["trial"] == trial, f"Trial mismatch: {in_data['trial']} vs {trial}"
            assert in_data["model"] == "Free", f"Model mismatch: {in_data['model']} vs Free"

            seed = in_data["seed"]
            data_file = in_data["data_file"]
            np.random.seed(seed)
            Xtrain, Xtest, Xvld, Ytrain, Ytest, Yvld = conn2_driver.get_data(normalization=normalization, standardization=standardization, data_file=data_file)
            
            results = out_data["results"]
            # out_data["results"] has fields train, test, vld that have X_hash, Y_hash
            # Check those against conn2_driver.array_fingerprint(Xtrain), etc
            for fld, X, Y in zip(["train", "test", "vld"], [Xtrain, Xtest, Xvld], [Ytrain, Ytest, Yvld]):
                X_hash = conn2_driver.array_fingerprint(X)
                Y_hash = conn2_driver.array_fingerprint(Y)
                assert results[fld]["X_hash"] == X_hash, f"X hash mismatch for {fld} in trial {trial}"
                assert results[fld]["Y_hash"] == Y_hash, f"Y hash mismatch for {fld} in trial {trial}"

            for corr_type in ["r2", "pearson", "spearman"]:
                corr_fun = corr_funs[corr_type]
                for fld, include_diag in corr_stats_include_diag.items():
                    if fld == "in-in":
                        C1, C2 = results["test"]["Cin"],   results["vld"]["Cin"]
                    elif fld == "out-out":
                        C1, C2 = results["test"]["Cstar"], results["vld"]["Cstar"]
                    elif fld == "in-out":
                        C1, C2 = results["test"]["Cstar"], results["test"]["Cin"]
                    elif fld == "model-out":
                        C1, C2 = results["vld"]["Cstar"],  results["vld"]["Cest"]
                    else:
                        raise ValueError(f"Unknown field {fld}")

                    compute_corr = lambda *args, **kwargs: pfm.compute_corr(corr_fun, *args, include_diag = include_diag, **kwargs)

                    self.corrs[corr_type][fld].append(compute_corr(C1, C2))

            print(f"Computing props for trial {trial} with {best_la=:.3g}")
            self.props_vals.append(compute_props(Xtrain, Ytrain, results["p_final"], best_la))

            if trial == 0:
                print(f"RUNNING MODEL ON TRIAL-AVERAGED DATA WITH {best_la=:.3g}")
                # Run the model on the trial-averaged data, using the best regulization value we found
                Xtrn, Ytrn    = conn2_driver.get_data(normalization=normalization, standardization=standardization, data_file=data_file, average=True)
                n_inputs, n_odours = Xtrn.shape
                use_la = best_la * 1
                in_data["init_args"]["λ"] = [use_la]
                self.results, mdl = conn2_driver.run(in_data, Xtrn, Ytrn, return_model = True)
                self.Z = self.results["p_final"].reshape(Xtrn.shape[0], Xtrn.shape[0])
                self.la = use_la * (n_odours**2) / (n_inputs**2)

                CY = np.cov(Ytrn.T, bias=True) * Ytrn.shape[0]
                VY, s, _ = np.linalg.svd(CY)
                S = np.diag(s)
                X = Xtrn @ VY
                E0 = S - X.T @ X
                self.Z_ = np.eye(X.shape[0]) + 2 * X @ E0 @ X.T / self.la
                self.X  = X
                self.S  = S
                self.in_data = in_data
                normalizer = lambda X: X / np.sqrt(np.diag(X)[:, None] * np.diag(X)[None, :])

                self.Rep_out = normalizer(self.results["train"]["Cstar"])
                self.Rep_est = normalizer(self.results["train"]["Cest"])
                
            Z = mdl.ZFUN(self.results["p_final"])
            E = [Z @ X for X in [Xtrain, Xtest, Xvld]]

            self.corr_energy["train"].append([compute_corr_energy(U) for U in [Xtrain, Ytrain, E[0]]])
            self.corr_energy["test"].append([compute_corr_energy(U) for U in [Xtest, Ytest, E[1]]])
            self.corr_energy["vld"].append([compute_corr_energy(U) for U in [Xvld, Yvld, E[2]]])

        self.corr_energy = {fld: np.array(vals) for fld, vals in self.corr_energy.items()}

        self.computed = True
        return self
