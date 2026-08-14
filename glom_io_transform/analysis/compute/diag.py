import os, sys
import pickle
import numpy as np

from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr

from .compute import Computation
from .compute import paths
from .compute import get_Cstar
from .compute import pfm
from .compute import center, standardization, normalization
from .compute import conn2_driver
from .compute import compute_corr_energy

from ob_io_conn_models.models import diag as Diag 

def compute_props(Xtrn, Ytrn, z, best_la):
    n_input, n_odours = Xtrn.shape

    CY = np.cov(Ytrn.T, bias=True) * Ytrn.shape[0]

    VY,s,_ =np.linalg.svd(CY)
    S = np.diag(s)

    assert np.allclose(Xtrn.mean(axis=0), 0), "Input data should be mean subtracted per odour."
    X = Xtrn @ VY
    Ux, Sx, Vxt = np.linalg.svd(X, full_matrices=False); Vx = Vxt.T; assert np.allclose(Ux @ np.diag(Sx) @ Vx.T, X)

    la = best_la * n_odours**2/n_input
    
    E0 = S - X.T @ X # J X = X since data are mean subtracted per odour
    r  = E0.flatten()
    
    G  = [(2 * Xi.reshape(-1,1) @ Xi.reshape(1,-1)).flatten() for Xi in X]
    G = np.array(G).T
    GtG = G.T @ G
    H   = GtG + la * np.eye(GtG.shape[0])
    a   = G.T @ r
    delta_z = np.linalg.solve(H, a)

    gg = np.diag(GtG)
    b = GtG @ delta_z - gg * delta_z

    assert np.allclose(delta_z, (a - b)/(gg + la)), "Delta_z should match the closed form solution for the diagonal case."

    b_est = np.sum(GtG,axis=1) - gg
    k = np.cov(b_est, b)[0,1] / np.var(b_est)
    k = -1./4
    delta_z_est0 = a/(gg + la)
    delta_z_est = (a - b_est * k) / (gg + la)
    
    return SimpleNamespace(z=z, # The gains applied
                           Xtrain = Xtrn, Ytrain = Ytrn, # Training data
                           a = a, # The a values for each input neuron
                           b = b, # The true b values for each input neuron
                           b_est = b_est, # The estimated b values for each input neuron
                           delta_z = delta_z, # z - 1 for the linearlization
                           k = k,
                           delta_z_est  = delta_z_est,
                           delta_z_est0 = delta_z_est0,
                           la = la, # The estimated z - 1 for the linearizations
                           G=G,
                           r=r,
                           ) 

class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.props_vals = []
        self.corr_energy = {"train":[], "test":[], "vld":[]} 
        self.corrs = {fld:{"in-in": [], "out-out": [], "in-out": [], "model-out": []} for fld in ["r2", "pearson", "spearman"]}
        
    def compute(self, best_stat="r2", best_stat_includes_diag=True,
                corr_stats_include_diag = {"in-in":False, # Will be pearson correlations, so don't include the diagonal.
                                           "in-out":False, # Both in and out will be pearson correlations, so don't include the diagonal.
                                           "out-out":False, # Will be pearson correlations, so don't include the diagonal.
                                           "model-out":True, # Model will be a covariance fit, so include the diagonal.
                                           }):
        """
        best_stat: Which statistic to use when comparing outputs to predictions.
        best_stat_include_diag: Whether to use the diagonal when computing the best stat.
        corr_stats_include_diag: Whether to use the diagonal when computing the self correlations.
        """

        corrs_fields = set(self.corrs["r2"].keys())
        include_diag_fields = set(corr_stats_include_diag.keys())
        assert corrs_fields == include_diag_fields, f"Fields in corrs and corr_stats_include_diag must match. Found {corrs_fields} and {include_diag_fields}."
        
        print("COMPUTING diag.Data.")
        
        norm_str = "_".join(normalization) if isinstance(normalization, (list, tuple)) else normalization
        self.base_dir = os.path.join(paths.fits_root, "fits", f"{center=}", f"standardization={standardization}", f"normalization={norm_str}")
        print(f"Loading models from {self.base_dir=}") 
        self.models = pfm.load_models(self.base_dir, load_config_from_input=True, load_only = ["Diag", "Free"], stats_include_diag = best_stat_includes_diag)  

        pfm.compute_best_params(self.models, best_stat=best_stat)
        best_la = self.models["Diag"]["best_params"]
        self.best_la = best_la
        df = self.models["Diag"]["df"]

        fits_dir = os.path.join(self.base_dir, pfm.subdirs["Diag"])
        assert os.path.exists(fits_dir), f"Fits dir {fits_dir} does not exist."

        corr_funs = {"r2":pfm.r2_fun, "pearson":pfm.pearson_fun, "spearman":pfm.spearman_fun}

        n_trials = self.models["Diag"]["config"]["n_trials"]
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
            assert in_data["model"] == "Diag", f"Model mismatch: {in_data['model']} vs Diag"

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

            self.props_vals.append(compute_props(Xtrain, Ytrain, results["p_final"], best_la))

            if trial == 0:
                print(f"RUNNING MODEL ON TRIAL-AVERAGED DATA WITH {best_la=:.3g}")
                # Run the model on the trial-averaged data, using the best regulization value we found
                X, Y    = conn2_driver.get_data(normalization=normalization, standardization=standardization, data_file=data_file, average=True)
                self.results, mdl = conn2_driver.run(in_data, X, Y, return_model = True)
                self.in_data = in_data
                normalizer = lambda X: X / np.sqrt(np.diag(X)[:, None] * np.diag(X)[None, :])

                self.Rep_out = normalizer(self.results["train"]["Cstar"])
                self.Rep_est = normalizer(self.results["train"]["Cest"])
                
                n_odours = X.shape[1]
                
                n_inputs, n_odours = X.shape
                self.la = self.best_la * (n_odours**2) / n_inputs

            Z = mdl.ZFUN(self.results["p_final"])
            E = [Z @ X for X in [Xtrain, Xtest, Xvld]]

            self.corr_energy["train"].append([compute_corr_energy(U) for U in [Xtrain, Ytrain, E[0]]])
            self.corr_energy["test"].append([compute_corr_energy(U) for U in [Xtest, Ytest, E[1]]])
            self.corr_energy["vld"].append([compute_corr_energy(U) for U in [Xvld, Yvld, E[2]]])

        self.corr_energy = {fld: np.array(vals) for fld, vals in self.corr_energy.items()}
            
        self.computed = True
        return self
    
