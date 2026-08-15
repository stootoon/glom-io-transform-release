import os, sys
import pickle
import numpy as np

from types import SimpleNamespace

from .compute import Computation
from .compute import paths
from .compute import get_Cstar
from .compute import driver

import glom_io_transform.model_fitting.results as results


def compute_props(Xs, Ys, Z, la0, center=True):
    """Compare the fitted connectivity to the theory prediction.

    The fit minimizes mean_k mean((Cstar_k - C_k)^2)/2 + la0 * mean((Z-I)^2)/2.
    Linearizing around Z = I gives the predicted connectivity perturbation
        W_ = 2 X E0 X^T / la,   la = la0 * n_odours^2 / n_cells^2,
    where E0 is the residual of the *average* observed covariance at Z = I.
    Fitting the K (X,Y) pairs jointly corresponds to fitting the average
    covariance, so E0 uses mean_k Cstar_k; the inputs X are identical across
    pairs.
    """
    X0 = Xs[0]
    assert all(np.allclose(Xk, X0) for Xk in Xs), "Expected identical inputs across (X,Y) pairs."
    n_cells, n_odours = X0.shape
    assert np.allclose(X0.mean(axis=0), 0), "Input data should be mean subtracted per odour."

    Z = Z.reshape(n_cells, n_cells)

    # Average observed covariance across the (X,Y) pairs, in its eigenbasis
    CY = np.mean([get_Cstar(Yk, center) for Yk in Ys], axis=0)
    VY, s, _ = np.linalg.svd(CY)
    S = np.diag(s)

    X = X0 @ VY
    Ux, sx, Vxt = np.linalg.svd(X)

    E0 = S - X.T @ X
    la = la0 * n_odours**2 / n_cells**2

    W  = Z - np.eye(n_cells)
    Uw, sw, Vwt = np.linalg.svd(W)

    W_ = 2 * X @ E0 @ X.T / la
    Uw_, sw_, Vwt_ = np.linalg.svd(W_)

    UwtUw_ = Uw[:, :3].T @ Uw_
    UwtUx  = Uw[:, :3].T @ Ux
    ind_max_w = np.argmax(np.abs(UwtUw_), axis=1)
    ind_max_x = np.argmax(np.abs(UwtUx),  axis=1)
    val_max_w = np.max(np.abs(UwtUw_), axis=1)
    val_max_x = np.max(np.abs(UwtUx),  axis=1)
    return SimpleNamespace(
        Z=Z,               # The fitted gains/connectivity for this seed
        X=X,               # Inputs rotated into the eigenbasis of the mean observed covariance
        la0=la0,           # The best lambda found for this seed
        la=la,             # The rescaled lambda used in the theory prediction
        W_=W_,             # The theory-predicted connectivity perturbation
        ind_max_w=ind_max_w,  # Index of maximum overlap with each mode of W
        ind_max_x=ind_max_x,  # Index of maximum overlap with each mode of X
        val_max_w=val_max_w,  # Value of the maximum overlap
        val_max_x=val_max_x,  # Value of the maximum overlap
    )


def compute_diag_quartic(config, n_train=10, gtol=1e-8):
    """Refit the Diag model and compute the per-unit quartic geometry.

    The fit is re-run (rather than loaded) because the phase-plane analysis
    needs tightly-converged gains (gtol) so each unit sits at a minimum of its
    quartic loss. Returns everything the phase-plane and approximation panels
    consume.
    """
    from numpy.lib.scimath import sqrt as csqrt

    config = dict(config)
    config["sampler"] = dict(config["sampler"])
    config["sampler"]["n_train"] = n_train
    config.setdefault("min_args", {}).setdefault("options", {})["gtol"] = gtol

    res, dataset, mdl = driver.run(config, return_dataset=True, return_model=True)

    Q  = mdl.compute_quartic_coefs()
    xn = np.linalg.norm(mdl.Xs[0], axis=1)   # channel powers |x_i|
    Xn = np.linalg.norm(Q.X, axis=1)

    # Geometry: centering shift, angle to the population mean, cubic parameterization
    XU       = Q.X / xn[:, None]
    mu       = Q.mi_ / (Q.mi_.shape[0] - 1)
    mu_n     = mu / np.linalg.norm(mu, axis=1)[:, None]
    c        = np.sum(XU * mu, axis=1)              # centering shift, in scaled units
    cos_th_i = np.sum(XU * mu_n, axis=1)
    K = 2 * csqrt(-Q.g_ / 3)
    u = -4 * Q.h_ / K**3

    # Scaled phase-plane coordinates and region masks
    zz = Q.z * xn
    gg =  Q.g_ * Xn**2
    hh = -Q.h_ * Xn**3
    in_W = (gg < 0) & (hh**2 < (4/27) * np.abs(gg)**3)
    in_J = (gg < 0) & (~in_W)
    in_U = (gg > 0)

    # Exact per-region roots and their one-parameter approximations (scaled units)
    with np.errstate(invalid="ignore"):
        fits = {
            "W": {
                "full":   np.sqrt(np.abs(Q.g_)) * xn * np.sign(-Q.h_),
                "approx": -np.sign(Q.h_) * np.sqrt(np.abs(Q.g_)) * xn,
                "approx_tex": r"$-\mathrm{sign}(\text{Tilt})\; \sqrt{|\text{Redundancy}|}$",
            },
            "J": {
                "full":        np.real(np.cosh(1/3 * np.arccosh(np.abs(u))) * K) * np.sign(-Q.h_) * xn,
                "approx_low":  2/np.sqrt(3) * np.sign(-Q.h_) * np.sqrt(np.abs(Q.g_)) * xn,
                "approx_high": -np.sign(Q.h_) * np.abs(Q.h_)**(1/3) * xn,
                "approx_tex":  r"$-\mathrm{sign}(\text{Tilt}) \; \sqrt[3]{|\text{Tilt}|} $",
            },
            "U": {
                "full":        np.imag(K) * np.sinh(1/3 * np.arcsinh(np.imag(u))) * xn,
                "approx_low":  -np.sign(Q.h_) * (np.abs(Q.h_) / np.abs(Q.g_)) * xn,
                "approx_high": -np.sign(Q.h_) * np.abs(Q.h_)**(1/3) * xn,
                "approx_tex":  r"$-\mathrm{sign}(\text{Tilt})\; \sqrt[3]{|\text{Tilt}|}$",
            },
        }
    fits["J"]["approx"] = fits["J"]["approx_high"].copy()
    fits["U"]["approx"] = fits["U"]["approx_high"].copy()

    return SimpleNamespace(
        Q=Q, xn=xn, Xn=Xn,
        c=c, cos_th_i=cos_th_i, K=K, u=u,
        zz=zz, gg=gg, hh=hh,
        in_W=in_W, in_J=in_J, in_U=in_U,
        fits=fits,
        config=config,
    )


class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.props_vals = []

    def _seed_config(self, model, seed, la, expect_model="Free"):
        """Load the in.N.p run config for the given seed and lambda."""
        sel = (model.df["seed"] == seed) & (model.df["λ"] == la)
        files = model.df[sel]["file"].unique()
        assert len(files) == 1, f"Expected exactly one input file for {seed=}, λ={la}, found {len(files)}."
        with open(os.path.join(model.base_dir, files[0]), "rb") as f:
            config = pickle.load(f)
        assert config["seed"] == seed, f"Seed mismatch: {config['seed']} vs {seed}"
        assert config["model"] == expect_model, f"Model mismatch: {config['model']} vs {expect_model}"
        # driver.run reads data_file from the config and asserts it exists;
        # drop stale paths so it falls back to the $GLOM_IO_DATA default.
        data_file = config.get("data_file")
        if data_file is not None and not os.path.exists(data_file):
            print(f"Data file {data_file} not found; falling back to $GLOM_IO_DATA default.")
            config.pop("data_file")
        return config

    def _seed_data(self, config):
        """Regenerate the (X,Y) splits used for this run from its config."""
        data_file = config.get("data_file")
        if data_file is not None and not os.path.exists(data_file):
            print(f"Data file {data_file} not found; falling back to $GLOM_IO_DATA default.")
            data_file = None
        return driver.get_data(normalization=config["normalization"],
                               standardization=config["standardization"],
                               data_file=data_file,
                               seed=config["seed"],
                               sampler=config["sampler"])

    def compute(self,
                selection_metric="ratio",
                seeds=None,
                ref_seed=0,
                ref_train=0,
                diag_seed=0,
                diag_la=None,
                diag_n_train=10,
                diag_gtol=1e-8,
                ):
        print("COMPUTING explain_models.Data (Free half).")

        models_dir = paths.proj_path + "/model_fitting"
        base = results.BaseContext(fits_root=os.path.join(models_dir, "fits"),
                                   models_dir=models_dir,
                                   standardization="separate",
                                   normalization="odour_std",
                                   center=True)
        split = base.split("trials", "random", "max")
        model = split.model("Free")
        self.model = model

        if seeds is None:
            seeds = sorted(model.df["seed"].unique())

        print(f"Computing props for seeds: {seeds} with selection_metric={selection_metric}, ref_seed={ref_seed}, ref_train={ref_train}")

        self.props_vals = []
        for i, seed in enumerate(seeds):
            print(f"Processing seed {seed} ({i+1}/{len(seeds)})")
            res = model.extract(seed=seed, train=ref_train, metric=selection_metric, with_params=True)
            config = self._seed_config(model, seed, res.la)
            XX, YY = self._seed_data(config)

            n_cells = XX.trains[0].shape[0]
            Z = res.params["p_final"].reshape(n_cells, n_cells)

            print(f"Computing props for {seed=} with best λ={res.la:.3g}")
            props = compute_props(XX.trains, YY.trains, Z, res.la, center=base.center)
            self.props_vals.append(props)

            if seed == ref_seed:
                # Reference fit shown in the connectivity panels.
                # The joint fit over (X,Y) pairs plays the role of the old
                # trial-averaged run: no separate refit is needed.
                self.extraction = res
                self.best_la = res.la
                self.la = props.la
                self.Z  = props.Z
                self.Z_ = np.eye(n_cells) + props.W_
                self.X  = props.X
                self.config = config
                # Correlation-normalized observed/estimated covariances (vld),
                # consumed by the (interim) Reps panels.
                self.Rep_out = res.vld_corrs["Cstar"]
                self.Rep_est = res.vld_corrs["Cest"]

        print("COMPUTING explain_models.Data (Diag half).")
        model_diag = split.model("Diag")
        res_d = model_diag.extract(seed=diag_seed, train=ref_train,
                                   metric=selection_metric, la=diag_la)
        config_d = self._seed_config(model_diag, diag_seed, res_d.la, expect_model="Diag")
        print(f"Refitting Diag for the quartic analysis: seed={diag_seed}, λ={res_d.la:.3g}, "
              f"n_train={diag_n_train}, gtol={diag_gtol:.1e}")
        self.diag = compute_diag_quartic(config_d, n_train=diag_n_train, gtol=diag_gtol)

        self.computed = True
        return self 
