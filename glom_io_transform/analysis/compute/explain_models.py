import os, sys
import pickle
import numpy as np

from types import SimpleNamespace

from .compute import Computation
from .compute import paths
from .compute import get_Cstar
from .compute import driver

import model_fitting.results as results


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


class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.props_vals = []

    def _seed_config(self, model, seed, la):
        """Load the in.N.p run config for the given seed and lambda."""
        sel = (model.df["seed"] == seed) & (model.df["λ"] == la)
        files = model.df[sel]["file"].unique()
        assert len(files) == 1, f"Expected exactly one input file for {seed=}, λ={la}, found {len(files)}."
        config = results.load_pickle(os.path.join(model.base_dir, files[0]))
        assert config["seed"] == seed, f"Seed mismatch: {config['seed']} vs {seed}"
        assert config["model"] == "Free", f"Model mismatch: {config['model']} vs Free"
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

        self.props_vals = []
        for seed in seeds:
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

        self.computed = True
        return self
