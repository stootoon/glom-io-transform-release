"""Free-model analyses, exposed as figure-agnostic computations.

Figure compute modules (show_models, explain_models, ...) compose these;
nothing here knows which figure is calling.
"""
import numpy as np

from types import SimpleNamespace

from .compute import get_Cstar
from .compute import seed_config, seed_data


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


def connectivity_theory(split, selection_metric="ratio", seeds=None,
                        ref_seed=0, ref_train=0):
    """Fitted-vs-predicted connectivity analysis across the Free fits.

    `split` is a results.SplitContext. Returns a namespace with per-seed mode
    overlaps (props_vals) and the reference seed's connectivity (Z), theory
    prediction (Z_), rotated inputs (X), and correlation-normalized observed/
    estimated covariances (Rep_out/Rep_est).
    """
    model = split.model("Free")
    center = split.base.center

    if seeds is None:
        seeds = sorted(model.df["seed"].unique())

    print(f"Free connectivity theory: seeds={seeds}, {selection_metric=}, {ref_seed=}, {ref_train=}")

    out = SimpleNamespace(model=model, props_vals=[])
    for i, seed in enumerate(seeds):
        print(f"Processing seed {seed} ({i+1}/{len(seeds)})")
        res = model.extract(seed=seed, train=ref_train, metric=selection_metric, with_params=True)
        config = seed_config(model, seed, res.la, expect_model="Free")
        XX, YY = seed_data(config)

        n_cells = XX.trains[0].shape[0]
        Z = res.params["p_final"].reshape(n_cells, n_cells)

        print(f"Computing props for {seed=} with best λ={res.la:.3g}")
        props = compute_props(XX.trains, YY.trains, Z, res.la, center=center)
        out.props_vals.append(props)

        if seed == ref_seed:
            # Reference fit shown in the connectivity panels.
            # The joint fit over (X,Y) pairs plays the role of the old
            # trial-averaged run: no separate refit is needed.
            out.extraction = res
            out.best_la = res.la
            out.la = props.la
            out.Z  = props.Z
            out.Z_ = np.eye(n_cells) + props.W_
            out.X  = props.X
            out.config = config
            # Correlation-normalized observed/estimated covariances (vld)
            out.Rep_out = res.vld_corrs["Cstar"]
            out.Rep_est = res.vld_corrs["Cest"]

    return out
