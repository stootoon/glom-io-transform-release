"""Diag-model analyses, exposed as figure-agnostic computations.

Figure compute modules (show_models, explain_models, ...) compose these;
nothing here knows which figure is calling.
"""
import numpy as np

from types import SimpleNamespace

from .compute import driver
from .compute import seed_config


def quartic_geometry(split, selection_metric="ratio", seed=0, la=None, train=0,
                     n_train=10, gtol=1e-8):
    """Refit the Diag model and compute the per-unit quartic geometry.

    `split` is a results.SplitContext. The fit is re-run (rather than loaded)
    because the phase-plane analysis needs tightly-converged gains (gtol) so
    each unit sits at a minimum of its quartic loss. Returns everything the
    phase-plane and approximation panels consume.
    """
    from numpy.lib.scimath import sqrt as csqrt

    model = split.model("Diag")
    res = model.extract(seed=seed, train=train, metric=selection_metric, la=la)
    config = seed_config(model, seed, res.la, expect_model="Diag")

    print(f"Refitting Diag for the quartic analysis: {seed=}, λ={res.la:.3g}, "
          f"{n_train=}, {gtol=:.1e}")
    config = dict(config)
    config["sampler"] = dict(config["sampler"])
    config["sampler"]["n_train"] = n_train
    config.setdefault("min_args", {}).setdefault("options", {})["gtol"] = gtol

    _, _, mdl = driver.run(config, return_dataset=True, return_model=True)

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
                # |Alignment| = |Redundancy|, so only the name changes here.
                "approx_tex": r"$-\mathrm{sign}(\text{Tilt})\; \sqrt{|\text{Alignment}|}$",
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
        la=res.la,
    )
