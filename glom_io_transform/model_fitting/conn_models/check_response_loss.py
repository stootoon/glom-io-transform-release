"""Checks for the response-fitting (loss="resp") path of the Diag/Free models.

Run directly:  python -m glom_io_transform.model_fitting.conn_models.check_response_loss

Verifies:
  1. analytic gradients match autograd, for both losses and both models;
  2. the Diag response fit matches its closed-form ridge solution;
  3. the default (loss="cov") behaviour is unchanged.
"""
import numpy as np

from .diag import Model as Diag
from .free import Model as Free
from .free_lat import Model as FreeLat


def make_data(m=6, n=11, K=3, seed=0):
    rng = np.random.default_rng(seed)
    Xs = [rng.normal(size=(m, n)) for _ in range(K)]
    Xs = [X - X.mean(axis=0, keepdims=True) for X in Xs]   # odour-centred, as in the real data
    Ys = [rng.normal(size=(m, n)) for _ in range(K)]
    return Xs, Ys


def check_gradients(Xs, Ys):
    print("--- gradient checks (analytic vs autograd) ---")
    rng = np.random.default_rng(1)
    ok = True
    for name, Model, kw in [("Diag", Diag, dict(λ=0.7)),
                            ("Free", Free, dict(λ=[0.7])),
                            ("FreeLat", FreeLat, dict(λ=[0.7]))]:
        for loss in ["cov", "resp"]:
            mdl = Model(list(Xs), list(Ys), loss=loss, **kw)
            p = mdl.p_reg() + 0.1 * rng.normal(size=len(mdl.p_reg()))
            try:
                mdl.check_grad(p)
                print(f"  {name}/{loss}: OK")
            except AssertionError as e:
                ok = False
                print(f"  {name}/{loss}: FAIL -- {e}")
    return ok


def check_diag_closed_form(Xs, Ys, lam=0.7):
    """With reg=1 the Diag response fit is ridge regression per channel:
        d_i = (<y_i,x_i>_k/n + λ) / (||x_i||^2_k/n + λ)
    """
    print("--- Diag response fit vs closed form ---")
    n = Xs[0].shape[1]
    num = np.mean([np.sum(Y * X, axis=1) for X, Y in zip(Xs, Ys)], axis=0) / n + lam
    den = np.mean([np.sum(X * X, axis=1) for X in Xs], axis=0) / n + lam
    d_closed = num / den

    mdl = Diag(list(Xs), list(Ys), λ=lam, loss="resp")
    mdl.minimize(method="L-BFGS-B", options={"gtol": 1e-12, "ftol": 1e-16, "maxiter": 5000})
    err = np.abs(mdl.results.x - d_closed).max()
    print(f"  max |optimizer - closed form| = {err:.3e}")
    return err < 1e-6


def check_defaults(Xs, Ys):
    print("--- backwards compatibility ---")
    mdls = {"Diag": Diag(list(Xs), list(Ys), λ=0.5),
            "Free": Free(list(Xs), list(Ys), λ=[0.5]),
            "FreeLat": FreeLat(list(Xs), list(Ys), λ=[0.5])}
    same = all([np.allclose(m.FIT_LOSS(m.p_reg()), m.COV_LOSS(m.p_reg())) for m in mdls.values()])
    losses = ", ".join(f"{k}={m.loss!r}" for k, m in mdls.items())
    print(f"  default loss: {losses}; FIT_LOSS == COV_LOSS: {same}")
    return all([m.loss == "cov" for m in mdls.values()]) and same


if __name__ == "__main__":
    Xs, Ys = make_data()
    results = [check_gradients(Xs, Ys),
               check_diag_closed_form(Xs, Ys),
               check_defaults(Xs, Ys)]
    print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED")
