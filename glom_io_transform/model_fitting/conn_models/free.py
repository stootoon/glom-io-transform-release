"""
In this model we learn the connectivity freely without any constraints.
"""
import numpy as np
from autograd import grad
from autograd import numpy as anp

from .common import get_IJN, get_Cstar, init_r, cond, FitBase

class Model(FitBase):
    @classmethod
    def Z_from_p(cls, p, m, **kwargs):
        return np.reshape(p, (m, m), order="C")

    def __init__(self, X, Y, λ = [0], center = True, init_scale = 1e-3, loss = "cov"):
        if not isinstance(X, list):
            print("WARNING: Converting X to singleton list.")
            X = [X]
        if not isinstance(Y, list):
            print("WARNING: Converting Y to singleton list.")
            Y = [Y]

        assert all([Xi.shape == X[0].shape for Xi in X]), "All X must be the same shape."
        assert all([Yi.shape == Y[0].shape for Yi in Y]), "All Y must be the same shape."

        self.Xs = X
        self.Ys = Y
        self.K  = len(X)

        assert loss in ("cov", "resp"), f"Unknown loss '{loss}'."
        self.loss = loss
        if loss == "resp":
            # Responses are compared channel by channel, so the caller must pass
            # X and Y whose rows correspond (e.g. matched input/output glomeruli).
            assert all([Yk.shape == Xk.shape for Xk, Yk in zip(X, Y)]), \
                "Response fitting requires X and Y with matched rows (channels) and columns (odours)."

        self.m, self.n = self.Xs[0].shape
        self.λ = λ
        self.center = center
        self.I, self.J, _ = get_IJN(self.m)
        if not center: self.J = self.I
 
        self.Cstars = [get_Cstar(Yk, center) for Yk in self.Ys]
        JY = get_IJN(self.Ys[0].shape[0])[self.center]
        for Yk, Cstar_k in zip(self.Ys, self.Cstars):
            assert np.allclose(Cstar_k, Yk.T @ JY @ Yk), "Cstar != Y.T @ J @ Y."

        self.init_scale = init_scale

        self.computers = {
            "Z":    self.ZFUN,
            "Ys":   lambda p: [self.get("Z",p) @ Xk for Xk in self.Xs],
            "JtYs": lambda p: [self.J.T @ Yk for Yk in self.get("Ys",p)],
            "Cs":   lambda p: [Yk.T @ JtYk for Yk, JtYk in zip(self.get("Ys",p), self.get("JtYs",p))],
            "Fs":   lambda p: [JtY_k @ (Cstar_k - Ck) @ Xk.T
                               for JtY_k, Cstar_k, Ck, Xk in
                               zip(self.get("JtYs",p), self.Cstars, self.get("Cs",p), self.Xs)],
        }
        # self.test(): This calls __init__ so calling it here would create an infinite loop.
        # Instead, we call it below in the minimize function.

    def get(self, v, p):
        return self.computers[v](p)

    # --- Parameterisation ---------------------------------------------------
    # A variant of this model changes two things and nothing else: how a
    # parameter vector becomes a connectivity (n_params / ZFUN / p_from_Z, and
    # _anp_Z for the gradient check), and how a gradient with respect to Z
    # becomes a gradient with respect to those parameters (pack_grad). The
    # losses, the regulariser and everything downstream are written in terms of
    # Z and are inherited unchanged.

    def n_params(self):
        """How many free parameters the parameterisation has."""
        return self.m ** 2

    def ZFUN(self, p):
        # return np.reshape(p, (self.m, self.m), order="C")
        return self.Z_from_p(p, self.m)

    def _anp_Z(self, p):
        """ZFUN again, in autograd's numpy, for the gradient check in test()."""
        return anp.reshape(p, (self.m, self.m), order="C")

    def p_from_Z(self, Z):
        """The parameters standing for a given Z. Inverse of ZFUN."""
        return Z.flatten(order="C")

    def pack_grad(self, G, p):
        """dLoss/dZ -> dLoss/dp, in the order ZFUN reads p back."""
        return G.flatten(order="C")

    def REG(self, p):
        Z = self.get("Z",p)
        return np.mean((Z - self.I)**2)/2 * self.λ[0]

    def JAC_REG(self, p):
        Z = self.get("Z",p)
        return (Z - self.I) * self.λ[0]/self.m**2
   
    def COV_LOSS(self, p):
        Cs = self.get("Cs", p)
        return np.mean([(Cstar_k - Ck)**2 for Cstar_k, Ck in zip(self.Cstars, Cs)])/2

    def RESP_LOSS(self, p):
        Z = self.get("Z", p)
        return np.mean([(Yk - Z @ Xk)**2 for Xk, Yk in zip(self.Xs, self.Ys)])/2

    def JAC_LOSS(self,p):
        F = np.mean(self.get("Fs",p), axis=0)
        G = -2*F/self.n**2 + self.JAC_REG(p)
        return self.pack_grad(G, p)

    def JAC_RESP(self, p):
        Z = self.get("Z", p)
        G = np.mean([(Z @ Xk - Yk) @ Xk.T for Xk, Yk in zip(self.Xs, self.Ys)],
                    axis=0)/(self.m * self.n)
        return self.pack_grad(G + self.JAC_REG(p), p)

    def value_and_grad(self, p):
        cov = self.FIT_LOSS(p)
        reg = self.REG(p)
        loss = cov + reg
        g = self.JAC_RESP(p) if self.loss == "resp" else self.JAC_LOSS(p)
        self._last_loss, self._last_cov, self._last_reg = loss, cov, reg
        self._last_gnorm = np.abs(g).max()
        return loss, g

    def _anp_loss(self, p):
        Z = self._anp_Z(p)
        fit_terms = []
        if self.loss == "resp":
            for Xk, Yk in zip(self.Xs, self.Ys):
                fit_terms.append(anp.mean((Yk - anp.dot(Z, Xk))**2))
        else:
            for Xk, Cstar_k in zip(self.Xs, self.Cstars):
                Yk = anp.dot(Z, Xk)
                Ck = anp.dot(Yk.T, anp.dot(self.J, Yk))
                fit_terms.append(anp.mean((Cstar_k - Ck)**2))
        gof = anp.mean(anp.stack(fit_terms))/2
        reg = anp.mean((Z - self.I)**2)/2 * self.λ[0]
        return gof + reg

    def on_solution(self, p):
        self.r = p
        self.Z = self.ZFUN(p)

    def init_guess(self, scale = 1e-3, center = 1.):
        print("Initializing guess with scale = ", scale)
        r0 = np.eye(self.m) * center
        return init_r(self.n_params(), self.λ[0], r0 = self.p_from_Z(r0), scale = scale)

    def init_from(self, center, scale):
        return self.init_guess(scale = scale, center=center)

    def p_reg(self):
        return self.p_from_Z(self.I)
    
    def predict(self, X):
        """Predicted responses (loss="resp") or output covariances (loss="cov")."""
        if not isinstance(X, list): X = [X]
        Xself = self.Xs
        self.Xs = X
        preds = self.get("Ys" if self.loss == "resp" else "Cs", self.r)
        self.Xs = Xself
        return preds


class SymModel(Model):
    """Free connectivity constrained to be symmetric.

    Same parameterisation as Free -- all m^2 entries -- read symmetrically:
    Z = (P + P')/2 for P the reshaped parameters. Symmetrising in the UNPACKING
    rather than only in the gradient matters for two reasons. It makes
    (G + G')/2 the true gradient with respect to p, by the chain rule, so the
    autograd check in test() agrees with it. And it makes any starting point
    symmetric, where projecting the gradient alone would leave the
    antisymmetric part of the initial guess untouched forever -- init_r
    perturbs every entry independently, so that part is not zero.

    The antisymmetric half of p is then in the null space of the map: it moves
    nothing, and the gradient along it is exactly zero, so the optimiser
    ignores it.

    Symmetric is not the same as rotation-free: an indefinite symmetric Z still
    has an orthogonal polar factor, a reflection through its negative
    eigendirections. PSDModel is the rotation-free one.
    """
    @classmethod
    def Z_from_p(cls, p, m, **kwargs):
        P = np.reshape(p, (m, m), order="C")
        return (P + P.T) / 2
    
    def ZFUN(self, p):
        return self.Z_from_p(p, self.m)
        
    def _anp_Z(self, p):
        P = anp.reshape(p, (self.m, self.m), order="C")
        return (P + anp.transpose(P)) / 2

    def pack_grad(self, G, p):
        return ((G + G.T) / 2).flatten(order="C")


class PSDModel(Model):
    """Free connectivity constrained to be symmetric positive semidefinite.

    Z = L L' with L lower triangular. Z is then PSD by construction, so its
    polar rotation is exactly the identity: this is the model with no rotation
    AVAILABLE, as against a fitted Z with its rotation deleted afterwards.

    L is lower triangular rather than full to remove the gauge freedom -- L and
    LQ give the same Z for any orthogonal Q -- which also brings the parameter
    count down to m(m+1)/2, the dimension of the PSD cone. A discrete freedom
    survives, flipping the sign of a column of L, but it changes no Z and
    creates no flat directions.

    The gradient follows from dZ = dL L' + L dL':
        dLoss/dL = (G + G')L,  restricted to the entries that are free.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tril = np.tril_indices(self.m)
        # Item assignment is not differentiable under autograd, so the anp path
        # builds L as a product with this 0/1 selection matrix instead.
        self.tril_basis = np.zeros((self.m * self.m, self.n_params()))
        self.tril_basis[self.tril[0] * self.m + self.tril[1],
                        np.arange(self.n_params())] = 1.0

    def n_params(self):
        return self.m * (self.m + 1) // 2

    @classmethod
    def Z_from_p(cls, p, m, **kwargs):
        L = np.zeros((m, m))
        tril = np.tril_indices(m) if "tril" not in kwargs else kwargs["tril"]
        L[tril] = p
        return L @ L.T

    def L_of_p(self, p):
        L = np.zeros((self.m, self.m))
        L[self.tril] = p
        return L
    
    def ZFUN(self, p):
        return self.Z_from_p(p, self.m, tril=self.tril)

    def _anp_Z(self, p):
        L = anp.reshape(anp.dot(self.tril_basis, p), (self.m, self.m))
        return anp.dot(L, anp.transpose(L))

    def p_from_Z(self, Z):
        return np.linalg.cholesky(Z)[self.tril]

    def pack_grad(self, G, p):
        return ((G + G.T) @ self.L_of_p(p))[self.tril]

    def init_guess(self, scale = 1e-3, center = 1.):
        # Z = center*I at zero noise, so L = sqrt(center)*I. Built directly
        # rather than through p_from_Z, whose Cholesky rejects center = 0 --
        # and 0 is one of the restart centres the yamls ask for.
        print("Initializing guess with scale = ", scale)
        L0 = np.eye(self.m) * np.sqrt(max(center, 0.0))
        return init_r(self.n_params(), self.λ[0], r0 = L0[self.tril], scale = scale)
