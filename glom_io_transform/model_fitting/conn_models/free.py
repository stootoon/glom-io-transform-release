"""
In this model we learn the connectivity freely without any constraints.
"""
import numpy as np
from numpy import *
from autograd import grad
from autograd import numpy as anp

from .common import get_IJN, get_Cstar, init_r, cond, FitBase

class Model(FitBase):
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

    def ZFUN(self, p):
        return reshape(p, (self.m, self.m), order="C")

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
        return G.flatten(order="C")

    def JAC_RESP(self, p):
        Z = self.get("Z", p)
        G = np.mean([(Z @ Xk - Yk) @ Xk.T for Xk, Yk in zip(self.Xs, self.Ys)],
                    axis=0)/(self.m * self.n)
        return (G + self.JAC_REG(p)).flatten(order="C")

    def value_and_grad(self, p):
        cov = self.FIT_LOSS(p)
        reg = self.REG(p)
        loss = cov + reg
        g = self.JAC_RESP(p) if self.loss == "resp" else self.JAC_LOSS(p)
        self._last_loss, self._last_cov, self._last_reg = loss, cov, reg
        self._last_gnorm = np.abs(g).max()
        return loss, g

    def _anp_loss(self, p):
        Z = anp.reshape(p, (self.m, self.m), order="C")
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
        self.Z = np.reshape(p, (self.m, self.m), order="C")

    def init_guess(self, scale = 1e-3, center = 1.):
        print("Initializing guess with scale = ", scale)
        r0 = np.eye(self.m) * center
        return init_r(self.m**2, self.λ[0], r0 = r0.flatten(), scale = scale)

    def init_from(self, center, scale):
        return self.init_guess(scale = scale, center=center)

    def p_reg(self):
        return self.I.flatten()
    
    def predict(self, X):
        if not isinstance(X, list): X = [X]
        Xself = self.Xs
        self.Xs = X
        Cpreds = self.get("Cs", self.r)
        self.Xs = Xself
        return Cpreds
   
   
    
