import numpy as np
from numpy import *
from autograd import numpy as anp
from autograd import grad

from .common import get_IJN, get_Cstar, init_r, FitBase

class Model(FitBase):
    use_bounds = True
    
    def __init__(self, X, Y, bounds = (-np.inf, np.inf), λ = 0, center = True, reg = 1):
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
        
        self.m, self.n = X[0].shape
        self.λ = λ
        self.I, self.J, _ = get_IJN(self.m)
        if not center: self.J = self.I
        
        self.Cstars = [get_Cstar(Yk, center = center) for Yk in self.Ys]
        
        self.bounds = bounds
        self.center = center

        self.reg = reg
        print(f"Regularizing z**{self.reg} - 1")
        
        JY = get_IJN(self.Ys[0].shape[0])[self.center]
        for k, (Yk, Cstar_k) in enumerate(zip(self.Ys, self.Cstars)):
            assert np.allclose(Cstar_k, Yk.T @ JY @ Yk), "Cstar != Y.T @ J @ Y."

        self.predicting = False

        print("bounds", bounds)
        m = self.m
        self.cache={
            "Z":    ({}, self.ZFUN),
            "Ys":   ({}, lambda p: [self.get("Z",p) @ Xk for Xk in self.Xs]),
            "Cs":   ({}, lambda p: [Yk.T @ JtYk for Yk, JtYk in zip(self.get("Ys",p), self.get("JtYs",p))]),
            "JtYs": ({}, lambda p: [self.J.T @ Yk for Yk in self.get("Ys",p)]),
            "Fs":   ({}, lambda p: [JtY_k @ (Cstar_k - Ck) @ Xk.T
                                   for JtY_k, Cstar_k, Ck, Xk in
                                   zip(self.get("JtYs",p), self.Cstars, self.get("Cs", p), self.Xs)]),
        }

    def get(self, v, p):
        if self.predicting:
            return self.cache[v][1](p)
        
        k = p.tobytes()
        if k not in self.cache[v][0]: self.cache[v][0][k] = self.cache[v][1](p)
        return self.cache[v][0][k]
        
    def ZFUN(self, r):
        return diag(r)

    def COV_LOSS(self, p):
        Cs = self.get("Cs", p)
        return np.mean([(Cstar_k - Ck)**2 for Cstar_k, Ck in zip(self.Cstars, Cs)])/2

    def REG(self, p):
        return self.λ * np.mean((p**self.reg-1)**2)/2
    
    def JAC_LOSS(self,r):
        Fs = self.get("Fs",r)
        g = -2 * np.mean([diag(Fk) for Fk in Fs], axis=0)
        return g/self.n**2 + self.λ * (r**self.reg-1)/self.m * (self.reg) * r**(self.reg-1)

    def value_and_grad(self, p):
        cov = self.COV_LOSS(p)
        reg = self.REG(p)
        loss = cov + reg
        g = self.JAC_LOSS(p)
        self._last_loss, self._last_cov, self._last_reg = loss, cov, reg
        self._last_gnorm = np.abs(g).max()
        return loss, g

    def _anp_loss(self, p):
        Z = anp.diag(p)
        fit_terms = []
        for Xk, Cstar_k in zip(self.Xs, self.Cstars):
            Yk = anp.dot(Z, Xk)
            Ck = anp.dot(Yk.T, anp.dot(self.J, Yk))
            fit_terms.append(anp.mean((Cstar_k - Ck)**2))
        fit = anp.mean(anp.stack(fit_terms))/2
        reg = self.λ * anp.mean((p**self.reg-1)**2)/2
        return fit + reg


    def on_solution(self, p):
        self.r = p

    def init_guess(self, scale=1e-3, r0=1):
        return init_r(self.m, self.λ, scale=scale, r0=r0)

    def init_from(self, center, scale):
        return self.init_guess(scale, r0=center)
    
    def predict(self, X):
        self.predicting = True
        if not isinstance(X, list): X = [X]
        Xself = self.Xs
        self.Xs = X
        Cpreds = self.get("Cs", self.r)
        self.Xs = Xself
        self.predicting = False
        return Cpreds   
    
