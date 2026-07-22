"""
In this model we learn the connectivity freely without any constraints,
except that there is only lateraly connectivity, no diagonal term
Hence we optimize in W space, and W_ii = 0.
"""
import sys
import numpy as np
from numpy import *
from autograd import grad
from autograd import numpy as anp
from autograd import hessian_vector_product
from scipy.optimize import minimize
import time
from .common import get_IJN, get_Cstar, init_r, FitBase


class Model(FitBase):
    def __init__(self, X, Y, λ = [0], center = True, init_scale = 1e-3):
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
        self.predicting = True
        print(f"Warning: Model is initialized in predicting mode. This will disable caching. Set self.predicting = False to enable caching.")
        

        mask = ~np.eye(self.m, dtype=bool)
        self.offdiag_idx = np.where(mask.flatten())[0]
        self.n_params    = self.offdiag_idx.size
        S = np.zeros((self.m * self.m, self.n_params))
        S[self.offdiag_idx, np.arange(self.n_params)] = 1.
        self.S = S

        self.cache = {
            "W":    ({}, self.WFUN),
            "Z":    ({}, self.ZFUN),
            "Ys":   ({}, lambda p: [self.get("Z",p) @ Xk for Xk in self.Xs]),
            "Cs":   ({}, lambda p: [Yk.T @ self.J @ Yk for Yk in self.get("Ys",p)]),
        }

        self._it = 0 # iteration counter for debugging

    def get(self, v, p):
        if self.predicting:
            return self.cache[v][1](p)
    
        k = p.tobytes()
        if k not in self.cache[v][0]: self.cache[v][0][k] = self.cache[v][1](p)
        return self.cache[v][0][k]

    def WFUN(self, p):
        W = np.zeros((self.m, self.m))
        W.flat[self.offdiag_idx] = p
        return W 
    
    def ZFUN(self, p):
        return np.linalg.inv(self.I + self.get("W", p)) 

    def REG(self, p):
        Z = self.get("Z", p)
        return np.mean((Z - self.I)**2)/2 * self.λ[0]

    def COV_LOSS(self, p):
        Cs = self.get("Cs", p)
        return np.mean([(Cstar_k - Ck)**2 for Cstar_k, Ck in zip(self.Cstars, Cs)])/2
        
    def value_and_grad(self, p):
        W = self.get("W", p)
        Z = np.linalg.inv(self.I + W)
        Ys = [Z @ Xk for Xk in self.Xs]
        Cs = [Yk.T @ self.J @ Yk for Yk in Ys]

        cov = np.mean([(Cstar_k - Ck)**2 for Cstar_k, Ck in zip(self.Cstars, Cs)])/2
        reg = np.mean((Z - self.I)**2)/2 * self.λ[0]
        loss= cov + reg

        G = np.zeros((self.m, self.m))
        for Yk, Ck, Cstar_k, Xk in zip(Ys, Cs, self.Cstars, self.Xs):
            G += Yk @ (Ck - Cstar_k) @ Xk.T
        G = self.J @ G
        G *= 2.0/(self.K * self.n**2)
        G += (self.λ[0]/self.m**2) * (Z - self.I)

        dW = -Z.T @ G @ Z.T
        grad = dW.flat[self.offdiag_idx] #self.S.T @ dW.flatten()

        self._last_loss = loss
        self._last_gnorm = np.abs(grad).max()
        self._last_cov = cov
        self._last_reg = reg
        
        return loss, grad
    
    def _anp_loss(self, p):
        W = anp.reshape(anp.dot(self.S, p), (self.m, self.m))
        Z = anp.linalg.inv(self.I + W)
        fit_terms = []
        for Xk, Cstar_k in zip(self.Xs, self.Cstars):
            Yk = anp.dot(Z, Xk)
            Ck = anp.dot(Yk.T, anp.dot(self.J, Yk))
            fit_terms.append(anp.mean((Cstar_k - Ck)**2))
        gof = anp.mean(anp.stack(fit_terms))/2
        reg = anp.mean((Z - self.I)**2)/2 * self.λ[0] 
        return gof + reg
    
    def init_guess(self, scale = 1e-3):
        print("Initializing guess with scale = ", scale)
        return scale*np.random.randn(self.n_params,)

    def init_from(self, center, scale):
        return self.init_guess(scale) + center

    def p_reg(self):
        return np.zeros(self.n_params)

    def on_solution(self, p):
        self.W = self.get("W", p)
        self.Z = self.get("Z", p)

    def report_solution(self):
        print("cond(W + I) at solution:", np.linalg.cond(self.W + self.I))
    
    def predict(self, X):
        prev = self.predicting
        self.predicting = True 
        if not isinstance(X, list): X = [X]
        Xself = self.Xs
        self.Xs = X
        Cpreds = self.get("Cs", self.p)
        self.Xs = Xself
        self.predicting = prev
        return Cpreds
   
   
    
