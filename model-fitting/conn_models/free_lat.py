"""
In this model we learn the connectivity freely without any constraints,
except that there is only lateraly connectivity, no diagonal term
Hence we optimize in W space, and W_ii = 0.
"""
import numpy as np
from numpy import *
from autograd import grad
from autograd import numpy as anp
from scipy.optimize import minimize

from .common import get_IJN, get_Cstar, init_r


class Model:
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
        # self.test(): This calls __init__ so calling it here would create an infinite loop.
        # Instead, we call it below in the minimize function.

    def get(self, v, p):
        if self.predicting:
            return self.cache[v][1](p)
    
        k = p.tobytes()
        if k not in self.cache[v][0]: self.cache[v][0][k] = self.cache[v][1](p)
        return self.cache[v][0][k]

    def WFUN(self, p):
        return np.reshape(self.S @ p, (self.m, self.m))
    
    def ZFUN(self, p):
        return np.linalg.inv(self.I + self.get("W", p)) 

    def REG(self, p):
        W = self.get("W", p)
        return np.mean(W**2)/2 * self.λ[0]

    def COV_LOSS(self, p):
        Cs = self.get("Cs", p)
        return np.mean([(Cstar_k - Ck)**2 for Cstar_k, Ck in zip(self.Cstars, Cs)])/2
        
    def LOSS(self, p):
        return self.COV_LOSS(p) + self.REG(p)

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
            G += self.J @ Yk @ (Ck - Cstar_k) @ Xk.T
        G *= 2.0/(self.K * self.n**2)
        G += (self.λ[0]/self.m**2) * (Z - self.I)

        dW = -Z.T @ G @ Z.T
        grad = self.S.T @ dW.flatten()
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
    
    def minimize(self, p0=None, **kwargs):
        self.test()
        print("RUNNING MINIMIZATION")
        if p0 is None: p0 = self.init_guess(scale = self.init_scale)
        self.p0 = p0
        print("COV_LOSS at initial guess:", self.COV_LOSS(self.p0))        
        self.results = minimize(self.value_and_grad, p0, jac=True, **kwargs)            
        self.p = self.results.x
        self.W = self.get("W", self.p)
        self.Z = self.get("Z", self.p)
        print(f"Minimization finished with status {self.results.status}.")
        print(f"Message: {self.results.message}")
        print("COV_LOSS at solution:", self.COV_LOSS(self.results.x))
        print("cond(I + W) at solution:", np.linalg.cond(self.I + self.W))
        return self.results        

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
   
    def test(self):
        print("TESTING GRADIENTS")
        _grad = grad(self._anp_loss)
        z = np.random.rand(self.n_params,)
        grad_anp = _grad(z)
        grad_manual = self.value_and_grad(z)[1]
        err = np.linalg.norm(grad_anp - grad_manual) / (np.linalg.norm(grad_anp) + np.linalg.norm(grad_manual))
        assert err < 1e-6, f"Gradient check failed with error {err}"
        print(f"Gradient check passed with error {err}")
        print("GRADIENTS TESTED SUCCESSFULLY")
        return err
    
    
