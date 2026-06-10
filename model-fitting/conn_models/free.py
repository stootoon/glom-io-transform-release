"""
In this model we learn the connectivity freely without any constraints.
"""
import numpy as np
from numpy import *
from autograd import grad
from autograd import numpy as anp
from scipy.optimize import minimize, NonlinearConstraint

from .common import get_IJN, get_Cstar, init_r, complete_basis, cond

import pdb

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
        self.I = I
        self.center = center
        self.I, self.J, _ = get_IJN(self.m)
        if not center: self.J = self.I
 
        self.Cstars = [get_Cstar(Yk, center) for Yk in self.Ys]
        JY = get_IJN(self.Ys.shape[0])[self.center]
        for Yk, Cstar_k in zip(self.Ys, self.Cstars):
            assert np.allclose(Cstar_k, self.Yk.T @ JY @ self.Yk), "Cstar != Y.T @ J @ Y."


        self.init_scale = init_scale
        self.predicting = True

        self.cache={
            "Z":   ({}, self.ZFUN),
            "Y":   ({}, lambda p: self.get("Z",p)   @ self.X),
            "F":   ({}, lambda p: self.get("JtY",p) @ (self.Cstar - self.get("C",p)) @ self.X.T),
            "C":   ({}, lambda p: self.get("Y",p).T @ self.get("JtY",p)),
            "JtY": ({}, lambda p: self.J.T          @ self.get("Y",p)),            
        } 

        # self.test(): This calls __init__ so calling it here would create an infinite loop.
        # Instead, we call it below in the minimize function.

    def get(self, v, p):
        if self.predicting:
            return self.cache[v][1](p)
    
        k = p.tobytes()
        if k not in self.cache[v][0]: self.cache[v][0][k] = self.cache[v][1](p)
        return self.cache[v][0][k]

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
        
    def LOSS(self, p):
        return self.COV_LOSS(p) + self.REG(p)
    
    def JAC_LOSS(self,p):
        F = np.mean(self.get("Fs",p), axis=0)
        G = -2*F/self.n**2 + self.JAC_REG(p)
        return G.flatten(order="C")

    def init_guess(self, scale = 1e-3):
        print("Initializing guess with scale = ", scale)
        r0 = np.eye(self.m) 
        return init_r(self.m**2, self.λ[0], r0 = r0.flatten(), scale = scale)        
    
    def minimize(self, p0=None, **kwargs):
        self.test()
        print("RUNNING MINIMIZATION")
        if p0 is None: p0 = self.init_guess(scale = self.init_scale)
        self.p0 = p0
        print("COV_LOSS at initial guess:", self.COV_LOSS(self.p0))        
        self.results = minimize(self.LOSS, p0, jac=self.JAC_LOSS, **kwargs)            
        self.Z = np.reshape(self.results.x, (self.m, self.m), order="C")
        print(f"Minimization finished with status {self.results.status}.")
        print(f"Message: {self.results.message}")
        print("COV_LOSS at solution:", self.COV_LOSS(self.results.x))
        return self.results        

    def predict(self, X):
        self.predicting = True
        if not isinstance(X, list): X = [X]
        Xself = self.Xs
        self.Xs = X
        Cpreds = self.get("Cs", self.r)
        self.Xs = Xself
        self.predicting = False
        return Cpreds
   
    def test(self):
        print("TESTING GRADIENTS")

        def loss(z):
            Z = anp.reshape(z, (self.m, self.m), order="C")
            fit_terms = []
            for Xk, Cstar_k in zip(self.Xs, self.Cstars):
                Yk = anp.dot(Z, Xk)
                Ck = anp.dot(Yk.T, anp.dot(self.J, Yk))
                fit_terms.append(anp.mean((Cstar_k - Ck)**2))
            gof = anp.mean(anp.stack(fit_terms))/2
            reg = anp.mean((Z - self.I)**2)/2 * self.λ[0]
            return gof + reg 
        
        mdl0 = Model(self.Xs, self.Ys, λ = self.λ, center=self.center)
        z = np.random.rand(self.m**2,)
        g_true = grad(loss)(z)
        g_mdl  = mdl0.JAC_LOSS(z)
        assert allclose(g_true, g_mdl), "Model gradient does not match true gradient, for global regularization."
        print("Model gradient matches true gradient, for global regularization.")
        print("GRADIENTS TESTED SUCCESSFULLY")
        return g_true, g_mdl, z
    
    
