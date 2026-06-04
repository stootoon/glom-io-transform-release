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
    def __init__(self, X, Y, λ = [0], center = True, Λuu = None, Λup = None, Λpu = None, Λpp = None, init_scale = 1e-3):
        self.X = X
        self.Ux, self.sx, self.Vx = np.linalg.svd(X, full_matrices=False); self.Vx = self.Vx.T
        # Unicode for the perp symbol: ⊥
        self.Ux_p = complete_basis(self.Ux)
        
        
        self.Y = Y
        self.m, self.n = X.shape
        self.Cstar = get_Cstar(Y, center)
        self.I, self.J, _ = get_IJN(self.m)
        if not center: self.J = self.I
        self.center = center
        JY = get_IJN(self.Y.shape[0])[self.center]
        assert np.allclose(self.Cstar, self.Y.T @ JY @ self.Y), "Cstar != Y.T @ J @ Y."

        JY = get_IJN(self.Y.shape[0])[self.center]
        assert np.allclose(self.Cstar, self.Y.T @ JY @ self.Y), "Cstar != Y.T @ J @ Y."

        m = self.m
        self.Im = np.eye(m)
        self.λ = λ

        self.predicting = True

        if all([Λ is None for Λ in [Λuu, Λup, Λpu, Λpp]]):
            self.REG = self.REG_STANDARD
            self.JAC_REG = self.JAC_REG_STANDARD
        else:
            self.REG = self.REG_COMPONENT
            self.JAC_REG = self.JAC_REG_COMPONENT

        n = self.n
        self.In = np.eye(n)
        self.Ip = np.eye(m - n)

        self.Λuu = cond([((type(Λuu) is str) and (Λuu.lower() == "sx"), self.sx, "Using Λuu = sx")], np.ones(n,))[np.newaxis, :]
        self.Λup = 1
        self.Λpu = cond([((type(Λpu) is str) and (Λpu.lower() == "sx"), np.outer(np.ones(m-n,),self.sx), "Using Λpu = sx")], 1)
        self.Λpp = 1

        self.init_scale = init_scale

        self.cache={
            "Z":   ({}, self.ZFUN),
            "Zuu": ({}, lambda p: self.Ux.T         @ self.get("Z",p) @ self.Ux),
            "Zup": ({}, lambda p: self.Ux.T         @ self.get("Z",p) @ self.Ux_p),
            "Zpu": ({}, lambda p: self.Ux_p.T       @ self.get("Z",p) @ self.Ux),
            "Zpp": ({}, lambda p: self.Ux_p.T       @ self.get("Z",p) @ self.Ux_p),
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

    def REG_STANDARD(self, p):
        Z = self.get("Z",p)
        return np.mean((Z - self.I)**2)/2 * self.λ[0]
    def JAC_REG_STANDARD(self, p):
        Z = self.get("Z",p)
        return (Z - self.I) * self.λ[0]/self.m**2

    def REG_COMPONENT(self, p):
        Zuu = self.get("Zuu",p)
        Zup = self.get("Zup",p)
        Zpu = self.get("Zpu",p)
        Zpp = self.get("Zpp",p)
        reg  = np.sum((Zuu * self.Λuu  - self.In)**2) # It really is In, not Im
        reg += np.sum((Zup * self.Λup)**2)
        reg += np.sum((Zpu * self.Λpu)**2)
        reg += np.sum((Zpp * self.Λpp - self.Ip)**2) # p = m - n
        reg *= self.λ[0]/2/self.m**2
        return reg

    def JAC_REG_COMPONENT(self, p):
        Zuu = self.get("Zuu",p)
        Zup = self.get("Zup",p)
        Zpu = self.get("Zpu",p)
        Zpp = self.get("Zpp",p)
        jac  = self.Ux   @ ((Zuu * self.Λuu - self.In) * self.Λuu) @ self.Ux.T
        jac += self.Ux   @ ((Zup * self.Λup)           * self.Λup) @ self.Ux_p.T
        jac += self.Ux_p @ ((Zpu * self.Λpu)           * self.Λpu) @ self.Ux.T
        jac += self.Ux_p @ ((Zpp * self.Λpp - self.Ip) * self.Λpp) @ self.Ux_p.T
        jac *= self.λ[0]/self.m**2
        return jac
    
    def COV_LOSS(self, p):
        loss = np.mean((self.Cstar - self.get("C",p))**2)/2
        return loss        
        
    def LOSS(self, p):
        loss = self.COV_LOSS(p) + self.REG(p)
        return loss
    
    def JAC_LOSS(self,p):
        Z = self.get("Z",p)
        F = self.get("F",p)
        G = -2*F/self.n**2 + self.JAC_REG(p)
        return G.flatten(order="C")

    def init_guess(self, scale = 1e-3):
        print("Initializing guess with scale = ", scale)
        p0  = np.zeros(self.m**2)
        iΛuu = 0*self.Λuu
        iΛuu[self.Λuu != 0] = 1/(self.Λuu[self.Λuu != 0] + 1e-6)
        r0  = self.Ux   @ (np.eye(self.n)          * iΛuu) @ self.Ux.T
        r0 += self.Ux_p @ (np.eye(self.m - self.n)) @ self.Ux_p.T        
        p0[:self.m**2] = init_r(self.m**2, self.λ[0], r0 = r0.flatten(), scale = scale)        
        return p0
    
    def minimize(self, p0=None, **kwargs):
        self.test()
        print("RUNNING MINIMIZATION")
        if self.REG == self.REG_STANDARD:
            print("Using STANDARD regularization.")
        elif self.REG == self.REG_COMPONENT:
            print("Using COMPONENT regularization.")
        else:
            raise ValueError("Unknown regularization.")
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
        Xself  = self.X
        self.X = X
        Cpred  = self.get("C", self.results.x)
        self.X = Xself
        self.predicting = False
        return Cpred
    
    def test(self):
        print("TESTING GRADIENTS")
        m = self.m
        X = self.X
        Cstar = self.Cstar
        J = self.J

        def loss(z):
            Z = anp.reshape(z, (self.m, self.m), order="C")
            Y = anp.dot(Z,self.X)
            C = anp.dot(Y.T , anp.dot(self.J, Y))
            return anp.mean((self.Cstar - C)**2)/2 + anp.mean((Z - self.I)**2)/2 * self.λ[0]

        def loss_component(z):
            Z = anp.reshape(z, (self.m, self.m), order="C")
            Y = anp.dot(Z,self.X)
            C = anp.dot(Y.T , anp.dot(self.J, Y))
            cov_loss = anp.mean((C - self.Cstar)**2)/2
            Λuu = self.sx[np.newaxis, :]
            Λpu = self.sx[np.newaxis, :]
            Zuu = self.Ux.T @ Z @ self.Ux
            Zup = self.Ux.T @ Z @ self.Ux_p
            Zpu = self.Ux_p.T @ Z @ self.Ux            
            Zpp = self.Ux_p.T @ Z @ self.Ux_p
            reg = anp.sum((Zuu * Λuu  - self.In)**2) # It really is In, not Im
            reg += anp.sum((Zup)**2)
            reg += anp.sum((Zpu * Λpu)**2)
            reg += anp.sum((Zpp - self.Ip)**2) # p = m - n
            reg *= self.λ[0]/2/self.m**2
            return cov_loss + reg

        
        mdl0 = Model(self.X, self.Y, λ = self.λ, center = self.center, Λuu = None, Λup = None, Λpu = None, Λpp = None)
        z = np.random.rand(self.m**2,)
        g_true = grad(loss)(z)
        g_mdl  = mdl0.JAC_LOSS(z)
        assert allclose(g_true, g_mdl), "Model gradient does not match true gradient, for global regularization."
        print("Model gradient matches true gradient, for global regularization.")
        mdl1 = Model(self.X, self.Y, λ = self.λ, center = self.center, Λuu = "sx", Λup = None, Λpu = "sx", Λpp = None)
        g_true = grad(loss_component)(z)
        g_mdl  = mdl1.JAC_LOSS(z)
        assert allclose(g_true, g_mdl), "Model gradient does not match true gradient, for component regularization."
        print("Model gradient matches true gradient, for component regularization.")
        print("GRADIENTS TESTED SUCCESSFULLY")
        return g_true, g_mdl, z
    
    
