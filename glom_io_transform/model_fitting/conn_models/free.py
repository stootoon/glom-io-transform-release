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


class RotModel(Model):
    """Free connectivity constrained to a scaled orthogonal map,
    Z = s D (I - A)(I + A)^-1, for A antisymmetric and D = I or diag(-1, 1...1).

    The Cayley transform sends any antisymmetric A to an orthogonal matrix -- it
    is the matrix form of the tangent half-angle substitution, and I + A is
    always invertible because an antisymmetric matrix has purely imaginary
    eigenvalues. So A's entries are unconstrained: every value gives a valid
    rotation, and no projection or retraction is needed during the fit.

    Z is then an orthogonal map times a scale, with no stretch at all, which
    makes this the complement of PSDModel. The scale is not optional: an
    orthogonal map cannot change any length, and the inputs and outputs are
    normalised differently, so without it the model would be measuring that
    mismatch rather than the rotation.

    Cayley reaches SO(m) only -- it can never produce an eigenvalue of -1, so
    det = +1 always. `reflect` puts D in the other component of O(m). The two
    components are disconnected, so reflect cannot be a continuous parameter;
    it is swept as a HYPERPARAMETER instead, chosen on the test split alongside
    lambda and reported on validation. Which axis D flips does not matter: any
    two reflections differ by a rotation, which the Cayley factor absorbs.

    For even m a negative scale does NOT reach the reflected component, since
    det(-C) = (-1)^m det(C) = +1; it merely re-describes a solution the positive
    branch already had. The sign of s is therefore redundant, and A recovered
    from a fit with s < 0 describes -C, so negate before reading angles off it.
    """

    def __init__(self, X, Y, reflect=False, **kwargs):
        super().__init__(X, Y, **kwargs)
        self.reflect = bool(reflect)
        # STRICT lower triangle: A is antisymmetric, so its diagonal is zero.
        self.tril = np.tril_indices(self.m, -1)
        self.D = self.D_for(self.m, self.reflect)
        # Item assignment is not differentiable under autograd, so the anp path
        # builds A as a product with this 0/1 selection matrix instead.
        self.tril_basis = np.zeros((self.m * self.m, len(self.tril[0])))
        self.tril_basis[self.tril[0] * self.m + self.tril[1],
                        np.arange(len(self.tril[0]))] = 1.0

    @staticmethod
    def D_for(m, reflect):
        D = np.eye(m)
        if reflect:
            D[0, 0] = -1.0
        return D

    def n_params(self):
        return 1 + self.m * (self.m - 1) // 2

    def A_of_p(self, p):
        A = np.zeros((self.m, self.m))
        A[self.tril] = p[1:]
        return A - A.T

    @classmethod
    def Z_from_p(cls, p, m, reflect=False, tril=None, D=None, **kwargs):
        tril = np.tril_indices(m, -1) if tril is None else tril
        D = cls.D_for(m, reflect) if D is None else D
        I = np.eye(m)
        A = np.zeros((m, m))
        A[tril] = p[1:]
        A = A - A.T
        return p[0] * D @ ((I - A) @ np.linalg.inv(I + A))

    def ZFUN(self, p):
        return self.Z_from_p(p, self.m, tril=self.tril, D=self.D)

    def _anp_Z(self, p):
        A = anp.reshape(anp.dot(self.tril_basis, p[1:]), (self.m, self.m))
        A = A - anp.transpose(A)
        I = anp.eye(self.m)
        return p[0] * anp.dot(self.D, anp.dot(I - A, anp.linalg.inv(I + A)))

    def pack_grad(self, G, p):
        # dZ = -s D (I + C) dA M, with M = (I + A)^-1 and M' = (I - A)^-1, so
        #   dLoss/ds = <G, D C>            -- written this way rather than
        #                                     tr(G'Z)/s, which is 0/0 at s = 0
        #   dLoss/dA = -(sD + Z') G (I-A)^-1
        # A's entries are not independent: one parameter sets A[i,j] and
        # A[j,i] = -A[i,j], so it collects Om_ij - Om_ji. That difference is the
        # projection onto the antisymmetric part and the doubling in one step.
        s = p[0]
        A = self.A_of_p(p)
        C = (self.I - A) @ np.linalg.inv(self.I + A)
        Z = s * self.D @ C
        g_s = np.sum(G * (self.D @ C))
        Om  = -(s * self.D + Z.T) @ G @ np.linalg.inv(self.I - A)
        return np.r_[g_s, (Om - Om.T)[self.tril]]

    def p_from_Z(self, Z):
        """The parameters standing for a Z that is in this class.

        Inverse Cayley: A = (I + C)^-1 (I - C) for C = D' Z / s.
        """
        s = np.linalg.norm(Z) / np.sqrt(self.m)      # ||D C||_F = sqrt(m)
        C = self.D.T @ Z / s
        assert np.allclose(C.T @ C, self.I, atol=1e-8), \
            "Z is not a scaled orthogonal matrix, so it has no parameters here."
        A = np.linalg.inv(self.I + C) @ (self.I - C)
        return np.r_[s, A[self.tril]]

    def init_guess(self, scale = 1e-3, center = 1.):
        # A = 0 gives C = I, so Z = center * D at zero noise.
        print("Initializing guess with scale = ", scale)
        r0 = np.zeros(self.n_params())
        r0[0] = center
        return init_r(self.n_params(), self.λ[0], r0 = r0, scale = scale)

    def p_reg(self):
        # s = 1, A = 0 gives Z = D, which is the regularisation target I only
        # when D = I. With reflect the target is outside the class -- sDC = I
        # would need det(D)/s^m = 1 with det(D) = -1 -- so this is the nearest
        # in-class stand-in rather than an exact hit.
        r = np.zeros(self.n_params())
        r[0] = 1.0
        return r
