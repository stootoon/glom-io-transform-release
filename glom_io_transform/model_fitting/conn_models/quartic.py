import numpy as np

class Quartic:
    @staticmethod
    def OFF(z, i):
        z = z.copy()
        z[i] = 0
        return z

    def __init__(self, X, C, z, λ):
        self.X, self.C, self.z, self.λ = X, C, z, λ
        self.N, self.n_od = X.shape
        self.α = (self.N - 1)/self.N
        self.m = self.X.T @ self.z
        self.E = self.EFUN(z)
        self.build_vars()

    def build_vars(self, **kwargs):
        # Geometric quantities from notes/quartic_analysis.pdf, one entry per unit i
        self.mi_ = np.array([self.MiFUN(i) for i in range(self.N)])  # mᵢ = m - zᵢxᵢ
        Ei       = [self.EiFUN(i) for i in range(self.N)]
        self.pi_ = np.sum(self.X**2, axis=1)                                                  # pᵢ = xᵢᵀxᵢ, raw channel power
        self.qi_ = np.array([Xi @ E_i @ Xi  for Xi, E_i      in zip(self.X, Ei)])             # qᵢ = xᵢᵀEᵢxᵢ, residual channel power
        self.ai_ = np.array([Xi @ m_i       for Xi, m_i      in zip(self.X, self.mi_)])       # aᵢ = xᵢᵀmᵢ, raw alignment with the mean
        self.bi_ = np.array([Xi @ E_i @ m_i for Xi, E_i, m_i in zip(self.X, Ei, self.mi_)])   # bᵢ = xᵢᵀEᵢmᵢ, residual alignment with the mean
        self.m2_ = np.sum(self.mi_**2, axis=1)                                                # the notes' m = mᵢᵀmᵢ, raw power of the mean
        self.ki_ = self.α * self.qi_ + (self.pi_ * self.m2_ + self.ai_**2)/self.N**2          # κᵢ, curvature at the origin

        # Quartic coefficients Azᵢ⁴ + Bzᵢ³ + Czᵢ² + Dzᵢ, Eq. (2)-(3). Overridable
        # via kwargs so pair-averaged coefficients can be substituted: averaging
        # must happen at this level (A needs ⟨p²⟩, B needs ⟨p·a⟩), not on the
        # geometric quantities above.
        self.A_ = kwargs.get("A_", 1/2*self.α**2  * self.pi_**2)
        self.B_ = kwargs.get("B_",-2*self.α/self.N * self.pi_ * self.ai_)
        self.C_ = kwargs.get("C_", self.ki_ + self.λ/2)
        self.D_ = kwargs.get("D_",-(2 * self.bi_/self.N + self.λ))

        # Monic quartic z⁴ + bz³ + cz² + dz
        self.b_ = self.B_/self.A_
        self.c_ = self.C_/self.A_
        self.d_ = self.D_/self.A_
        # Depressed quartic s⁴ + c̃s² + d̃s in s = z - z̄
        self.zbar_ = -self.b_ / 4
        self.ct_   =  self.c_ - 3/8*self.b_**2
        self.dt_   =  self.d_ - self.b_ * self.c_/2 + self.b_**3/8
        self.g_    =  self.ct_/2
        self.h_    =  self.dt_/4

        self.LiFUN = lambda i, z: self.A_[i] * z**4 + self.B_[i]* z**3 + self.C_[i]*z**2 + self.D_[i]*z
        self.Li1FUN= lambda i, z: 4 * self.A_[i] * z**3 + 3 * self.B_[i] * z**2 + 2 * self.C_[i]*z + self.D_[i]
        return self

    def EFUN(self, z):
        return self.MFUN(z) - self.C

    def EiFUN(self, i):
        return self.EFUN(self.OFF(self.z,i))

    def MFUN(self, z):
        m = self.X.T @ z
        return self.X.T @ np.diag(z**2) @ self.X - np.outer(m, m)/self.N

    def MiFUN(self, i):
        return self.m - self.z[i] * self.X[i]

    def wells(self, i):
        """Real minima in gain units, and whether they straddle zero."""
        r = np.roots([4*self.A_[i], 3*self.B_[i], 2*self.C_[i], self.D_[i]])
        r = np.real(r[np.abs(np.imag(r)) < 1e-9])
        mins = r[12*self.A_[i]*r**2 + 6*self.B_[i]*r + 2*self.C_[i] > 0]
        return mins, (len(mins) > 1 and mins.min()*mins.max() < 0)

    def escapes(self):
        out = []
        for i in range(self.N):
            mins, straddle = self.wells(i)
            if straddle:
                best = mins[np.argmin(self.LiFUN(i, mins))]
                if abs(best - self.z[i]) > 1e-6:
                    out.append((i, self.z[i], best, self.LiFUN(i, best) - self.LiFUN(i, self.z[i])))
        return out

