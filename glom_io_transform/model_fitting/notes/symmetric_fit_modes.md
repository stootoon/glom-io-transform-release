# The symmetric fit: its modes, its spectrum, and the mean

Companion to [`r2_ladder.md`](r2_ladder.md) and
[`surrogate_asymmetry.md`](surrogate_asymmetry.md). The ladder establishes that
the transformation is symmetric and the surrogate sweep shows that result has
power; this note is about what that symmetric connectivity actually *is*.
Math is in `$…$` / `$$…$$`; open with `Cmd+Shift+V`.

---

## 1. Notation

| symbol | shape | meaning |
|---|---|---|
| $m$ | | matched glomeruli, $=16$ |
| $n$ | | odours, $=18$ |
| $X, Y$ | $m \times n$ | input and output responses, columns = odours |
| $\Sigma_{xx}$ | $m \times m$ | $XX^\top + \lambda I$ |
| $C$ | $m \times m$ | $XY^\top + YX^\top + 2\lambda I$ |
| $Z$ | $m \times m$ | the symmetric fit |
| $D_i, v_i$ | | eigenvalues/vectors of $XX^\top$, **largest first** |
| $\widetilde Z$ | $m \times m$ | $V^\top Z V$ — the connectivity in the input's eigenbasis |
| $c$ | $m$ | an input's coefficients, $c = V^\top x$ |

$XX^\top$ and $\Sigma_{xx}$ share eigenvectors, so "the eigenbasis of $X$" is unambiguous.

---

## 2. The closed form

The symmetric fit solves the Sylvester equation

$$Z\,\Sigma_{xx} + \Sigma_{xx}\,Z = C.$$

In the eigenbasis this is **entrywise**, which is the whole reason to work here:

$$\boxed{\;\widetilde Z_{ij} = \frac{\widetilde C_{ij}}{D_i + D_j + 2\lambda}\;}$$

Two consequences worth stating separately.

**It is a Hadamard product.** $\widetilde Z = \widetilde C \circ K$ with the Cauchy matrix $K_{ij} = 1/(D_i + D_j + 2\lambda)$. $K$ is positive definite, since

$$K_{ij} = \int_0^\infty e^{-(D_i+\lambda)t}\,e^{-(D_j+\lambda)t}\,dt = \int_0^\infty u_i(t)\,u_j(t)\,dt,$$

a Gram matrix. By the Schur product theorem this preserves definiteness — but $\widetilde C$ is *indefinite* here, and the Hadamard product does **not** preserve inertia in general (about 30% failure on ill-conditioned random cases). It happens to nearly hold on this data; see §6.

**Small $D$ means large weight.** $1/(D_i+D_j+2\lambda)$ is *largest* where the input has *least* power. So the fit amplifies exactly the directions where $\widetilde C$ is least well determined, capped at $1/2\lambda$. This is the usual ill-conditioned-regression blow-up, and it is the reason to be suspicious of the low-rank corner — except for the last mode, which is a different animal entirely (§3).

---

## 3. The last mode is the mean, not noise

### 3.1 Why it has exactly zero variance

The per-odour normalisation $z$-scores each odour across glomeruli, so every column of $X$ has zero mean across ROIs:

$$\mathbf{1}^\top X = 0 \quad\Longrightarrow\quad XX^\top \mathbf{1} = 0.$$

So $\mathbf{1}/\sqrt m$ is an eigenvector of $XX^\top$ with eigenvalue exactly $0$, and $\operatorname{rank}(X) = 15$, not 16. Measured:

| quantity | value |
|---|---|
| $D_{16}$ | $5.7\times 10^{-17}$ |
| $\lvert\langle v_{16},\, \mathbf{1}/\sqrt m\rangle\rvert$ | $1.000000$ |
| $\max_x \lvert v_{16}^\top x\rvert$ on vld | $1.9\times 10^{-14}$ |

$v_{16}$ **is** the uniform-across-glomeruli direction, and the input has no component along it. Not "little power" — *no* power.

### 3.2 Row and column are not interchangeable

Writing $y = \widetilde Z c$ in mode coordinates, the last coefficient is identically zero, $c_{16} \equiv 0$. Therefore:

- **Column 16** multiplies $c_{16} = 0$. It is inert.
- **Row 16** produces $y_{16} = \sum_j \widetilde Z_{16,j}\, c_j$, the output's across-glomerulus mean, built from the input's *structured* modes. It is essential.

$\widetilde Z$ is symmetric, so the row and the column hold the same numbers, which is why both show up in the heat map — but only one does work. Ablation on held-out data, median over 12 seeds:

| | $R^2$(vld) |
|---|---|
| intact | $+0.2479$ |
| zero last **column** | $+0.2479$ |
| zero last **row** | $-0.2193$ |
| zero both | $-0.2193$ |

So the corner carries $\approx 0.47$ of $R^2$.

### 3.3 The action, corrected

The reading "$y_i = \widetilde Z_{ii} c_i + \widetilde Z_{iN} c_N$" has its second term vanish. The actual structure is

$$y_i = \widetilde Z_{ii}\, c_i + \text{(small off-diagonal)}, \quad i < 16, \qquad
y_{16} = \sum_{j<16} \widetilde Z_{16,j}\, c_j .$$

### 3.4 This is $\bar Z$ arriving from a different direction

$\bar Z = (I - J)Z = \mathbf{1}b'^\top$ is the mean component: rank one, constant across ROIs, varying by odour — and the component the covariance loss cannot see, since $\mathcal{L}_\text{cov}$ contains it only through the regulariser. Row 16 of $\widetilde Z$ **is** $b'$ expressed in the input eigenbasis.

That closes a loop with the ladder: `Z_cov` is blind to this row, which is exactly why the `Z_cov + Z̄_resp` rung had to borrow it, and why borrowing it was worth $\approx 0.45$ of $R^2$ there. Same object, two routes.

### 3.5 What remains open

The cumulative ablation confounded mode 16 with modes 13–15, so it says nothing about them. Those have genuinely *low* but *nonzero* variance, and are where the $1/(D_i+D_j)$ amplification of §2 should bite. The original intuition — low-power directions are mostly noise — is untested there and worth a separate ablation that skips mode 16.

### 3.6 Figure consequence

Mode 16 is different in kind, not merely last in an ordering. Shown as the final column of a variance-ordered sequence, a reader will take the corner for the strongest ordinary mode. It wants a label ("mean") or a separating rule.

---

## 4. What the panel measures, and the sign convention

$\widetilde Z$ is averaged over seeds. Two conventions are forced:

**Rank matching.** Modes have no identity across seeds; they are matched by rank, most variable first.

**Sign alignment.** An eigenvector is defined only up to sign, and $v_i \to -v_i$ flips row and column $i$ of $\widetilde Z$. Averaging raw therefore cancels the off-diagonals. Each seed's basis is aligned to a reference seed's, $s_i = \operatorname{sign}(v_i^\top v_i^{\text{ref}})$, before averaging. Off-diagonal energy: $4.14$ aligned vs $1.30$ naive — so without this the panel would show a spuriously diagonal matrix.

The diagonal is immune: $\widetilde Z_{ii}$ is unchanged by a flip.

---

## 5. Why the symmetric refit beats the free fit, and post-hoc symmetrisation does not

From the same eigenbasis relation, with $F = Z_\text{free}$:

$$\widetilde S_{ij} = \frac{\widetilde F_{ji} D_i + \widetilde F_{ij} D_j}{D_i + D_j}.$$

Since $\operatorname{Var}(\widetilde F_{ij}) \propto 1/D_j$, the precisions are $D_j$ and $D_i$, so this is the **inverse-variance weighted** average of $\widetilde F_{ij}$ and $\widetilde F_{ji}$ — the optimal combination of two noisy estimates of the same weight, if the truth is symmetric.

Naive symmetrisation $(Z + Z^\top)/2$ is the **equal-weight** average of the same two numbers. They agree only when $D_i = D_j$. For $D_j \ll D_i$:

$$\operatorname{Var}(\text{optimal}) \approx \frac{1}{D_i}, \qquad
\operatorname{Var}(\text{equal-weight}) \approx \frac{1}{4D_j},$$

worse by $\sim D_i/4D_j$. And worse than leaving the matrix alone, because averaging pours noise from the ill-determined direction into the well-determined one, degrading *both* entries of the pair. Hence `Z_sym` (refit) at the top of the ladder and `Z_resp_sym` (ablation) down near PSD — not a contradiction, and not evidence against symmetry.

**Corollary.** `Z_resp_sym` is *not* a valid ablation test of whether the antisymmetric part carries signal, since deleting it also redistributes noise. The parameter-level comparison against the $\alpha=0$ surrogate null is the test that works.

---

## 6. Inertia: where the sign split comes from

The positive/negative split of $Z$'s spectrum is basis-free (it is the inertia), and it is *nearly* inherited from $C = XY^\top + YX^\top + 2\lambda I$, which requires no fit at all:

| | |
|---|---|
| inertia of $Z_\text{sym}$ vs $C$ | exact on 4/8 seeds, off by one mode otherwise |
| correlation of sorted spectra | $+0.85$ to $+0.91$ |

Not a theorem (§2), but it holds here because $\Sigma_{xx}$ is only moderately conditioned (60–500 at the selected $\lambda$); with all $D_i$ equal, $K$ is constant and $Z \propto C$ exactly.

If it survives scrutiny it is worth stating, because it says the negative modes are visible in the raw symmetrised cross-covariance and are not an artefact of fitting: a negative mode is a direction along which input and output **anti-covary**.

Two cautions:

- The negative-mode **count** is not stable across seeds (2–6 of 16, mode 5). "Several, a minority, never zero" is the defensible claim — and *never zero* is what matters, since zero is what $W = GG^\top$ predicts.
- The **sign** split (positive vs negative) is basis-free; the **magnitude** split (amplified vs attenuated, $\lvert z\rvert$ vs 1) is not — it depends on the output normalisation, since it asks whether a gain exceeds one.

---

## 7. Spectrum summary

16 modes × 12 seeds, $\lambda$ selected per seed:

| regime | share of modes | per-seed count |
|---|---|---|
| $z > 1$, amplified | 30% | 4–6, mostly 5 |
| $0 < z \le 1$, reciprocal-admissible | 43% | — |
| $z < 0$, inverted | 27% | 2–6, mostly 5 |

Reciprocal dendrodendritic inhibition gives $W = GG^\top$, hence $Z = (I + GG^\top)^{-1}$, whose eigenvalues all lie in $(0,1]$. **Only ~40% of the spectrum is inside that band, and it is violated at both ends.** Inverting $w = 1/z - 1$: the amplified modes have $w \in [-0.65, -0.04]$, negative but above $-1$ and so dynamically stable (readable as disinhibition); the inverted modes have $w < -1$, which is unstable under the reciprocal model and not explicable by it at all.
