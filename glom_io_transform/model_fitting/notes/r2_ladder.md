# The R² ladder — what the matched-glomeruli transformation requires

Notes for the manuscript coda. Math is in `$…$` / `$$…$$` and renders in the
VSCode preview.

The coda rests on the 16 glomeruli matched between input and output. Because
they are matched, we can fit connectivity to the **responses** directly, rather
than to their covariances, which is what the rest of the paper does. The ladder
is the device for reading off what that transformation actually needs.

Built by `analysis/compute/matched_rois.build_r2_fits`, drawn by
`analysis/figures/matched_rois.Main`.

---

## 1. Setup

| symbol | meaning |
|---|---|
| $m$ | matched glomeruli, $=16$ |
| $X, Y$ | input and output responses, $m \times n$, columns = odours |
| $Z$ | the fitted connectivity, $m \times m$ |
| $J$ | centering across ROIs, $J = I - m^{-1}\mathbf{1}\mathbf{1}^\top$ |

Every rung is scored the same way: **held-out $R^2$ on the validation odours**,
with $\lambda$ selected on test. All values below are at 18 training odours.

Two facts do the structural work.

**The covariance loss cannot see the mean component.** Split
$Z = JZ + \bar{Z}$, with $\bar{Z} = (I-J)Z = \mathbf{1}b'^\top$: rank one,
constant across ROIs, varying by odour. The two components are orthogonal, so
both losses decompose over them — but the covariance loss contains $\bar{Z}$
only through its regulariser, which sends it to $\bar{I}$. So a covariance fit
carries no information about $\bar{Z}$, and any comparison of connectivities
across the two losses has to be made on $JZ$. (Derivation in `fit_cov_resp.tex`.)

**Polar decomposition separates the two things $Z$ can do.** Write
$JZ = QP$ for a rotation $Q$ and a PSD stretch $P$. Since
$(JZ)^\top(JZ) = P^\top P$, the covariance fit determines only $P$; the
response fit determines both. That is exactly the axis the ladder walks along.

---

## 2. The rungs

| # | rung | what it is | ≈ $R^2$(vld) |
|---|---|---|---|
| 1 | Input | $Y$ predicted by $X$, i.e. $Z = I$ | $-1.6$ |
| 2 | Free (cov) | $Z$ fitted to covariances, applied to responses | $-1.2$ |
| 3 | Free (cov, bl) | $Z_\text{cov} + \bar{Z}_\text{resp}$ | $-0.75$ |
| 4 | PSD refit | $Z = LL^\top$, refitted on responses | $-0.05$ |
| 5 | Rot | $Z = s\,C$, $C \in SO(m)$, refitted | $+0.03$ |
| 6 | Orth | $Z = s\,D\,C$, $D$ sweeping both components of $O(m)$ | $+0.05$ |
| 7 | Free (resp) | unconstrained $Z$ fitted to responses | $+0.23$ |
| 8 | Sym refit | $Z = Z^\top$, refitted on responses | $+0.27$ |
| 9 | Output | one trial of $Y$ against another — the noise ceiling | $+0.55$ |

Rung 1 is the basement, rung 9 the spire. Everything between them is the
transformation the bulb performs, and the ladder says which structural
ingredient buys which part of it.

### Why rung 3 is a borrowed baseline, not a refit

$Z_\text{cov}$ is blind to $\bar{Z}$, so applying it to responses is unfair in a
way that has nothing to do with connectivity. The fix is to lend it the mean
component from the response fit: $Z_\text{cov} + \bar{Z}_\text{resp}$. No new
fitting, no new hyperparameters, no new model. The point of the rung is not to
compete — it is to show that $JZ_\text{cov}$ and $JZ_\text{resp}$ capture
overlapping structure, specifically the same stretch, so the covariance analysis
in the body of the paper and the response analysis in the coda are looking at the
same object from different sides. Its success then motivates the PSD rung.

A learned version ($aZ_\text{cov} + \mathbf{1}b'^\top$) was tried and dropped: at
18 odours the fitted $a$ fluctuates around $-0.2$ and the rung is unstable. The
borrowed baseline is both simpler and better.

### Why there is no affine rung

$aX + b$ looks like a natural "units and offset" control, but it is vacuous. The
fitted $a$ is the pooled input–output correlation, $-0.05$ to $-0.11$, so
$R^2 \approx a^2 \approx 0.005$ — the rung just restates that $R^2$ is defined
against the mean. Nothing is learned by including it.

---

## 3. What the ladder says

**Neither ingredient alone suffices.** Pure stretch (PSD, rung 4) and pure
rotation (rung 5) both land near zero, far below the free fit at $+0.23$. The
transformation is not a gain change and it is not a rotation.

**Reflections are cheap and they matter.** $SO(m) \to O(m)$ (rung 5 → 6) is worth
a small but consistent gain. Note this needed a fixed $D = \mathrm{diag}(-1,1,\dots,1)$:
the Cayley parameterisation $C = (I-A)(I+A)^{-1}$ reaches only $SO(m)$, and for
even $m$ a negative scale does not escape it, since $\det(-C) = +1$.

**The transformation is symmetric.** The symmetric refit (rung 8) matches and
slightly exceeds the unconstrained fit (rung 7), $\approx +0.27$ against
$+0.23$. Constraining $Z = Z^\top$ costs nothing and buys generalisation. See
`surrogate_asymmetry.md` for the calibration establishing that this null result
has power.

**Nearly half the ceiling is unexplained.** Rung 8 at $+0.27$ against a noise
ceiling of $+0.55$. A linear symmetric $Z$ is not the whole story.

---

## 4. Reading the symmetric solution

Reciprocal dendrodendritic synapses give $W = GG^\top$ and hence
$Z = (I + GG^\top)^{-1}$: symmetric, PSD, eigenvalues in $(0,1]$. The fitted
$Z_\text{sym}$ is symmetric, as predicted, but its spectrum leaves that interval.
Inverting $w = 1/z - 1$ per eigenvalue splits the modes in two:

- **5 amplified modes**, $w \in [-0.65, -0.04]$. Negative but above $-1$, so
  dynamically stable. Consistent with disinhibition — an excitatory effective
  mode on top of the $+GG^\top$ inhibition, plausibly not granule cells.
- **5 inverted modes**, all $w < -1$. Dynamically unstable under the reciprocal
  model, so they cannot be explained by it at all. Feedforward inhibition is the
  natural alternative.

So the symmetry is consistent with the reciprocal architecture, but the gains are
not — which is a discussion point, not a result.

---

## 5. Relating rungs 7 and 8

`fit_cov_resp.tex` works this out. In the eigenbasis of
$\Sigma_{xx} = XX^\top + \lambda I$,

$$\widetilde{S}_{ij} = \frac{\widetilde{F}_{ji} D_i + \widetilde{F}_{ij} D_j}{D_i + D_j},$$

where $F$ is the free solution, $S$ the symmetric one, and $D_i$ the eigenvalues
of $\Sigma_{xx}$. Since $\mathrm{Var}(\widetilde{F}_{ij}) \propto 1/D_j$, this is
a **precision-weighted average** of $\widetilde F_{ij}$ and its transpose: the
symmetric fit is not throwing information away, it is combining two noisy
estimates of the same weight. That is why rung 8 can beat rung 7 rather than
merely tie it.

What it does *not* explain is why the true weights should be symmetric — one can
easily construct a system whose true $Z$ is asymmetric. That question is
architectural, and is where the $W = GG^\top$ argument above comes in.
