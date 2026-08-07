# Diagonal model — logic of the gains (handoff)

Context for continuing on claude.ai. This is a self-contained primer on where the
theory of the **diagonal (channel-gain) model** stands. Paste it to seed a new chat.
All math is in `$…$` / `$$…$$` and renders on web.

---

## 1. Setup and notation

We fit a diagonal (per-channel gain) model of the olfactory-bulb input→output
transformation. Symbols:

| symbol | space | meaning |
|---|---|---|
| $N$ | scalar | number of cells (glomeruli), $=148$ |
| $n$ | scalar | number of odours, $=48$ |
| $X$ | $N\times n$ | input responses; row $x_i$ = cell $i$'s profile over odours. Columns are per-odour $z$-scored, so $\mathbf{1}^\top X = 0$ |
| $C$ | $n\times n$ | **target** output covariance |
| $z$ | $\mathbb{R}^N$ | per-cell gains (the unknowns) |
| $[z]$ | $N\times N$ | $\mathrm{diag}(z)$ |
| $J$ | $N\times N$ | cell-space centering, $J = I - N^{-1}\mathbf{1}\mathbf{1}^\top$ |

The model's predicted output covariance and the loss:

$$
C_{\text{pred}}(z) = X^\top [z]\, J\, [z]\, X,
\qquad
L(z) = \tfrac12\big\| C_{\text{pred}}(z) - C \big\|_F^2 .
$$

There is no separate regularizer at the operating point — see §4.

Derived objects (both depend on $z$):

$$
E(z) \;\triangleq\; C_{\text{pred}} - C \;\in\; \mathbb{R}^{n\times n}\ \text{(prediction} - \text{target)},
\qquad
M(z) \;\triangleq\; X E X^\top \;\in\; \mathbb{R}^{N\times N},
\qquad
D(z) \;\triangleq\; \mathrm{Diag}(M).
$$

Note $M_{ii} = x_i^\top E\, x_i$. **Sign convention:** the manuscript Methods define
$E_{\text{ms}} = C - C_{\text{pred}}$ (opposite sign); the eigenproblem below is
invariant to it, but the first-order wording ("aligned with the residual") flips.

---

## 2. Exact stationarity condition

Differentiating $L$ (with $A \triangleq [z]X$, so $C_{\text{pred}} = A^\top J A$):

$$
\frac{\partial L}{\partial z_i} = 2\,[\,J A E X^\top\,]_{ii}
= 2\big(D z - N^{-1} M z\big)_i .
$$

Setting $\nabla L = 0$:

$$
\boxed{\; M z = N D z \;\;\Longleftrightarrow\;\; D^{-1} M z = N z \;}
$$

A **generalized eigenvalue problem** — but implicit, since $M = M(z)$ through $E$.

Equivalent Hadamard form (useful): with $H = X E X^\top$ and $J = I - N^{-1}\mathbf 1\mathbf 1^\top$,

$$
K(z)\,z = 0, \qquad K = J \odot H = \mathrm{Diag}(H) - N^{-1} H .
$$

---

## 3. Structure of the eigenproblem

**The eigenvalue is pinned.** $\mathrm{tr}(D^{-1}M) = \sum_i M_{ii}/M_{ii} = N$
identically. So the $N$ eigenvalues always sum to $N$; the solution $z$ is the
eigenvector whose eigenvalue equals the trace, forcing the other $N-1$ to sum to zero.

**It is NOT the top eigenvalue.** Empirically $D^{-1}M$ has one real eigenvalue
above $N$ and a large negative one, with the solution's eigenvalue at exactly $N$
sitting *interior*. So the "$z$ = leading Fisher discriminant" reading is **wrong** —
$z$ is a saddle of the Rayleigh quotient $z^\top M z / z^\top D z$, not a max.

**The spectrum is complex.** Conjugate pairs appear because $D = \mathrm{Diag}(x_i^\top E x_i)$
has **negative entries** (most of them, in fact — a few tiny positive). $E$ is a
residual, hence indefinite; the pencil $(M,D)$ is symmetric-indefinite. Whitening
$D^{-1/2} M D^{-1/2}$ is complex, so no clean LDA picture globally.

**But $z$ itself is clean:** it's the real, isolated eigenvalue at exactly $N$,
computable as $z \in \mathrm{null}(M - N D)$ — a real solve, immune to the complex bulk.

**$M$ is effectively rank 3.** $\mathrm{spec}(M)$: two large positive
($\approx 10,\,4.3$), one large negative ($\approx -5$), rest $\approx 0$.
So the residual lives in $\sim 3$ odour-space modes. (Compare: the *free* model's
connectivity is also captured by 3 modes — a genuine parallel between the two figures.)

---

## 4. Empirical facts from the refit (covariance convention, low $\lambda$)

- **Diag is regularization-insensitive:** test/vld $r^2 \approx 0.185$, flat over
  $\lambda \in [10^{-8}, 10^{-3}]$. Confirmed real (survives optimizer change +
  reg-target start), not a convergence artifact. Operating point is $\lambda \to 0$.
  Reason: $148$ params vs $\sim1176$ covariance entries → **over**determined → cannot
  overfit → plateau. (The *free* model, $21904$ params, is underdetermined → interior
  $\lambda^\ast$.)
- **Gains are non-perturbative:** fitted $z$ ranges $\approx -4.5$ to $+2.3$. Not near
  the old regularization target of $1$. Sign flips: $\sim2/3$ negative. A middle
  plateau near $z\approx 0$ (silenced cells), large-$|z|$ tails (working cells).
  → The old linearization-around-$z=1$ theory is dead; that's why we need the
  unregularized treatment.

---

## 5. The data-only result (drop the mean term)

Split $C_{\text{pred}} = X^\top[w]X - N^{-1} b b^\top$ with $w = z\odot z$ and
$b = X^\top z$. The $-N^{-1}bb^\top$ is the **only subtractive term** (structurally
load-bearing for decorrelation) but is second order in the gain deviations.
Dropping it gives a **convex nonnegative least squares**:

$$
\min_{w \ge 0}\ \tfrac12\big\| \textstyle\sum_i w_i\, x_i x_i^\top - C \big\|_F^2,
\qquad
G w = c,\quad G_{ij} = (x_i\!\cdot\! x_j)^2,\quad c_i = x_i^\top C\, x_i .
$$

So, to leading order, $w = G^{-1} c$ — **entirely from $X$ and $C$, no $z$, no $E$.**

**Interpretation.** $c_i = x_i^\top C x_i$ is unit $i$'s alignment with the target
covariance; $G_{ij} = (x_i\!\cdot\!x_j)^2$ is squared input-similarity (redundancy).
$G^{-1}$ discounts for redundancy:

> A glomerulus's (squared) gain is its alignment with the needed covariance change,
> divided among all the glomeruli that could produce it.

### 5b. Same object via the linear-ramp ansatz (independent derivation)

Ansatz: under some permutation $I$ of cells, gains are a linear ramp
$w_n = a + (b-a)n/N$. Objective (mean term dropped):

$$
\min_{I}\ \tfrac12\sum_{ij}\Big(\sum_n X_{I(n),i}X_{I(n),j}\,(a + (b-a)n/N) - C_{ij}\Big)^2 .
$$

The constant part $a\,X^\top X$ is **permutation-invariant** (only the slope sees the
order). With $r_m$ = rank of cell $m$, this reduces to a quadratic assignment

$$
\min_{r\,\in\,\text{perms}(1..N)}\ r^\top G\, r - 2 r^\top t,
\qquad t = c - a\,G\mathbf 1 .
$$

Continuous relaxation: $r^\ast = G^{-1} t = G^{-1} c - a\,\mathbf 1$. The $-a\mathbf 1$
is a constant shift, so it **doesn't change the ordering**:

$$
\boxed{\ \text{optimal order} = \operatorname{argsort}\big(G^{-1} c\big),\quad
\text{independent of } a,b.\ }
$$

The ramp ansatz's ordering **is** the unconstrained gain $w^\ast = G^{-1}c$ from §5.
Two routes → same data-only object. The ramp fits iff sorted $G^{-1}c$ is roughly
linear (the empirical sorted-gain plot is "not too far off," with an S-shape + a jump
near index 98 that likely marks two clusters).

---

## 6. Open threads / next checks

1. **Mean-term magnitude.** Dropped $-N^{-1}bb^\top$; correction to $c$ is
   $N^{-1}(Xb)_i^2$ (2nd order in $b$). Compute $\|N^{-1}(Xb)^2\|/\|c\|$ on the fit.
   If small, $G^{-1}c$ is the honest predictor; if not, the mean term isn't a correction.
2. **Does $G^{-1}c$ actually track the gains?** Scatter (and Spearman-rank) $G^{-1}c$
   vs fitted $w=z^2$. Rank is the robust test given the nonlinear sorted-gain shape.
3. **Sign vs magnitude split.** $w=z^2$ (usage) is what §5 predicts; $\mathrm{sign}(z)$
   lives in the mean term $b$ ($\le 48$-dim, maybe $\sim3$-dim). Explain separately.
4. **Rank 3 vs 4.** Two clear positive $M$-modes + one negative, maybe a 2nd small
   negative. Report the honest mode count.
5. **Negative-$D$ / tiny-$D$ units** (the red-dotted cells): $z_i = (Mz)_i/(N D_{ii})$
   blows up unless $(Mz)_i \to 0$ there too. Check they don't dominate any reconstruction.

## 7. Figure plan (for reference)

- **Fig 6:** top-level Diag vs Free — schematics, transformations, violin plots
  (generalization across trials + across odours; use `outclass` split; metric =
  correlation energy for consistency).
- **Fig 7:** logic of the diagonal (and plain free) gains — this document.
- **Alternates** (DiagInhOnly, FreeLat) → SI, few sentences in main text.
