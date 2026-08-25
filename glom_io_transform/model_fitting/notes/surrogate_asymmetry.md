# Calibrating the symmetry result — how much asymmetry could we detect?

Companion to `r2_ladder.md`. Math is in `$…$` / `$$…$$` and renders in the
VSCode preview.

---

## 1. The question

Rung 8 of the ladder ties or beats rung 7: constraining $Z = Z^\top$ does not
hurt, so we read the transformation as symmetric. But a null result is only
worth reporting if it has power. If our pipeline could not tell a symmetric
$Z$ from a strongly asymmetric one at 18 odours, "no difference" would mean
nothing.

So: **at what absolute level of asymmetry would `Free` beat `Sym`?**

The word *absolute* is what fixes the design. The tempting construction is

$$Y_\text{surr} = Y_\text{real} + \alpha A X,$$

but that answers a different question — *if there were an extra asymmetric
component on top of whatever is already there, at what $\alpha$ would we catch
it?* We need ground truth we control, running from purely symmetric upward:

$$Y_\text{surr} = (S + \alpha A) X + \kappa\,(Y - \bar{Y}).$$

At $\alpha = 0$ the truth is exactly symmetric; $\alpha = \|A\|/\|S\|$ is the
size of the antisymmetric part relative to the symmetric one.

Two constraints on the implementation: use the **real solvers**, not closed
forms, so the calibration reflects what the pipeline actually does; and generate
as little new code and as few new data files as possible.

---

## 2. Construction

`driver.make_surrogate(XX, YY, alpha, target_r2, seed)`.

**$S$, the symmetric truth.** Computed deterministically from the data, so it
never needs storing separately. Solve the symmetric normal equations
(the Sylvester equation of `fit_cov_resp.tex` §Response Fits) on the
trial-averaged $\bar X, \bar Y$:

$$S\,M + M\,S = \bar X \bar Y^\top + \bar Y \bar X^\top + 2\lambda_0 I,
\qquad M = \bar X \bar X^\top + \lambda_0 I.$$

Averaging over trials already suppresses the noise, which is why a small fixed
$\lambda_0$ suffices rather than a selected one.

**$A$, the antisymmetric part.** Shuffle the entries of $S$, take the
antisymmetric part, rescale to $\|A\| = \|S\|$ so that $\alpha$ reads directly as
the asymmetry ratio. Shuffling preserves the marginal distribution of the
weights. The exact construction is not load-bearing — the claim is only that
*some* level of asymmetry is clearly detectable, not that a particular $A$ is
canonical.

**$\kappa$, the difficulty.** The noise is the real trial-to-trial deviation
$Y_k - \bar Y$, so it has the true magnitude and cross-ROI structure and needs no
noise model; subtracting the mean keeps the real input–output relationship out of
it. $\kappa$ scales it to hit `target_r2`, bisected in log space (monotone
decreasing). Crucially it is **solved at $\alpha = 0$ and then held fixed**, so
varying $\alpha$ varies the asymmetry and nothing else. A consequence worth
expecting: the truth's own $R^2$ then *rises* with $\alpha$ (0.25 at $\alpha=0$,
0.32 at $\alpha=0.5$) because the signal has grown.

**A note on the means.** $\bar X, \bar Y$ are means over the *train* trials
only, not the whole dataset — deliberately, since the held-out trials should not
inform the ground truth. One consequence: the train deviations $Y_k - \bar Y$ are
smaller than the held-out deviation $Y_\text{vld} - \bar Y$ by a factor of about
1.59 at 18 odours, where exchangeable trials would give $\sqrt{(1+1/K)/(1-1/K)}
= 1.11$. That gap is real and systematic across seeds (pairwise distances 2.49 among
trains, 3.31 to vld, 3.22 to test, 12 seeds). The cause is in `TrialsSampler`:
vld and test are each drawn once and removed, then the `n_train` train draws each
sample independently from the *same* remaining pool, so train draws repeat one
another while vld and test are disjoint from that pool and from each other. The
trials are exchangeable; the draws are not. The surrogate's noise *is* the real
deviations, so it starts out carrying this asymmetry — but the outputs are then
run back through `preproc`, which scales each split to std 1 independently, so
the scale part of it is removed exactly as it is for real data. The measured
factors make this concrete: 0.886 for the trains against 0.690 for vld at
$\alpha = 0.5$. $\kappa$ is calibrated on the vld side, which is the one we
report, and R² is unchanged by a per-split rescaling, so it can be solved before
normalizing.

Note the surrogate has no model misspecification, so a correctly-specified fit
lands near `target_r2` — unlike the real data, where fits reach roughly half the
noise ceiling.

---

## 3. Why $\lambda_0 = 0.1$

Necessity first: under the per-odour normalisation $X^\top\mathbf{1} = 0$, so
$\bar X \bar X^\top$ has rank $m-1$ and $\lambda_0 = 0$ fails outright. Any small
value fixes that.

The specific choice was a realism argument — $\lambda_0$ controls the structure of
the truth, and 0.1 gives $S$ a spectrum matching the real symmetric fits. But
since $\kappa$ now carries the difficulty, the two knobs are cleanly separated,
and the calibration turns out to be insensitive to $\lambda_0$ across two orders
of magnitude (6 seeds, closed-form fits, `target_r2` = 0.25):

| $\lambda_0$ | $\|S\|$ | eig($S$) | $\alpha=0$ | $\alpha=0.5$ | $\alpha=1$ |
|---|---|---|---|---|---|
| 0.01 | 8.50 | $[-4.46, +5.82]$ | $+0.006$ | $-0.069$ | $-0.143$ |
| **0.1** | 4.57 | $[-1.95, +2.93]$ | $+0.013$ | $-0.041$ | $-0.121$ |
| 1.0 | 3.65 | $[-0.75, +1.63]$ | $+0.014$ | $-0.036$ | $-0.121$ |

(entries are mean $R^2_\text{Sym} - R^2_\text{Free}$; real `FreeSym` fits have
eigenvalues in $[-1.86, +2.83]$, so $\lambda_0 = 0.1$ sits on top of them and
$\lambda_0 = 0.01$ is well outside.)

So the choice is tidiness, not a load-bearing assumption. Worth stating in the
methods precisely because it looks like a free parameter and isn't.

---

## 4. Result

Through the real solvers (`Free` and `FreeSym` via `driver.run`), 18 odours,
`target_r2` = 0.25:

| $\alpha$ | Sym − Free | truth's own $R^2$(vld) |
|---|---|---|
| 0.00 | $+0.0142$ | 0.250 |
| 0.25 | $-0.0054$ | 0.279 |
| 0.50 | $-0.0467$ | 0.360 |
| 1.00 | $-0.1377$ | 0.558 |

(Closed-form fits, mean over 8 seeds, run locally to check the construction —
the published numbers come from the real solvers on the NEMO sweep, which ran
$\alpha \in \{0, 0.2, \ldots, 1\}$. The truth's own $R^2$ rises with $\alpha$
because $\kappa$ is held at its $\alpha=0$ value, so the signal grows while the
noise does not.)

`Sym` wins when the truth is symmetric and loses progressively as asymmetry
grows. **The sign flips between $\alpha = 0$ and $\alpha = 0.2$** — an
antisymmetric component a fifth the size of the symmetric one is already enough
for `Free` to come out ahead — and from $\alpha = 0.4$ the separation is
unambiguous.

Two things to take from where the observed violin sits. It is on the symmetric
side of zero, and well clear of the crossover, so an asymmetry of even modest
size would have shown up as `Free` winning and did not: the tie in the ladder is
a result, not an absence of power. And it sits *above* the $\alpha = 0$ violin,
not level with it — the real data shows a larger symmetric advantage than an
exactly symmetric ground truth does. Worth saying explicitly in the text, so a
reader does not stop at "indistinguishable from zero asymmetry".

Drawn by `figures.matched_rois.plot_surrogate_alpha` into the `surrogate_alpha`
panel, from `compute.matched_rois.surrogate_r2`. The observed violin is built
from the ladder's own `r2_df` — medianed over trains within a seed, violins over
seeds, exactly as the ladder is — so it is the gap between the ladder's Free and
Sym rungs and not a separate computation. So the null result has power: an asymmetry of even moderate size would
have shown up as `Free` winning, and it does not.

The natural figure is the distribution of **paired** differences (paired per
seed, as the ladder already is) at each $\alpha$ on surrogate data, with the
observed distribution of differences plotted alongside.

---

## 5. Plumbing

Deliberately minimal, and inert by default.

- `get_data(alpha=, target_r2=)` builds the surrogate from the preprocessed
  splits, then runs the outputs back through `preproc`. So `preproc` is the last
  step on both the real and the surrogate path, and the surrogate satisfies the
  same normalization asserts. This works because `preproc` is idempotent: the
  scaler divides by the std it just measured, and `scale_by_cells` by a fixed
  $\sqrt{m}$.
- `alpha=None` is the default and leaves everything untouched — `run()`
  reproduces the pre-change baseline bit-for-bit, generalization figures
  unchanged.
- The ground truth rides back on the returned outputs as `YY.surrogate` rather
  than changing any signature; `run()` stores it in `results["surrogate"]`, so
  every `out.N.p` records the $S$, $A$, $\kappa$, $\lambda_0$ and the per-split
  normalization factors it was fitted against.
- Fit tree: `alpha=<a>` sits after `matched=` and contributes nothing when
  absent, so real fits keep their existing paths. `BaseContext.alpha` reads them
  back.
- `--target-r2` without `--alpha` asserts rather than being silently ignored.

Sweeping — one invocation per $\alpha$; `--variant sym` composes unchanged:

```
--gen ffree_trials_random_max.yaml --loss resp --match-file ... --n-od-train 18_rand_0 \
      --alpha 0.0 --target-r2 0.25
```
