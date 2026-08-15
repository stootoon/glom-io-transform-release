# Matched-glomeruli analysis — design notes

*Companion to `math_scratch.md` (which holds the derivations). Open with `Cmd+Shift+V` for the rendered view.*

## 1. Motivation

Reviewers asked why we didn't match input and output glomeruli and fit connectivity directly, instead of fitting representations. We now can, for a subset: a colleague has identified **16 glomeruli with both input and output recordings** (cf. Figures S3.3–S3.6, where the matching procedure and the preserved decorrelation in the matched subset are already reported).

**Planned framing:** a concluding, *supporting* section — "we fitted representations because channel identities were unavailable; for 16 matched channels they are available, so we fit connectivity directly and compare." Not a load-bearing result; a validation of the approach plus a genuinely new measurement (see §3).

## 2. What the reader will want to know

- **(a)** Does connectivity estimated from responses agree with connectivity estimated from representations?
- **(b)** Do the paper's Diag-vs-Free conclusions hold in this small subset?
- **(c)** Do conclusions from the full-dataset representation fits carry over?

## 3. Key conceptual results from the discussion

### 3.1 Identifiability: representations see the stretch, not the rotation

The rep objective depends on $W$ only through $M = W^T J W$. Polar-decomposing $W = QS$ (rotation × symmetric stretch), the objective sees $S$ **exactly** and is **completely blind** to $Q$; the regularizer picks the pure-stretch representative ($Q = I$), which is why our fitted $W_{\text{rep}}$ is symmetric. This is *not* the same as "seeing the symmetric part of $W$" (a pure rotation has the same representation as the identity but a very different symmetric part). Derivation in `math_scratch.md`.

**Exact partition of degrees of freedom:** $S$ carries $m(m+1)/2$, $Q$ carries $m(m-1)/2$. Fitting reps determines the stretch; fitting responses additionally determines the rotation. No overlap, no remainder.

**Consequences.**
- Rep fits are *not* uninformative about real weights even if response fits differ — they pin down the stretch and are silent on the routing.
- The matched data let us **measure the rotation for the first time**. That is the section's novel measurement, not just a validation.
- Framing sentence: *representation fitting determines how the circuit reshapes similarity structure (the stretch) but not how activity is routed across channels (the rotation); the matched glomeruli let us measure the routing, and the gains — which have no routing freedom — are where the two approaches can be compared exactly.*

### 3.2 Diag is fully identifiable from representations

A diagonal $W$ has no rotation freedom (its polar rotation is just $\operatorname{sign}(z_i)$, and the signs are fixed by the centering/tilt term — see the quartic analysis). So **gains are the one place where rep fitting makes fully identifiable claims about the real weights**, which is why gain agreement is the headline test.

### 3.3 Gains do not transfer across population contexts

The same channel embedded in the full 148-channel population vs the 16-channel subset will *not* get the same gain: both the redundancy $g_i$ and the tilt $h_i$ depend on the residual left by all other units and on the population mean over those units. The fitting *target* also changes (16-channel output covariance vs the 167-ROI representation). **Therefore the only meaningful gain comparison is subset-rep vs subset-resp**, same channels, same context, same target.

Question (c) must therefore be answered at the level of *conclusions* (does Diag ≈ Free replicate?) rather than parameters. One optional parameter-level test survives: the Diag model is channel-local ($\hat y_i = z_i x_i$), so the **published full-population gains make predictions for the 16 matched channels** without needing correspondence for the other 132; those can be scored against the measured matched outputs (up to overall scale).

### 3.4 Correction: rep-Free is not handicapped by symmetry

Symmetry is where the regularizer *resolves an ambiguity*, not a constraint that costs performance: any representation achievable by an asymmetric $W$ is achieved exactly by a symmetric one. So "Diag ≈ Free at the representation level" **cannot** be an artifact of enforced symmetry.

### 3.5 Scenario taxonomy (for $W_{\text{resp}}$ vs $W_{\text{rep}}$)

| # | Outcome | Reading |
|---|---------|---------|
| 0 | $W_{\text{resp}}$ barely beats identity | Matched data too noisy to constrain connectivity beyond gains — still a serviceable answer to the reviewers. Distinguished from (3) by the noise ceiling. |
| 1 | $W_{\text{resp}}$ ≈ symmetric and matches $W_{\text{rep}}$ | True effective map is near pure stretch; rep fit measured the real thing. Test asymmetry against a trial-split null (raw asymmetry is guaranteed). |
| 2 | Asymmetric but predictions match | Asymmetric part is noise in ill-constrained directions; ridge shrinks exactly that. **More likely than it first appears** (see §3.6), and partly favoured by similarity-based matching. |
| 3 | Asymmetric and predicts better | Some gap is *guaranteed* ($W_{\text{resp}}$ optimizes the scored metric), so score as *fraction of explainable variance*, not a binary. Conclusions survive: reps identify the stretch; Diag conclusions untouched. |
| 4 | Free(resp) ≫ Diag(resp) | Real lateral structure that matters for channel-level prediction but is invisible to representations. Sharpens the existing caveat from possibility to demonstrated fact; does **not** overturn the headline (which is about explaining the representational transformation). Decide this framing *before* seeing the answer. |

### 3.6 The response regression is less determined than naive counting suggests

$47 \times 16 \approx 750$ equations vs $256$ unknowns looks comfortable, but the constraint enters through the input covariance, and inputs are low-dimensional (~10 PCs for 90% variance, Figure 3). Effective well-constrained directions per row are more like 10–15. So $W_{\text{resp}}$ needs ridge, and its ill-constrained components are noise — **generically asymmetric**. Asymmetry per se is therefore uninformative; the informative question is whether the asymmetric part *predicts*.

### 3.7 Relating $W_{\text{free}}$ to the diagonal gains

The ambiguity is **descriptive, not inferential**: $W_{\text{resp}}$ is a single well-defined matrix; how we split it into "gains + rest" is a convention. Since $W = D + \mathbf{u}\mathbf{v}^T \Rightarrow W_{ii} = D_{ii} + u_iv_i$, reading off the diagonal is not a well-posed gain estimate.

1. **Functional projection** $D^\star$ — the diagonal minimizing the *action* difference (input-weighted). Robust exactly where $W_{\text{resp}}$ is unreliable. **Already computed**: by least-squares orthogonality, the diagonal response fit *is* this projection, so the Diag(resp)–Free(resp) response-error gap already measures how far the effective map is from channel-aligned. (Ridge adds a small correction.)
2. **Simultaneous diagonal + low rank** — the factor-analysis structure; identifiable for rank $\lesssim 10$ at $m = 16$.
3. **Sequential** — define $R = W_{\text{resp}} - D_{\text{diag}}$, report its rank/spectrum and the fraction of its energy on the diagonal. **This is the one that adds new information** (does Free *keep* or *redistribute* the gains?), and it matches the paper's narrative order.

Comparing $\|W_{\text{resp}} - D_{\text{diag}}\|$ against $\|W_{\text{resp}} - D^\star\|$ is fine but **one-sided by construction** (each candidate is optimal in some metric), so present it as "how far the diagonal model's gains sit from the optimal channel-aligned summary", not as a horse race.

**Neurally plausible variant (optional, only if the residual is real):** constrain the residual to rank one along the all-ones direction — "each channel receives inhibition proportional to total population activity, with a channel-specific weight". That is the canonical broad-normalization motif (short-axon / periglomerular circuitry), the direction is fixed rather than fitted so it is cheap and identifiable, and it would upgrade "low-rank residual" to "global gain control".

## 4. Analysis plan

**Fits (2 × 2 + 1):**

| | Diag | Free |
|---|---|---|
| **Representations** | gains from subset rep fit | $W_{\text{rep}}$ (symmetric by regularizer) |
| **Responses** | gains from response regression | $W_{\text{resp}}$ (unique; also fit **symmetric-constrained** as the ladder's middle rung) |

Plus the **identity** baseline (pass input through unchanged) at both levels — the analogue of "in-out" in the existing figures, and what makes "Diag ≈ Free" legible.

**The ladder** (cross-validated response error):
$$\text{identity} \;\le\; W_{\text{rep}} \;\le\; W_{\text{resp,sym}} \;\le\; W_{\text{resp}} \;\le\; \text{noise ceiling}$$
- gap $(W_{\text{resp}} - W_{\text{resp,sym}})$ = **cost of symmetry** → is the rotation real or noise?
- gap $(W_{\text{resp,sym}} - W_{\text{rep}})$ = **cost of fitting representations instead of responses**.

**Controls / nulls (one per comparison):**
- **Shuffled matching** — permute the input–output pairing and repeat the gain agreement. Addresses the circularity that pairs were partly selected by odour-response similarity (the null inherits the same selection).
- **Noise ceiling** — split-half (across trials) reliability of the response-fit gains / predictions. Distinguishes scenario 0 from 3 and makes moderate correlations readable.
- **Split-half polar reference** — fit responses on each half, polar-decompose both, compute the same stretch/rotation distances between halves: how much apparent rotation does noise alone manufacture?

**Normalization:** report representation error as *fraction of the input→output representational gap closed*, so subset numbers are comparable with the published full-dataset values (raw error isn't, since the targets differ). Preprocessing must match across rep and response fits, or compare on correlation/rank/sign only.

**Caveat to state in the text:** with 16 channels, Free has 256 parameters and leans hard on regularization, so its advantage over Diag may shrink for partly statistical rather than biological reasons.

## 5. Panels

| Panel | Content | Reader takeaway |
|---|---|---|
| **A** | Representation error (as fraction of gap closed): Input, Diag(rep), Diag(resp), Free(rep), Free(resp) | As in the full dataset, the diagonal model explains most of the representational transformation |
| **B** (merged with old D) | Response error ladder: Input, Diag(rep), Diag(resp), Free(rep), Free(resp,sym), Free(resp), + noise-ceiling bar ("self, other trial") | The diagonal model goes a long way to explaining *responses*, not just representations; and the sym-vs-full gap shows whether routing matters |
| **C** | Scatter: Diag(resp) gains vs Diag(rep) gains, one cloud per seed, **+ shuffled-matching null** | Rep-fitting and response-fitting recover the same gains — i.e. the *estimator* is validated (not the specific full-data gain values, cf. §3.3) |
| **E** | Polar summary: $\|W_{\text{rep}} - S\|$ (normalized by $\|W_{\text{rep}} - I\|$) and rotation size as mean principal angle, **+ split-half noise reference** | How large is the routing component that representations could not see |
| **(freed slot)** | Candidates: mode/subspace comparison (only if E's stretch distance is large), residual characterization (§3.7 option 3), or the full-population-gains prediction test (§3.3) | — |

Bar colours/order consistent across A and B so the reader can compare across panels. E is secondary to B: B answers whether the rotation *matters*, E how big it is.

## 6. Interpretation — how the section (and paper) closes

**The reader's question:** "so is the connectivity diagonal or free?" — a conflation of levels. Answer: the two models answer different questions.

- **Diag** = a statement about the **transformation**: the effective input→output map is well captured by channel-wise gain.
- **Free** = a statement about the **effective connectivity** implementing it. Note: even with full glomerular identities we would recover the *effective feedforward map*, **not** synaptic weights (many recurrent granule-cell arrangements realize the same map). Keep the word "effective" in the text.
- Nobody would conclude from a successful diagonal fit that the anatomy is diagonal; the claim is that despite the extra degrees of freedom lateral connectivity provides, most of the transformation amounts to gain control.

**The substantive content of "diagonal":** any map is diagonal in *some* basis. What makes this non-trivial is that the basis is the **glomerular one** — handed over by biology, not fitted. The transformation is approximately *aligned with the channel decomposition of the input*: circuits could have mixed channels arbitrarily; this one largely doesn't. In polar terms, the stretch is close to diagonal in the glomerular basis.

**Compact statement of the conclusion:** the input–output transformation is dominated by channel-aligned gain, with a small low-rank residual; representations identify the gain component exactly and the residual only up to routing; and the matched glomeruli show that the response data demand the same gains.

**Closing the paper:** end on how real (lateral) connectivity could produce an effectively diagonal transformation. This turns the paper's most counterintuitive result into a mechanism instead of a puzzle, stays consistent with the existing counting argument ($\sqrt{1500} \approx 40$ odours, so gain control alone cannot scale — the implementation must be lateral even where the computation is diagonal), and lands on the circuit motifs the predictions section already targets (periglomerular presynaptic gain control, PV-mediated linear scaling, broad normalization).
