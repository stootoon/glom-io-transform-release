# Statistics for the generalization violin figures

How the tests behind the significance brackets are defined, and why. Written
2026-08-19, when the tests were added.

## What is being compared

Each panel of `figures/generalization.Supp` shows, for one split and one metric
family, a violin per model plus an `Input` violin (and an `Output` violin for
`corr_en`). The brackets test differences between those violins.

`Input` and `Output` are not symmetric, and this is why `Output` appears only in
the `corr_en` panels:

- `cov` and `corr` are **distances to the output**: `cov_in_out` is
  `RMS(Cin - Cstar)` and `cov_est_out` is `RMS(Cest - Cstar)`. An `Output`
  violin would be `RMS(Cstar - Cstar)`, identically zero, so it carries no
  information. `Input` is the baseline: how wrong you are if you assume the
  output covariance equals the input's.
- `corr_en` is a **property of each matrix on its own**: `compute_corr_energ_`
  returns the off-diagonal energy of a single matrix, evaluated separately on
  `Cin`, `Cstar` and `Cest`. All three have meaningful values, and `Output` is
  the target level rather than a reference of zero.

A consequence for interpretation: in `corr_en`, a model-vs-`Output` test asks
whether the model reaches the right decorrelation level, so *non*-significance
is the favourable outcome. Everywhere else, significance against `Input` is.

## Pairing

The samples are paired, not independent. Every violin in a panel is indexed by
the same `(seed, train, outclass)`: Diag's value and Free's value for seed 7
come from the same data split, the same odours and the same trials. The violins
look like independent distributions, but they are repeated measures.

Tests are therefore **paired Wilcoxon signed-rank** on the per-unit differences.
This is both correct and much more powerful than an unpaired comparison of the
two marginal distributions.

## The unit of analysis

Each split contributes `50 seeds x 10 trains = 500` rows per model. Those ten
trains share a split and are not independent: they are subsamples of the same
odour/trial partition. Treating 500 as the sample size would put essentially
every comparison at `p < 0.001` regardless of effect size, which says more about
the resampling than about the models.

Values are therefore **aggregated over `train` within each seed** (median),
giving `n = 50` independent units per comparison. This deliberately gives up
significance that the raw row count would otherwise appear to provide.

## Which comparisons

Testing every pair gives 6-10 brackets per panel and an unreadable figure. Only
the comparisons that carry an argument are drawn:

- **each model vs `Input`** -- does the model beat doing nothing;
- **Diag vs Free** -- the constrained against the unconstrained fit, which is
  the comparison the matched-glomeruli section rests on;
- **each model vs `Output`**, `corr_en` only -- does the model reach the target
  decorrelation level (see above: non-significance is the good outcome).

## Multiple comparisons

Several brackets per panel, over several panels and three metric families, so
uncorrected p-values overstate the evidence. Correction is **Holm within a
panel**: the panel is the unit a reader compares within, and correcting across
the whole figure would penalise comparisons that are never read together.
The corrected value is what the asterisks reflect.

## Thresholds

Conventional levels, on the corrected p-values:

| corrected p | mark   |
|-------------|--------|
| < 0.001     | `***`  |
| < 0.01      | `**`   |
| < 0.05      | `*`    |
| otherwise   | `n.s.` |
