# Statistics for the generalization violin figures

How the tests behind the significance brackets are defined, and why. Written
2026-08-19, when the tests were added; updated 2026-10-23 to match the code.

## What is being compared

Each panel of `figures/generalization.Supp` shows, for one split and one metric
family, a violin per model plus an `Input` and an `Output` violin. The brackets
test differences between those violins. Which column each group reads is
registered in `METRIC_COLUMNS` (`compute/generalization.py`).

`Output` means different things in different families:

- `cov` and `corr` are **distances to the output**: `cov_in_out` is
  `RMS(Cin - Cstar)` and `cov_est_out` is `RMS(Cest - Cstar)`. `Input` is the
  baseline: how wrong you are if you assume the output covariance equals the
  input's. `Output` is a **noise floor**: the same distance between two
  trial-disjoint estimates of `Cstar` (`compute.output_noise_floor`). It
  measures output trial noise only, so it sits below what any model could
  actually reach; see that function's docstring.
- `corr_en` is a **property of each matrix on its own**: `compute_corr_energ_`
  returns the off-diagonal energy of a single matrix, evaluated separately on
  `Cin`, `Cstar` and `Cest`. `Output` is the target level.

A consequence for interpretation: a model-vs-`Output` test asks whether the
model reaches the target (`corr_en`) or the noise floor (`cov`, `corr`), so
*non*-significance is the favourable outcome. Against `Input`, significance is.

## Pairing

The samples are paired, not independent. Every violin in a panel is indexed by
the same `(seed, train, outclass)`: Diag's value and Free's value for seed 7
come from the same data split, the same odours and the same trials. The violins
look like independent distributions, but they are repeated measures.

Tests are therefore **paired Wilcoxon signed-rank** on the per-unit differences.
This is both correct and much more powerful than an unpaired comparison of the
two marginal distributions.

## The unit of analysis

Each split contributes `seeds x trains` rows per model. The trains within a
seed share a split and are not independent: they are subsamples of the same
odour/trial partition. Treating every row as a sample would put essentially
every comparison at `p < 0.001` regardless of effect size, which says more about
the resampling than about the models.

Values are therefore **aggregated over `train` within each seed** (median),
giving one independent unit per seed (per seed and outclass, when a panel pools
outclasses). This deliberately gives up significance that the raw row count
would otherwise appear to provide.

## Which comparisons

Testing every pair gives an unreadable figure, so no brackets are drawn unless
the caller asks for them, via `comparisons=` on `Supp.plot` (and
`violin_plots`). Each comparison is a short string:

- `A<B` (or `B>A`) -- **one-sided**, direction fixed in advance;
- `A:B` -- **two-sided**, the data chooses which way round to state it. This
  costs a factor of two in p relative to `<`, the price of not committing;
- `Model` is a wildcard for every model in the panel, e.g. `Input>Model` tests
  each model against `Input`.

## Multiple comparisons

**Reported p-values are uncorrected** by default, and the asterisks reflect the
raw p-values (one- or two-sided, as requested).

The comparisons here are few (three to six per panel) and pre-planned rather
than searched over, and the effects that matter are orders of magnitude clear of
any threshold, so a correction changes no conclusion. Against that, any
correction has to be taken over a family, and the only available family is "the
tests we chose to draw" -- which would make a given comparison's p-value depend
on what else happens to be on the figure. That dependence is arbitrary and hard
to defend, so it is not used.

`correction="holm"` is available on `compare_panel`, `stats_df`, `report` and
`Supp.plot` if it is ever asked for. It applies Holm-Bonferroni over the
distinct pairs in a panel, ranking each pair by its smaller one-sided p-value.
Every row records which correction produced it.

## Thresholds

Conventional levels, on the reported p-values:

| corrected p | mark   |
|-------------|--------|
| < 0.001     | `***`  |
| < 0.01      | `**`   |
| < 0.05      | `*`    |
| otherwise   | `n.s.` |
