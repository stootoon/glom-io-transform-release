# glom-io-transform-release
Released code for the glomerular input-output paper.

## Installation Instructions
First, download and unpack the [connectivity models repository](https://github.com/stootoon/ob_io_conn_models/)

Add path-related environment variables by update the paths in `add_paths.sh` and sourcing that file using
`source add_paths.sh`
This will
1. Add an environment variable GLOM_IO to point to the root of this repository.
2. Add an environment variable GLOM_IO_DATA to point to `data` directory of this repository.
3. Add an environment variable OB_IO_CONN_MODELS to the root directory where you installed the repository.

## The pipeline

Four stages, each reading the output of the one before. A figure needs every
stage below it to have been run for the models it draws.

| stage | command | produces |
|---|---|---|
| 1. fit | `driver.py --gen` / `--run` / `--collect` / `--loadmodels` | `fits/.../collected.p`, `loaded_models.p` |
| 2. summarise | `analysis.compute.generalization` | `model_fitting/generalization_results/*.pkl` |
| 3. analyse | `matched_rois.Data().compute()` etc. in a notebook | an in-memory (or pickled) `Data` object |
| 4. draw | `analysis/fig_driver.ipynb` | the figures |

Stages 1 and 2 are the expensive ones and are done once per *set of fits*.
Stages 3 and 4 are cheap and are re-run whenever a figure changes.

## Stage 1: fitting connectivity models

### Where the fits go

`--gen` computes the output directory from the config rather than taking it as
an argument, so the path *is* the record of what was fitted
(`model_fitting/layout.py`):

```
fits/loss=<cov|resp>/matched=<bool>[/alpha=<a>]/center=True/
    standardization=separate/normalization=odour_std/
    sampler=<trials|odours>/mode=<random|inclass|outclass>/
    n_od_train=<spec>/<model name>/
```

`alpha=<a>` appears only for surrogate runs, so real fits keep the paths they
have always had. The leaf is the model's own directory (`Free`, `FreeSym`, ...),
taken from the model name so the two cannot disagree.

### 1. Generate the run configurations

```
cd model_fitting
python driver.py --gen yaml/base/ffree_trials_random_max.yaml [flags]
```

This expands one YAML into an `in.N.p` per (seed x swept hyperparameter) and
prints the directory it created. The YAMLs under `yaml/base/` cover the Diagonal
(`fit_diag_*`) and Free (`ffree_*`) models for each sampler and split mode; the
constrained Free variants have no YAML of their own and are selected with
`--variant` instead.

The flags all override the YAML, and each one that changes what was fitted is
recorded in the path:

| flag | effect |
|---|---|
| `--loss cov\|resp` | which loss to fit against |
| `--match-file FILE` | fit only the matched roi pairs in that csv; also sets `matched=True` |
| `--variant sym\|psd\|rot\|orth` | constrain the Free model (`FreeSym`, `FreePSD`, `FreeRot`, `FreeOrth`) |
| `--n-od-train SPEC` | odour subset: `max`, an integer, `<n>_rand_<seed>`, `<n>_var_input`, `<n>_var_output` |
| `--alpha A` | fit SURROGATE data from a known `Z = S + alpha A` |
| `--target-r2 R` | scale the surrogate's noise so the true `Z` reaches this held-out R2; only with `--alpha` |

`--variant orth` sweeps `reflect` over both components of O(m), so it generates
twice the runs the other variants do.

### 2. Run the fits

One at a time:

```
python driver.py --run fits/.../in.0.p
```

`--run` takes several files, so the two helper scripts batch them onto a
cluster: `./run_dir.sh <dir> [n per job]` submits every `in.*.p` in a directory,
and `./run_stale.sh <dir>` submits only the ones `check_dir.py` reports as
missing an output — which is the one to use after a partial failure.

Each run writes `out.N.p` beside its input.

### 3. Collect

```
python driver.py --collect fits/.../<model dir>
```

Reads every `out.*.p` in the directory into a single `collected.p`. Use
`--inputfields` / `--outputfields` to keep extra fields from the config and the
results respectively.

### 4. Load the models

```
python driver.py --loadmodels fits/.../n_od_train=<spec>
```

Point this at the directory ONE LEVEL ABOVE the model directories: it walks the
`collected.p` files one level down and writes a `loaded_models.p` beside them.
It is incremental — only collections newer than the existing `loaded_models.p`
are re-read — so it is cheap to repeat after adding a model.

### Evaluating the fits

Once collected, training/test/validation performance across the regularization
sweep can be viewed with the `proc_fit_models.ipynb` notebook.

## Stage 2: the generalization dataframes

```
python -m glom_io_transform.analysis.compute.generalization --scheme matched_rois
```

A *scheme* in `analysis/compute/generalization_schemes.yaml` names one set of
fits worth summarising; the file's header lists them. Any flag overrides the
scheme, so a variant is usually a scheme plus one flag rather than a new scheme:

| flag | notes |
|---|---|
| `--scheme NAME` | optional — the flags stand on their own without one |
| `--loss`, `--matched` / `--no-matched`, `--alpha` | which fits to read |
| `--n_od_train SPEC` | note the UNDERSCORES here; `driver.py` spells the same thing `--n-od-train` |
| `--splits sampler:mode ...` | e.g. `trials:random odours:outclass` |
| `--models Diag Free FreeSym ...` | which to summarise |
| `--check-only` | report whether the cache exists, compute nothing |
| `--overwrite` | recompute even if it does |

Results are cached under `model_fitting/generalization_results/`. The loader
checks staleness against the fits, so a cache older than its fits is reported
rather than silently used.

## Stage 3: the analysis objects

In a notebook (see `analysis/fig_driver.ipynb`):

```python
from glom_io_transform.analysis.compute import matched_rois
data = matched_rois.Data().compute(n_od_train="18_rand_0")
```

`Data.compute` pulls the generalization dataframes for both losses, refits the
models it needs at the selected lambdas, and builds everything the matched-roi
figure draws. It needs the fits tree, unlike the generalization figures, which
need only the cached dataframes.

## Stage 4: the figures

Run `analysis/fig_driver.ipynb`.

A figure can also be rendered outside the notebook, from a pickled `Data`,
which is useful for checking layout changes:

```
GLOM_IO=$PWD/glom_io_transform GLOM_IO_DATA=$PWD/glom_io_transform/data python render.py
```

`paths.py` only asserts that `$GLOM_IO/model_fitting` *exists*, so pointing
`GLOM_IO` at the package directory is enough to draw from a pickle without the
fits being present.

## Use cases

### Reproducing the paper's main figures

Fit `Diag` and `Free` at `loss=cov` for every sampler and split mode
(`yaml/base/*_max.yaml`), collect, load, then

```
python -m glom_io_transform.analysis.compute.generalization --scheme show_models
```

### The matched-roi figure on a new odour subset

Everything below is `--match-file <csv>` and `--n-od-train <spec>`. For a subset
`18_rand_1`, the runs are:

| loss | models | why |
|---|---|---|
| `cov` | `Diag`, `Free` | the `Free (cov)` panels and the covariance violins |
| `resp` | `Diag`, `Free`, `FreeSym`, `FreePSD`, `FreeRot`, `FreeOrth` | the ladder's rungs and the response violins |
| `resp`, one per alpha | `Free`, `FreeSym` | the surrogate calibration panel |

`Diag` at `loss=resp` is the one most easily forgotten: it is needed both for
the response violins and for the ladder's `Diag resp` rung.

Then the dataframes, and finally `Data().compute(n_od_train="18_rand_1")`:

```
python -m glom_io_transform.analysis.compute.generalization --scheme matched_rois     --n_od_train 18_rand_1
python -m glom_io_transform.analysis.compute.generalization --scheme matched_rois_cov --n_od_train 18_rand_1
```

### The surrogate asymmetry sweep

One pair of runs per alpha, and `--target-r2` must travel with `--alpha` or the
noise is left unscaled:

```
for A in 0.0 0.2 0.4 0.6 0.8 1.0; do
  python driver.py --gen yaml/base/ffree_trials_random_max.yaml \
                   --loss resp --match-file <csv> --n-od-train 18_rand_0 \
                   --alpha $A --target-r2 0.25
  python driver.py --gen yaml/base/ffree_trials_random_max.yaml --variant sym \
                   --loss resp --match-file <csv> --n-od-train 18_rand_0 \
                   --alpha $A --target-r2 0.25
done
```

`--target-r2` must be the SAME for every alpha: kappa is solved against the
symmetric truth so that varying alpha varies the asymmetry and nothing else, and
a sweep with a moving target is not a calibration. The `alpha=0.0` run is not a
no-op — it is the leftmost violin, where the truth is exactly symmetric.

Then one dataframe per alpha:

```
for A in 0.0 0.2 0.4 0.6 0.8 1.0; do
  python -m glom_io_transform.analysis.compute.generalization --scheme surrogate --alpha $A
done
```

### Adding a constrained Free variant

Register it in `FREE_VARIANTS` in `driver.py`, then `--gen` with `--variant
<name>` — no new YAML, since the variants differ from `Free` only in how their
parameters unpack into a connectivity. Add it to the relevant scheme's
`which_models` so the dataframes pick it up.

### After changing the fitting or compute code

`--loadmodels` and the generalization cache are both incremental, so a code
change that alters what a fit MEANS needs the cache invalidating: rerun the
scheme with `--overwrite`. `--check-only` reports what exists without computing.
A `Data` object is not cached at all — recompute it in the notebook, and restart
the kernel first if a dataclass changed shape, since autoreload cannot fix that.
