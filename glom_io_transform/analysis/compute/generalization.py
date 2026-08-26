"""Generalization performance of the models across split types.

Builds (or loads from cache) the dataframe of validation metrics per
(split, model, seed, train, outclass), with three metric families:
  cov_*     : covariance mismatch (in_out, est_out)
  corr_*    : correlation mismatch (in_out, est_out)
  corr_en_* : correlation energy (in, out, est)
"""
import os
import pickle
import re
import numpy as np
import pandas as pd

from itertools import product
from tqdm import tqdm

import glom_io_transform.model_fitting.proc_fit_models as pfm

from .compute import Computation, base_context

SPLITS = [
    ("trials", "random", "max"),
    ("odours", "random", "max"),
    ("odours", "inclass", "max"),
    ("odours", "outclass", "max"),
]

WHICH_MODELS = ["Diag", "DiagOnlyInh", "Free", "FreeLat", "FreeSym", "FreePSD"]

# Dataframe name -> the label used on the figure AND in comparison strings, in
# the order the violins are drawn. One source, so a comparison can never name a
# group the figure spells differently.
MODEL_LABELS = {"Diag": "Diag", "DiagOnlyInh": "DiagInh", "Free": "Free", "FreeLat": "FreeLat", "FreeSym":"FreeSym", "FreePSD":"FreePSD"}


def as_labels(models, df=None):
    """{model name: axis label} from whatever a caller passed for `models`.

    None    -> the models the dataframe contains, labelled by models_in
    sequence-> those names, in that order, each labelled by itself, for
               conditions whose name is already what belongs on the axis
    mapping -> used as given, for names that need a different label
    """
    if models is None:
        return models_in(df)
    if isinstance(models, dict):
        return models
    return {m: m for m in models}


def models_in(df):
    """{model name: axis label} for the models a dataframe contains.

    Names known to MODEL_LABELS come first, in its order and with its labels, so
    existing figures and statistics are unaffected. Anything else follows in the
    order it appears -- a suffixed variant such as "Free_cov", which is what you
    get by concatenating the frames from two losses. Pass an explicit mapping
    wherever you want a particular order.
    """
    present = list(dict.fromkeys(df["model"]))
    known   = [m for m in MODEL_LABELS if m in present]
    extra   = [m for m in present if m not in MODEL_LABELS]
    return {m: MODEL_LABELS.get(m, pfm.variant_label(m)) for m in known + extra}

# Which column each metric family reads for the non-model groups, and for the
# model estimates. This is the one place a metric is registered: both the
# statistics and the violins read it, so adding an entry here is all it takes.
#
# cov/corr are distances to the output, so they have no Output group. corr_en is
# a property of each matrix, so it has all three. r2 has an Output because one
# trial of the output against another is the reliability ceiling.
METRIC_COLUMNS = {"cov":     {"Input": "cov_in_out",     "est": "cov_est_out"},
                  "corr":    {"Input": "corr_in_out",    "est": "corr_est_out"},
                  "corr_en": {"Input": "corr_en_in",     "est": "corr_en_est",
                              "Output": "corr_en_out"},
                  "r2":      {"Input": "r2_in_out",      "est": "r2_est_out",
                              "Output": "r2_out"}}

# The groups that are not models: read once from the data rather than fitted,
# drawn in grey at the ends, and left out of the Model wildcard.
REFERENCE_GROUPS = ("Input", "Output")

# Group names are whatever the axis labels are, and a label may contain spaces
# and brackets ("Free (cov)"), so match anything either side of the operator
# rather than \w+. None of the labels contain <, > or : themselves.
COMPARISON_RE = re.compile(r"^\s*(\S.*?)\s*([<>:])\s*(\S.*?)\s*$")
WILDCARD = "Model"

MARKS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def vld_fun_ratio(vld):
    in_out = np.mean((vld.Cin - vld.Cstar)**2)**0.5
    est_out = np.mean((vld.Cest - vld.Cstar)**2)**0.5
    return in_out, est_out

def vld_fun_corr(corrs):
    in_out = np.mean((corrs["Cin"] - corrs["Cstar"])**2)**0.5
    est_out = np.mean((corrs["Cest"] - corrs["Cstar"])**2)**0.5
    return in_out, est_out

def compute_corr_energ_(C):
    # If C is a square matrix, use the off-diagonal elements to compute the energy
    if C.shape[0] == C.shape[1]:
        total_energy = np.sum(C**2) - np.sum(np.diag(C)**2)
        n_elements = C.shape[0] * (C.shape[0] - 1)
        return total_energy / n_elements
    else:
        # If C is not square, compute the energy using all elements
        return np.mean(C**2)

def vld_fun_corr_energy(corrs):
    [in_, out_, est_] = [compute_corr_energ_(corrs[key]) for key in ["Cin", "Cstar", "Cest"]]
    return in_, out_, est_


def default_cache_file(base, splits=None):
    """Where the dataframe for THIS set of models lives.

    One file per set of fits it could summarise, since they are different
    results. Everything at its default contributes nothing to the name, so the
    original cache keeps its original path and is still found.
    """
    suffix = ""
    if getattr(base, "loss", "cov") != "cov":
        suffix += f"_loss={base.loss}"
    if getattr(base, "matched", False):
        suffix += "_matched"
    if getattr(base, "alpha", None) not None:
        suffix += f"_alpha={base.alpha}"
    all_specs = sorted({str(sp[2]) for sp in (splits or ()) if len(sp) > 2})
    specs = [sp for sp in all_specs if sp != "max"]
    # 'max' contributes nothing to the name, so a summary mixing it with a
    # subset spec would be named after the subset alone and overwrite the
    # summary of that subset on its own.
    assert not (specs and "max" in all_specs), (
        f"These splits mix n_od_train={specs} with 'max', which cannot be named "
        f"unambiguously. Summarise them separately, or pass cache_file= explicitly.")
    if specs:
        suffix += "_n_od_train=" + "+".join(specs)
    return os.path.join(base.models_dir, f"generalization_results{suffix}.pkl")


def generalization_df(base, splits=SPLITS, which_models=WHICH_MODELS,
                      selection_metric=None,
                      compute=False, cache_file=None,
                      check_staleness=True):
    """Load (or compute and cache) the generalization metrics dataframe.

    selection_metric=None lets ModelResults.extract choose one that matches the
    loss its models were fitted against -- ratio for a covariance fit, ratio_resp
    for a response fit. Naming one here overrides that.
    """
    if cache_file is None:
        cache_file = default_cache_file(base, splits)

    if not compute:
        assert os.path.exists(cache_file), f"Could not find {cache_file}."
        if check_staleness:
            # File exists. But make sure it's more recent than all the loaded_models.p files
            # Get the last modified time of the cache file
            cache_mtime = os.path.getmtime(cache_file)
            # Get the last modified time of all loaded_models.p files
            for split_name in splits:
                # load=False: we only want the path, not the models themselves --
                # loading them here would cost exactly what the cache exists to save.
                models_file = base.split(*split_name, load=False).models_file
                assert os.path.exists(models_file), f"Could not find {models_file}."
                assert cache_mtime >= os.path.getmtime(models_file), (
                    f"Cache file {cache_file} is older than {models_file}. "
                    f"Please recompute the generalization dataframe (compute_df=True).")

        with open(cache_file, "rb") as f:
            df = pickle.load(f)
        print(f"Loaded generalization results from {cache_file}.")
        return df, cache_file
    
    print(f"Computing generalization results for {len(splits)} splits and {len(which_models)} models.")
    print(f"Using {cache_file} as the cache file.")
    records = []
    for split_name in splits:
        split = base.split(*split_name)
        for model_name in which_models:
            model = split.model(model_name)
            n_train = len(model.df["ref"].unique())
            n_seeds = len(model.df["seed"].unique())
            outclasses = model.df["outclass"].unique()
            iterable = product(range(n_seeds), range(n_train), outclasses)
            for seed, train, outclass in tqdm(iterable, total=n_seeds*n_train*len(outclasses)):
                extra_fields = {"outclass": outclass} if "outclass" in split_name else {}
                res = model.extract(seed=seed, train=train,
                                    la="min" if model_name.startswith("Diag") else None,
                                    metric=selection_metric, **extra_fields)
                cov_in_out, cov_est_out = vld_fun_ratio(res.vld)
                corr_in_out, corr_est_out = vld_fun_corr(res.vld_corrs)
                corr_en_in, corr_en_out, corr_en_est = vld_fun_corr_energy(res.vld_corrs)
                records.append({
                    "sampler": split_name[0],
                    "mode": split_name[1],
                    "n_od_train": split_name[2],
                    "model": model_name,
                    "seed": seed,
                    "train": train,
                    "outclass": outclass,
                    "cov_in_out": cov_in_out,
                    "cov_est_out": cov_est_out,
                    "corr_in_out": corr_in_out,
                    "corr_est_out": corr_est_out,
                    "corr_en_in": corr_en_in,
                    "corr_en_out": corr_en_out,
                    "corr_en_est": corr_en_est
                })
    df = pd.DataFrame(records)
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)
    print(f"Wrote {cache_file}.")
    return df, cache_file


class Data(Computation):
    """Compute for the supplementary generalization figures."""
    def compute(self, selection_metric=None, compute_df=False, splits=SPLITS,
                which_models=WHICH_MODELS, check_staleness=True, **base_kwargs):
        """base_kwargs go to base_context -- loss='resp', matched=True, ...

        splits defaults to all four; the matched runs only have some of them,
        and asking for a split that was never fitted fails at the directory.
        """
        print("COMPUTING generalization.Data.")
        base = base_context(**base_kwargs)
        self.df, self.cache_file = generalization_df(base, splits=splits, which_models=which_models,
                                                      selection_metric=selection_metric, 
                                                      compute=compute_df,
                                                        check_staleness=check_staleness,
                                                      )
        self.computed = True
        return self


# ----------------------------------------------------------------------------
# Comparisons between the violins. See notes/generalization_statistics.md for
# why the tests are paired, one-sided, seed-aggregated and Holm-corrected.
# ----------------------------------------------------------------------------

def panel_units(df, prefix, sampler, mode, outclass=None, models=None):
    """One row per independent unit, one column per group, for a single panel.

    The unit is a seed (and an outclass, when the panel pools them): the ten
    trains within a seed are subsamples of the same split, so they are averaged
    rather than counted, which would otherwise inflate n fivefold and make
    everything significant regardless of effect size.
    """
    cols = METRIC_COLUMNS[prefix]
    mask = (df["sampler"] == sampler) & (df["mode"] == mode)
    if outclass is not None:
        mask = mask & (df["outclass"] == outclass)
    d = df[mask]
    assert len(d), f"No rows for sampler={sampler}, mode={mode}, outclass={outclass}."

    key = ["seed"] + (["outclass"] if outclass is None and d["outclass"].notnull().any() else [])
    out = {}
    # Input/Output do not depend on the model, so take them from one model's rows.
    ref = d[d["model"] == sorted(d["model"].unique())[0]]
    for group in ("Input", "Output"):
        if group in cols:
            out[group] = ref.groupby(key)[cols[group]].median()
    # models_in rather than MODEL_LABELS, so that a frame carrying suffixed
    # variants ("Free_cov") is compared rather than silently dropped. A caller
    # that labelled its violins itself passes the same mapping here, or the
    # comparisons would name groups the figure spells differently.
    for name, label in as_labels(models, df).items():
        rows = d[d["model"] == name]
        if len(rows):
            out[label] = rows.groupby(key)[cols["est"]].median()
    return pd.DataFrame(out).dropna()


def parse_comparison(text, groups):
    """'A<B', 'A>B' or 'A:B' -> [(lo, hi, auto), ...], expanding the Model wildcard.

    '<' and '>' fix the direction in advance and give a ONE-SIDED test; '>' is
    sugar for a swap. ':' leaves the direction to the data -- the group with the
    lower median is reported as the smaller one -- and gives a TWO-SIDED test.

    The two-sidedness is not incidental. Picking the direction after seeing the
    data and then testing one-sided rejects at 2*alpha, and the correction for
    that is exactly the doubling that makes it two-sided. So ':' costs a factor
    of two in p relative to '<', which is the price of not having committed.
    """
    m = COMPARISON_RE.match(text)
    assert m, f"Cannot parse comparison {text!r}; expected e.g. 'Diag<Input', 'Free>Diag' or 'Input:Diag'."
    a, op, b = m.group(1), m.group(2), m.group(3)
    lo, hi = (a, b) if op in "<:" else (b, a)
    models = [g for g in groups if g not in REFERENCE_GROUPS]
    expand = lambda g: models if g == WILDCARD else [g]
    pairs = [(x, y, op == ":") for x in expand(lo) for y in expand(hi) if x != y]
    unknown = {g for pr in pairs for g in pr[:2]} - set(groups)
    assert not unknown, f"{text!r} names groups not in this panel: {sorted(unknown)}; have {groups}."
    return pairs


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the order given."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def mark_for(p):
    for thresh, mark in MARKS:
        if p < thresh:
            return mark
    return "n.s."


def compare_panel(df, prefix, sampler, mode, comparisons, outclass=None,
                  correction=None, models=None):
    """Every requested comparison for one panel.

    correction=None (the default) reports the raw one-sided p-values. The
    comparisons here are few and pre-planned, and a correction taken over "the
    tests we chose to draw" would make a given test's p-value depend on what
    else is on the figure, which is arbitrary. correction="holm" applies
    Holm-Bonferroni over the distinct PAIRS -- 'A<B' and 'B<A' are one
    comparison asked in two directions, not two findings -- for the case where a
    reviewer asks for it.
    """
    from scipy.stats import wilcoxon

    units  = panel_units(df, prefix, sampler, mode, outclass, models=models)
    groups = list(units.columns)
    wanted, seen = [], {}
    for text in comparisons:
        for lo, hi, auto in parse_comparison(text, groups):
            if auto:
                # The data chooses which way round to state it.
                lo, hi = (lo, hi) if units[lo].median() <= units[hi].median() else (hi, lo)
            wanted.append((text, lo, hi, auto))
            seen.setdefault(frozenset((lo, hi)), len(seen))

    rows = []
    for text, lo, hi, auto in wanted:
        a, b = units[lo].values, units[hi].values
        d = a - b
        alternative = "two-sided" if auto else "less"
        if np.allclose(d, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(a, b, alternative=alternative)
        q1, q3 = np.percentile(d, [25, 75])
        rows.append({"sampler": sampler, "mode": mode, "outclass": outclass,
                     "metric": prefix, "comparison": f"{lo}<{hi}", "requested": text,
                     "alternative": alternative,
                     "lo": lo, "hi": hi, "n": len(d), "statistic": stat, "p": p,
                     "median_diff": float(np.median(d)), "iqr_lo": float(q1), "iqr_hi": float(q3),
                     "pair": seen[frozenset((lo, hi))]})
    out = pd.DataFrame(rows)
    if not len(out):
        return out

    out["n_pairs"] = out["pair"].nunique()
    if correction is None:
        out["p_adj"] = out["p"]
    elif correction == "holm":
        # A pair's rank is set by its smaller one-sided p-value -- the direction
        # that could be significant -- and the multiplier is then applied to
        # whichever direction was asked for.
        best = out.groupby("pair")["p"].min().sort_values()
        mult = {pair_id: len(best) - rank for rank, pair_id in enumerate(best.index)}
        out["p_adj"] = np.minimum(1.0, out["p"] * out["pair"].map(mult))
        running, floor = 0.0, {}
        for pair_id in best.index:
            running = max(running, float(out[out["pair"] == pair_id]["p_adj"].min()))
            floor[pair_id] = running
        out["p_adj"] = np.maximum(out["p_adj"], out["pair"].map(floor)).clip(upper=1.0)
    else:
        raise ValueError(f"Unknown correction {correction!r}; use None or 'holm'.")
    out["correction"] = correction or "none"
    out["mark"] = [mark_for(q) for q in out["p_adj"]]
    return out.drop(columns=["pair"])


def stats_df(df, prefix, comparisons, splits=None, per_outclass=True,
             correction=None, models=None):
    """compare_panel over every panel present in the dataframe."""
    present = set(map(tuple, df[["sampler", "mode"]].drop_duplicates().values))
    splits = [sm for sm in (splits or sorted(present)) if sm in present]
    out = []
    for sampler, mode in splits:
        out.append(compare_panel(df, prefix, sampler, mode, comparisons,
                                 correction=correction, models=models))
        if per_outclass and mode == "outclass":
            for oc in sorted(df[df["outclass"].notnull()]["outclass"].unique()):
                out.append(compare_panel(df, prefix, sampler, mode, comparisons,
                                         outclass=oc, correction=correction,
                                         models=models))
    return pd.concat([o for o in out if len(o)], ignore_index=True)


def report(plot_data, sampler="trials", mode="random", metric="corr", comparison=None,
           outclass=None, correction=None, verbose=True):
    """What a test says, in words. With no `comparison`, lists what is available.

    Uncorrected by default, matching the figure; pass correction="holm" to see
    what a Holm adjustment over the panel's pairs would do.
    """
    df = plot_data.df if hasattr(plot_data, "df") else plot_data
    units  = panel_units(df, metric, sampler, mode, outclass)
    groups = list(units.columns)

    if comparison is None:
        if verbose:
            print(f"{metric} / {sampler} {mode}" + (f" / outclass={outclass}" if outclass else ""))
            print(f"  groups   : {groups}   (n = {len(units)} units)")
            print(f"  available: " + ", ".join(f"{a}<{b}" for i, a in enumerate(groups)
                                               for b in groups[i+1:]))
            print(f"  operators: A<B / A>B  one-sided, direction fixed in advance")
            print(f"             A:B        two-sided, direction taken from the data")
            print(f"  wildcards: Model<Input, Output<Model, Model:Model")
        return groups

    fam = [f"{a}<{b}" for i, a in enumerate(groups) for b in groups[i+1:]] \
        if correction else [comparison]
    res = compare_panel(df, metric, sampler, mode, list(dict.fromkeys(fam + [comparison])),
                        outclass=outclass, correction=correction)
    # The direction may have been resolved from the data, so match on the pair
    # rather than on the order the comparison was written in.
    a, b, _ = parse_comparison(comparison, groups)[0]
    row = res[res[["lo", "hi"]].apply(lambda r: {r.lo, r.hi} == {a, b}, axis=1)].iloc[0]
    if verbose:
        print(f"{row['comparison']}  ({metric}, {sampler} {mode}"
              + (f", outclass={outclass}" if outclass else "") + ")")
        sided = "two-sided" if row["alternative"] == "two-sided" else "one-sided"
        print(f"  paired {sided} Wilcoxon signed-rank, n = {row['n']} seeds")
        print(f"  median difference {row['median_diff']:+.4g}  "
              f"IQR [{row['iqr_lo']:+.4g}, {row['iqr_hi']:+.4g}]")
        if correction:
            print(f"  p = {row['p']:.3g}   {correction} over {int(row['n_pairs'])} pairs -> "
                  f"p = {row['p_adj']:.3g}   {row['mark']}")
        else:
            print(f"  p = {row['p']:.3g}   {row['mark']}   (uncorrected)")
    return row
