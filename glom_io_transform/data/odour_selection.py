"""Which odours a fit uses, from a single spec string.

The spec lives in the sampler's `n_od_train` field, which previously held only
"max" or an integer:

    max            every odour, the original behaviour
    18             the first 18 of the stored order (an integer, as before)
    18_rand_0      18 drawn at random, reproducibly, with seed 0
    18_var_input   the 18 with the largest input variance
    18_var_output  the 18 with the largest output variance

`resolve` is the single source of truth: everything that needs to know which
odours a run uses calls it, so the data subset and the train/test/vld split
cannot disagree about what the odour axis is. Given the same spec it always
returns the same odours.

Variance specs rank odours by the variance of the trial-averaged response
across odours, computed over the MATCHED channels only -- X for var_input, Y
for var_output. They therefore need the matched data, which `resolve` takes as
an argument rather than loading, so that the caller stays in control of which
data the selection is made from.
"""
import re

import numpy as np

from .odours import odours

SPEC_RE = re.compile(r"^(\d+)_(rand|var)_(\w+)$")
MODES = ("rand", "var")


def parse(spec):
    """(n, mode, arg) for a spec string, or None for 'max' / a bare integer."""
    if spec == "max":
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec), "first", None
    m = SPEC_RE.match(str(spec))
    assert m, (f"Cannot parse odour spec {spec!r}. Expected 'max', an integer, "
               f"'<n>_rand_<seed>', or '<n>_var_input' / '<n>_var_output'.")
    n, mode, arg = int(m.group(1)), m.group(2), m.group(3)
    if mode == "rand":
        assert arg.isdigit(), f"'{spec}': a rand spec needs an integer seed, got {arg!r}."
    else:
        assert arg in ("input", "output"), \
            f"'{spec}': a var spec must be var_input or var_output, got {arg!r}."
    return n, mode, arg


def resolve(spec, X=None, Y=None, order="X0Y0"):
    """The odour NAMES a spec selects, in the stored order.

    Returned in the stored order rather than in selection order, so that the
    odour axis keeps its usual meaning and only gets shorter.

    X, Y are the matched (roi, odour) responses, needed only for the var modes.
    """
    names = odours.get_order(order)
    parsed = parse(spec)
    if parsed is None:
        return list(names)
    n, mode, arg = parsed
    assert 0 < n <= len(names), f"Asked for {n} of {len(names)} odours."

    if mode == "first":
        chosen = set(names[:n])
    elif mode == "rand":
        rng = np.random.default_rng(int(arg))
        chosen = set(np.asarray(names)[rng.choice(len(names), size=n, replace=False)])
    else:
        data = X if arg == "input" else Y
        assert data is not None, \
            f"'{spec}' ranks odours by {arg} variance, so the matched {arg} responses are needed."
        v = variance_by_odour(data, names)
        chosen = set(np.asarray(names)[np.argsort(-v)[:n]])
    return [nm for nm in names if nm in chosen]


def variance_by_odour(data, names):
    """Variance across rois of the trial-averaged response, per odour.

    data is (roi, odour[, repetition]), or the list of per-experiment (or
    per-matched-pair) arrays that get_data works with, which is stacked along
    the roi axis first. Trials are averaged before the variance is taken.
    Selection is by name, so an array in a different odour order still gives
    the same answer.
    """
    if isinstance(data, (list, tuple)):
        rows = [as_roi_by_odour(d, names) for d in data]
        return np.nanvar(np.concatenate(rows, axis=0), axis=0)
    return np.nanvar(as_roi_by_odour(data, names), axis=0)


def as_roi_by_odour(data, names):
    """One array's responses as (roi, odour), trials averaged, odours by name."""
    arr = data
    if hasattr(arr, "dims"):
        if "repetition" in arr.dims:
            arr = arr.mean("repetition")
        arr = arr.sel(odour=list(names))
        arr = np.asarray(arr)
    else:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = np.nanmean(arr, axis=2)
        assert arr.shape[1] == len(names), \
            f"Expected {len(names)} odours on axis 1, got {arr.shape[1]}."
    return arr
