# data.py
#
# Self-contained builder for X0Y0_new.p, extracted from the glom-io-transform
# repo (conn-lin branch) so that downstream repos can regenerate the data and
# its metadata instead of just loading the pickle.
#
# It combines, without changing any numerics:
#   - datasets2.py                          : loading Tobias' .mat experiments
#   - tobias_utils.standardize_dimension    : z-scoring helper
#   - tobias_proc.py                        : z_score_experiment, olo odour
#                                             ordering, get_data_for_classification
#   - analyses/model-selection/data.py      : load_experiments (the X0Y0_new.p
#                                             reader/writer)
# plus a new get_metadata() that records which experiment each ROI (row of the
# stacked data) came from, its ROI id/label, imaging plane and pixel location.
#
# Requirements: numpy, pandas, and either mat73 (for MATLAB v7.3 files) or
# scipy (for v7 and earlier); load_mat() tries mat73 first and falls back.
# The raw data is expected in $DATA/tobias/allExp/*.mat (same convention as the
# original repo); override with set_data_dir() or the data_dir argument below.
#
# Usage:
#   import data
#   X0, Y0 = data.load_experiments(save_if_reload=True)   # builds X0Y0_new.p if absent
#   meta   = data.get_metadata()                           # provenance for every ROI
#
# X0 / Y0 are lists with one array per experiment (OMP = input, Tbet = output),
# each shaped (roi, odour, trial): single-trial, z-scored, time-integrated
# responses, with odours in the stored ("X0Y0") order.

import os
import pickle
from functools import partial
from collections import namedtuple
from collections.abc import Sequence
import numpy as np
from xarray import DataArray

from glom_io_transform.data import odours
from .common import (load_mat, get_data_dir, get_registry, registry_file,
                     INFO, WARN, DEBUG)

# ----------------------------------------------------------------------------
# Experiment geometry
# ----------------------------------------------------------------------------

Coords = namedtuple("Coords", "x y z units")

# tobias.ackels  2020 July 16 4:51 PM
# Sorry, to be clear: O174 to M72
# Anterior-Posterior -1.5mm (which corresponds to Y; so further to the tip of the bulb)
# Medio-Lateral -0.3 mm (corresponds to X; further towards the eye)
centers = {"M72":  Coords(x=0,   y=0,    z=0, units="um"),
           "O174": Coords(x=300, y=1500, z=0, units="um")}

# Order of the dimensions of the ca2 field
ca2_dims_order = ["odour", "repetition", "time"]

# ----------------------------------------------------------------------------
# Experiment loading (from datasets2.py)
# ----------------------------------------------------------------------------

class GlomerularExperiment:
    def __init__(self, data):

        _info2fld = {
            "name":        "name",
            "ISI":         "isi_sec",
            "fs_calcium":  "fs_ca2",
            "fs_sniffing": "fs_sniff",
            "odourOn":     "odour_on_index",
            "odours":      "odours",
            "region":      "region",
            "rois":        "roi_type",
            "type":        "indicator",
            "zstep":       "z_step",
        }

        for key, val in data["expInfo"].items():
            try:
                fld = _info2fld[key]
            except KeyError:
                pass
            else:
                self.__dict__[fld] = val

        assert self.roi_type.lower() == "gl", f"Expected roi_type to be GL, but is {self.roi_type}."

        _roi2fld = {
            "ca2":      "ca2",
            "plane":    "z",
            "roiID":    "roi_id",
            "roiLabel": "roi_label",
            "roiType":  "roi_type",
            "xpix":     "x_pix",
            "ypix":     "y_pix",
        }

        for key, val in data["roiData"].items():
            try:
                fld = _roi2fld[key]
            except KeyError:
                pass
            else:
                try:
                    self.__dict__[fld] = np.array(val)
                except Exception as e:
                    if not fld.endswith("pix"):
                        WARN(f"WARNING in assigning value for {key=} to {fld=}:")
                        WARN(e)
                    
                    self.__dict__[fld] = val

        assert "ca2" in self.__dict__, "Expected ca2 field in roiData."
        assert "odours" in self.__dict__, "Expected odours field in expInfo."
        self.odours = [odours.normalize_odour_name(o) for o in self.odours]  # Convert to Tobias' new list of odour names.
        self.n_roi, self.n_odours, self.n_reps, self.n_t = self.ca2.shape

        self.fs = self.fs_ca2
        self.t  = np.arange(self.n_t)/self.fs
        self.trial_length   = self.isi_sec  # seconds
        self.odour_start    = self.odour_on_index/self.fs
        self.odour_duration = 2

        # Make ca2 a DataArray with named dimensions, so we can index by name instead of
        # remembering the order. Per-ROI metadata goes in coordinates along the roi axis,
        # so it subsets with the data and survives concatenation across experiments
        # (attrs would not: xr.concat keeps only the first array's attrs).
        self.ca2 = DataArray(self.ca2,
                             dims   = ["roi"] + ca2_dims_order,
                             coords = {"odour": self.odours,
                                       "time":  self.t,
                                       **self._roi_coords()},
                             attrs  = {"indicator": self.indicator})

        INFO(f"Loaded {self.__str__()}.")

    def _roi_coords(self):
        """Per-ROI metadata, as xarray coordinates along the 'roi' dimension.

        Each entry must be one value per ROI. The pixel fields hold a list of
        pixels per ROI, which cannot be a coordinate, so they are reduced to a
        centroid. Fields whose length does not match n_roi are skipped with a
        warning rather than silently misaligning.
        """
        def centroid(v):
            try:
                return float(np.nanmean(np.asarray(v, dtype=float)))
            except Exception:
                return np.nan

        # Anything that can differ between experiments must be a coordinate, so it
        # stays correct when experiments are stacked along 'roi'. (The indicator is
        # invariant within a stack -- inputs and outputs are never mixed -- so it
        # lives in attrs instead.)
        coords = {"experiment":  ("roi", [self.name]        * self.n_roi),
                  "fs":          ("roi", [self.fs]          * self.n_roi),
                  "odour_start": ("roi", [self.odour_start] * self.n_roi)}

        for fld, coord, dtype in [("roi_id", "roi_id", int), ("roi_label", "roi_label", str), ("z", "z_plane", float)]:
            val = getattr(self, fld, None)
            if val is None:
                continue
            try:
                arr = np.asarray(val).reshape(-1).astype(dtype)
            except Exception as e:
                # Ragged fields (one entry per ROI, but entries of differing
                # length) cannot be a coordinate; numpy >= 2 raises rather than
                # building an object array. Skip rather than failing the load.
                WARN(f"{fld}: could not convert to a per-ROI {dtype.__name__} coordinate "
                     f"({type(e).__name__}: {e}); skipping. "
                     f"First element: {np.shape(val[0]) if len(val) else '?'}")
                continue
            if len(arr) == self.n_roi:
                coords[coord] = ("roi", arr)
            else:
                WARN(f"{fld} has {len(arr)} entries for {self.n_roi} ROIs; not added as a coordinate.")

        for fld, coord in [("x_pix", "x_centroid"), ("y_pix", "y_centroid")]:
            val = getattr(self, fld, None)
            if val is None:
                continue
            if len(val) == self.n_roi:
                coords[coord] = ("roi", [centroid(v) for v in val])
            else:
                WARN(f"{fld} has {len(val)} entries for {self.n_roi} ROIs; not added as a coordinate.")

        return coords

    def __str__(self):
        return (f"Glomerular Experiment {self.name:>8s}: {self.n_roi:>3d} ROIs for "
                f"{self.n_odours:>2d} odours. {self.n_t} time points at fs = {self.fs:1.1f} "
                f"is {self.t[-1]:1.1f} seconds.")

def concat_experiments(experiments, dim="roi", common_coords=True):
    """Stack the ca2 arrays of several experiments along the roi axis.

    Alignment is always exact, so mismatched labels on the other dimensions --
    a different odour list, or a different time base (whether from a different
    sampling rate or a different number of samples) -- raise instead of being
    silently unioned, intersected or overridden. Anything else would quietly
    change what the stack contains: join="inner" on experiments that disagree
    about the odours would shrink it to their intersection.

    The per-ROI coordinates are a different matter: they are metadata, not
    alignment keys, and some experiments simply don't carry all of them (a
    missing roi_label, say). With common_coords=True those are dropped, with a
    warning, so a missing label costs the label rather than the concatenation.
    Pass common_coords=False to require every experiment to carry the same set.
    """
    import xarray as xr
    arrays = [g.ca2 if isinstance(g, GlomerularExperiment) else g for g in experiments]

    if common_coords and len(arrays) > 1:
        names   = [set(a.coords) for a in arrays]
        shared  = set.intersection(*names)
        dropped = sorted(set.union(*names) - shared)
        if dropped:
            missing_from = {c: [str(a.coords["experiment"].values[0]) if "experiment" in a.coords else f"#{i}"
                                for i, a in enumerate(arrays) if c not in a.coords]
                            for c in dropped}
            WARN(f"Dropping coordinates that not every experiment carries: "
                 + "; ".join(f"{c} (missing from {', '.join(w)})" for c, w in missing_from.items()))
            arrays = [a.drop_vars([c for c in a.coords if c not in shared]) for a in arrays]

    return xr.concat(arrays, dim=dim, join="exact")


def load_glomerular_experiments(indicator, from_registry=True):
    glomerular_experiments = []
    n_rois = 0

    if from_registry:
        df = get_registry()
        file_names = df[(df.rois == "GL") & (df.indicator.str.lower() == indicator.lower())].file_name.values
    else:
        file_names = glob(get_data_dir() + "/*.mat")

    for file_name in sorted(file_names):
        DEBUG(f"Trying {file_name}.")
        data = load_mat(file_name)
        rois = data["expInfo"]["rois"]
        ind  = data["expInfo"]["type"]
        DEBUG(f"{rois=}, indicator={ind}")
        if rois == "GL" and indicator.lower() == ind.lower():
            DEBUG(f"Loading glomerular {indicator} experiment from {file_name}.")
            glomerular_experiments.append(GlomerularExperiment(data))
            n_rois += glomerular_experiments[-1].n_roi
        else:
            DEBUG(f"Not a glomerular {indicator} experiment, continuing.")

    INFO(f"Done loading {n_rois} ROIs from {len(glomerular_experiments)} glomerular {indicator} experiments.")
    return glomerular_experiments

class LazyDatasetList(Sequence):
    # Deriving from Sequence (not list) so 'in', iteration etc. go through
    # __getitem__ and trigger the lazy load.
    def __init__(self, name, loader):
        self.name   = name
        self.loader = loader
        self.data   = None

    def load(self):
        if self.data is None:
            INFO(f"Lazy-loading {self.name}")
            self.data = self.loader()
        return self.data

    def __getitem__(self, n):
        return self.load()[n]

    def __len__(self):
        return len(self.load())

glom_tbet = LazyDatasetList("glom_tbet", partial(load_glomerular_experiments, indicator="tbet"))
glom_omp  = LazyDatasetList("glom_omp",  partial(load_glomerular_experiments, indicator="omp"))

# ----------------------------------------------------------------------------
# Z-scoring and time integration (from tobias_utils.py / tobias_proc.py)
# ----------------------------------------------------------------------------

def standardize_dimension(X, axis=-1, which_elements=[]):
    sh = X.shape
    n_dims = len(X.shape)
    if axis < 0:  # to allow indexing relative to the end
        axis = n_dims + axis
    swap_dims = np.arange(n_dims)
    swap_dims[axis] = 0
    swap_dims[0]    = axis
    X_transpose = np.transpose(X, axes=swap_dims)
    X_transpose_reshape = np.reshape(X_transpose, (X_transpose.shape[0], -1))
    n_elements = X_transpose_reshape.shape[0]
    if len(which_elements) == 0:
        which_elements = np.arange(n_elements)
    mu = np.nanmean(X_transpose_reshape[which_elements], axis=0)
    sd = np.nanstd( X_transpose_reshape[which_elements], axis=0)
    Xtr_scaled   = (X_transpose_reshape - mu)/sd
    Xtrs_reshape = Xtr_scaled.reshape(X_transpose.shape)
    X_scaled     = np.transpose(Xtrs_reshape, axes=swap_dims)
    return X_scaled

def interp_last_axis(X, t, t_new):
    assert len(t) == X.shape[-1], f"Time axis length {len(t)} does not match last axis of X {X.shape[-1]}"
    Xr = np.reshape(X, (-1, len(t)))
    X_new = np.array([np.interp(t_new, t, Xr[i]) for i in range(Xr.shape[0])])
    return np.reshape(X_new, X.shape[:-1] + (len(t_new),))

def upsample_times(t, factor=1):
    assert np.var(np.diff(t)) < 1e-10, "Input times must be equally spaced"
    T = t[1] - t[0]
    T_new = T/factor
    return np.arange(t[0], t[-1] + T_new, T_new)

PRE_ODOUR = "pre_odour"


def z_score_experiment(g, int_width=5, max_width=5, up_sample=100, which_elements=PRE_ODOUR):
    """Z-score and time-integrate an experiment's calcium traces.

    which_elements says which time samples define the baseline that the traces
    are z-scored against. The default, PRE_ODOUR, uses every sample before
    odour onset, which is the only definition that travels between experiments:
    a fixed sample count does not, because the input and output datasets are
    sampled at very different rates (~7.7 Hz vs ~2.6 Hz), so ten samples is
    comfortably pre-onset for the input and two samples PAST onset for the
    output -- which inflates the baseline sd of exactly the odours that drive a
    cell early and hard, and so suppresses them. An explicit array of indices is
    still honoured; None skips the z-scoring altogether.

    The numerics are done on plain arrays -- standardize_dimension and
    interp_last_axis both reshape, which is meaningless for labelled dimensions
    -- and the labels are re-attached to the results at the end. Returned
    arrays carry the experiment's odour names and per-ROI metadata, plus a
    'time' coordinate (seconds) or 'bin' coordinate (bin start, seconds).
    """
    t        = g.t
    ca2t     = np.asarray(g.ca2)  # roi, odour, repetition, time
    odour_on = g.odour_start

    if isinstance(which_elements, str):
        assert which_elements == PRE_ODOUR, \
            f"Unknown which_elements {which_elements!r}; expected {PRE_ODOUR!r}, an array, or None."
        which_elements = np.flatnonzero(t < odour_on)
        assert len(which_elements) >= 3, (
            f"{g.name}: only {len(which_elements)} samples before odour onset "
            f"({odour_on:.3f}s at fs={g.fs:.3f}) -- too few to estimate a baseline.")
        DEBUG(f"Z-scoring {g.indicator} experiment {g.name} against the "
              f"{len(which_elements)} samples before odour onset ({odour_on:.3f}s).")
    else:
        which_el_str = f"{which_elements=}" + ("" if which_elements is not None else " (not actually Z-scoring).")
        DEBUG(f"Z-scoring {g.indicator} experiment {g.name} using {which_el_str}.")

    t_interp  = upsample_times(t, up_sample)
    dt_interp = t_interp[1] - t_interp[0]
    scale     = int_width/dt_interp
    standardizer = (lambda X: standardize_dimension(X, axis=3, which_elements=which_elements)) if which_elements is not None else (lambda X: X)

    t_starts = np.arange(odour_on, odour_on + max_width, int_width)
    assert len(t_starts), f"No time bins to integrate over with {int_width=}, {max_width=}, {odour_on=}"
    ca2t_z = standardizer(ca2t)
    # Interpolate the values of ca2t_z along the time dimension so that we can bin correctly.
    # Otherwise, e.g. bins that fall between time points get assigned a value of 0.
    ca2t_z_interp = interp_last_axis(ca2t_z, t, t_interp)
    # nansum, matching the trial-averaged path below: a NaN sample in one trial
    # should not wipe out that trial's whole integration window.
    integrators = [lambda X, t_start=t_start: np.nansum(X[:, :, :, (t_interp >= t_start) & (t_interp < t_start + int_width)], axis=-1) for t_start in t_starts]
    ca2t_zi = np.array([integrator(ca2t_z_interp) / scale for integrator in integrators])
    # Put the first dimension (bins) last
    ca2t_zi = np.moveaxis(ca2t_zi, 0, -1)
    # Same baseline as the single-trial path above, but estimated from the
    # trial-averaged trace, so its sd is smaller by ~sqrt(n_trials).
    ca2ta   = np.nanmean(ca2t, axis=2)  # trial average
    mu      = np.nanmean(ca2ta[:, :, t < odour_on], axis=-1)
    sd      = np.nanstd( ca2ta[:, :, t < odour_on], axis=-1)
    ca2ta_z = (ca2ta - mu[:, :, None])/sd[:, :, None]

    ca2ta_z_interp = interp_last_axis(ca2ta_z, t, t_interp)
    ca2ta_zi = np.array([np.nansum(ca2ta_z_interp[:, :, (t_interp >= t_start) & (t_interp < t_start + int_width)], axis=-1) / scale for t_start in t_starts])
    ca2ta_zi = np.moveaxis(ca2ta_zi, 0, -1)

    # Re-attach the labels: per-ROI coordinates and odour names come from the
    # input, and the last axis is either time (seconds) or integration bin
    # (bin start, seconds).
    roi_coords = {name: coord for name, coord in g.ca2.coords.items() if coord.dims == ("roi",)}
    def wrap(arr, dims, last):
        return DataArray(arr, dims=dims,
                         coords={**roi_coords, "odour": g.ca2.odour.values, **last},
                         attrs=dict(g.ca2.attrs))

    trial_dims = ["roi", "odour", "repetition"]
    mean_dims  = ["roi", "odour"]
    return (wrap(ca2t_z,        trial_dims + ["time"], {"time": t}),
            wrap(ca2t_zi,       trial_dims + ["bin"],  {"bin":  t_starts}),
            wrap(ca2ta_z,       mean_dims  + ["time"], {"time": t}),
            wrap(ca2ta_zi,      mean_dims  + ["bin"],  {"bin":  t_starts}),
            t_starts,
            wrap(ca2t_z_interp, trial_dims + ["time"], {"time": t_interp}),
            t_interp, t)

def get_data_for_classification(odour_order="X0Y0", which_elements=PRE_ODOUR):
    """Single-trial, z-scored, time-integrated responses; one array per experiment.

    odour_order is the name of an order in odours.ORDERS, or an explicit list of
    odour names. Selecting by name does two jobs at once: it puts the odours in
    the requested order, and it reconciles the different acquisition orders of
    the input (gl_omp) and output (gl_tbet) datasets -- no index arithmetic, and
    an unknown or missing odour raises rather than silently misaligning.
    """
    names = (odours.odours.get_order(odour_order) if isinstance(odour_order, str)
             else list(odour_order))
    # isel(bin=0): the first (and only) integration bin.
    Zin  = [z_score_experiment(g, which_elements=which_elements)[1].isel(bin=0).sel(odour=names)
            for g in glom_omp]
    Zout = [z_score_experiment(g, which_elements=which_elements)[1].isel(bin=0).sel(odour=names)
            for g in glom_tbet]
    return Zin, Zout

# ----------------------------------------------------------------------------
# X0Y0_new.p reader/writer (from analyses/model-selection/data.py)
# ----------------------------------------------------------------------------

def load_experiments(full_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "X0Y0_new.p"),
                     reload_if_doesnt_exist=True, save_if_reload=False):
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
        X0, Y0 = data['X0'], data['Y0']
        print(f"Loaded data from {full_path}")
    else:
        X0, Y0 = None, None
        if reload_if_doesnt_exist:
            print(f"File {full_path} does not exist.")
            print("Reloading data...")
            X0, Y0 = get_data_for_classification()
            if save_if_reload:
                with open(full_path, 'wb') as f:
                    pickle.dump({'X0': X0, 'Y0': Y0}, f)
                print(f"Saved data to {full_path}")
        else:
            raise FileNotFoundError(f"File {full_path} does not exist.")

    return X0, Y0

# ----------------------------------------------------------------------------
# Metadata (new)
# ----------------------------------------------------------------------------

def _roi_centroid(coord, j):
    # x_pix / y_pix hold the pixel coordinates of each ROI; reduce to a centroid.
    try:
        return float(np.nanmean(np.asarray(coord[j], dtype=float)))
    except Exception:
        return np.nan

def get_metadata():
    """Provenance for the arrays returned by load_experiments()/get_data_for_classification().

    Returns a dict with keys:
      - "input", "output": one entry per dataset (input = OMP experiments -> X0,
        output = Tbet experiments -> Y0), each a dict with
          "experiments": per-experiment records (name, indicator, sizes, sampling),
          "rois": a DataFrame with one row per ROI, in the same order as the ROI
                  axis of the corresponding X0/Y0 entries. global_index is the
                  row the ROI lands on when the per-experiment arrays are stacked
                  with np.concatenate(..., axis=0), as done downstream.
      - "odours": odour names in the column (odour-axis) order of X0/Y0,
                  i.e. odours.get_order("X0Y0").
      - "centers": reference coordinates of the M72/O174 glomeruli.
    """
    import pandas as pd
    meta = {}
    for key, experiments in [("input", glom_omp), ("output", glom_tbet)]:
        exp_records, roi_records = [], []
        global_index = 0
        for i, g in enumerate(experiments):
            exp_records.append({
                "experiment_index": i,
                "name":             g.name,
                "indicator":        g.indicator,
                "region":           getattr(g, "region", None),
                "n_roi":            g.n_roi,
                "n_odours":         g.n_odours,
                "n_reps":           g.n_reps,
                "n_t":              g.n_t,
                "fs":               g.fs,
                "odour_start":      g.odour_start,
                "z_step":           getattr(g, "z_step", None),
            })
            roi_id    = getattr(g, "roi_id",    None)
            roi_label = getattr(g, "roi_label", None)
            z_plane   = getattr(g, "z",         None)
            x_pix     = getattr(g, "x_pix",     None)
            y_pix     = getattr(g, "y_pix",     None)
            for j in range(g.n_roi):
                roi_records.append({
                    "global_index":     global_index,
                    "experiment_index": i,
                    "experiment":       g.name,
                    "indicator":        g.indicator,
                    "roi_index":        j,
                    "roi_id":           roi_id[j]    if roi_id    is not None else None,
                    "roi_label":        roi_label[j] if roi_label is not None else None,
                    "z_plane":          z_plane[j]   if z_plane   is not None else None,
                    "x_centroid_pix":   _roi_centroid(x_pix, j) if x_pix is not None else np.nan,
                    "y_centroid_pix":   _roi_centroid(y_pix, j) if y_pix is not None else np.nan,
                })
                global_index += 1
        meta[key] = {"experiments": pd.DataFrame(exp_records),
                     "rois":        pd.DataFrame(roi_records)}

    meta["odours"]  = odours.odours.get_order("X0Y0")
    meta["centers"] = centers
    return meta

if __name__ == "__main__":
    X0, Y0 = load_experiments(save_if_reload=True)
    print(f"X0 (input, OMP):   {len(X0)} experiments, shapes {[Xi.shape for Xi in X0]}")
    print(f"Y0 (output, Tbet): {len(Y0)} experiments, shapes {[Yi.shape for Yi in Y0]}")
