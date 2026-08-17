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
# Requirements: numpy, pandas, mat73.
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
# responses, with odours in the clustered "olo" order.

import os
import logging
import pickle
from glob import glob
from functools import partial
from collections import namedtuple
from collections.abc import Sequence

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Resolved lazily so importing this module doesn't require $DATA to be set.
_data_dir = None

def set_data_dir(path):
    """Point the loader at the directory containing Tobias' .mat files."""
    global _data_dir
    _data_dir = path

def get_data_dir():
    if _data_dir is not None:
        return _data_dir
    if "DATA" in os.environ:
        return os.path.join(os.environ["DATA"], "tobias", "allExp")
    raise RuntimeError("Set the $DATA environment variable or call set_data_dir().")

# Registry CSV caching the (file, roi type, indicator) table. Kept next to this
# file by default so reruns don't rescan every .mat.
registry_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry.csv")

def create_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        for h in logger.handlers:
            logger.removeHandler(h)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s %(module)24s %(levelname)8s: %(message)s',
                                      datefmt='%Y/%m/%d %H:%M:%S'))
    logger.addHandler(ch)
    return logger

logger = create_logger("data")
INFO   = logger.info
WARN   = logger.warning
DEBUG  = logger.debug

# ----------------------------------------------------------------------------
# Experiment registry and odour lists (from datasets2.py)
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

def build_experiments_registry():
    import mat73
    import pandas as pd
    records = []
    for file_name in sorted(glob(get_data_dir() + "/*.mat")):
        try:
            data = mat73.loadmat(file_name)
        except Exception as e:
            # There can be an OSError if the file is not a MATLAB 7.3 file
            # In that case use scipy.io.loadmat instead (but that will fail for 7.3 files)

            # Report the exception and try to load
            print(f"Failed to load {file_name} with mat73: {e}. Trying scipy.io.loadmat instead.")
            from scipy.io import loadmat as scipy_loadmat
            data = scipy_loadmat(file_name)
                
        rois = data["expInfo"]["rois"]
        ind  = data["expInfo"]["type"]
        INFO(f"{file_name} is {rois} {ind}.")
        records.append({"file_name": file_name, "rois": rois, "indicator": ind})
    INFO(f"Found {len(records)} records.")
    df = pd.DataFrame(records)
    df.to_csv(registry_file)
    INFO(f"Wrote {registry_file}.")

def get_registry():
    import pandas as pd
    if not os.path.isfile(registry_file):
        INFO(f"{registry_file=} not found, building.")
        build_experiments_registry()
    return pd.read_csv(registry_file)

_odours = None

def get_odours():
    """Odour name lists per dataset. Cached after the first call."""
    global _odours
    if _odours is not None:
        return _odours
    import mat73
    reg = get_registry()

    data = mat73.loadmat(reg[(reg.rois == "GL") & (reg.indicator.str.lower() == "omp")].file_name.values[0])
    odours_gl_omp = [o.lower() for o in data["expInfo"]["odours"]]

    data = mat73.loadmat(reg[(reg.rois == "GL") & (reg.indicator.str.lower() == "tbet")].file_name.values[0])
    odours_gl_tbet = [o.lower() for o in data["expInfo"]["odours"]]

    # Not needed for X0Y0, so don't fail if no MTC experiments are present.
    try:
        data = mat73.loadmat(reg[reg.rois == "MTC"].file_name.values[0])
        odours_mtc = [o.lower() for o in data["expInfo"]["odours"]]
    except (IndexError, FileNotFoundError):
        WARN("No MTC experiments found; odours['mtc'] will be None.")
        odours_mtc = None

    _odours = {"gl_omp": odours_gl_omp, "gl_tbet": odours_gl_tbet, "mtc": odours_mtc}
    return _odours

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
                    WARN(f"Warning in assigning value for {key=} to {fld=}: {e}.")
                    WARN("Attempting to assign without array conversion.")
                    self.__dict__[fld] = val

        self.n_roi, self.n_odours, self.n_reps, self.n_t = self.ca2.shape
        self.fs = self.fs_ca2
        self.t  = np.arange(self.n_t)/self.fs
        self.trial_length   = self.isi_sec  # seconds
        self.odour_start    = self.odour_on_index/self.fs
        self.odour_duration = 2

        INFO(f"Loaded {self.__str__()}.")

    def __str__(self):
        return (f"Glomerular Experiment {self.name:>8s}: {self.n_roi:>3d} ROIs for "
                f"{self.n_odours:>2d} odours. {self.n_t} time points at fs = {self.fs:1.1f} "
                f"is {self.t[-1]:1.1f} seconds.")

def load_glomerular_experiments(indicator, from_registry=True):
    import mat73
    glomerular_experiments = []
    n_rois = 0

    if from_registry:
        df = get_registry()
        file_names = df[(df.rois == "GL") & (df.indicator.str.lower() == indicator.lower())].file_name.values
    else:
        file_names = glob(get_data_dir() + "/*.mat")

    for file_name in sorted(file_names):
        DEBUG(f"Trying {file_name}.")
        data = mat73.loadmat(file_name)
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

def z_score_experiment(g, int_width=5, max_width=5, up_sample=100, which_elements=np.arange(10)):

    which_el_str = f"{which_elements=}" + ("" if which_elements is not None else " (not actually Z-scoring).")
    DEBUG(f"Z-scoring {g.indicator} experiment {g.name} using {which_el_str}.")
    t        = g.t
    ca2t     = g.ca2  # roi, odour, trial, time
    odour_on = g.odour_start

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
    integrators = [lambda X, t_start=t_start: np.sum(X[:, :, :, (t_interp >= t_start) & (t_interp < t_start + int_width)], axis=-1) for t_start in t_starts]
    ca2t_zi = np.array([integrator(ca2t_z_interp) / scale for integrator in integrators])
    # Put the first dimension (bins) last
    ca2t_zi = np.moveaxis(ca2t_zi, 0, -1)
    ca2ta   = np.nanmean(ca2t, axis=2)  # trial average
    mu      = np.nanmean(ca2ta[:, :, t < odour_on], axis=-1)
    sd      = np.nanstd( ca2ta[:, :, t < odour_on], axis=-1)
    ca2ta_z = (ca2ta - mu[:, :, None])/sd[:, :, None]

    ca2ta_z_interp = interp_last_axis(ca2ta_z, t, t_interp)
    ca2ta_zi = np.array([np.nansum(ca2ta_z_interp[:, :, (t_interp >= t_start) & (t_interp < t_start + int_width)], axis=-1) / scale for t_start in t_starts])
    ca2ta_zi = np.moveaxis(ca2ta_zi, 0, -1)

    return ca2t_z, ca2t_zi, ca2ta_z, ca2ta_zi, t_starts, ca2t_z_interp, t_interp, t

# Odour order we use when plotting (clustered order; indices into odours["gl_tbet"])
olo = [21,24,25,11,46,8,17,5,33,16,22,26,27,29,13,43,28,42,47,10,4,2,35,23,31,38,41,40,14,39,7,44,19,15,3,34,0,12,9,6,1,36,32,30,18,37,20,45]

def get_data_for_classification(olo=None, which_elements=np.arange(10)):
    odours = get_odours()
    odour_inds = [odours["gl_omp"].index(o) for o in odours["gl_tbet"]]
    Zin  = [z_score_experiment(g, which_elements=which_elements)[1][:, :, :, 0] for g in glom_omp]  # ...0]: Get the result from the first (and only) bin
    Zin  = [Z[:, odour_inds, :] for Z in Zin]  # Only keep the odours we are interested in
    Zout = [z_score_experiment(g, which_elements=which_elements)[1][:, :, :, 0] for g in glom_tbet]

    if olo is not None:
        Zin  = [Zi[:, olo, :] for Zi in Zin]
        Zout = [Zi[:, olo, :] for Zi in Zout]

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
            logger.setLevel("WARN")
            X0, Y0 = get_data_for_classification(olo=olo)
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
                  i.e. odours["gl_tbet"] reordered by olo.
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

    meta["odours"]  = [get_odours()["gl_tbet"][i] for i in olo]
    meta["centers"] = centers
    return meta

if __name__ == "__main__":
    X0, Y0 = load_experiments(save_if_reload=True)
    print(f"X0 (input, OMP):   {len(X0)} experiments, shapes {[Xi.shape for Xi in X0]}")
    print(f"Y0 (output, Tbet): {len(Y0)} experiments, shapes {[Yi.shape for Yi in Y0]}")
