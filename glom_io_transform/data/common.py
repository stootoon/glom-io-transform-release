"""Shared plumbing for the data layer: raw-data location, logging, MATLAB file
loading, and the experiment registry.

Note there are two distinct data roots:
  - $DATA/tobias/allExp  -- the raw .mat experiment files, resolved by
                            get_data_dir() here;
  - $GLOM_IO_DATA        -- the packaged/derived files that ship with this
                            package (odour_labels.mat, odour_orders.csv,
                            X0Y0_new.p, ...), resolved by odours.get_data_file().
"""
import os
import logging
from glob import glob

# ----------------------------------------------------------------------------
# Raw data location
# ----------------------------------------------------------------------------

# Resolved lazily so importing this module doesn't require $DATA to be set.
_data_dir = None

def set_data_dir(path):
    """Point the loader at the directory containing Tobias' raw .mat files."""
    global _data_dir
    _data_dir = path

def get_data_dir():
    """Directory holding the raw .mat experiment files."""
    if _data_dir is not None:
        return _data_dir
    if "DATA" in os.environ:
        return os.path.join(os.environ["DATA"], "tobias", "allExp")
    raise RuntimeError("Set the $DATA environment variable or call set_data_dir().")

# Registry CSV caching the (file, roi type, indicator) table. Kept next to this
# file by default so reruns don't rescan every .mat.
registry_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry.csv")

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

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
INFO   = print #logger.info
WARN   = print #logger.warning
DEBUG  = print #logger.info

# ----------------------------------------------------------------------------
# MATLAB file loading
# ----------------------------------------------------------------------------

def _structs_to_arrays(obj):
    """Normalise scipy's MATLAB layout to mat73's.

    A MATLAB struct array comes back from scipy as a list of dicts (one per
    element), whereas mat73 returns a dict of arrays (one entry per field,
    stacked over elements). Downstream code expects the latter, so convert
    "array of structs" -> "struct of arrays", recursively.
    """
    if isinstance(obj, dict):
        return {k: _structs_to_arrays(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and len(obj) and all(isinstance(o, dict) for o in obj):
        keys = list(obj[0].keys())
        if all(list(o.keys()) == keys for o in obj):
            return {k: [_structs_to_arrays(o[k]) for o in obj] for k in keys}
    return obj


def load_mat(file_name):
    """Load a MATLAB file regardless of which format it was saved in.

    Tries mat73 first (the only option for v7.3 / HDF5 files), then falls back
    to scipy.io.loadmat for v7 and earlier. The scipy result is passed through
    simplify_cells and _structs_to_arrays so that both paths return the same
    shape of nested dicts / lists / arrays, and callers don't have to care
    which format the file was in.
    """
    try:
        import mat73
    except ImportError:
        mat73 = None
        DEBUG("mat73 not installed; using scipy.io.loadmat only.")

    if mat73 is not None:
        try:
            return mat73.loadmat(file_name)
        except Exception as e:
            # Typically a TypeError/OSError because the file is not v7.3.
            DEBUG(f"mat73 could not read {file_name} ({e}); falling back to scipy.io.loadmat.")

    from scipy.io import loadmat as scipy_loadmat
    return _structs_to_arrays(scipy_loadmat(file_name, simplify_cells=True))

# ----------------------------------------------------------------------------
# Experiment registry
# ----------------------------------------------------------------------------

def build_experiments_registry():
    import pandas as pd
    records = []
    for file_name in sorted(glob(get_data_dir() + "/*.mat")):
        data = load_mat(file_name)
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
