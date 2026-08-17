"""Everything about odours: names, chemical classes, and the orderings.

Two sources are read here, and this is the only module that reads either:
  - odour_labels.mat / odour_orders.csv from $GLOM_IO_DATA -- names, chemical
    classes, and the plotting orders;
  - the raw .mat experiment files -- the per-dataset odour lists (gl_omp,
    gl_tbet, mtc), i.e. the order the odours were acquired in.

Both are loaded lazily on first use (rather than at import time), so importing
this module -- or anything that imports it -- does not itself require the data
to be present.

Usage:
    from glom_io_transform.data.odours import odours
    odours.names, odours.classes
    odours.get_order("X0Y0")   # or "tbet", "chemical_class", "input", "output"
"""
import os
from functools import lru_cache
from typing import List, NamedTuple

from .common import load_mat, get_registry, WARN

# Named orderings. There is deliberately no "default": every caller says which
# order it wants, so that no ordering is ever applied by accident.
#   X0Y0          : the order the response data is stored in (see olo)
#   tbet          : the gl_tbet acquisition order, which is the order
#                   odour_labels.mat (and hence odours.names) is in
#   chemical_class: grouped by chemical class
#   input/output  : the clustered orders used for plotting
ORDERS = ("X0Y0", "tbet", "chemical_class", "input", "output")

# The order the odours appear in X0Y0 (a clustered order; indices into the
# gl_tbet acquisition order, i.e. into odours.names). This is what get_order()
# returns for "X0Y0", because it is the order the response data is stored in.
olo = [21,24,25,11,46,8,17,5,33,16,22,26,27,29,13,43,28,42,47,10,4,2,35,23,31,38,41,40,14,39,
       7,44,19,15,3,34,0,12,9,6,1,36,32,30,18,37,20,45]


def get_data_file(name):
    """Path to <name> in $GLOM_IO_DATA, the single source of truth for data files.

    Errors rather than falling back to any other location.
    """
    assert "GLOM_IO_DATA" in os.environ, \
        "GLOM_IO_DATA is not set: it must point at the directory holding the data files."
    path = os.path.join(os.environ["GLOM_IO_DATA"], name)
    assert os.path.exists(path), f"Data file not found: {path}"
    return path


def normalize_odour_name(odour: str) -> str:
    """Converts odour names to Tobias' new list of odour names."""
    odour = odour.lower().strip()
    rename = {
        "2-methyl-4-butanol": "2-methyl-2-butanol",
        "cineol": "cineole",
        "1,4-cineol": "1,4-cineole",
    }
    return rename[odour] if odour in rename else odour


@lru_cache(maxsize=None)
def load_orders():
    """The odour order table: name, chemical_class, chemical_sort, input, output."""
    import pandas as pd
    return pd.read_csv(get_data_file("odour_orders.csv"), delimiter=";")


class Odours(NamedTuple):
    names: List[str]
    classes: List[str]

    def get_order(self, which_order: str) -> List[int]:
        """Indices that put the odours into the requested order.

        The indices are into self.names, which is in the gl_tbet acquisition
        order (see verify_odours).
        """
        if which_order not in ORDERS:
            raise ValueError(f"Unknown order: {which_order}. Must be one of {ORDERS}.")
        if which_order == "X0Y0":
            # The order the response data is stored in.
            return list(olo)
        if which_order == "tbet":
            # names are already in acquisition order, so this is the identity.
            return list(range(len(self.names)))
        # 'chemical_sort' is the explicit 1..N ranking that groups the odours by
        # class; sorting on the class *name* instead would order the classes
        # alphabetically (putting the blank mid-list) and leave the within-class
        # order to the sort's tie-breaking.
        column = "chemical_sort" if which_order == "chemical_class" else which_order
        ordered = load_orders().sort_values(by=column)["name"]
        missing = [n for n in ordered if n not in self.names]
        assert not missing, f"Odours in {column} order but not in the labels: {missing}"
        return [self.names.index(n) for n in ordered]


@lru_cache(maxsize=None)
def load_odours():
    from scipy.io import loadmat
    mat = loadmat(get_data_file("odour_labels.mat"))
    return Odours(names   = [normalize_odour_name(str(n[0])) for n in mat["odour_labels"][0]],
                classes = [str(n[0]) for n in mat["odour_labels"][1]])

def verify_odours():
    """Check odour_labels.mat is in the same order as the gl_tbet acquisition.

    Everything that indexes odours positionally relies on this, so it is worth
    checking explicitly. Requires the raw .mat files, which is why it is a
    separate call rather than part of load_odours().
    """
    od = load_odours()
    tbet_odours = get_odours_for_datasets()["gl_tbet"]
    assert od.names == tbet_odours, "Odours in odour_labels and the tbet odours don't match"
    return True

@lru_cache(maxsize=None)
def get_odours_for_datasets():
    """Odour name lists per dataset, read from the raw .mat experiment files.

    These are the odours in the order they were *acquired*, which differs
    between the input (gl_omp) and output (gl_tbet) datasets, and from the
    order the data is stored in (see olo).
    """
    reg = get_registry()

    data = load_mat(reg[(reg.rois == "GL") & (reg.indicator.str.lower() == "omp")].file_name.values[0])
    odours_gl_omp = [normalize_odour_name(o) for o in data["expInfo"]["odours"]]

    data = load_mat(reg[(reg.rois == "GL") & (reg.indicator.str.lower() == "tbet")].file_name.values[0])
    odours_gl_tbet = [normalize_odour_name(o) for o in data["expInfo"]["odours"]]

    # Not needed for X0Y0, so don't fail if no MTC experiments are present.
    try:
        data = load_mat(reg[reg.rois == "MTC"].file_name.values[0])
        odours_mtc = [normalize_odour_name(o) for o in data["expInfo"]["odours"]]
    except (IndexError, FileNotFoundError):
        WARN("No MTC experiments found; odours['mtc'] will be None.")
        odours_mtc = None

    return {"gl_omp": odours_gl_omp, "gl_tbet": odours_gl_tbet, "mtc": odours_mtc}


def __getattr__(name):
    # Module-level lazy attributes (PEP 562), so `from ... import odours` still
    # works but only reads the files when the name is actually requested.
    if name == "odours":
        return load_odours()
    if name == "orders_df":
        return load_orders()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
