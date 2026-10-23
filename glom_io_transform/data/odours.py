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
import numpy as np
from functools import lru_cache
from typing import List, NamedTuple

from .common import load_mat, get_registry, WARN

# Named orderings. There is deliberately no "default": every caller says which
# order it wants, so that no ordering is ever applied by accident.
#   X0Y0          : the order the response data is stored in (see olo)
#   tbet          : the gl_tbet acquisition order, which is the order
#                   odour_labels.mat (and hence odours.names) is in
#   chemical_class: grouped by chemical class
#   input/output  : the clustered orders used for plotting. NOTE these rank the
#                   TBET list, not the CSV's rows -- see get_order.
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


def normalize_class_name(chemical_class: str) -> str:
    """Chemical class names, lowercased so the sources agree.

    odour_labels.mat capitalises them ('Ketone'), odour_orders.csv does not
    ('ketones'). Lowercasing removes one of the two differences; the singular /
    plural difference remains.
    """
    return str(chemical_class).lower().strip()


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

    def get_order(self, which_order: str) -> List[str]:
        """The odour NAMES, in the requested order.

        Names rather than indices, so that callers select by odour
        (DataArray.sel(odour=...)) instead of by position. Indices are only
        meaningful relative to some other order, which is what caused the
        frames to drift apart in the first place.
        """
        if which_order not in ORDERS:
            raise ValueError(f"Unknown order: {which_order}. Must be one of {ORDERS}.")
        if which_order == "X0Y0":
            # The order the response data is stored in.
            return [self.names[i] for i in olo]
        if which_order == "tbet":
            # names are already in acquisition order.
            return list(self.names)
        # The remaining columns are all 1..N rankings, but they are rankings
        # INTO DIFFERENT LISTS, which is the trap here:
        #
        #   chemical_sort  ranks the CSV's OWN ROWS. Row k carries rank k+1, so
        #                  the file is already in chemical order and reading the
        #                  rank off each row is right.
        #   input, output  rank the TBET (acquisition) list. Rank r belongs to
        #                  the r-th name of odours.get_order("tbet"), NOT to the
        #                  name sitting on that CSV row.
        #
        # Reading input/output off the rows -- which this did until 2026-08-28 --
        # produces an ordering with no structure in the correlations at all
        # (fall-off with distance from the diagonal +0.01, against +0.45 for the
        # reading below and +0.52 for clustering the data directly). The two
        # cases cannot be distinguished by chemical_sort, because its ranks run
        # 1..N straight down the rows and so give the same answer either way;
        # that is why a chemical-ordered heat map matched Tobias's and an
        # input-ordered one did not.
        #
        # Each column also clusters ITS OWN side under this reading -- 'input'
        # organises the input correlations, 'output' the output ones -- which is
        # what marks it as the intended reading rather than a lucky permutation.
        orders = load_orders()
        if which_order == "chemical_class":
            ordered = list(orders.sort_values(by="chemical_sort")["name"])
        else:
            acquisition = list(self.names)          # tbet order
            ranks = orders[which_order].values
            assert sorted(ranks) == list(range(1, len(acquisition) + 1)), (
                f"The '{which_order}' column should be a 1..{len(acquisition)} "
                f"ranking, got {sorted(ranks)[:5]}...")
            ordered = [acquisition[i] for i in np.argsort(ranks)]
        missing = [n for n in ordered if n not in self.names]
        assert not missing, f"Odours in {which_order} order but not in the labels: {missing}"
        return ordered

    def index_of(self, names: List[str]) -> List[int]:
        """Positions of the given odour names within self.names.

        An escape hatch for code that must index a plain array positionally.
        Prefer selecting by name; if you find yourself calling this, check
        whether the array could carry an odour coordinate instead.
        """
        missing = [n for n in names if n not in self.names]
        assert not missing, f"Unknown odours: {missing}"
        return [self.names.index(n) for n in names]


@lru_cache(maxsize=None)
def load_odours():
    """Odour names (from odour_labels.mat) and chemical classes (from odour_orders.csv).

    The names come from the .mat because their ORDER is meaningful: it is the
    gl_tbet acquisition order that everything else is defined relative to (see
    verify_odours). The classes come from the order table, keyed by name, so
    there is a single source of truth for which class an odour belongs to.
    The .mat also carries a class list, but it disagreed with the order table
    for five odours (some phenols and the cineoles), so it is not used.
    """
    from scipy.io import loadmat
    mat = loadmat(get_data_file("odour_labels.mat"))
    names = [normalize_odour_name(str(n[0])) for n in mat["odour_labels"][0]]

    orders = load_orders()
    class_of = {normalize_odour_name(n): normalize_class_name(c)
                for n, c in zip(orders["name"], orders["chemical_class"])}
    missing = [n for n in names if n not in class_of]
    assert not missing, f"No chemical class in odour_orders.csv for: {missing}"

    return Odours(names=names, classes=[class_of[n] for n in names])

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
