"""Odour names, chemical classes, and the canonical odour orderings.

Reads odour_labels.mat and odour_orders.csv from $GLOM_IO_DATA, which is the
single source of truth for data files: if it is unset, or the file is not there,
this errors rather than looking anywhere else.

Both files are loaded lazily on first use (rather than at import time), so
importing this module -- or anything that imports it -- does not itself require
$GLOM_IO_DATA to be set.

Usage:
    from glom_io_transform.data.odours import odours
    odours.names, odours.classes
    odours.get_order("chemical_class")   # or "input", "output", "default"
"""
import os
from functools import lru_cache
from typing import List, NamedTuple

ORDERS = ("default", "chemical_class", "input", "output")


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
        """Indices that put the odours into the requested order."""
        if which_order == "default":
            return list(range(len(self.names)))
        if which_order not in ORDERS:
            raise ValueError(f"Unknown order: {which_order}. Must be one of {ORDERS}.")
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


def __getattr__(name):
    # Module-level lazy attributes (PEP 562), so `from ... import odours` still
    # works but only reads the files when the name is actually requested.
    if name == "odours":
        return load_odours()
    if name == "orders_df":
        return load_orders()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
