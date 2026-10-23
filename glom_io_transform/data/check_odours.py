"""Check the odour metadata is self-consistent.

Everything downstream assumes:
  - odour_labels.mat is in the gl_tbet acquisition order (this is what makes
    the stored X0Y0 order, and hence every ordering, well defined);
  - every odour has a chemical class in odour_orders.csv;
  - each named ordering is a permutation of the same 48 odours.

Reading the acquisition order needs the raw .mat files, so this needs $DATA (or
--data-dir) as well as $GLOM_IO_DATA.

Run:
    python glom_io_transform/data/check_odours.py
    python -m glom_io_transform.data.check_odours --data-dir /path/to/allExp
"""
import argparse
import os
import sys
from collections import Counter

# The data modules use relative imports, so they must be imported as part of the
# package. Put the repo root on the path so this works as a plain script too.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from glom_io_transform.data import common, odours as odours_mod
from glom_io_transform.data.odours import odours, load_orders, ORDERS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None,
                   help="directory holding the raw .mat files (default: $DATA/tobias/allExp)")
    p.add_argument("--skip-acquisition", action="store_true",
                   help="skip the check against the raw .mat files (no raw data needed)")
    args = p.parse_args(argv)

    if args.data_dir:
        common.set_data_dir(args.data_dir)

    ok = True
    n = len(odours.names)
    print(f"{n} odours, {len(set(odours.names))} distinct names.")
    if len(set(odours.names)) != n:
        dupes = [k for k, v in Counter(odours.names).items() if v > 1]
        print(f"  ERR: duplicate odour names: {dupes}")
        ok = False

    print("\nChemical classes (from odour_orders.csv):")
    for c, k in sorted(Counter(odours.classes).items()):
        print(f"  {c:10s} {k:2d}")
    csv = dict(zip(load_orders()["name"], load_orders()["chemical_class"]))
    unclassed = [nm for nm in odours.names if nm not in csv]
    if unclassed:
        print(f"  ERR: no class for: {unclassed}")
        ok = False

    print("\nOrderings:")
    for which in ORDERS:
        try:
            names = odours.get_order(which)
            idx   = odours.index_of(names)
            good  = sorted(idx) == list(range(n))
            print(f"  {which:15s} {'OK ' if good else 'ERR'} "
                  f"{len(names)} odours, permutation: {good}   first 3: {names[:3]}")
            ok &= good
        except Exception as e:
            print(f"  {which:15s} ERR {type(e).__name__}: {e}")
            ok = False

    if args.skip_acquisition:
        print("\nSkipping the acquisition-order check (--skip-acquisition).")
    else:
        print("\nChecking odour_labels.mat against the gl_tbet acquisition order...")
        try:
            odours_mod.verify_odours()
            print("  OK: odour_labels.mat is in gl_tbet order.")
        except AssertionError:
            ok = False
            tbet = odours_mod.get_odours_for_datasets()["gl_tbet"]
            print("  ERR: odour_labels.mat is NOT in gl_tbet order.")
            if sorted(tbet) != sorted(odours.names):
                print(f"    only in labels: {sorted(set(odours.names) - set(tbet))}")
                print(f"    only in tbet  : {sorted(set(tbet) - set(odours.names))}")
            else:
                first = next(i for i, (a, b) in enumerate(zip(odours.names, tbet)) if a != b)
                print(f"    same odours, different order; first difference at position {first}: "
                      f"labels={odours.names[first]!r} tbet={tbet[first]!r}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
