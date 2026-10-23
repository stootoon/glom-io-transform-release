"""Check that the stored X0Y0_new.p matches what responses.py rebuilds.

X0Y0_new.p is the anchor for everything downstream, so it is worth being able
to confirm, on demand, that the code still reproduces it exactly. Any
difference means the pickle and the pipeline have drifted apart.

Rebuilding reads the raw .mat experiment files, so this needs $DATA (or
--data-dir) as well as $GLOM_IO_DATA.

Run:
    python glom_io_transform/data/check_X0Y0.py
    python -m glom_io_transform.data.check_X0Y0 --tol 1e-12 --odour-order X0Y0
"""
import argparse
import os
import pickle
import sys

import numpy as np

# The model modules use relative imports, so they must be imported as part of
# the package. Put the repo root on the path so this works as a plain script too.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import glom_io_transform.paths as paths
from glom_io_transform.data import odours
from glom_io_transform.data import responses


def compare_arrays(loaded, built, label, tol=0.0):
    """Compare two lists of per-experiment arrays. Returns True if they match."""
    ok = True
    if len(loaded) != len(built):
        print(f"  ERR: {label}: {len(loaded)} experiments stored, {len(built)} rebuilt.")
        return False
    for i, (a, b) in enumerate(zip(loaded, built)):
        a, b = np.asarray(a), np.asarray(b)   # built arrays may be DataArrays
        if a.shape != b.shape:
            print(f"  {label} {i}: ERR shape mismatch, stored {a.shape} vs rebuilt {b.shape}")
            ok = False
            continue
        diff = np.linalg.norm(np.nan_to_num(a - b))
        n_nan = int(np.isnan(a).sum()), int(np.isnan(b).sum())
        status = "OK " if diff <= tol else "ERR"
        if diff > tol:
            ok = False
        print(f"  {status} {label} {i}: shape {str(a.shape):16s} norm difference {diff:.3e}"
              + (f"  (NaNs stored/rebuilt: {n_nan[0]}/{n_nan[1]})" if any(n_nan) else ""))
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x0y0", default=None,
                   help="path to X0Y0_new.p (default: <data_root>/X0Y0_new.p)")
    p.add_argument("--odour-order", default="X0Y0",
                   help="odour order to rebuild with (default: X0Y0, i.e. the stored order)")
    p.add_argument("--tol", type=float, default=0.0,
                   help="tolerated norm difference per experiment (default: 0, i.e. exact)")
    p.add_argument("--data-dir", default=None,
                   help="directory holding the raw .mat files (default: $DATA/tobias/allExp)")
    p.add_argument("--skip-verify-odours", action="store_true",
                   help="skip checking odour_labels.mat against the gl_tbet acquisition order")
    args = p.parse_args(argv)

    if args.data_dir:
        responses.set_data_dir(args.data_dir)

    x0y0_file = args.x0y0 or os.path.join(paths.data_root, "X0Y0_new.p")
    assert os.path.exists(x0y0_file), f"File not found: {x0y0_file}"

    ok = True

    # The rebuild selects odours by name, which is only equivalent to the stored
    # order if the labels agree with the acquisition order, so check that first.
    if not args.skip_verify_odours:
        print("Verifying odour labels against the gl_tbet acquisition order...")
        odours.verify_odours()
        print("  OK: odour_labels.mat is in gl_tbet order.\n")

    print(f"Loading stored data from {x0y0_file}")
    with open(x0y0_file, "rb") as f:
        stored = pickle.load(f)
    X0_stored, Y0_stored = stored["X0"], stored["Y0"]

    print(f"Rebuilding from the raw .mat files (odour order: {args.odour_order})...")
    X0_built, Y0_built = responses.get_data_for_classification(odour_order=args.odour_order)

    print(f"\nComparing (tolerance {args.tol:g}):")
    ok &= compare_arrays(X0_stored, X0_built, "input ", tol=args.tol)
    ok &= compare_arrays(Y0_stored, Y0_built, "output", tol=args.tol)

    # The rebuilt arrays are labelled, so we can also report what the odour axis holds.
    names = list(np.asarray(X0_built[0].odour.values)) if hasattr(X0_built[0], "odour") else None
    if names:
        print(f"\nRebuilt odour axis ({len(names)} odours), first 4: {names[:4]}")

    print("\n" + ("MATCH: the stored X0Y0 is reproduced exactly."
                  if ok else "MISMATCH: the stored X0Y0 and the rebuild differ."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
