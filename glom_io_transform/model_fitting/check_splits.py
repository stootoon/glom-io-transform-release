"""Check that the odour splits the fits actually used are in the right frame.

gen_split returns positions along the data's odour axis, which is the stored
X0Y0 order, and those positions are written into each in.N.p. If the classes
were looked up in the wrong frame, an 'outclass' run holds out a set of odours
that is not the class it names -- which is what happened before, and is
invisible in the results unless you go and look.

For every in.N.p under the fits root this reports what the held-out odours
actually are:

  outclass : every test/vld odour must belong to the named class, and no odour
             of that class may appear in the training set;
  inclass  : exactly one odour held out per class.

Timestamps are reported too, so a directory that was never regenerated shows up
as old rather than as wrong.

Run:
    python glom_io_transform/model_fitting/check_splits.py
    python -m glom_io_transform.model_fitting.check_splits --fits-root ~/.../fits
"""
import argparse
import glob
import os
import pickle
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from glom_io_transform.data.odours import odours

GRN, RED, YEL, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


def class_by_position():
    """Chemical class of the odour at each position of the stored odour axis."""
    storage  = odours.get_order("X0Y0")
    class_of = dict(zip(odours.names, odours.classes))
    return [class_of[n] for n in storage], storage


def check_config(config, classes):
    """(mode, ok, detail) for one run configuration."""
    sampler = config.get("sampler", {})
    split   = sampler.get("split", {})
    mode    = split.get("mode", sampler.get("type", "?"))
    held    = list(split.get("test_odours", [])) + list(split.get("vld_odours", []))
    train   = list(split.get("train_odours", []))
    if not held:
        return mode, None, "no held-out odours recorded"

    got = Counter(classes[i] for i in held)
    if mode == "outclass":
        want = split.get("outclass")
        wrong = sum(k for c, k in got.items() if c != want)
        leaked = sum(1 for i in train if classes[i] == want)
        ok = wrong == 0 and leaked == 0
        detail = f"outclass={want}: held out {dict(got)}"
        if leaked:
            detail += f", and {leaked} {want}(s) still in train"
        return mode, ok, detail
    if mode == "inclass":
        ok = set(got.values()) == {1} and len(got) == len(set(classes))
        return mode, ok, f"held out {len(held)} odours over {len(got)} classes: {dict(got)}"
    return mode, None, f"held out {len(held)} odours: {dict(got)}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = None
    try:
        import glom_io_transform.paths as paths
        default_root = os.path.join(paths.fits_root, "fits")
    except Exception:
        pass
    p.add_argument("--fits-root", default=default_root or "fits")
    p.add_argument("--max-bad", type=int, default=5, help="examples to print per directory")
    args = p.parse_args(argv)

    classes, _ = class_by_position()
    root = os.path.expanduser(args.fits_root)
    ins  = sorted(glob.glob(os.path.join(root, "**", "in.*.p"), recursive=True))
    if not ins:
        print(f"No in.*.p under {root}")
        return 1

    by_dir = {}
    for path in ins:
        by_dir.setdefault(os.path.dirname(path), []).append(path)

    print(f"{BOLD}splits{OFF}  {root}\n")
    all_ok = True
    for d, paths_ in sorted(by_dir.items()):
        modes, bad, undecided, mtimes = Counter(), [], 0, []
        for path in paths_:
            mtimes.append(os.path.getmtime(path))
            try:
                with open(path, "rb") as f:
                    config = pickle.load(f)
            except Exception as e:
                bad.append((path, f"unreadable: {type(e).__name__}"))
                continue
            mode, ok, detail = check_config(config, classes)
            modes[mode] += 1
            if ok is False:
                bad.append((path, detail))
            elif ok is None:
                undecided += 1

        name   = os.path.relpath(d, root)
        newest = time.strftime("%Y-%m-%d %H:%M", time.localtime(max(mtimes)))
        oldest = time.strftime("%Y-%m-%d %H:%M", time.localtime(min(mtimes)))
        stamp  = newest if newest == oldest else f"{oldest} .. {newest}"
        colour = RED if bad else (YEL if undecided == len(paths_) else GRN)
        state  = f"{len(bad)} WRONG" if bad else ("not checkable" if undecided == len(paths_) else "ok")
        print(f"  {colour}{name:<44}{OFF} {len(paths_):>4} configs  "
              f"{'/'.join(sorted(modes)):<18} {colour}{state:<14}{OFF} {DIM}{stamp}{OFF}")
        for path, detail in bad[:args.max_bad]:
            print(f"      {DIM}{os.path.basename(path)}: {detail}{OFF}")
        if len(bad) > args.max_bad:
            print(f"      {DIM}... and {len(bad)-args.max_bad} more{OFF}")
        all_ok &= not bad

    print("\n" + (f"{GRN}ALL SPLITS IN THE RIGHT FRAME{OFF}" if all_ok
                  else f"{RED}SOME SPLITS ARE IN THE WRONG FRAME -- those fits need regenerating{OFF}"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
