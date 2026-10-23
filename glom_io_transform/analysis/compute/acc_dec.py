"""Fig 5: how much decorrelation contributes to decoding accuracy.

Runs the transformation sequences (see acc_dec_transform), collects their
results, and loads them for figures.acc_dec. From the repository root:

    python -m glom_io_transform.analysis.compute.acc_dec all       --ntrials 10
    python -m glom_io_transform.analysis.compute.acc_dec all       --ntrials 10 --flip
    python -m glom_io_transform.analysis.compute.acc_dec collect   --ntrials 10
    python -m glom_io_transform.analysis.compute.acc_dec collect   --ntrials 10 --flip

`which` is "all", "collect", or orderings named by the first letters of their
transformations -- Angles, Lengths, Rotation, Covariance -- joined by hyphens
(e.g. "alrc-aclr"), so the sequences can be spread over cluster jobs and merged
afterwards. An ordering with ROTATION before ANGLES is not valid and is
recorded as an error, so "all" yields 12 of the 24. --flip runs from the output
to the input.

--ntrials is the number of random subsamplings (seeds), not imaging trials. The
paper used 10; "all" at 10 takes about five minutes per direction on a laptop.
"""
import os
import pickle
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob

import numpy as np

from glom_io_transform import paths
from glom_io_transform.data.odours import odours
from glom_io_transform.model_fitting import driver as fits
from .compute import Computation
from .acc_dec_transform import TransformInputToOutput

RESULTS_DIR = os.path.join(paths.proj_path, "out", "acc_dec")
LETTERS = {"a": "ANGLES", "l": "LENGTHS", "r": "ROTATION", "c": "COVARIANCE"}


def load_data():
    """Input and output responses, one rois x odours x trials array per experiment.

    Loaded exactly as the model fits load them -- X0Y0_new.p, through
    model_fitting.driver.get_data -- so this analysis sees the same cells, odours
    and preprocessing as every other one. The odours are in the stored X0Y0
    order, which is the order the 2023 analysis used, so the example odours in
    Fig 5A are unchanged.
    """
    X, Y = fits.get_data(full=True)
    order = odours.get_order("X0Y0")
    for Z in (X, Y):
        for z in Z:
            assert z.dims == ("roi", "odour", "repetition"), f"Unexpected dims {z.dims}."
            assert list(z.odour.values) == order, "Odours are not in the X0Y0 order."
    return [np.asarray(z) for z in X], [np.asarray(z) for z in Y]


@dataclass
class Results:
    """One direction's results.

    accs:   ordering -> seeds x steps x splits, the accuracy after each step
    Xseq_:  ordering -> seeds x steps x rois x odours x trials, the dataset after each step
    errors: ordering -> why it could not be run
    Xorig, Yorig: the data the run started from, to check runs before merging them.

    Saved as a plain dict rather than as this class, so that results stay
    readable wherever the code that made them lives.
    """
    flip: bool
    Xorig: list
    Yorig: list
    accs: dict
    Xseq_: dict
    errors: dict

    @classmethod
    def from_run(cls, T):
        return cls(flip=T.flip, Xorig=T.Xorig, Yorig=T.Yorig,
                   accs=T.accs, Xseq_=T.Xseq_, errors=T.errors)

    def save(self, path):
        with open(path, "wb") as fp:
            pickle.dump(vars(self), fp)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fp:
            return cls(**pickle.load(fp))

    def merge(self, other):
        """Add other's orderings; both must come from the same data and direction."""
        assert self.flip == other.flip, "Flip values don't match."
        for name in ("Xorig", "Yorig"):
            P, Q = getattr(self, name), getattr(other, name)
            assert len(P) == len(Q) and all(np.array_equal(p, q) for p, q in zip(P, Q)), \
                f"{name} doesn't match."
        overlap = set(self.accs) & set(other.accs)
        assert not overlap, f"Both have results for {overlap}."
        self.accs.update(other.accs)
        self.Xseq_.update(other.Xseq_)
        self.errors.update(other.errors)


def results_file(flip, n_trials, which=None, results_dir=RESULTS_DIR):
    """Where one run (`which` given) or the collected results are written."""
    direction = "o2i" if flip else "i2o"
    name = f"T{direction}.{n_trials}.p" if which is None else f"T{direction}.{which}.{n_trials}.p"
    return os.path.join(results_dir, name)


def parse_orderings(which):
    """'alrc-aclr' -> [('ANGLES', 'LENGTHS', 'ROTATION', 'COVARIANCE'), ...]."""
    orderings = [tuple(LETTERS[ch] for ch in code.lower()) for code in which.split("-")]
    for order in orderings:
        TransformInputToOutput.validate_order(order)
    return orderings


def collect(flip, n_trials, results_dir=RESULTS_DIR):
    """Merge every run of one direction and seed count into a single result."""
    pattern = results_file(flip, n_trials, which="*", results_dir=results_dir)
    files = sorted(glob(pattern))
    assert files, f"No runs match {pattern}."
    merged = None
    for f in files:
        print(f"Collecting from {f}.")
        r = Results.load(f)
        if merged is None:
            merged = r
        else:
            merged.merge(r)
    return merged


class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, n_trials=10, results_dir=RESULTS_DIR):
        print("COMPUTING acc_dec.Data.")
        self.Ti2o = Results.load(results_file(False, n_trials, results_dir=results_dir))
        self.To2i = Results.load(results_file(True,  n_trials, results_dir=results_dir))
        # Both directions subsample with the same seeds before flipping, so each
        # one's starting dataset is the other's destination.
        for k in self.Ti2o.Xseq_:
            assert np.allclose(self.Ti2o.Xseq_[k][0, 0],  self.To2i.Xseq_[k][0, -1])
            assert np.allclose(self.Ti2o.Xseq_[k][0, -1], self.To2i.Xseq_[k][0, 0])
        self.computed = True
        return self


def main():
    parser = ArgumentParser(description="Transform input data to output data sequentially and report decoding accuracy.")
    parser.add_argument("which", type=str, help="'all', 'collect', or hyphen-separated orderings such as 'alrc-aclr'.")
    parser.add_argument("--flip", action="store_true", help="Go from the output to the input.")
    parser.add_argument("--ntrials", type=int, default=2, help="Number of random subsamplings (seeds).")
    parser.add_argument("--outdir", type=str, default=RESULTS_DIR, help="Where results are written and collected from.")
    args = parser.parse_args()
    print(args)
    os.makedirs(args.outdir, exist_ok=True)

    if args.which.lower() == "collect":
        results = collect(args.flip, args.ntrials, results_dir=args.outdir)
        out_file = results_file(args.flip, args.ntrials, results_dir=args.outdir)
    else:
        run_only = None if args.which.lower() == "all" else parse_orderings(args.which)
        print("Direction: " + ("output-to-input" if args.flip else "input-to-output"))
        Xorig, Yorig = load_data()
        T = TransformInputToOutput(Xorig, Yorig, flip=args.flip)
        T.run_all(n_trials=args.ntrials, run_only=run_only)
        results = Results.from_run(T)
        out_file = results_file(args.flip, args.ntrials, which=args.which, results_dir=args.outdir)

    results.save(out_file)
    print(f"Wrote {out_file}")
    print("ALLDONE")


if __name__ == "__main__":
    main()
