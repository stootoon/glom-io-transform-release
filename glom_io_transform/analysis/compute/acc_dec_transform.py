"""The Fig 5 calculation: transform input into output one property at a time.

Ported from glom-io-transform (2023); the calculation is unchanged. Each
dataset is split into odour means and the trial-to-trial scatter around them.
The means are described by their Gram matrix, the elementwise product of their
cosines (ANGLES) and their pairwise length products (LENGTHS), which fixes them
up to an orthogonal transformation (ROTATION). Matching all three, and then the
scatter (COVARIANCE), takes one dataset exactly to the other. Decoding accuracy
is measured after every step.

The decoder is LDA on pseudo-trials, ported from tobias_proc.split /
classifier and the tobias_utils helpers they call, with the arguments the
analysis always passed fixed in place: experiments pooled into one
pseudo-population, one trial held out per (roi, odour), the rest bootstrapped
up to `n_train_trials` and shuffled, `n_splits` such splits. Every random draw
comes from numpy's global generator, in the same order as the original, so
given the same data and seed the accuracies are identical. The options the
analysis never used -- shuffled labels, confusion matrices, roi or odour
subsets -- are left out; none of them drew from numpy's generator.

Run and collected by compute.acc_dec; drawn by figures.acc_dec.
"""
from itertools import permutations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

TRANSFORMATIONS = ("ANGLES", "LENGTHS", "ROTATION", "COVARIANCE")


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------
def take_trial(Z):
    """(held_out, rest): one random trial per (roi, odour), and the others.

    Z is rois x odours x trials. The held-out trial is drawn independently for
    every cell, so a test pseudo-trial mixes physical trials across rois. That
    is deliberate: the rois come from different animals, so there is no shared
    trial to keep together.
    """
    n_rois, n_odours, n_trials = Z.shape
    which = np.random.randint(n_trials, size=(n_rois, n_odours, 1))
    held_out = np.zeros((n_rois, n_odours, 1))
    rest = np.zeros((n_rois, n_odours, n_trials - 1))
    for i, j in np.ndindex(n_rois, n_odours):
        held_out[i, j, 0] = Z[i, j, which[i, j, 0]]
        rest[i, j] = np.delete(Z[i, j], which[i, j, 0])     # in their original order
    return held_out, rest


def bootstrap_trials(Z, n_trials):
    """Z padded to n_trials per (roi, odour) by resampling, then shuffled.

    Each cell is padded and shuffled on its own, which is what makes these
    pseudo-trials: a padded trial index does not refer to the same draw in two
    different rois. All cells are padded before any is shuffled, matching the
    original's order of draws.
    """
    n_rois, n_odours, n_have = Z.shape
    n_pad = n_trials - n_have
    assert n_pad > 0, f"Asked for {n_trials} trials but already have {n_have}."
    out = np.zeros((n_rois, n_odours, n_trials))
    out[:, :, :n_have] = Z
    for i, j in np.ndindex(n_rois, n_odours):
        out[i, j, n_have:] = np.random.choice(Z[i, j], n_pad, replace=True)
    for i, j in np.ndindex(n_rois, n_odours):
        out[i, j] = np.random.permutation(out[i, j])
    return out


def split(Z, n_train_trials, n_splits):
    """Train/test sets for each split, as (samples x rois) matrices and labels.

    Rows are ordered odour-major, trial-minor, and labelled by odour index.
    """
    n_rois, n_odours, _ = Z.shape
    for _ in range(n_splits):
        test, rest = take_trial(Z)
        train = bootstrap_trials(rest, n_train_trials)
        X_train = train.reshape(n_rois, -1).T
        X_test = test.reshape(n_rois, -1).T
        y_train = np.repeat(np.arange(n_odours), n_train_trials)
        y_test = np.repeat(np.arange(n_odours), test.shape[2])
        yield X_train, X_test, y_train, y_test


def classify(Zs, clf, seed=None, n_train_trials=25, n_splits=5):
    """Held-out accuracy of `clf`, one value per split.

    Zs is one rois x odours x trials array per experiment; they are pooled into
    a single pseudo-population first. Seeding here, at every call, is what makes
    the steps of a transformation sequence comparable: each is split with the
    same held-out trials and the same bootstrap draws.
    """
    if seed is not None:
        np.random.seed(seed)
    Z = np.concatenate(Zs, axis=0)
    scores = []
    for X_train, X_test, y_train, y_test in split(Z, n_train_trials, n_splits):
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    return np.array(scores)


# ---------------------------------------------------------------------------
# The transformations
# ---------------------------------------------------------------------------
def subsample_rois(Y, n_req):
    """n_req rois drawn at random from all experiments' rois pooled together."""
    n_total = sum(Yi.shape[0] for Yi in Y)
    keep = np.random.permutation(np.arange(n_total) < n_req)
    out = []
    for Yi in Y:
        out.append(Yi[keep[:len(Yi)]])
        keep = keep[len(Yi):]
    return out


def subsample_trials(Y, n_req):
    """n_req trials per experiment, drawn at random."""
    return [Yi[:, :, np.random.permutation(np.arange(Yi.shape[-1]))[:n_req]] for Yi in Y]


def split_vector(v, n):
    """v cut into consecutive pieces of lengths n."""
    pieces = []
    for ni in n:
        pieces.append(v[:ni])
        v = v[ni:]
    return pieces


def move_centers(X, C):
    """The matrix nearest X whose Gram matrix is C: min ||X - Y|| s.t. Y'Y = C.

    With C = P G P', every such Y is Q sqrt(G) P' for some Q with orthonormal
    columns, and the nearest one takes Q to be the polar factor of X P sqrt(G).
    """
    gamma, P = np.linalg.eigh(C)
    root = np.sqrt(np.diag(gamma))
    U, _, Vt = np.linalg.svd(X @ P @ root, full_matrices=False)
    return U @ Vt @ root @ P.T


def rotate_X2Y(X, Y):
    """X rotated to best match Y: Q X for min ||Y - Q X|| s.t. Q'Q = I."""
    U, _, Vt = np.linalg.svd(X @ Y.T)
    return Vt.T @ U.T @ X


class TransformInputToOutput:
    @staticmethod
    def validate_order(order):
        if len(order) != 4: raise ValueError("Exactly four operations must be specified.")
        if not all(oi in TRANSFORMATIONS for oi in order): raise ValueError(f"Not all of {order} was a valid operation.")
        return True

    def __init__(self, Xorig, Yorig, flip = False):
        # Flip says whether to flip X and Y after subsampling.
        # We do things this way (rather than allowing Yorig to be passed in for Xorig and vv.)
        # is so that when we subsample, we always sample Xorig and Yorig the same way given the same seed.
        # Otherwise, when arguments are flipped, we sample them in the flipped sequence, meaning
        # the e.g. X that we're supposed to arrive at going in one direction isn't the same X as
        # the one we start with going the other, given the same seed.
        self.Xorig,   self.Yorig = Xorig, Yorig
        self.min_trials     = np.min([Xi.shape[-1] for Z in [Xorig, Yorig] for Xi in Z])
        self.min_total_rois = np.min([sum([Xi.shape[0] for Xi in Z]) for Z in [Xorig, Yorig]])
        self.ready          = False
        self.flip           = flip
        self.results        = {}

    def create_datasets(self):
        self.X,   self.Y   = [subsample_trials(subsample_rois(Z, self.min_total_rois), self.min_trials) for Z in [self.Xorig, self.Yorig]]

        if self.flip: self.X, self.Y = self.Y, self.X

        self.μX,  self.μY  = [[np.mean(Zi,axis=-1) for Zi in XY] for XY in [self.X,self.Y]]
        self.nX,  self.nY  = [[Zi.shape[0] for Zi in Z] for Z in [self.μX, self.μY]]
        self.dX,  self.dY  = [[Zi - μZi[:,:,None] for Zi, μZi in zip(Z, μZ)] for (Z, μZ) in zip([self.X,self.Y],[self.μX, self.μY])]

        self.μX_, self.μY_ = [np.vstack(dZ) for dZ in [self.μX, self.μY]]
        self.nX_, self.nY_ = [Z.shape[0] for Z in [self.μX_, self.μY_]]
        self.dX_, self.dY_ = [np.vstack(dZ) for dZ in [self.dX, self.dY]]

        self.Zx   = np.dot(self.μX_.T, self.μX_)
        self.aX_  = np.linalg.norm(self.μX_,axis=0)
        self.cosX = np.clip(self.Zx/np.outer(self.aX_, self.aX_),-1,1)

        self.Zy   = np.dot(self.μY_.T, self.μY_)
        self.aY_  = np.linalg.norm(self.μY_,axis=0)
        self.cosY = np.clip(self.Zy/np.outer(self.aY_, self.aY_),-1,1)

        self.ready = True

    # Transformations:
    # 1. Match angles to input
    # 2. Match lengths
    # 3. Match rotation
    # 4. Match Covariance

    # We can put match covariance at any point, since it doesn't affect the means
    # Match rotation has to always follow match angles and match lengths, otherwise after the adjustments we might need to apply it again.

    def match_covariance(self, X):
        # Adds the deltas of the output relative to the input means to the means of X
        nX = [len(Xi) for Xi in X]
        μX = [np.mean(Xi, axis=-1) for Xi in X]
        μX_= np.vstack(μX)
        Z_ = self.dY_ + μX_[:,:,np.newaxis]
        Z  = split_vector(Z_, nX)
        return Z

    def match_lengths_or_angles_(self, X, which, keep_other = True):
        μX, nX = zip(*[(np.mean(Xi, axis=-1), len(Xi)) for Xi in X])
        μX_    = np.vstack(μX)
        aX_    = np.linalg.norm(μX_,axis=0)
        Zx_    = np.dot(μX_.T, μX_)
        cosX_  = Zx_/np.outer(aX_, aX_)

        if which=="angles":
            # Keep old lengths if keep_other, or use the output ones.
            a_  = aX_ if keep_other else self.aY_
            Zy_ = self.cosY * np.outer(a_, a_)
        elif which == "lengths":
            Zy_  = (cosX_ if keep_other else self.cosY) * np.outer(self.aY_, self.aY_)
        else: raise ValueError(f"Don't know how to match {which=}.")

        μX1_   = move_centers(μX_,  Zy_)
        aX1_   = np.linalg.norm(μX1_,axis=0)
        Zx1_   = np.dot(μX1_.T, μX1_)
        cosX1_ = Zx1_/np.outer(aX1_, aX1_)

        if which=="lengths":
            assert np.allclose(aX1_, self.aY_) # Lengths should match
            if keep_other:
                assert np.allclose(cosX1_, cosX_) # Angles should be unchanged
            else:
                assert np.allclose(cosX1_, self.cosY)  # Angles should match output
        elif which=="angles":
            assert np.allclose(cosX1_, self.cosY) # Angles should match
            if keep_other:
                assert np.allclose(aX1_, aX_) # Lengths should be unchanged
            else:
                assert np.allclose(aX1_, self.aY_) # Lengths should match output

        μX1 = split_vector(μX1_, nX)
        # Add the means to the deltas to get the new dataset
        X1   = [Xi - μXi[:,:,None] + μX1i[:,:,None] for Xi, μXi, μX1i in zip(X, μX, μX1)]
        return X1

    def match_lengths(self, X, keep_angles = True): return self.match_lengths_or_angles_(X, "lengths", keep_other = keep_angles)
    def match_angles(self, X, keep_lengths = True): return self.match_lengths_or_angles_(X, "angles",  keep_other = keep_lengths)

    def match_rotation(self, X):
        μX, nX = zip(*[(np.mean(Xi, axis=-1), len(Xi)) for Xi in X])
        μX_    = np.vstack(μX)
        μX1_   = rotate_X2Y(μX_, self.μY_)
        μX1    = split_vector(μX1_, nX)
        X1     = [Xi - μXi[:,:,None] + μX1i[:,:,None] for Xi, μXi, μX1i in zip(X, μX, μX1)]
        return X1

    def run_all(self, n_trials = 2, run_only = None):
        # n_trials is the number of random subsamplings (seeds), not imaging trials.
        errors = {}
        for order in permutations(TRANSFORMATIONS):
            if run_only and order not in run_only: continue
            print(order)
            for i in range(n_trials):
                np.random.seed(i)
                print(f"Trial {i}")
                self.create_datasets()
                try:
                    self.run1(order, seed = i)
                except Exception as e:
                    errors[order] = e
                    print(e)
                    break
            if order not in errors:
                self.results[order]["acc"]   = np.array(self.results[order]["acc"])
                self.results[order]["Xseq_"] = np.array(self.results[order]["Xseq_"])
        self.errors = errors
        self.accs   = {k:self.results[k]["acc"]   for k in self.results}
        self.Xseq_  = {k:self.results[k]["Xseq_"] for k in self.results}

    def run1(self, order, seed = None):
        if not self.ready: raise ValueError("Not ready to run. Create datasets first.")

        TransformInputToOutput.validate_order(order)
        # Rotations should come after angles are matched, otherwise we'd need another rotation to line things up.
        if order.index("ANGLES")>order.index("ROTATION"): raise ValueError("Ops.ROTATION must come after Ops.ANGLES.")

        X = self.X
        clf = LDA(store_covariance=True)
        if order not in self.results:
            self.results[order] = {"acc":[], "Xseq_":[]}

        acc = []
        acc.append(classify(X, clf, seed = seed))
        print(f"0: {'Initial':>16} - {np.mean(acc[-1]):1.2f} +/- {np.std(acc[-1]):1.2f}")
        matched_angles = False
        matched_lengths = False
        Xseq_ = [np.vstack(X)]
        for i,o in enumerate(order):
            print(f"{i+1}: {o:>16} - ", end ="")
            if o == "ANGLES":
                X = self.match_angles(X, keep_lengths = not matched_lengths) #
                matched_angles = True
            elif o == "LENGTHS":
                X = self.match_lengths(X, keep_angles = not matched_angles) #
                matched_lengths = True
            elif o == "ROTATION":
                X = self.match_rotation(X)
            elif o == "COVARIANCE":
                X = self.match_covariance(X)
            else: raise ValueError(f"Don't know what to do for operation {o}.")
            Xseq_.append(np.vstack(X))
            acc.append(classify(X, clf, seed = seed))
            print(f"{np.mean(acc[-1]):1.2f} +/- {np.std(acc[-1]):1.2f}")
        assert np.allclose(np.vstack(X), np.vstack(self.Y)), "X should match output after final transformation, but does not."
        self.results[order]["acc"].append(np.array(acc))
        self.results[order]["Xseq_"].append(np.array(Xseq_))
