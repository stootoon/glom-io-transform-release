"""Fig S14: how closely cosine similarity tracks Pearson correlation between odours.

Fig 5 measures the similarity of two odours as the cosine of their mean
population responses, and the paper justifies that by S14: cosine closely
approximates Pearson correlation in our data. So S14 is computed on exactly the
representation Fig 5 transforms -- the trial means of the single-trial
responses in X0Y0_new.p, loaded by the same function, with no further
normalization -- and compares, for every pair of odours, the cosine of their
mean responses with the Pearson correlation between them across rois.

All rois and trials are used. Fig 5 additionally subsamples the output to the
input's size for each seed; that matches the two datasets to each other, and is
not part of what either representation is.

The 2023 version of this figure (two-correlations.ipynb, glom-io-transform) used
a different preprocessing -- trial-averaged then z-scored responses, each roi
standardized across odours, as for the main-text correlation matrices -- and so
did not test the quantity Fig 5 uses.
"""
import numpy as np
from scipy.stats import spearmanr

from .compute import Computation
from .acc_dec import load_data


def trial_means(Z):
    """odours x rois: each roi's response to each odour, averaged over trials."""
    return np.vstack([z.mean(axis=-1) for z in Z]).T


def cosines(X):
    """Cosine between every pair of rows."""
    L = np.linalg.norm(X, axis=1)
    return (X @ X.T) / np.outer(L, L)


def off_diag(C):
    """The entries above the diagonal of a square matrix."""
    return C[np.triu_indices(len(C), k=1)]


class Data(Computation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        print("COMPUTING two_correlations.Data.")
        X, Y = load_data()
        # odours x rois, in the stored X0Y0 order.
        self.responses = {"input": trial_means(X), "output": trial_means(Y)}
        self.pearson   = {k: np.corrcoef(Z) for k, Z in self.responses.items()}
        self.cosine    = {k: cosines(Z)     for k, Z in self.responses.items()}
        self.spearman  = {k: spearmanr(off_diag(self.pearson[k]), off_diag(self.cosine[k]))[0]
                          for k in self.responses}
        self.computed = True
        return self
