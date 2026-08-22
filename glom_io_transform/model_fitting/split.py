from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import TypeVar, Generic, List

class Role(str, Enum):
    TRAIN       = 'train'
    TEST        = 'test'
    VLD         = 'vld'
    UNAVAILABLE = 'unavailable'
    AVAILABLE   = 'available'
    UNASSIGNED  = 'unassigned'

role_type = pd.CategoricalDtype(
    categories=[r.value for r in Role],
    ordered=False
)

class IoType(str, Enum):
    INPUT  = 'input'
    OUTPUT = 'output'

io_type = pd.CategoricalDtype(
    categories=[t.value for t in IoType],
    ordered=False
)

def data_to_df(X, X_type):
    """
    X: List of (n_neurons x n_odours x n_trials) tensors containing
    the responses per experiment.

    Returns a single dataframe with columns:
    - 'experiment': experiment number
    - 'neuron': neuron number
    - 'odour': odour number
    - 'trial': trial number
    - 'response': response value
    - 'type': 'input' or 'output'
    - 'role': 'train','test', 'vld','unavailable', 'available','unassigned'
    """

    assert X_type in IoType, f"io_type must be one of {IoType}, got {io_type}"
    assert type(X) == list, f"X must be a list of tensors, got {type(X)}"
    # Check all tensors have the same n_odours
    n_odours = {Xi.shape[1] for Xi in X}
    assert len(n_odours) == 1, f"All tensors must have the same n_odours, got {n_odours}"

    dfs = []
    glob_id = 0
    for i, Xi in enumerate(X):
        # X0Y0 now holds DataArrays. The labels are not used here -- the odour
        # column below is positional, from np.indices -- so drop to a plain
        # array, which also gives us .ravel().
        Xi = np.asarray(Xi)
        n_neurons, n_odours, n_trials = Xi.shape
        neuron_idx, odour_idx, trial_idx = np.indices(Xi.shape)
        dfs.append(pd.DataFrame({
            'experiment': i,
            'id_in_exp': neuron_idx.ravel(),
            'glob_id': glob_id + neuron_idx.ravel(),
            'odour': odour_idx.ravel(), 
            'trial': trial_idx.ravel(),
            'response': Xi.ravel(),
            'type': X_type.value,
            'role': Role.UNASSIGNED.value
        }))
        glob_id += n_neurons
    df = pd.concat(dfs, ignore_index=True)

    df = df.astype({
        'experiment': int,
        'id_in_exp':  int,
        'glob_id':    int,
        'odour':      int,
        'trial':      int,
        'response':   float,
        'type':       io_type,
        'role':       role_type,
    })

    # Check that glob_id to (experiment, id_in_exp) mapping is unique
    assert df.groupby('glob_id')[['experiment', 'id_in_exp']].nunique().eq(1).all(axis=None), "glob_id to (experiment, id_in_exp) mapping is not unique"

    return df

T = TypeVar('T')

@dataclass 
class Split(Generic[T]):
    vld: T
    test: T
    trains: List[T]

    def iter_named(self):
        yield "vld", self.vld
        yield "test", self.test
        for j, train in enumerate(self.trains):
            yield f"train[{j}]", train
  
SplitSamples = Split[np.ndarray]

@dataclass
class SplitIndices(Split[pd.Index]):
    def materialize(self, df, df2mat):
        print("Materializing split...")
        result = []
        vld = df2mat(df.loc[self.vld])
        test = df2mat(df.loc[self.test])
        trains = [df2mat(df.loc[train_idx]) for train_idx in self.trains]
        return SplitSamples(vld=vld, test=test, trains=trains)
                            
def df2mat(df):
    """An n_neurons x n_odours matrix of responses.

    Rows are the sorted glob_ids and columns the sorted odours, which is what
    np.unique gives. searchsorted turns each row's labels into its position in
    those two, so the whole frame lands in one indexed assignment -- iterating
    the rows instead builds a pandas Series per row, and this is called for
    every split of every seed.
    """
    which_neurons = np.unique(df['glob_id'].values)
    which_odours  = np.unique(df['odour'].values)

    rows = np.searchsorted(which_neurons, df['glob_id'].values)
    cols = np.searchsorted(which_odours,  df['odour'].values)

    mat = np.full((len(which_neurons), len(which_odours)), np.nan)
    mat[rows, cols] = df['response'].values

    # All values in the dataframe must be present, otherwise raise an error
    if np.isnan(mat).any():
        raise ValueError("Not all values in the dataframe are present in the matrix")

    return mat


class BaseSampler(ABC):
    """ Abstract base class for trial sampling schemes."""

    @abstractmethod
    def generate(self, df, seed = 0):
        """Generate a split."""
        pass

    def validate(self, split, df):
        """Check universal invariants. Subclasses can override to add more."""
        self._check_disjointness(split)
        self._check_no_duplicates(split)

    def _all_indices(self, split):
        yield "vld", split.vld
        yield "test", split.test
        for j, train in enumerate(split.trains):
            yield f"train[{j}]", train
        
    def _check_disjointness(self, split):
        vld_set  = set(split.vld)
        test_set = set(split.test)
        assert vld_set.isdisjoint(test_set), f"Validation set overlaps with test set."
        for j, train in enumerate(split.trains):
            train_set = set(train)
            assert vld_set.isdisjoint(train_set), f"Validation set overlaps with train set {j}"
            assert test_set.isdisjoint(train_set), f"Test set overlaps with train set {j}"

    def _check_no_duplicates(self, split):
        for name, idx in self._all_indices(split):
            assert idx.is_unique, f"{name} contains duplicate indices"

    def _check_df_odours(self, df, name = "", must_have = None, must_not_have = None, can_only_have = None):
        odours = set(df['odour'].unique())
        if must_have is not None:
            assert set(must_have).issubset(odours), f"Dataframe {name} must contain odours {must_have}, but only has {odours}"
        if can_only_have is not None:
            assert set(can_only_have).issuperset(odours), f"Dataframe {name} can only contain odours {can_only_have}, but has {odours}"
        if must_not_have is not None:
            assert set(must_not_have).isdisjoint(odours), f"Dataframe {name} must not contain odours {must_not_have}, but has {odours}"
            

## SPLIT TYPES
# 1. GEN_TRIALS:
#    - Pick one trial per neuron per odour as validation, leave out.
#    - For each of n_test
#      - Pick one trial per neuron per odour as test, leave out
#      - For each of n_train
#        - Pick one trial per neuron per odour as train
SAMPLER_REGISTRY = {}

class TrialsSampler(BaseSampler):
    """ Train, test, validation splits are made by sampling trials. """

    def __init__(self, n_train = 1, train_odours = None, **kwargs):
        self.n_train = n_train
        self.train_odours     = train_odours

    def generate(self, df, seed = 0):
        print(f"Generating splits with TrialsSampler, n_train={self.n_train}, train_odours={self.train_odours}")
        
        df = df.copy()
        if self.train_odours is not None:
            df = df[df['odour'].isin(self.train_odours)]

        vld_rng, tst_rng, trn_rng = np.random.default_rng(seed).spawn(3)

        # Pick one trial per neuron per odour as validation, leave out.
        vld_idx = df.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=vld_rng).index

        post_vld = df.drop(vld_idx)
        # Pick one trial per neuron per odour as test, leave out
        test_idx = post_vld.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=tst_rng).index

        post_test = post_vld.drop(test_idx)
        # Pick one trial per neuron per odour as train
        trn_rngs = trn_rng.spawn(self.n_train)
        trns = [post_test.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=rng).index for rng in trn_rngs]

        return SplitIndices(vld=vld_idx, test=test_idx, trains=trns)

    def validate(self, split, df):
        super().validate(split, df)
        for name, idx in self._all_indices(split):
            self._check_df_odours(df.loc[idx], name=name, can_only_have=self.train_odours)

SAMPLER_REGISTRY['trials'] = TrialsSampler

class OdoursSampler(BaseSampler):
    """
    Train, test, validation splits are made by sampling odours.
    We specify the set of odours to use, and which to use in the trainin set.
    """

    def __init__(self, n_train = 1, train_odours=None, test_odours=None, vld_odours=None, **kwargs):
        assert train_odours is not None and test_odours is not None and vld_odours is not None, "train_odours, test_odours, vld_odours must be specified"

        self.train_odours = train_odours
        self.test_odours = test_odours
        self.vld_odours = vld_odours
        self.n_train = n_train
        # test and vld odours must be disjoint
        assert set(test_odours).isdisjoint(set(vld_odours)), f"test_odour {test_odours} must be disjoint from vld_odour {vld_odours}"
        # Assert that train_odours, test_odours, vld_odours are disjoint
        assert set(train_odours).isdisjoint(set(test_odours)), f"train_odours {train_odours} must be disjoint from test_odour {test_odours}"
        assert set(train_odours).isdisjoint(set(vld_odours)), f"train_odours {train_odours} must be disjoint from vld_odour {vld_odours}"
        
        
    def generate(self, df, seed = 0):
        print(f"Generating splits with OdoursSampler, n_train={self.n_train}\ntrain_odours={self.train_odours}, test_odours={self.test_odours}, vld_odours={self.vld_odours}")
        
        df = df.copy()

        vld_rng, tst_rng, trn_rng = np.random.default_rng(seed).spawn(3)

        df_vld = df[df['odour'].isin(self.vld_odours)]
        vld_idx = df_vld.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=vld_rng).index

        df_test = df[df['odour'].isin(self.test_odours)]
        test_idx = df_test.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=tst_rng).index
        
        df_train = df[df['odour'].isin(self.train_odours)]
        trn_rngs = trn_rng.spawn(self.n_train)
        trns = [df_train.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=rng).index for rng in trn_rngs]

        return SplitIndices(vld=vld_idx, test=test_idx, trains=trns)

    def validate(self, split, df):
        super().validate(split, df)

        self._check_df_odours(df.loc[split.vld],       name="validation", can_only_have=self.vld_odours, must_have=self.vld_odours)
        self._check_df_odours(df.loc[split.test],      name="test",       can_only_have=self.test_odours, must_have=self.test_odours)
        [self._check_df_odours(df.loc[trainsi], name="train",
                               must_have=self.train_odours, can_only_have=self.train_odours) for trainsi in split.trains]

SAMPLER_REGISTRY['odours'] = OdoursSampler

def make_sampler(config):
    config = dict(config) # copy
    sampler_type = config.pop('type', None)
    sampler_split= config.pop('split', None)
    if sampler_type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler type {sampler_type}. Must be one of {list(SAMPLER_REGISTRY.keys())}")
    return SAMPLER_REGISTRY[sampler_type](**config, **(sampler_split or {}))
        
            

