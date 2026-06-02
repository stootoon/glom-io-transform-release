from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List, NamedTuple

class Role(str, Enum):
    TRAIN       = 'train'
    TEST        = 'test'
    VAL         = 'val'
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
    - 'role': 'train','test', 'val','unavailable', 'available','unassigned'
    """

    assert X_type in IoType, f"io_type must be one of {IoType}, got {io_type}"
    assert type(X) == list, f"X must be a list of tensors, got {type(X)}"
    # Check all tensors have the same n_odours
    n_odours = {Xi.shape[1] for Xi in X}
    assert len(n_odours) == 1, f"All tensors must have the same n_odours, got {n_odours}"

    dfs = []
    glob_id = 0
    for i, Xi in enumerate(X):
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

class DataTriplet(NamedTuple):
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray

@dataclass
class TrainTestIndices:
    test: pd.Index
    trains: List[pd.Index]
   
@dataclass
class SplitIndices:
    val: pd.Index
    test_trains: List[TrainTestIndices]

    def materialize(self, df, df2mat):
        result = []
        val = df2mat(df.loc[self.val])
        for tt in self.test_trains:
            test = df2mat(df.loc[tt.test])
            for train_idx in tt.trains:
                train = df2mat(df.loc[train_idx])
                result.append(DataTriplet(train=train, test=test, val=val))
        return result

def df2mat(df):
    # return an n_neurons x n_odours tensor of responses
    # All values in the dataframe must be present, otherwise raise an error

    which_neurons = list(np.unique(df['glob_id']))
    which_odours  = list(np.unique(df['odour']))
    n_neurons     = len(which_neurons)
    n_odours      = len(which_odours)
    
    mat = np.full((n_neurons, n_odours), np.nan)
    for _, row in df.iterrows():
        neuron   = row['glob_id']
        odour    = row['odour']
        mat[which_neurons.index(neuron),
            which_odours.index(odour)] = row['response']

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
        yield "val", split.val
        for i, pair in enumerate(split.test_trains):
            yield f"test[{i}]", pair.test
            for j, train in enumerate(pair.trains):
                yield f"train[{i}][{j}]", train
        
    def _check_disjointness(self, split):
        val_set = set(split.val)
        for i, pair in enumerate(split.test_trains):
            test_set = set(pair.test)
            assert val_set.isdisjoint(test_set), f"Validation set overlaps with test set in pair {i}"
            for j, train in enumerate(pair.trains):
                train_set = set(train)
                assert val_set.isdisjoint(train_set), f"Validation set overlaps with train set in pair {i}, train {j}"
                assert test_set.isdisjoint(train_set), f"Test set overlaps with train set in pair {i}, train {j}"

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

    def __init__(self, n_test = 1, n_train = 1, which_odours = None):
        self.n_test       = n_test
        self.n_train      = n_train
        self.which_odours = which_odours

    def generate(self, df, seed = 0):
        df = df.copy()
        if self.which_odours is not None:
            df = df[df['odour'].isin(self.which_odours)]

        val_rng, tst_rng, trn_rng = np.random.default_rng(seed).spawn(3)

        # Pick one trial per neuron per odour as validation, leave out.
        val_idx = df.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=val_rng).index

        tts = []
        for i in range(self.n_test):
            post_val = df.drop(val_idx)

            # Pick one trial per neuron per odour as test, leave out
            test_idx = post_val.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=tst_rng).index

            post_test = post_val.drop(test_idx)
            # Pick one trial per neuron per odour as train
            trns = [post_test.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=trn_rng).index for _ in range(self.n_train)]    
            
            tts.append(TrainTestIndices(test=test_idx, trains=trns))

        return SplitIndices(val=val_idx, test_trains=tts)

    def validate(self, split, df):
        super().validate(split, df)
        for name, idx in self._all_indices(split):
            self._check_df_odours(df.loc[idx], name=name, can_only_have=self.which_odours)

SAMPLER_REGISTRY['trials'] = TrialsSampler


def make_sampler(config):
    sampler_type = config.pop('type')
    if sampler_type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler type {sampler_type}. Must be one of {list(SAMPLER_REGISTRY.keys())}")
    return SAMPLER_REGISTRY[sampler_type](**config)
        
            

