import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List

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
    # Check all tensors have the same n_odours and n_trials
    n_odours = {Xi.shape[1] for Xi in X}
    n_trials = {Xi.shape[2] for Xi in X}
    assert len(n_odours) == 1, f"All tensors must have the same n_odours, got {n_odours}"
    assert len(n_trials) == 1, f"All tensors must have the same n_trials, got {n_trials}"

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
    


@dataclass
class TrainTestSamples:
    test: pd.DataFrame
    trains: List[pd.DataFrame]

@dataclass
class SplitSamples:
    val: pd.DataFrame
    test_trains: List[TrainTestSamples]
    
@dataclass
class TrainTestIndices:
    test: pd.Index
    trains: List[pd.Index]
   
@dataclass
class SplitIndices:
    val: pd.Index
    test_trains: List[TrainTestIndices]

    def materialize(self, df):
        val = df.loc[self.val].drop(columns=['role'])
        test_trains = []
        for tt in self.test_trains:
            test = df.loc[tt.test].drop(columns=['role'])
            trains = [df.loc[train].drop(columns=['role']) for train in tt.trains]
            test_trains.append(TrainTestSamples(test=test, trains=trains))
        return SplitSamples(val=val, test_trains=test_trains)

def df2mat(df):
    # return an n_neurons x n_odours tensor of responses
    # All values in the dataframe must be present, otherwise raise an error
    n_neurons = df['glob_id'].nunique()
    n_odours = df['odour'].nunique()

    mat = np.full((n_neurons, n_odours), np.nan)
    for _, row in df.iterrows():
        neuron = row['glob_id']
        odour = row['odour']
        response = row['response']
        mat[neuron, odour] = response

    if np.isnan(mat).any():
        raise ValueError("Not all values in the dataframe are present in the matrix")

    return mat

    
## SPLIT TYPES
# 1. GEN_TRIALS:
#    - Pick one trial per neuron per odour as validation, leave out.
#    - For each of n_test
#      - Pick one trial per neuron per odour as test, leave out
#      - For each of n_train
#        - Pick one trial per neuron per odour as train
def gen_trials(df, n_test=1, n_train=1, which_odours=None, seed = 0):
    df = df.copy()
    
    if which_odours is not None:
        df = df[df['odour'].isin(which_odours)]

    val_rng, tst_rng, trn_rng = np.random.default_rng(seed).spawn(3)
    
    # Pick one trial per neuron per odour as validation, leave out.
    val_idx = df.groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=val_rng).index
    df.loc[val_idx, 'role'] = Role.VAL
    
    tts = []
    for i in range(n_test):
        dfi = df.copy()

        # Pick one trial per neuron per odour as test, leave out
        test_idx = dfi[dfi['role'] == Role.UNASSIGNED].groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=tst_rng).index
        dfi.loc[test_idx, 'role'] = Role.TEST

        # Pick one trial per neuron per odour as train
        trns = [dfi[dfi['role'] == Role.UNASSIGNED].groupby(['glob_id', 'odour'])['trial'].sample(n=1, random_state=trn_rng).index for _ in range(n_train)]    
        tts.append(TrainTestIndices(test=test_idx, trains=trns))

    return SplitIndices(val=val_idx, test_trains=tts)
            
