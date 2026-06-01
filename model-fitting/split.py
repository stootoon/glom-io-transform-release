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
    for i, Xi in enumerate(X):
        n_neurons, n_odours, n_trials = Xi.shape
        neuron_idx, odour_idx, trial_idx = np.indices(Xi.shape)
        dfs.append(pd.DataFrame({
            'experiment': i,
            'neuron': neuron_idx.ravel(), 
            'odour': odour_idx.ravel(), 
            'trial': trial_idx.ravel(),
            'response': Xi.ravel(),
            'type': X_type.value,
            'role': Role.UNASSIGNED.value
        }))
    df = pd.concat(dfs, ignore_index=True)

    return df.astype({
        'experiment': int,
        'neuron':     int,
        'odour':      int,
        'trial':      int,
        'response':   float,
        'type':       io_type,
        'role':       role_type,
    })



@dataclass
class TrainTestSamples:
    test: pd.DataFrame
    trains: List[pd.DataFrame]

@dataclass
class SplitSamples:
    val: pd.DataFrame
    test_trains: List[TrainTestSamples]

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
    val_idx = df.groupby(['experiment', 'neuron', 'odour'])['trial'].sample(n=1, random_state=val_rng).index
    df.loc[val_idx, 'role'] = Role.VAL
    df_val = df.loc[val_idx]
    
    tts = []
    for i in range(n_test):
        dfi = df.copy()

        # Pick one trial per neuron per odour as test, leave out
        test_idx = dfi[dfi['role'] == Role.UNASSIGNED].groupby(['experiment', 'neuron', 'odour'])['trial'].sample(n=1, random_state=tst_rng).index
        dfi.loc[test_idx, 'role'] = Role.TEST
        df_tst = dfi.loc[test_idx]
        
        df_trns = []
        for j in range(n_train):
            dfij = dfi.copy()
            # Pick one trial per neuron per odour as train
            train_idx = dfij[dfij['role'] == Role.UNASSIGNED].groupby(['experiment', 'neuron', 'odour'])['trial'].sample(n=1, random_state=trn_rng).index
            dfij.loc[train_idx, 'role'] = Role.TRAIN
            df_trns.append(dfij.loc[train_idx])
        tts.append(TrainTestSamples(test=df_tst, trains=df_trns))

    return SplitSamples(val=df_val, test_trains=tts)
            
