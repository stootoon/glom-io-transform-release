from numpy import *
import numpy as np

print0 = lambda *args, **kwargs: None
DEBUG  = print0

def cond(conds, default = None):
    for (pred, val, msg) in conds:
        if pred:
            print(msg)
            return val
    if default is not None:

        return default
    raise ValueError("No condition was met.")

def take_trial(Z, which_trial = None, trial_dim = 2):

    sh = Z.shape # E.g. (n, m, T, p)
    took = zeros(sh[:trial_dim]+(1,)+sh[trial_dim+1:]) # E.g. (n, m, 1, p)
    left = zeros(sh[:trial_dim]+(sh[trial_dim]-1,) + sh[trial_dim+1:]) # E.g. (n, m, T-1, p)
    # We pick a random trial for each element of took, and put the rest in left.
    which_trial = (which_trial * ones(took.shape)).astype(int) if which_trial is not None else random.randint(sh[trial_dim], size=took.shape)
    for ii in ndindex(sh[:trial_dim]):        
        for kk in ndindex(sh[trial_dim+1:]):
            ind1 = ii + (0,) + kk
            wt  = which_trial[ind1]
            took[ind1] = Z[ii + (wt,) + kk]
            for jj in range(sh[trial_dim]):
                if jj != wt:
                    left[ii + (jj-(jj>wt),) + kk]= Z[ii + (jj,) + kk]
            
    return took, left, which_trial

def split(Zs):
    # Split the data into train, test, and validation sets.

    # 2024-06-25: Returns a single trial as the test set, a single
    # trial as the validation set, and a single trial as the training
    # set. We often have three trials so this is what we can do
    # anyway. When we've had more, the first thing we tried was
    # average the training trials, but this gave funny resullts. So
    # we just take one trial from each set now.
    
    Ztest,  Ztrain0, which_test_trial  = zip(*[take_trial(Z) for Z in Zs])
    Zvld,   Ztrain1, which_vld_trial   = zip(*[take_trial(Z) for Z in Ztrain0])
    Ztrain, Ztrain2, which_train_trial = zip(*[take_trial(Z) for Z in Ztrain1])
    
    # Ztrain = [np.mean(Zi,axis=-1) for Zi in Ztrain1] # Trial average the remaining trials - don't do this, the results for train, vs test and vld are not comparable.
    
    # Combine ROIS
    Xtrain = np.concatenate(Ztrain, axis=0).squeeze()
    Xtest  = np.concatenate(Ztest,  axis=0).squeeze()
    Xvld   = np.concatenate(Zvld,   axis=0).squeeze()    
    return Xtrain, Xtest, Xvld, which_test_trial, which_vld_trial

def split_by_odour(Zs, frac_test = 0.2, frac_vld = 0.2, train_inds = None, test_inds = None, vld_inds = None, return_inds = False):
    """
    Split the data into train, test, and validation sets by odour.
    Zs: list of np.arrays, each of shape (n_neurons, n_odours, n_trials)
    frac_test: fraction of odours to use for test set
    frac_vld: fraction of odours to use for validation set
    Returns:
    Xtrain: np.array of shape (n_neurons_total, n_odours_train)
    Xtest: np.array of shape (n_neurons_total, n_odours_test)
    Xvld: np.array of shape (n_neurons_total, n_odours_vld)
    """
    
    # Trial average
    Zm = [np.mean(Z, axis=-1) for Z in Zs]
    # Concatenate along neurons
    Zall = np.concatenate(Zm, axis=0) # (n_neurons_total, n_odours)
    n_odours = Zall.shape[1]
    # If any inds are provided, first make sure all are provided.
    if train_inds is not None or test_inds is not None or vld_inds is not None:
        assert train_inds is not None and test_inds is not None and vld_inds is not None, "If any of train_inds, test_inds, vld_inds are provided, all must be provided."
        Xtrain = Zall[:, train_inds]
        Xtest  = Zall[:, test_inds]
        Xvld   = Zall[:, vld_inds]
        return Xtrain, Xtest, Xvld
    
    # Randomly split odours
    odour_inds = np.arange(n_odours)
    np.random.shuffle(odour_inds)
    n_test = int(frac_test * n_odours)
    n_vld  = int(frac_vld  * n_odours)
    test_inds = odour_inds[:n_test]
    vld_inds  = odour_inds[n_test:n_test+n_vld]
    train_inds= odour_inds[n_test+n_vld:]
    Xtest  = Zall[:, test_inds]
    Xvld   = Zall[:, vld_inds]
    Xtrain = Zall[:, train_inds]
    if return_inds:
        return Xtrain, Xtest, Xvld, train_inds, test_inds, vld_inds
    return Xtrain, Xtest, Xvld
 
def eval_fields(d, context = None):
    # Generate code that will accept a dictionary
    # and evalutes all fields that are not dictionaries or dictionaries.
    # If the field evaluates without error, then the result is kept,
    # otherwise the original string is kept.
    DEBUG(f"eval_fields: Evaluating dictionary {d}, {context=}")
    for k,v in d.items():
        # Check that the value is not a dictionary.
        if isinstance(v, dict):
            # If it is a dictionary, then recurse.
            DEBUG(f"Value for {k} is a dictionary, so recursing.")
            v = eval_fields(v, context=context)
        else:
            # If the value is not a string, keep it as is.
            if not isinstance(v, str):
                DEBUG(f"Key={k:>12s}: Value {v} is not a string, so keeping it as is.")
            else:
                # Otherwise, try to evalute the string.
                try:
                    v1 = eval(v, context)
                    DEBUG(f"Key={k:>12s}: Evaluating the value {v} succeeded, so keeping the result {v1}")
                    v = v1
                except Exception as E:
                    # print the exception message
                    DEBUG(f"Key={k:>12s}: Evaluating the value {v} raised exception {E}, so keeping the original string.")
                    # If it fails, keep the original string.
                    pass
            d[k] = v
    return d
