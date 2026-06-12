from numpy import *
import numpy as np
from scipy.stats import spearmanr

get_IJN = lambda n: (eye(n), eye(n) - ones((n,n))/n, n)

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


def off_diag(A):
    # Return the off-diagonal elements of a matrix.
    return A[~eye(A.shape[0],dtype=bool)]
    

def before_after(X,Y,Z):
    _, Jx, _ = get_IJN(X.shape[0])
    _, Jy, _ = get_IJN(Y.shape[0])    
    before = 0.5*mean((Y.T @ Jy @ Y - X.T @ Jx @ X)**2)
    after  = 0.5*mean((Y.T @ Jy @ Y - X.T @ Z.T @ Jx @ Z @ X)**2)
    return before, after, after/before

def get_Cstar(X, center, X2=None):
    if X2 is None:
        X2 = X
    m, n = X.shape
    assert X2.shape[0] == m, "X and X2 must have the same number of rows (cells)."
    I, J, _ = get_IJN(m)
    Cstar = X.T @ (J if center else I) @ X2
    return Cstar

def init_r(m, λ = None, scale=1e-3, r0 = 1):
    # We want e's contribution to the loss at λ = 1to be of order 'scale'.
    # The contribution is sum e^2 / 2 / m .
    # The expected value of this is m E(e^2) / 2 m =  var(e) / 2.
    # For a uniform distribution over -1/2 to 1/2, var(e) = 1/12.
    # So the expected value is 1 / 24.
    # So the factor we want is sqrt(scale * 24 ).
    e = (np.random.rand(m) - 0.5) * sqrt(scale * 24)
    return r0 + e

def init_rλ(m, λ, scale=1e-3, r0 = 1):
    e = (np.random.rand(m) - 0.5)
    if λ == 0:
        e *= sqrt(scale)
    elif λ >  0:
        # We want e's contribution to the loss to be of order 'scale'.
        # The contribution is λ sum e^2 / 2 / m .
        # The expected value of this is λ E(e^2) / 2 = λ var(e) / 2.
        # For a uniform distribution over -1/2 to 1/2, var(e) = 1/12.
        # So the expected value is λ / 24.
        # So the factor we want is sqrt(scale * 24 / λ).
        # But we also don't want e to get too big, so we take the min.
        e *= sqrt(scale * 24) * min(1, 1 / sqrt(λ))
    else:
        raise ValueError("λ must be non-negative")

    return r0 + e

def compute_corr(corr_fun, Cstar, Cest, Cin = None, is_cross = False, include_diag = True):
        """
        Compute the R2 value between the estimated and true connectivity matrices.
        """

        # Assert all C's have the same shape
        assert all(C.shape == Cstar.shape for C in [Cest] + ([Cin] if Cin is not None else [])), "All C's must have the same shape."
        assert is_cross or Cstar.shape[0] == Cstar.shape[1], "Cstar must be square if not is_cross."
        
        if is_cross:
            # If is_cross, we take all elements of Cstar and Cest, and Cin if it is not None.
            Cstar = Cstar.flatten()
            Cest  = Cest.flatten()
            Cin   = Cin.flatten() if Cin is not None else None
        else:
            ind_above_diag = np.triu_indices_from(Cstar, k=0 if include_diag else 1)

            Cstar = Cstar[ind_above_diag]
            Cest  = Cest[ind_above_diag]
            Cin   = Cin[ind_above_diag] if Cin is not None else None

        return corr_fun(Cstar, Cest, Cin)

compute_r2 = lambda *args, **kwargs: compute_corr(r2_fun, *args, **kwargs) 
    
r2_fun       = lambda x_true, x_pred, x_in : 1 - np.sum((x_true - x_pred)**2) / np.sum((x_true - np.mean(x_true))**2)
pearson_fun  = lambda x_true, x_pred, x_in : np.corrcoef(x_true, x_pred)[0,1]
spearman_fun = lambda x_true, x_pred, x_in : spearmanr(x_true, x_pred).correlation
ratio_fun    = lambda x_true, x_pred, x_in : np.mean((x_true - x_pred)**2)/np.mean((x_true - x_in)**2)


def compute_ratio(Cstar, Cest, Cin):
    return np.mean((Cstar - Cest)**2)/np.mean((Cstar - Cin)**2)

def complete_basis(X):
    # Complete the basis X to a basis for R^m
    # by adding orthogonal vectors to X
    m, n = X.shape
    assert m >= n, "X must have at least as many columns as rows"    
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    Y = np.random.randn(m, m-n)
    Y = Y - U @ (U.T @ Y)
    V, _, _ = np.linalg.svd(Y, full_matrices=False)
    return V


def decompose_connectivity(W, U, V=None):
    if V is None:
        V = complete_basis(U)

    UUt = U @ U.T
    VVt = V @ V.T

    Wuu = UUt @ W @ UUt
    Wvv = VVt @ W @ VVt
    Wuv = UUt @ W @ VVt
    Wvu = VVt @ W @ UUt

    return Wuu, Wuv, Wvu, Wvv

def get_ZtZ_opt(X, Cstar):
    Ux, Sx, Vx = np.linalg.svd(X, full_matrices=False)
    Xi = Ux @ np.diag(1/Sx) @ Vx
    ZtZ_opt = Xi @ Cstar @ Xi.T # ZtZ such that X'ZtZ X = Cstar
    return ZtZ_opt

def get_W_from_ZtZ_opt(ZtZ, S_min = 1e-8):
    # Get Z from ZtZ, and then W from Z.
    U,S,_ = np.linalg.svd(ZtZ, full_matrices=False)
    ind_nonzero = S > S_min
    U = U[:,ind_nonzero]    
    print(f"Keeping {U.shape[1]} eigenvalues, from {S[ind_nonzero].min()} to {S[ind_nonzero].max()}.")
    U_ = complete_basis(U)
    S = S[ind_nonzero]
    Z = U @ np.diag(np.sqrt(S)) @ U.T   + U_ @ U_.T
    iZ= U @ np.diag(1/np.sqrt(S)) @ U.T + U_ @ U_.T
    # Z = (I + W)^-1
    W = iZ - np.eye(iZ.shape[0])
    return W

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
