import os, sys, pickle
from tqdm import tqdm

import yaml

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.linalg import svd

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

import pandas as pd
import hashlib


import glom_io_transform.model_fitting.common as common
import glom_io_transform.model_fitting.driver as driver
import glom_io_transform.model_fitting.conn_models as conn_models

from glom_io_transform.model_fitting.conn_models.common import get_Cstar, compute_corr, r2_fun, pearson_fun, spearman_fun, ratio_fun 


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_leaf_order(C, **kwargs):
    Z = linkage(C, **kwargs)
    return leaves_list(Z)

def name2color(name):
    # Hash the name to get a color
    h = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16)
    r = (h >> 16) & 0xff
    g = (h >>  8) & 0xff
    b = (h >>  0) & 0xff
    print(h)
    return f'#{r:02x}{g:02x}{b:02x}'

def name2color_cmap(name, cmap, b = 10000):
    # Hash the name to get a color
    h = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16)
    # Use the hash to get a color from the cmap
    i = (h % b) / b
    #print(name, i)
    return cmap((h % b) / b)

# Colours pinned by hand, where hashing into tab20 gives something unusable.
# Free is the turquoise the paper's figures already use; FreePSD would otherwise
# hash to a light grey, and grey is reserved for the Input and Output reference
# violins. Keyed lowercase, so "Free" and "free" agree.
MODEL_COLORS = {"free": "turquoise", "freepsd": "#49aaff"}


def model_color(name):
    pinned = MODEL_COLORS.get(name.lower())
    return pinned if pinned is not None else name2color_cmap(name, cm.tab20, b=71)


# A model can be fitted under either loss, and both variants may be drawn in the
# same panel. Rather than give them two unrelated colours -- which the hash above
# would do, and which would lose the "turquoise means Free" association -- the
# hue stays the model's, and the loss chooses between a light and a dark version
# of it.
LOSS_SUFFIXES = ("resp", "cov")     # mirrors conn_models.diag.Model's `loss`

# tab20 is ten hues, each as a (dark, light) pair at consecutive indices, so a
# colour taken from it already has a partner designed to sit beside it.
TAB20  = [mcolors.to_hex(cm.tab20(i / 20)) for i in range(20)]

# A pinned colour is not in tab20, so it has no paired entry to borrow. Give it
# one, keyed by the pinned (dark) member, so that the model shows the pinned
# colour itself under the response loss -- which is what these models are for --
# and its lighter partner under the covariance loss. The light member follows
# tab20's own convention: same hue, roughly half the saturation, landing near
# luminance 0.78 like every light entry in the map.
PINNED_PAIRS = {"#49aaff": ("#9bd0ff", "#49aaff")}
V_DARK = 0.60      # for hues that are not from tab20 and have no partner
S_MIN  = 0.55      # keeps the dark variant coloured rather than muddy


def loss_pair(color):
    """(light, dark) versions of `color`, for the cov and resp fits.

    A tab20 colour uses its own paired entry. Everything else keeps its hue and
    drops in brightness, which is only a fallback: doing that to tab20's light
    orange gives brown rather than a darker orange.
    """
    hx = mcolors.to_hex(color)
    if hx in PINNED_PAIRS:
        light, dark = PINNED_PAIRS[hx]
        return mcolors.to_rgb(light), mcolors.to_rgb(dark)
    if hx in TAB20:
        i = TAB20.index(hx)
        dark, light = (TAB20[i], TAB20[i + 1]) if i % 2 == 0 else (TAB20[i - 1], TAB20[i])
        return mcolors.to_rgb(light), mcolors.to_rgb(dark)
    hue, sat, _ = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    return mcolors.to_rgb(color), mcolors.hsv_to_rgb((hue, max(sat, S_MIN), V_DARK))


def variant_color(name):
    """Colour for a model name that may carry a loss suffix, as in "Free_cov".

    A name with no suffix gets exactly what model_color gives it, and so does
    the cov variant, so the colour the rest of the paper uses is unchanged. The
    resp variant is its darker partner.
    """
    model, _, loss = name.partition("_")
    if not loss:
        return mcolors.to_rgb(model_color(name))
    assert loss in LOSS_SUFFIXES, (
        f"{name!r}: {loss!r} is not one of the losses {LOSS_SUFFIXES}. "
        f"Names are '<Model>_<loss>', for example 'Free_resp'.")
    light, dark = loss_pair(model_color(model))
    return light if loss == "cov" else dark


def variant_label(name):
    """"Free_cov" reads as "Free (cov)" on an axis; a plain name is unchanged."""
    model, _, loss = name.partition("_")
    return f"{model} ({loss})" if loss else name

from matplotlib.colors import LinearSegmentedColormap
def create_blue_shifted_cmap(base_cmap_name='pink', green = False):
    # Get the base colormap
    base_cmap = cm.get_cmap(base_cmap_name)

    # Create an array of colors from the base colormap
    colors = base_cmap(np.arange(base_cmap.N))

    # Shift reds to blues
    # Assuming the red component is in the first channel and blue in the third
    # Swap these channels
    if green:
        colors[:, [0, 1]] = colors[:, [1, 0]]
    else:
        colors[:, [0, 2]] = colors[:, [2, 0]]

    # Create a new colormap from the modified colors
    blue_shifted_cmap = LinearSegmentedColormap.from_list('blue_shifted', colors, base_cmap.N)
    return blue_shifted_cmap

# Function that returns the off-diagonal elements of a matrix.
def off_diag(M):
    M = M.copy()
    M[np.eye(M.shape[0], dtype=bool)] = np.nan
    return M[~np.isnan(M)].flatten()

def upper_tri_vec(C):
    return C[np.triu_indices_from(C, k=0)]

def sym_from_upper_tri_vec(v):
    n = int(np.sqrt(2*len(v)))
    C = np.zeros((n,n))
    C[np.triu_indices_from(C, k=0)] = v
    C = C + C.T - np.diag(np.diag(C))
    return C

def getCs(mdl):
    Cin   = mdl.X.T @ mdl.J @ mdl.X
    Cout  = mdl.get("C", mdl.results.x)
    Cstar = mdl.Cstar
    return Cin, Cout, Cstar

def rank_approx(C, k):
    w, v = np.linalg.eigh(C)
    return v[:,-k:] @ np.diag(w[-k:]) @ v[:,-k:].T

# Creat a function that makes the diagonal elemnts of a matrix nan.
def nan_diag(M):
    M = M.copy()
    M[np.eye(M.shape[0], dtype=bool)] = np.nan
    return M

# A function to return the nearest matrix to a given one, that also has a given covariance.
def nearest_matrix_with_cov(X, Σ):
    m, n = X.shape
    assert n >= m, "X must have at least as many columns as rows."
    xm = X.mean(axis=1)
    E, U = np.linalg.eigh(Σ)
    S = np.diag(np.sqrt(n * E))
    Xms = X - xm[:,np.newaxis]
    X1 = U.T @ Xms
    X2 = S @ X1
    U2, _, V2 = np.linalg.svd(X2, full_matrices=False)
    Yms = U @ S @ U2 @ V2
    Y = Yms + xm[:,np.newaxis]
    return Y

def complete_basis(X):
    # Complete the basis X to a basis for R^m
    # by adding orthogonal vectors to X
    m, n = X.shape
    assert m >= n, f"X must have at least as many columns ({n=}) as rows ({m=})."    
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    Y = np.random.randn(m, m-n)
    Y = Y - U @ (U.T @ Y)
    V, _, _ = np.linalg.svd(Y, full_matrices=False)
    return V

def decompose_connectivity(W, U):
    V   = complete_basis(U)
    Wu  = U @ U.T @ W @ U @ U.T
    Wv  = V @ V.T @ W @ V @ V.T
    Wuv = U @ U.T @ W @ V @ V.T
    Wvu = V @ V.T @ W @ U @ U.T
    return Wu, Wv, Wuv, Wvu, V

def diag_dominance(C):
    return np.sum(np.diag(C)**2)/np.sum(C**2)

diag_dom = lambda M: sum(diag(M)**2)/sum(M**2)

# A function that returns the square root of a hermitian matrix
def sqrtm(A):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.sqrt(np.maximum(w, 0))) @ np.linalg.inv(v)

def powm(A, p=1, tol=1e-6):
    assert np.allclose(A, A.T), "A must be symmetric"
    w, v = np.linalg.eigh(A)
    ind = np.where(w > tol)[0]
    return v[:,ind] @ np.diag(w[ind]**p) @ v[:,ind].T


def load_model(data_dir, unpack_params, load_config_from_input = False, stats_include_diag = True):
    if not os.path.exists(data_dir):
        # A split tree only holds the models that were fitted into it, so a
        # missing directory means "not fitted here", not an error.
        print(f"Path {data_dir} not found, skipping.")
        return None
    data_file = os.path.join(data_dir, "collected.p")
    with open(data_file, 'rb') as f:
        records = pickle.load(f)

    results = []
    print(f"Loading {len(records):>4d} records from {data_file}...")
    loaded_from_in_file = [] # In case out.XYZ.p is missing, we can try in.XYZ.p
    for i, record in enumerate(tqdm(records)):
        # if i % 50 == 49: print(".", end="", flush=True)
        seed = record['seed']
        assert 'file' in record, f"Record {i} was missing a 'file' field."
        filename= record['file']
        if not os.path.exists(os.path.join(data_dir,filename)) or load_config_from_input:                
            if "out" in filename:
                filename = filename.replace("out", "in")
                assert os.path.exists(os.path.join(data_dir,filename)), f"File {filename} and {filename.replace('out','in')} not found."
                loaded_from_in_file.append(filename)
        with open(os.path.join(data_dir,filename), 'rb') as f:
            config = pickle.load(f)
            if "in" in filename: config = {"config":config} # unpack_params expects a 'config' dict.
            params, vals = unpack_params(config)
        
        sampler = config["config"].get("sampler", {})
        split_info = sampler.get("split", {})
        sampler_typ = sampler.get("type")

        # Determine of the loss was cov or resp
        loss = config["config"].get("init_args", {}).get("loss", "cov")

        for name in ["trains", "test", "vld"]:
            res_list = getattr(record["split"], name, []) 
            corr1 = lambda fun, res: compute_corr(fun, res.Cstar, res.Cest, res.Cin, is_cross = res.is_cross, include_diag = stats_include_diag)
            corrs = {k:[corr1(f, res) for res in res_list] for k,f in zip(["r2", "pearson", "spearman","ratio"], [r2_fun, pearson_fun, spearman_fun, ratio_fun])}
            if loss == "resp": 
                corr2 = lambda fun, res: compute_corr(fun, res.Yeval, res.Yeval_est, res.Xeval, is_cross = True) 
                if len(res_list):   
                    assert all([res.Yeval_est is not None for res in res_list]), "Yeval_est is None for some res in res_list."
                    corrs2 = {(k+"_resp"):[corr2(f, res) for res in res_list] for k,f in zip(["r2", "pearson", "spearman","ratio"], [r2_fun, pearson_fun, spearman_fun, ratio_fun])}
            else:
                corrs2 = {}
    
            n_vals = len(corrs["r2"])
            for n in range(n_vals):
                results.append({"seed":seed,
                          "split":name,
                          "ref":f"train[{n}]",
                          "is_cross":res_list[n].is_cross,
                          "file": filename,
                          "sampler_type": sampler_typ,
                          "mode": split_info.get("mode"),
                          "outclass": split_info.get("outclass"),
                          "train_odours": frozenset(split_info.get("train_odours", [])),
                          "test_odours":  frozenset(split_info.get("test_odours", [])),
                          "vld_odours":   frozenset(split_info.get("vld_odours", [])),
                          "n_od_train":   len(split_info.get("train_odours", [])),   # realized size (esp. for max runs)                          
                          **{k:corrs[k][n] for k in corrs}, # Comparing Cstar, Cest, Cin
                          **{k:corrs2[k][n] for k in corrs2}, # Comparing Yeval, Yeval_est, Xeval 
                          **dict(zip(params, vals))})
           
    print(f"done ({len(loaded_from_in_file)}/{len(records)} configs from in.*.p files).", end = " ", flush=True)
    
    df = pd.DataFrame(results)

    # Load the config yaml file from the file with the same name as the data directory
    # To get the config dir, remove everything at 'fits' or below in the data_dir
    config_dir = os.path.dirname(data_dir)
    while os.path.basename(config_dir) != 'fits':
        config_dir = os.path.dirname(config_dir)
    config_dir = os.path.dirname(config_dir)
    # The file name is the same as the data_dir, but with a .yaml extension
    config_file = os.path.join(config_dir, os.path.basename(data_dir) + '.yaml')
    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found. Using empty config.")
        config = {}
    else:
        print("Loading config file:", config_file)
        sys.stdout.flush()
        config = load_config(config_file)
    # Overwrite normalization and standardization with the values from an input file.
    with open(os.path.join(data_dir, "in.0.p"), "rb") as f:
        in0 = pickle.load(f)
    config["normalization"] = in0["normalization"]
    config["standardization"] = in0["standardization"]
    return {"params":params, "df":df, "config":config, "config_orig":config.copy()}

def run_model(models, which_model, normalization = None, standardization = None, average = False, XY = None, shuffle_X = False, data_file= None, tol = None):
    model     = models[which_model]
    config    = model["config"]
    print("which_model: ", which_model)
    if XY is not None:
        X, Y = XY
        print("Using X and Y passed in.")
    else:
        normalization = config["normalization"] if normalization is None else normalization
        standardization = config["standardization"] if standardization is None else standardization
        print("Using normalization: ", normalization)
        print("Using standardization: ", standardization)
        if average:
            X, Y = driver.get_data(normalization=normalization,
                                   average=True, shuffle_X=shuffle_X)
        else:
            X, Xtst, Xvld, Y, Ytst, Yvld = driver.get_data(normalization=normalization,
                                                            standardization=standardization, shuffle_X=shuffle_X, data_file=data_file)
    context = {"np":np, "n_cells":X.shape[0]}    
    Model     = driver.known_models[model["config"]["model"]]
    init_args = common.eval_fields(model["config"]["init_args"], context=context) if "init_args" in model["config"] else {}
    min_args  = common.eval_fields(model["config"]["min_args"], context=context)  if "min_args" in model["config"] else {}
    print("params: ", model["params"])    
    best_params = model["best_params"]
    print(f"{best_params=}")
    if not hasattr(best_params, "__len__"):
        best_params = [best_params]
    for p,bp in zip(model["params"], best_params):
        if f"{p}__" in init_args:
            del init_args[f"{p}__"]        
        if which_model in ["Free", "FreeSym", "FreeAsym", "IdFreeSym", "DiagPosBgCov", "DiagPosBgSqCov", "IdPosBgRank1", "IdPosBgRank1Sym"] and p == "λ":
            init_args[p] = [bp] #if which_model != "FreeAsym" else [3e6]
        else:
            init_args[p] = bp
    if which_model.startswith("DiagPosBgRank1"):
        init_args["λ"] = [init_args["λ0"], init_args["λ1"]]
        del init_args["λ0"]
        del init_args["λ1"]
        if "λ__" in init_args:
            del init_args["λ__"]
          
        
    print(f"{init_args=}")
    #print(f"{min_args=}")
    mdl  = Model(X, Y, **init_args)
    if tol is not None:
        min_args["tol"] = tol
        print(f"Set minimization tolerance to {tol}.")
    mdl.minimize(**min_args)

    Cin   = mdl.X.T @ mdl.J @ mdl.X
    Cout  = mdl.get("C", mdl.results.x)
    Cstar = mdl.Cstar
    
    mdl.ratio = np.mean((Cstar - Cout)**2) / np.mean((Cstar - Cin)**2)
    print("Ratio trn:", mdl.ratio)
    # Compute the improvement, defined as the projection of Cout on (Cstar - Cin), minus Cin.
    # Get the off-diagonal elements of Cout - Cin
    target = upper_tri_vec(Cstar - Cin)
    mdl.name   = which_model
    mdl.proj   = ((upper_tri_vec(Cout - Cin)) @ target / (target @ target))
    mdl.improvement = sym_from_upper_tri_vec(mdl.proj * target)
    print("Improvement:", mdl.proj)

    for X_,Y_,w in zip([Xtst, Xvld],[Ytst, Yvld], ["tst", "vld"]):
        Ypred = mdl.get("Z", mdl.results.x) @ X_
        Cin = X_.T @ mdl.J @ X_
        Cout = Ypred.T @ mdl.J @ Ypred
        JY = np.eye(Y_.shape[0]) - np.ones(Y_.shape[0]) / Y_.shape[0]
        Cstar = Y_.T @ JY @ Y_
        ratio = np.mean((Cstar - Cout)**2) / np.mean((Cstar - Cin)**2)
        print(f"Ratio {w}:", ratio)
        r2 = np.mean((Cstar - Cout)**2)/ np.var(Cstar)
        print(f"R2 {w}:", r2)
    
    if XY is not None:
        return mdl, init_args, min_args, X, Y
    else:
        return mdl, init_args, min_args, X, Xtst, Xvld, Y, Ytst, Yvld
    
def proc_models(models, best_stat = 'ratio', all_stats=["r2", "pearson", "spearman","ratio"]):
    for name, mdl in models.items():
        # if name not in ["Diag", "Diag101"]: continue
        params = mdl["params"]
        df     = mdl["df"]
        # For each setting of the parameters, compute the mean of the ratio_test values
        # Store the parameter values that gave the lowest best_stat in best_params
        # Store the lowest for each stat in all_stats in best_stats
        stat_mean = df.groupby(params)[f'{best_stat}_test'].mean()
        best_params = stat_mean.idxmin()
        # For each of train, test, and vld, compute the mean and std of the ratio values at the best_params
        best_means = {}
        best_stds  = {}
        avail_stats = []
        for set_name in ['train', 'test', 'vld']:
            if len(params) == 1:
                df_sub = df[df[params[0]] == best_params]
            elif len(params) == 2:
                df_sub = df[(df[params[0]] == best_params[0]) & (df[params[1]] == best_params[1])]

            if len(avail_stats) == 0:
                for s in all_stats:
                    if f'{s}_{set_name}' in df_sub.columns:
                        avail_stats.append(s)

                assert len(avail_stats) > 0, f"No stats found for set {set_name} in model {name}."

                print(f"Available stats for set {set_name} in model {name}: {avail_stats}")
                
            best_means[set_name] = {stat:df_sub[f'{stat}_{set_name}'].mean() for stat in avail_stats}
            best_stds[set_name]  = {stat:df_sub[f'{stat}_{set_name}'].std()  for stat in avail_stats}
    
        models[name]["best_params"] = best_params
        models[name]["best_means"]  = best_means
        models[name]["best_stds"]   = best_stds

    return models
    
subdirs = {
    "Diag": 'fit_diag',
    "DiagOnlyInh": 'fit_diag__inh_only',
    "Free": 'ffree',
    "FreeLat": 'ffree_lat',
    "FreeSym": 'ffree_sym',
    "FreePSD": 'ffree_psd',
    "FreeRot": 'ffree_rot',
    "FreeOrth": 'ffree_orth',
}


def _unpack_scalar_la(config):
    """The Diag family stores its regularization strength as a scalar."""
    return ['λ'], [config['config']['init_args']['λ']]


def _unpack_list_la(config):
    """The Free family stores it as a one-element list."""
    return ['λ'], [config['config']['init_args']['λ'][0]]


def _unpack_la_reflect(config):
    """FreeOrth also sweeps which component of O(m) it fitted in."""
    init = config['config']['init_args']
    return ['λ', 'reflect'], [init['λ'][0], bool(init.get('reflect', False))]


# How to read the parameters out of each model's configs. Together with subdirs
# this is the whole definition of a known model, so adding one is two entries
# here rather than another line in load_models.
unpackers = {
    "Diag": _unpack_scalar_la,
    "DiagOnlyInh": _unpack_scalar_la,
    "Free": _unpack_list_la,
    "FreeLat": _unpack_list_la,
    "FreeSym": _unpack_list_la,
    "FreePSD": _unpack_list_la,
    "FreeRot": _unpack_list_la,
    "FreeOrth": _unpack_la_reflect,
}
assert set(unpackers) == set(subdirs), \
    f"subdirs and unpackers must cover the same models; {set(unpackers) ^ set(subdirs)} is in only one."

def load_models(base_dir, load_only = None, dont_load = [], load_config_from_input = False, stats_include_diag = True):

    models = {}                

    rest_loader_args = (load_config_from_input, stats_include_diag)
    print(f"{stats_include_diag=}")
    
    loadq = lambda name: (name not in dont_load) and ((load_only is None) or (name in load_only))

    for name, subdir in subdirs.items():
        if not loadq(name):
            continue
        loaded = load_model(base_dir + "/" + subdir, unpackers[name], *rest_loader_args)
        # None means the model was not fitted into this tree; leave it out
        # rather than storing an entry that later reads would trip over.
        if loaded is not None:
            models[name] = loaded

    print(f"Loaded models: {sorted(models)}.")
    return models


def plot_mdl(mdl, plot_which = [], ax = [], do_colorbar = False, cmap = cm.Spectral_r, vmin = None, leaf_order = None, vmax = None):    
    if len(ax) == 0:
        fig, ax = plt.subplots(1, len(plot_which), figsize = (4 * len(plot_which), 4))
    else:
        assert len(ax) == len(plot_which), f"{len(ax)=} != {len(plot_which)=}"
        fig = None

    Cin   = mdl.X.T @ mdl.J @ mdl.X
    Cout  = mdl.get("C", mdl.results.x)
    Cstar = mdl.Cstar

    C = {"Cin": Cin, "Cout": Cout, "Cstar": Cstar, "diff": Cstar - Cout, "proj": mdl.improvement}

    if leaf_order is not None:
        # Compute leaf order using hierarchical clustering
        link = linkage(C[leaf_order], method = 'average')
        leaf_order = leaves_list(link)
    else:
        leaf_order = np.arange(Cin.shape[0])
        
    for i, p in enumerate(plot_which):
        vmin1, vmax1 = np.percentile(C[p], [5, 95])
        if vmin is None: vmin = vmin1
        if vmax is None: vmax = vmax1
        im = ax[i].imshow(C[p][leaf_order][:, leaf_order], vmin = vmin, vmax = vmax, cmap = cmap)
        ax[i].set_title(p)

def compute_best_params(models, best_stat = 'ratio', all_stats = ["r2", "pearson", "spearman", "ratio"]):
    for name, mdl in models.items():
        # if name not in ["Diag", "Diag101"]: continue
        params = mdl["params"]
        df     = mdl["df"]
        # For each setting of the parameters, compute the mean of the ratio_test values
        # Store the parameter values that gave the lowest ratio in best_params
        # Store the lowest ratio in best_stats
        stat_mean = df.groupby(params)[f'{best_stat}_test'].mean()
        best_params = stat_mean.idxmin() if best_stat == 'ratio' else stat_mean.idxmax()
        # For each of train, test, and vld, compute the mean and std of the ratio values at the best_params
        best_means = {}
        best_stds  = {}
        avail_stats = []
        for set_name in ['train', 'test', 'vld']:
            if len(params) == 1:
                df_sub = df[df[params[0]] == best_params]
            elif len(params) == 2:
                df_sub = df[(df[params[0]] == best_params[0]) & (df[params[1]] == best_params[1])]

            if len(avail_stats) == 0:
                for s in all_stats:
                    if f'{s}_{set_name}' in df_sub.columns:
                        avail_stats.append(s)

                assert len(avail_stats) > 0, f"No stats found for set {set_name} in model {name}."

                print(f"Available stats for set {set_name} in model {name}: {avail_stats}")
                    
            best_means[set_name] = {stat:df_sub[f'{stat}_{set_name}'].mean() for stat in avail_stats}
            best_stds[set_name]  = {stat:df_sub[f'{stat}_{set_name}'].std() for stat in avail_stats}

        print(f"{name}")
        print(f"Best params: " + ",".join([f"{p}={v:.3g}" for p, v in zip(params, best_params if hasattr(best_params, '__len__') else [best_params])]))
        [print(f"{k:>12s}: {v[best_stat]:0.3f} +/- {s[best_stat]:0.3f}") for k, v, s in zip(['train', 'test', 'vld'], best_means.values(), best_stds.values())]
        models[name]["best_params"] = best_params
        models[name]["best_means"]  = best_means
        models[name]["best_stds"]   = best_stds    

from matplotlib.colors import TwoSlopeNorm    
def create_custom_colormap(vmin, vmax, colormap_name):
    """
    Create a colormap where -vmin to 0 maps to the lower half of the colormap
    and 0 to vmax maps to the upper half.
    """
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    cmap = plt.get_cmap(colormap_name)
    return cmap, norm

def zero_index(A, i=0,j=0):
    A[i,j] = 0
    return A

def compare(lhs, rhs):
    scatter(lhs.flatten(), rhs.flatten())
    lhs = lhs.flatten()
    rhs = rhs.flatten()
    R2 = 1 - 2 * sum((lhs - rhs)**2) / (sum(lhs**2) + sum(rhs**2))
    title(f"GOF = {R2:.2f}")


def get_Z_from_Zu(Zu, U):
    U1 = pfm.complete_basis(U)
    return Ux @ Zu @ Ux.T + U1 @ U1.T

from scipy.optimize import minimize
def fit_diag_rank1(X, **kwargs):
    n = X.shape[0]
    unpack = lambda p: (p[:n], p[n:])
    def fun(p):
        D, u = unpack(p)
        Y = np.diag(D) + np.outer(u, u)
        return np.sum((X - Y)**2)/2

    def jac(p):
        D, u = unpack(p)
        Y = np.diag(D) + np.outer(u, u)
        Δ = X - Y
        dD = -np.diag(Δ)
        du = -(Δ + Δ.T) @ u
        return np.hstack([dD, du])
    
    U, S, V = svd(X); V = V.T
    u0 = U[:,0] * np.sqrt(S[0])
    D0 = np.diag(X - np.outer(u0, u0))
    p0 = np.hstack([D0, u0])
    return minimize(fun, p0, jac=jac, **kwargs)

def fit_flat_diag_rank1(X, **kwargs):
    n = X.shape[0]
    unpack = lambda p: (p[0], p[1:])
    def fun(p):
        d, u = unpack(p)
        D = np.eye(n) * d
        Y = D + np.outer(u, u)
        return np.sum((X - Y)**2)/2

    def jac(p):
        d, u = unpack(p)
        D = np.eye(n) * d
        Y = D + np.outer(u, u)
        Δ = X - Y
        dd = -np.trace(Δ)/n
        du = -(Δ + Δ.T) @ u
        return np.hstack([dd, du])
    
    U, S, V = svd(X); V = V.T
    u0 = U[:,0] * np.sqrt(S[0])
    d0 = np.trace(X - np.outer(u0, u0))/n
    p0 = np.hstack([d0, u0])
    return minimize(fun, p0, jac=jac, **kwargs)

def fit_diag(X, **kwargs):
    n = X.shape[0]
    def fun(p):
        Y = np.diag(p)
        return np.sum((X - Y)**2)/2

    def jac(p):
        Y = np.diag(p)
        Δ = X - Y
        dD = -np.diag(Δ)
        return np.hstack([dD])
    
    D0 = np.diag(X)
    p0 = np.hstack([D0])
    return minimize(fun, p0, jac=jac, **kwargs)

def fit_flat_diag(X, **kwargs):
    n = X.shape[0]
    def fun(p):
        D = np.eye(n) * p[0]
        Y = D
        return np.sum((X - Y)**2)/2

    def jac(p):
        d = p[0]
        D = np.eye(n) * d
        Y = D
        Δ = X - Y
        dd = -np.trace(Δ)/n
        return np.hstack([dd])
    
    d0 = np.trace(X)/n
    p0 = np.hstack([d0])
    return minimize(fun, p0, jac=jac, **kwargs)   
