import os, sys, pickle
from tqdm import tqdm

import yaml
import common
import driver

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize
from scipy.linalg import svd

from matplotlib import pyplot as plt
from matplotlib import cm

import pandas as pd
import hashlib

import conn_models

from conn_models.common import get_Cstar, compute_corr, r2_fun, pearson_fun, spearman_fun, ratio_fun 


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

def model_color(name):
    return name2color_cmap(name, cm.tab20, b=71) if name.lower()!="free" else "turquoise"

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
        print(f"Path {data_dir} not found, skipping.")
    data_file = os.path.join(data_dir, "collected.p")
    with open(data_file, 'rb') as f:
        records = pickle.load(f)

    results = []
    print(f"Loading {len(records):>4d} records from {data_file}", end = "", flush=True)
    loaded_from_in_file = [] # In case out.XYZ.p is missing, we can try in.XYZ.p
    for i, record in enumerate(records):
        if i % 50 == 49: print(".", end="", flush=True)
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

        for name in ["trains", "test", "vld"]:
            res_list = getattr(record["split"], name, []) 
            corr1 = lambda fun, res: compute_corr(fun, res.Cstar, res.Cest, res.Cin, is_cross = res.is_cross, include_diag = stats_include_diag)
            corrs = {k:[corr1(f, res) for res in res_list] for k,f in zip(["r2", "pearson", "spearman","ratio"], [r2_fun, pearson_fun, spearman_fun, ratio_fun])}
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
                          **{k:corrs[k][n] for k in corrs},
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
    "DiagPosBg": 'fit_diag_pos_bg',
    "Diag": 'fit_diag',
    "Free": 'ffree',
    "FreeSym": 'ffrees',
    "FreeAsym": 'ffreeas',
    "IdFreeSym": 'fidfrees',
    "IdPosBgCov": 'fidpbgcov',
    "IdPosBgSqrtCov": 'fidpbgsqcov',
    "DiagPosBgCov": 'fdpbgcov',
    "DiagPosBgSqCov": 'fdpbgsqcov',
    "DiagPosBgCov2": 'fdpbgcov2',
    "DiagPosBgSqCov2": 'fdpbgsqcov2',    
    "DiagPosBgRank1Sym": 'fdpbgr1u_sym3',
    "DiagPosBgRank1": 'fdpbgr1u2',
    "IdPosBgRank1Sym": 'fidpbgr1u_sym',
    "IdPosBgRank1": 'fidpbgr1u',
    "DiagPos": 'fit_diag_pos',
    "ZDiagRankSym": "fzdrsym",
    "Decorr": "fdecorr",
    "Inference": "finference",
    "FreeGen": "ffree_gen",
    "FreeGen__Diag": "ffree_gen__diag",
    "FreeGen__DiagRank1": "ffree_gen__diag_r1",
    
}

def load_models(base_dir, load_only = None, dont_load = [], load_config_from_input = False, stats_include_diag = True):

    models = {}                

    rest_loader_args = (load_config_from_input, stats_include_diag)
    print(f"{stats_include_diag=}")
    
    loadq = lambda name: (name not in dont_load) and ((load_only is None) or (name in load_only))
    def unpacker0(config):
        λ = config['config']['init_args']['λ']
        return ['λ'], [λ]

    if loadq("DiagPosBg"): models["DiagPosBg"] = load_model(base_dir + "/" + subdirs["DiagPosBg"], unpacker0, *rest_loader_args)
    if loadq("Diag"): models["Diag"] = load_model(base_dir + "/" + subdirs["Diag"], unpacker0, *rest_loader_args)
    
    def unpacker1(config):
        λ = config['config']['init_args']['λ'][0]
        return ['λ'], [λ]
    if loadq("Free"): models["Free"] = load_model(base_dir + "/" + subdirs["Free"], unpacker1, *rest_loader_args)
    if loadq("FreeSym"): models["FreeSym"] = load_model(base_dir + "/" + subdirs["FreeSym"], unpacker1, *rest_loader_args)
    if loadq("FreeAsym"): models["FreeAsym"] = load_model(base_dir + "/" + subdirs["FreeAsym"], unpacker1, *rest_loader_args)
    if loadq("IdFreeSym"): models["IdFreeSym"] = load_model(base_dir + "/" + subdirs["IdFreeSym"], unpacker1, *rest_loader_args)
    if loadq("FreeGen"): models["FreeGen"] = load_model(base_dir + "/" + subdirs["FreeGen"], unpacker1, *rest_loader_args)
    if loadq("FreeGen__Diag"): models["FreeGen__Diag"] = load_model(base_dir + "/" + subdirs["FreeGen__Diag"], unpacker1, *rest_loader_args)
    if loadq("FreeGen__DiagRank1"): models["FreeGen__DiagRank1"] = load_model(base_dir + "/" + subdirs["FreeGen__DiagRank1"], unpacker1, *rest_loader_args)
    
    def unpacker2(config):
        k = config['config']['init_args']['k']
        return ['k'], [k]
    if loadq("IdPosBgCov"): models["IdPosBgCov"] = load_model(base_dir + "/" + subdirs["IdPosBgCov"], unpacker2, *rest_loader_args)
    if loadq("IdPosBgSqrtCov"): models["IdPosBgSqrtCov"] = load_model(base_dir + "/" + subdirs["IdPosBgSqrtCov"], unpacker2, *rest_loader_args)
    
    def unpacker12(config):
        k = config['config']['init_args']['k']
        λ = config['config']['init_args']['λ'][0]
        return ['k', 'λ'], [k, λ]

    for names in ["DiagPosBgCov", "DiagPosBgSqCov", "DiagPosBgCov2", "DiagPosBgSqCov2", "Inference"]:
        if loadq(names): models[names] = load_model(base_dir + "/" + subdirs[names], unpacker12, *rest_loader_args)
    #if loadq("DiagPosBgCov"): models["DiagPosBgCov"] = load_model(base_dir + "/" + subdirs["DiagPosBgCov"], unpacker12, *rest_loader_args)
    #if loadq("DiagPosBgSqCov"): models["DiagPosBgSqCov"] = load_model(base_dir + "/" + subdirs["DiagPosBgSqCov"], unpacker12, *rest_loader_args)
    
    def unpacker3(config):
        λ0, λ1 = config['config']['init_args']['λ']
        return ['λ0', 'λ1'], [λ0, λ1]
    if loadq("DiagPosBgRank1Sym"): models["DiagPosBgRank1Sym"] = load_model(base_dir + "/" + subdirs["DiagPosBgRank1Sym"], unpacker3, *rest_loader_args)
    if loadq("DiagPosBgRank1"): models["DiagPosBgRank1"] = load_model(base_dir + "/" + subdirs["DiagPosBgRank1"], unpacker3, *rest_loader_args)
    
    def unpacker4(config):
        λ = config['config']['init_args']['λ'][0]
        return ['λ'], [λ]
    if loadq("IdPosBgRank1Sym"): models["IdPosBgRank1Sym"] = load_model(base_dir + "/" + subdirs["IdPosBgRank1Sym"], unpacker4, *rest_loader_args)
    if loadq("IdPosBgRank1"): models["IdPosBgRank1"] = load_model(base_dir + "/" + subdirs["IdPosBgRank1"], unpacker4, *rest_loader_args)
    
    def unpacker5(config):
        λ = config['config']['init_args']['λ']
        return ['λ'], [λ]
    if loadq("DiagPos"): models["DiagPos"] = load_model(base_dir + "/" + subdirs["DiagPos"], unpacker5, *rest_loader_args)

    def unpacker6(config):
        λ        = config['config']['init_args']['λ']
        rank     = config['config']['init_args']['rank']
        fit_diag = config['config']['init_args']['fit_diag']
        return ['λ', "rank", "fit_diag"], [λ[1], rank, fit_diag]
    if loadq("ZDiagRankSym"): models["ZDiagRankSym"] = load_model(base_dir + "/" + subdirs["ZDiagRankSym"], unpacker6, *rest_loader_args)

    def unpacker7(config):
        λ        = config['config']['init_args']['λ']
        k        = config['config']['init_args']['k']
        fit_id   = config['config']['init_args']['fit_identity']
        return ['λ', "k", "fit_id"], [λ[0], k, fit_id]
    if loadq("Decorr"): models["Decorr"] = load_model(base_dir + "/" + subdirs["Decorr"], unpacker7, *rest_loader_args)
    
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
