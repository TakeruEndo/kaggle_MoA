# TabNet
import os
os.system('pip install --no-index --find-links /kaggle/input/pytorchtabnet/pytorch_tabnet-2.0.0-py3-none-any.whl pytorch-tabnet')

import sys
sys.path.append('/kaggle/input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

### General ###
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Data Wrangling ###
import numpy as np
import pandas as pd
from scipy import stats
sys.path.append("/kaggle/input/rank-gauss")
from gauss_rank_scaler import GaussRankScaler

### Data Visualization ###
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
### Machine Learning ###
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

### Deep Learning ###
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.decomposition import TruncatedSVD

### Make prettier the prints ###
from colorama import Fore
c_ = Fore.CYAN
m_ = Fore.MAGENTA
r_ = Fore.RED
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN

import warnings
warnings.filterwarnings('ignore')

data_dir = '../input/lish-moa/'
os.listdir(data_dir)



def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)





if __name__ == '__main__':
    # Parameters
    data_path = "../input/lish-moa/"
    no_ctl = True
    scale = "rankgauss"
    variance_threshould = 0.7
    decompo = "PCA"
    ncompo_genes = 80
    ncompo_cells = 10
    encoding = "dummy"

    seed_everything(seed_value=42)
    seed = 42

    train = pd.read_csv(data_path + "train_features.csv")
    #train.drop(columns = ["sig_id"], inplace = True)
    targets = pd.read_csv(data_path + "train_targets_scored.csv")
    #train_targets_scored.drop(columns = ["sig_id"], inplace = True)
    #train_targets_nonscored = pd.read_csv(data_path + "train_targets_nonscored.csv")
    test = pd.read_csv(data_path + "test_features.csv")
    #test.drop(columns = ["sig_id"], inplace = True)
    submission = pd.read_csv(data_path + "sample_submission.csv")    

    if no_ctl:
        # cp_type == ctl_vehicle
        print(b_, "not_ctl")
        train = train[train["cp_type"] != "ctl_vehicle"]
        test = test[test["cp_type"] != "ctl_vehicle"]
        targets = targets.iloc[train.index]
        train.reset_index(drop = True, inplace = True)
        test.reset_index(drop = True, inplace = True)
        targets.reset_index(drop = True, inplace = True)

    def distributions(num, graphs, items, features, gorc):
        """
        Plot the distributions of gene expression or cell viability data
        """
        for i in range(0, num - 1, 7):
            if i >= 3:
                break
            idxs = list(np.array([0, 1, 2, 3, 4, 5, 6]) + i)
        
            fig, axs = plt.subplots(1, 7, sharey = True)
            for k, item in enumerate(idxs):
                if item >= items:
                    break
                graph = sns.distplot(train[features].values[:, item], ax = axs[k])
                graph.set_title(f"{gorc}-{item}")
                graphs.append(graph)        
    
    GENES = [col for col in train.columns if col.startswith("g-")]
    CELLS = [col for col in train.columns if col.startswith("c-")]

    data_all = pd.concat([train, test], ignore_index = True)
    cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]
    mask = (data_all[cols_numeric].var() >= variance_threshould).values
    tmp = data_all[cols_numeric].loc[:, mask]
    data_all = pd.concat([data_all[["sig_id", "cp_type", "cp_time", "cp_dose"]], tmp], axis = 1)
    cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]

    def scale_minmax(col):
        return (col - col.min()) / (col.max() - col.min())

    def scale_norm(col):
        return (col - col.mean()) / col.std()

    if scale == "boxcox":
        print(b_, "boxcox")
        data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis = 0)
        trans = []
        for feat in cols_numeric:
            trans_var, lambda_var = stats.boxcox(data_all[feat].dropna() + 1)
            trans.append(scale_minmax(trans_var))
        data_all[cols_numeric] = np.asarray(trans).T
        
    elif scale == "norm":
        print(b_, "norm")
        data_all[cols_numeric] = data_all[cols_numeric].apply(scale_norm, axis = 0)
        
    elif scale == "minmax":
        print(b_, "minmax")
        data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis = 0)
        
    elif scale == "rankgauss":
        ### Rank Gauss ###
        print(b_, "Rank Gauss")
        scaler = GaussRankScaler()
        data_all[cols_numeric] = scaler.fit_transform(data_all[cols_numeric])
        
    else:
        pass

    # Create object for transformation.
    transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")

    # fitting
    GENES = [col for col in data_all.columns if col.startswith("g-")]
    CELLS = [col for col in data_all.columns if col.startswith("c-")]
    transformer.fit(data_all.loc[:, GENES + CELLS])

    # transforming
    q_trans = transformer.transform(data_all.loc[:, GENES + CELLS])

    data_all[GENES + CELLS] = q_trans

    # PCA
    if decompo == "PCA":
        print(b_, "PCA")
        GENES = [col for col in data_all.columns if col.startswith("g-")]
        CELLS = [col for col in data_all.columns if col.startswith("c-")]
        ALL = [col for col in data_all.columns if col.startswith("c-") or col.startswith("g-")]
        
        pca_genes = PCA(n_components=ncompo_genes, random_state=seed).fit_transform(data_all[GENES])
        pca_cells = PCA(n_components=ncompo_cells, random_state=seed).fit_transform(data_all[CELLS])
    #     svd = TruncatedSVD(n_components=20, n_iter=10, random_state=42).fit_transform(data_all[ALL])              
        
        pca_genes = pd.DataFrame(pca_genes, columns=[f"pca_g-{i}" for i in range(ncompo_genes)])
        pca_cells = pd.DataFrame(pca_cells, columns=[f"pca_c-{i}" for i in range(ncompo_cells)])
    #     svd = pd.DataFrame(svd, columns = [f"svd-{i}" for i in range(20)])    
        data_all = pd.concat([data_all, pca_genes, pca_cells], axis = 1)
    else:
        pass

    from sklearn.cluster import KMeans
    def fe_cluster(data, n_clusters_g = 35, n_clusters_c = 5, SEED = 123):
        
        features_g = GENES
        features_c = CELLS
        
        def create_cluster(data, features, kind = 'g', n_clusters = n_clusters_g):
            data_ = data[features].copy()
            kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data_)
            data[f'clusters_{kind}'] = kmeans.labels_[:data_.shape[0]]

            data = pd.get_dummies(data, columns = [f'clusters_{kind}'])
            return data
        
        data = create_cluster(data, features_g, kind = 'g', n_clusters = n_clusters_g)
        data = create_cluster(data, features_c, kind = 'c', n_clusters = n_clusters_c)
        return data

    def fe_stats(df):
        
        features_g = GENES
        features_c = CELLS

        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
            
        return df

    data_all = fe_cluster(data_all)
    data_all = fe_stats(data_all)

    # Encoding
    if encoding == "lb":
        print(b_, "Label Encoding")
        for feat in ["cp_time", "cp_dose"]:
            data_all[feat] = LabelEncoder().fit_transform(data_all[feat])
    elif encoding == "dummy":
        print(b_, "One-Hot")
        data_all = pd.get_dummies(data_all, columns = ["cp_time", "cp_dose"])

    GENES = [col for col in data_all.columns if col.startswith("g-")]
    CELLS = [col for col in data_all.columns if col.startswith("c-")]

    for stats in tqdm.tqdm(["sum", "mean", "std", "kurt", "skew"]):
        data_all["g_" + stats] = getattr(data_all[GENES], stats)(axis = 1)
        data_all["c_" + stats] = getattr(data_all[CELLS], stats)(axis = 1)    
        data_all["gc_" + stats] = getattr(data_all[GENES + CELLS], stats)(axis = 1)    

    SEED = 42

    NB_SPLITS = 10 # 7
    FOLDS = 10

    # LOAD FILES
    scored = targets.copy()
    drug = pd.read_csv('/kaggle/input/lish-moa/train_drug.csv')
    targ = scored.columns[1:]
    scored = scored.merge(drug, on='sig_id', how='left') 

    # LOCATE DRUGS
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[vc<=18].index.sort_values()
    vc2 = vc.loc[vc>18].index.sort_values()

    # STRATIFY DRUGS 18X OR LESS
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, 
            random_state=SEED)
    tmp = scored.groupby('drug_id')[targ].mean().loc[vc1]
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targ])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)
    # STRATIFY DRUGS MORE THAN 18X
    skf = MultilabelStratifiedKFold(n_splits=FOLDS, shuffle=True, 
            random_state=SEED)
    tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targ])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)

    # ASSIGN FOLDS
    scored['fold'] = scored.drug_id.map(dct1)
    scored.loc[scored.fold.isna(),'fold'] =\
        scored.loc[scored.fold.isna(),'sig_id'].map(dct2)
    scored.fold = scored.fold.astype('int8')

    fold_index = scored.fold.astype('int8')

    del scored, drug, targ, vc1, vc2
    import gc
    gc.collect()        

    features_to_drop = ["sig_id", "cp_type"]
    data_all.drop(features_to_drop, axis = 1, inplace = True)
    try:
        targets.drop("sig_id", axis = 1, inplace = True)
    except:
        pass
    train_df = data_all[: train.shape[0]]
    train_df.reset_index(drop = True, inplace = True)
    # The following line it's a bad practice in my opinion, targets on train set
    #train_df = pd.concat([train_df, targets], axis = 1)
    test_df = data_all[train_df.shape[0]: ]
    test_df.reset_index(drop = True, inplace = True)

    X_test = test_df.values

    MAX_EPOCH = 200
    # n_d and n_a are different from the original work, 32 instead of 24
    # This is the first change in the code from the original
    tabnet_params = dict(
        n_d = 32,
        n_a = 32,
        n_steps = 1,
        gamma = 1.3,
        lambda_sparse = 0,
        optimizer_fn = optim.Adam,
        optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
        mask_type = "entmax",
        scheduler_params = dict(
            mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
        scheduler_fn = ReduceLROnPlateau,
        seed = seed,
        verbose = 10
    )    

    scores_auc_all = []
    test_cv_preds = []

    ms_2 = True
    if ms_2:
        
        oof_preds = []
        oof_targets = []
        scores = []
        scores_auc = []    
        for fold_nb in range(NB_SPLITS):
            
            print(b_,"FOLDS: ", r_, fold_nb + 1)
            print(g_, '*' * 60, c_)
            
            train_idx = [i for i, v in enumerate(fold_index) if v != fold_nb]
            val_idx = [i for i, v in enumerate(fold_index) if v == fold_nb]        

            X_train, y_train = train_df.values[train_idx, :], targets.values[train_idx, :]
            X_val, y_val = train_df.values[val_idx, :], targets.values[val_idx, :]
            ### Model ###
            model = TabNetRegressor(**tabnet_params)

            ### Fit ###
            # Another change to the original code
            # virtual_batch_size of 32 instead of 128
            model.fit(
                X_train = X_train,
                y_train = y_train,
                eval_set = [(X_val, y_val)],
                eval_name = ["val"],
                eval_metric = ["logits_ll"],
                max_epochs = MAX_EPOCH,
                patience = 20,
                batch_size = 1024, 
                virtual_batch_size = 32,
                num_workers = 1,
                drop_last = False,
                # To use binary cross entropy because this is not a regression problem
                loss_fn = F.binary_cross_entropy_with_logits
            )
            print(y_, '-' * 60)

            ### Predict on validation ###
            preds_val = model.predict(X_val)
            # Apply sigmoid to the predictions
            preds = 1 / (1 + np.exp(-preds_val))
            score = np.min(model.history["val_logits_ll"])

            ### Save OOF for CV ###
            oof_preds.append(preds_val)
            oof_targets.append(y_val)
            scores.append(score)

            ### Predict on test ###
            preds_test = model.predict(X_test)
            test_cv_preds.append(1 / (1 + np.exp(-preds_test)))
    else:
        mskf = MultilabelStratifiedKFold(n_splits = NB_SPLITS, random_state = 0, shuffle = True)

        oof_preds = []
        oof_targets = []
        scores = []
        scores_auc = []
        for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train_df, targets)):
            print(b_,"FOLDS: ", r_, fold_nb + 1)
            print(g_, '*' * 60, c_)

            X_train, y_train = train_df.values[train_idx, :], targets.values[train_idx, :]
            X_val, y_val = train_df.values[val_idx, :], targets.values[val_idx, :]
            ### Model ###
            model = TabNetRegressor(**tabnet_params)

            ### Fit ###
            # Another change to the original code
            # virtual_batch_size of 32 instead of 128
            model.fit(
                X_train = X_train,
                y_train = y_train,
                eval_set = [(X_val, y_val)],
                eval_name = ["val"],
                eval_metric = ["logits_ll"],
                max_epochs = MAX_EPOCH,
                patience = 20,
                batch_size = 1024, 
                virtual_batch_size = 32,
                num_workers = 1,
                drop_last = False,
                # To use binary cross entropy because this is not a regression problem
                loss_fn = F.binary_cross_entropy_with_logits
            )
            print(y_, '-' * 60)

            ### Predict on validation ###
            preds_val = model.predict(X_val)
            # Apply sigmoid to the predictions
            preds = 1 / (1 + np.exp(-preds_val))
            score = np.min(model.history["val_logits_ll"])

            ### Save OOF for CV ###
            oof_preds.append(preds_val)
            oof_targets.append(y_val)
            scores.append(score)

            ### Predict on test ###
            preds_test = model.predict(X_test)
            test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

    oof_preds_all = np.concatenate(oof_preds)
    oof_targets_all = np.concatenate(oof_targets)
    test_preds_all = np.stack(test_cv_preds)

    aucs = []
    for task_id in range(oof_preds_all.shape[1]):
        aucs.append(roc_auc_score(y_true = oof_targets_all[:, task_id],
                                y_score = oof_preds_all[:, task_id]
                                ))


    all_feat = [col for col in submission.columns if col not in ["sig_id"]]
    # To obtain the same lenght of test_preds_all and submission
    test = pd.read_csv(data_path + "test_features.csv")
    sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)
    tmp = pd.DataFrame(test_preds_all.mean(axis = 0), columns = all_feat)
    tmp["sig_id"] = sig_id

    submission = pd.merge(test[["sig_id"]], tmp, on = "sig_id", how = "left")
    submission.fillna(0, inplace = True)

    #submission[all_feat] = tmp.mean(axis = 0)

    # Set control to 0
    #submission.loc[test["cp_type"] == 0, submission.columns[1:]] = 0
    submission.to_csv("submission_tabnet.csv", index = None)
