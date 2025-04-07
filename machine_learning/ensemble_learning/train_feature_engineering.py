import os
import re
import copy
import time
import lief
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)


with open("../feature_engineering/feature_engineering_features.pkl", 'rb') as f:
    feature_engineering_features = pickle.load(f)
with open("../models/feature_engineering_keys.pkl", 'rb') as f:
    keys = pickle.load(f)

with open("../models/hash_list.pkl", "rb") as f:
    hash_list = pickle.load(f)

with open("../models/black_list.pkl", "rb") as f:
    black_list = pickle.load(f)

# pe_scaler = StandardScaler() # 标准化
# feature_engineering_features = pe_scaler.fit_transform(feature_engineering_features)

train_features = []
for ha in hash_list:
    if ha in black_list:
        train_features.append(1)
    else:
        train_features.append(0)

train_features = np.array(train_features, dtype=np.int32)
train_df = pd.DataFrame(feature_engineering_features, columns=keys)


params = {
    'objective': 'binary',
    'boost_from_average': False,
    'num_leaves': 20,   # 叶子节点数，20，8
    'min_data_in_leaf': 1,  # 叶子最小数据量，1,5
    'objective': 'binary', # 定义的目标函数
    'max_depth': 4,     # 树深度，4,3
    'learning_rate': 0.01,   # 加快收敛速度, 0.01,0.05
    "min_sum_hessian_in_leaf": 4,
    "boosting": "gbdt",
    "feature_fraction": 0.9,  # 提取的特征比率
    "bagging_freq": 1,
    "bagging_fraction": 0.9,
    "bagging_seed": 11,
    "nthread": 10,
    'metric': {'binary_logloss'},  
    "random_state": 6666,
}

n_splits = 3

kf = StratifiedKFold(n_splits=n_splits, random_state=2200, shuffle=True)

prob_oof = np.zeros((len(train_features), ))

feature_importance_df = pd.DataFrame()

lgb_models = []

for fold_idx, (train_index, test_index) in enumerate(kf.split(train_df, train_features)):
    print("fold {}".format(fold_idx+1))
    trn_data = lgb.Dataset(train_df.iloc[train_index], label=train_features[train_index])
    val_data = lgb.Dataset(train_df.iloc[test_index], label=train_features[test_index])

    lgb_model = lgb.train(params,
                          trn_data,
                          3000,
                          valid_sets=[trn_data, val_data],
                          early_stopping_rounds=50,
                          verbose_eval=500)
    prob_oof[test_index] = lgb_model.predict(train_df.iloc[test_index], num_iteration=lgb_model.best_iteration)

    lgb_models.append(copy.deepcopy(lgb_model))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = keys
    fold_importance_df["importance"] = lgb_model.feature_importance()
    fold_importance_df["fold"] = fold_idx + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

with open("../oof/feature_engineerin_train.pkl", "wb") as fp:
    pickle.dump(prob_oof.reshape((len(train_features), 1)), fp)

with open("../models/lgb_models.pkl", "wb") as fp:
    pickle.dump(lgb_models, fp)
