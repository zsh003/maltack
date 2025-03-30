import os
import time
import lightgbm as lgb
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import tensorflow as tf


with open("../models/hash_list.pkl", "rb") as f:
    hash_list = pickle.load(f)
test_num = len(hash_list)


def plot(test_label, y_pred, model):
    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}

    accs = accuracy_score(test_label, y_pred)
    fpr, tpr, _ = metrics.roc_curve(test_label, y_pred)
    auc = metrics.roc_auc_score(test_label, y_pred)
    score = tpr[1] - 0.9*fpr[0]
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="{0}, auc={1:.2f}, score={2:.2f}".format(model, auc*100, score*100), color='green', linewidth=2)
    ax.set_title("ROC curve", fontdict=font)
    leg = ax.legend(loc="best")
    text = leg.get_texts()
    _ = plt.setp(text, color="blue") 
  

# -----------------直方图基模型（二层）训练集-----------------

with open("../models/raw_his_feature.pkl", "rb") as fp:
    raw_his_feature = pickle.load(fp)

with open("../models/raw_his_labels.pkl", "rb") as fp:
    raw_his_labels = pickle.load(fp)

features, labels = np.array(raw_his_feature, dtype=np.float32), np.array(raw_his_labels, dtype=np.int32)
model = tf.keras.models.load_model('../models/histogram_model.h5')
histogram_train = model.predict(features)


# -----------------pe静态特征基模型（二层）训练集-----------------
with open("../models/labels.pkl", "rb") as f:
    labels = pickle.load(f)

with open("../oof/raw_feature_stacking_train_5.pkl", "rb") as fp:
    raw_feature_stacking_train_5 = pickle.load(fp)

train_data, test_data, train_labels, test_labels = train_test_split(raw_feature_stacking_train_5, 
                                                                    labels, 
                                                                    test_size=0.2, 
                                                                    random_state=1)

# model = XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=4, reg_lambda=0.4, seed=4396, scale_pos_weight=3)
# model.fit(train_data, train_labels)
# y_pred = model.predict(test_data)
rfc_pe_model = RandomForestClassifier(100, random_state=8)
rfc_pe_model.fit(train_data, train_labels)
train_pred = rfc_pe_model.predict(train_data)
y_pred = rfc_pe_model.predict(test_data)

# plot(train_labels, train_pred, "rfc_pe")

raw_feature_train = rfc_pe_model.predict(raw_feature_stacking_train_5).reshape(-1, 1)
with open("../models/rfc_pe_model.pkl", "wb") as f:
    pickle.dump(rfc_pe_model, f)


# -----------------特征工程基模型（二层）训练集-----------------

with open("../oof/feature_engineerin_train.pkl", "rb") as fp:
    feature_engineerin_train = pickle.load(fp)

# ---------------------二层基模型堆叠------------------------

train = np.hstack([feature_engineerin_train, histogram_train , raw_feature_train])
stacking_train_5 = train

train_data, test_data, train_labels, test_labels = train_test_split(stacking_train_5,
                                                                    labels,
                                                                    test_size=0.2,
                                                                    random_state=2020) # 0x4651

# 逻辑回归
lr_model = LogisticRegression(random_state=7)
lr_model.fit(train_data, train_labels)
train_pre_lr = lr_model.predict(train_data)
y_pred_lr = lr_model.predict(test_data)

# plot(train_labels, train_pre_lr, "lr")
# plot(test_labels, y_pred_lr, "lr")


# LightGBM
params = {'num_leaves': 8, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 1,
          'objective': 'binary', #定义的目标函数
          'max_depth': 3,
          'learning_rate': 0.01,
          "min_sum_hessian_in_leaf": 8,
          "boosting": "gbdt",
          "feature_fraction": 0.9,	#提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,				#l1正则
          # 'lambda_l2': 0.001,		#l2正则
          "verbosity": -1,
          "nthread": -1,				#线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss'},	##评价函数选择
          "random_state": 5555,	#随机数种子，可以防止每次运行的结果不一致
          # 'device': 'gpu' ##如果安装的是gpu版本的lightgbm,可以加快运算
}
trn_data = lgb.Dataset(train_data, label=train_labels)
val_data = lgb.Dataset(test_data, label=test_labels)
clf = lgb.train(params, trn_data, 1000,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=200,
                    early_stopping_rounds=20)
train_pred_lgb = clf.predict(train_data)
train_pred_lgb_b = np.where(train_pred_lgb < 0.5, 0, 1)

y_pred_lgb = clf.predict(test_data)
y_pred_lgb_b = np.where(y_pred_lgb < 0.5, 0, 1)

# plot(train_labels, train_pred_lgb_b, "lgb")
# plot(test_labels, y_pred_lgb_b, "lgb")

# 随机森林
rfc_model = RandomForestClassifier(100, random_state=7)
rfc_model.fit(train_data, train_labels)
train_pred_rfc = rfc_model.predict(train_data)
y_pred_rfc = rfc_model.predict(test_data)
# plot(train_labels, train_pred_rfc, "rfc")
# plot(test_labels, y_pred_rfc, "rfc")

labels_lr = lr_model.predict_proba(train)
labels_rfc = rfc_model.predict_proba(train)

#labels_lgb = clf.predict(stacking_train_5)
train_labels = []
for x, y in zip(labels_lr, labels_rfc):
    if x[1]*0.3 + y[1]*0.7 < 0.5:
        train_labels.append(0)
    else:
        train_labels.append(1)

accs = accuracy_score(labels, train_labels)
fpr, tpr, _ = metrics.roc_curve(labels, train_labels)
auc = metrics.roc_auc_score(labels, train_labels)
score = tpr[1] - 1.2*fpr[0]
print("accs = ", accs)
print("auc = ", auc)
print("score = ", score)


with open("../models/lr_rfc.pkl", "wb") as f:
    pickle.dump([lr_model, rfc_model], f)