import os

# 特征组定义
FEATURE_GROUPS = {
    'section_entropy': slice(0, 16),  # 节区特征 (16维)
    'string_patterns': slice(16, 42),  # 字符串模式特征 (26维)
    'yara_detection': slice(42, 44),  # YARA检测特征 (2维)
    'keyword_counts': slice(44, 49),  # 关键字计数特征 (5维)
    'opcodes': slice(49, 56),  # 操作码特征 (7维)
}

# 数据路径
FEATURE_ENGINEERING_PATH = "../feature_engineering/feature_engineering_features.pkl"
FEATURE_NAMES_PATH = "../models/feature_engineering_keys.pkl"
HASH_LIST_PATH = "../models/hash_list.pkl"
BLACK_LIST_PATH = "../models/black_list.pkl"
WHITE_LIST_PATH = "../models/white_list.pkl"

# 模型参数
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'boost_from_average': False,
    'num_leaves': 128,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 1e-3,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'metric': 'auc',
    'verbosity': -1
}

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc',
    'verbosity': 1
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'n_jobs': -1
}

# 创建所需目录
def create_directories():
    """创建所需的目录"""
    directories = ['./results', './figures', 
                  './results/lightgbm', './results/lightgbm_pca',
                  './results/xgboost', './results/random_forest',
                  './results/robustness', './results/inference_time']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory) 