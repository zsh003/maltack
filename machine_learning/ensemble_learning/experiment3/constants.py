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
    'device': 'cpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'n_estimators': 50,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    #'tree_method': 'gpu_hist',  # 使用GPU
    'gpu_id': 0,
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'colsample_bytree': 0.9,
    'subsample': 0.8,
    'n_estimators': 50,
    'n_jobs': -1
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 50,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'n_jobs': -1
}

# 创建所需目录
def create_directories():
    """创建所需的目录"""
    directories = ['./experiment3/results', './experiment3/figures', 
                  './experiment3/results/lightgbm', './experiment3/results/lightgbm_pca',
                  './experiment3/results/xgboost', './experiment3/results/random_forest',
                  './experiment3/results/robustness', './experiment3/results/inference_time']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory) 