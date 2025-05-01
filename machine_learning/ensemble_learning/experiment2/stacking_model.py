import os
import numpy as np
import pandas as pd
import pickle
import time
import copy
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class StackingEnsemble:
    """Stacking集成学习模型实现"""
    
    def __init__(self, base_models_dict=None, meta_model=None, n_folds=5):
        """
        初始化Stacking集成模型
        
        参数:
            base_models_dict: 字典，键为模型名称，值为对应的基模型
            meta_model: 元模型
            n_folds: 交叉验证折数
        """
        self.base_models_dict = base_models_dict if base_models_dict else {}
        self.meta_model = meta_model if meta_model else RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.n_folds = n_folds
        self.base_oof_train = {}
        self.base_oof_test = {}
        self.base_models_trained = {}
        
    def add_base_model(self, name, model):
        """添加基模型"""
        self.base_models_dict[name] = model
    
    def remove_base_model(self, name):
        """移除基模型"""
        if name in self.base_models_dict:
            del self.base_models_dict[name]
            if name in self.base_oof_train:
                del self.base_oof_train[name]
            if name in self.base_oof_test:
                del self.base_oof_test[name]
            if name in self.base_models_trained:
                del self.base_models_trained[name]
    
    def train_base_models(self, X_train, y_train, X_test=None, verbose=True):
        """
        训练所有基模型，并生成OOF预测
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征，如果不为None则生成测试集OOF预测
            verbose: 是否显示进度条
        
        返回:
            字典，包含每个基模型的OOF训练预测和测试预测
        """
        print("训练基模型并生成OOF预测...")
        
        n_train = X_train.shape[0]
        n_test = X_test.shape[0] if X_test is not None else 0
        
        # 初始化字典存储OOF预测
        oof_train = {}
        oof_test = {}
        trained_models = {}
        
        # 初始化进度条
        total_iters = len(self.base_models_dict) * self.n_folds
        
        for model_name, model in self.base_models_dict.items():
            if verbose:
                print(f"训练模型: {model_name}")
            
            # 初始化当前模型的OOF预测
            oof_train[model_name] = np.zeros((n_train, 1))
            if X_test is not None:
                oof_test_fold = np.zeros((self.n_folds, n_test))
            
            # 存储当前模型的所有训练后的模型
            trained_models[model_name] = []
            
            # 创建交叉验证对象
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            
            with tqdm(total=self.n_folds, ncols=80, disable=not verbose) as pbar:
                for i, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
                    # 拆分数据
                    fold_x_train = X_train[train_idx]
                    fold_y_train = y_train[train_idx]
                    fold_x_valid = X_train[valid_idx]
                    
                    # 训练模型
                    model_copy = copy.deepcopy(model)
                    model_copy.fit(fold_x_train, fold_y_train)
                    trained_models[model_name].append(model_copy)
                    
                    # 验证集上的预测
                    oof_train[model_name][valid_idx] = model_copy.predict_proba(fold_x_valid)[:, 1].reshape(-1, 1)
                    
                    # 测试集上的预测
                    if X_test is not None:
                        oof_test_fold[i, :] = model_copy.predict_proba(X_test)[:, 1]
                    
                    pbar.update(1)
            
            # 取平均作为测试集OOF预测
            if X_test is not None:
                oof_test[model_name] = oof_test_fold.mean(axis=0).reshape(-1, 1)
        
        # 存储结果
        self.base_oof_train = oof_train
        self.base_oof_test = oof_test
        self.base_models_trained = trained_models
        
        return oof_train, oof_test
    
    def train_meta_model(self, y_train):
        """
        训练元模型
        
        参数:
            y_train: 训练集标签
        """
        print("训练元模型...")
        
        # 堆叠OOF预测作为元模型的输入
        meta_train = np.hstack([self.base_oof_train[model_name] for model_name in self.base_models_dict.keys()])
        
        # 训练元模型
        self.meta_model.fit(meta_train, y_train)
        
        print("元模型训练完成!")
    
    def predict(self, X_test):
        """
        使用训练好的基模型和元模型进行预测
        
        参数:
            X_test: 测试特征
        
        返回:
            预测概率
        """
        # 使用所有基模型对测试集进行预测
        base_predictions = []
        for model_name in self.base_models_dict.keys():
            # 所有fold模型在测试集上的预测取平均
            fold_preds = []
            for model in self.base_models_trained[model_name]:
                fold_preds.append(model.predict_proba(X_test)[:, 1])
            model_pred = np.mean(fold_preds, axis=0).reshape(-1, 1)
            base_predictions.append(model_pred)
        
        # 堆叠预测作为元模型的输入
        meta_features = np.hstack(base_predictions)
        
        # 元模型预测
        return self.meta_model.predict_proba(meta_features)[:, 1]
    
    def fit_predict(self, X_train, y_train, X_test):
        """
        训练模型并预测
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
        
        返回:
            测试集的预测概率
        """
        # 训练基模型
        self.train_base_models(X_train, y_train, X_test)
        
        # 训练元模型
        self.train_meta_model(y_train)
        
        # 预测
        return self.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        
        返回:
            字典，包含各种评估指标
        """
        # 预测
        y_pred_prob = self.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_prob)
        }
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        return metrics, cm, y_pred_prob
    
    def save_model(self, save_dir='../models'):
        """
        保存模型
        
        参数:
            save_dir: 保存目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存元模型
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'wb') as f:
            pickle.dump(self.meta_model, f)
        
        # 保存基模型
        for model_name, models_list in self.base_models_trained.items():
            model_dir = os.path.join(save_dir, model_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            for i, model in enumerate(models_list):
                with open(os.path.join(model_dir, f'fold_{i}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
    
    def load_model(self, save_dir='../models'):
        """
        加载模型
        
        参数:
            save_dir: 模型保存目录
        """
        # 加载元模型
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'rb') as f:
            self.meta_model = pickle.load(f)
        
        # 加载基模型
        self.base_models_trained = {}
        for model_name in self.base_models_dict.keys():
            model_dir = os.path.join(save_dir, model_name)
            if os.path.exists(model_dir):
                self.base_models_trained[model_name] = []
                for i in range(self.n_folds):
                    model_path = os.path.join(model_dir, f'fold_{i}.pkl')
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            self.base_models_trained[model_name].append(pickle.load(f))
    
    def evaluate_base_models(self, X_test, y_test):
        """
        评估所有基模型的性能
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        
        返回:
            DataFrame，包含所有基模型的评估指标
        """
        print("评估基模型性能...")
        
        results = []
        
        for model_name in self.base_models_dict.keys():
            # 所有fold模型在测试集上的预测取平均
            fold_preds = []
            for model in self.base_models_trained[model_name]:
                fold_preds.append(model.predict_proba(X_test)[:, 1])
            
            y_pred_prob = np.mean(fold_preds, axis=0)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # 计算评估指标
            metrics = {
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_prob)
            }
            
            results.append(metrics)
        
        # 转换为DataFrame
        return pd.DataFrame(results)


def load_data():
    """加载数据集"""
    print("加载数据集...")
    
    # 加载特征向量
    with open("../pe_raw/pe_raw_vectors.pkl", "rb") as f:
        pe_raw_vectors = pickle.load(f)
    
    # 加载标签
    with open("../models/hash_list.pkl", "rb") as f:
        hash_list = pickle.load(f)
    
    with open("../models/black_list.pkl", "rb") as f:
        black_list = pickle.load(f)
    
    # 生成标签
    labels = []
    for ha in hash_list:
        if ha in black_list:
            labels.append(1)  # 恶意样本
        else:
            labels.append(0)  # 良性样本
    
    # 转换为numpy数组
    features = np.array(pe_raw_vectors, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"数据集加载完成，特征维度：{features.shape}，标签数量：{len(labels)}")
    
    return features, labels


def load_trained_models():
    """加载已训练的基模型"""
    print("加载已训练的基模型...")
    
    try:
        with open("../models/raw_feature.pkl", "rb") as f:
            pe_raw_models = pickle.load(f)
        
        with open("../models/raw_feature_names.pkl", "rb") as f:
            model_names = pickle.load(f)
        
        print(f"成功加载基模型，模型数量：{len(model_names)}")
        
        return pe_raw_models, model_names
    except Exception as e:
        print(f"加载模型失败：{e}")
        return None, None


def get_base_models():
    """获取所有基模型"""
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        AdaBoostClassifier,
        BaggingClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier
    )
    
    # 创建基模型字典
    base_models = {
        'lr': LogisticRegression(max_iter=5000, random_state=42),
        'gbc': GradientBoostingClassifier(random_state=42),
        'bc': BaggingClassifier(n_estimators=100, random_state=42),
        'xgb': XGBClassifier(max_depth=7, learning_rate=0.05, n_estimators=500, random_state=42),
        'dt': DecisionTreeClassifier(random_state=42),
        'svm': LinearSVC(max_iter=50000, random_state=42),
        'rfc': RandomForestClassifier(n_estimators=100, random_state=42),
        'etc': ExtraTreesClassifier(random_state=42),
        'ada': AdaBoostClassifier(random_state=42)
    }
    
    return base_models 