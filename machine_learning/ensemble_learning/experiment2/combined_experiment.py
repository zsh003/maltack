import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import pickle
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from matplotlib.ticker import MaxNLocator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 特征组定义
FEATURE_GROUPS = {
    'ByteHistogram': slice(0, 256),  # 0-255
    'ByteEntropyHistogram': slice(256, 512),  # 256-511
    'GeneralFileInfo': slice(512, 522),  # 512-521
    'HeaderFileInfo': slice(522, 584),  # 522-583
    'ExportsInfo': slice(584, 712),  # 584-711
    'SectionInfo': slice(712, 967)  # 712-966
}

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
        
        # 修复：直接初始化元模型，避免嵌套结构
        if meta_model is None:
            self.meta_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        else:
            self.meta_model = meta_model
            
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
                    # 处理没有predict_proba方法的模型（如LinearSVC）
                    if hasattr(model_copy, 'predict_proba'):
                        oof_train[model_name][valid_idx] = model_copy.predict_proba(fold_x_valid)[:, 1].reshape(-1, 1)
                    else:
                        # 使用decision_function作为替代，或者直接用predict
                        if hasattr(model_copy, 'decision_function'):
                            # 对于LinearSVC等模型使用decision_function
                            decision_values = model_copy.decision_function(fold_x_valid)
                            # 转换为伪概率值（通过Sigmoid函数或归一化）
                            proba = 1 / (1 + np.exp(-decision_values))
                        else:
                            # 如果没有decision_function，使用predict
                            proba = model_copy.predict(fold_x_valid)
                        oof_train[model_name][valid_idx] = proba.reshape(-1, 1)
                    
                    # 测试集上的预测
                    if X_test is not None:
                        if hasattr(model_copy, 'predict_proba'):
                            oof_test_fold[i, :] = model_copy.predict_proba(X_test)[:, 1]
                        else:
                            # 处理没有predict_proba方法的模型
                            if hasattr(model_copy, 'decision_function'):
                                decision_values = model_copy.decision_function(X_test)
                                oof_test_fold[i, :] = 1 / (1 + np.exp(-decision_values))
                            else:
                                oof_test_fold[i, :] = model_copy.predict(X_test)
                    
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
                # 处理没有predict_proba方法的模型
                if hasattr(model, 'predict_proba'):
                    fold_preds.append(model.predict_proba(X_test)[:, 1])
                else:
                    # 使用decision_function或predict作为替代
                    if hasattr(model, 'decision_function'):
                        decision_values = model.decision_function(X_test)
                        fold_preds.append(1 / (1 + np.exp(-decision_values)))
                    else:
                        fold_preds.append(model.predict(X_test))
                        
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
        保存模型到指定目录
        
        参数:
            save_dir: 保存目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存基模型
        for model_name, models in self.base_models_trained.items():
            with open(os.path.join(save_dir, f'{model_name}_models.pkl'), 'wb') as f:
                pickle.dump(models, f)
        
        # 保存元模型
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'wb') as f:
            pickle.dump(self.meta_model, f)
        
        print(f"模型已保存到 {save_dir}")
    
    def load_model(self, save_dir='../models'):
        """
        从指定目录加载模型
        
        参数:
            save_dir: 模型目录
        """
        # 加载基模型
        for model_name in self.base_models_dict.keys():
            with open(os.path.join(save_dir, f'{model_name}_models.pkl'), 'rb') as f:
                self.base_models_trained[model_name] = pickle.load(f)
        
        # 加载元模型
        with open(os.path.join(save_dir, 'meta_model.pkl'), 'rb') as f:
            self.meta_model = pickle.load(f)
        
        print(f"模型已从 {save_dir} 加载")
    
    def evaluate_base_models(self, X_test, y_test):
        """
        评估每个基模型的性能
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        
        返回:
            DataFrame，包含每个基模型的性能指标
        """
        base_model_metrics = []
        for model_name, models in self.base_models_trained.items():
            print(f"评估基模型: {model_name}")
            fold_preds = []
            for model in models:
                if hasattr(model, 'predict_proba'):
                    fold_preds.append(model.predict_proba(X_test)[:, 1])
                else:
                    if hasattr(model, 'decision_function'):
                        decision_values = model.decision_function(X_test)
                        fold_preds.append(1 / (1 + np.exp(-decision_values)))
                    else:
                        fold_preds.append(model.predict(X_test))
            
            model_pred = np.mean(fold_preds, axis=0)
            y_pred = (model_pred > 0.5).astype(int)
            
            metrics = {
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, model_pred)
            }
            base_model_metrics.append(metrics)
        
        return pd.DataFrame(base_model_metrics)

def train_test_split_data(features, labels, test_size=0.2, random_state=42):
    """
    划分训练集和测试集
    
    参数:
        features: 特征矩阵
        labels: 标签
        test_size: 测试集比例
        random_state: 随机种子
    
    返回:
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(features, labels, test_size=test_size, stratify=labels, random_state=random_state)

def preprocess_features(X_train, X_test):
    """
    特征预处理：标准化
    
    参数:
        X_train: 训练集特征
        X_test: 测试集特征
    
    返回:
        (X_train_scaled, X_test_scaled)
    """
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

def run_model_ablation_experiment(features, labels, result_dir='./results'):
    """
    运行模型消融实验：逐一移除每个基模型，测试对整体性能的影响
    
    参数:
        features: 特征矩阵
        labels: 标签
        result_dir: 结果保存目录
    
    返回:
        model_ablation_results: DataFrame，包含模型消融实验结果
        base_model_performance: DataFrame，包含基模型独立性能
    """
    print(f"开始模型消融实验... 数据集大小: {features.shape}, 标签分布: {np.bincount(labels)}")
    
    # 创建结果目录
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split_data(features, labels)
    print(f"数据集划分完成：训练集：{X_train.shape}, 测试集：{X_test.shape}")
    
    # 预处理特征
    X_train, X_test = preprocess_features(X_train, X_test)
    print("特征标准化完成")
    
    # 获取所有基模型
    base_models = get_base_models()
    model_names = list(base_models.keys())
    print(f"已加载 {len(base_models)} 个基模型: {', '.join(model_names)}")
    
    # 创建元模型
    meta_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    
    # 记录实验结果
    model_ablation_results = []
    
    # 1. 首先训练完整模型，包含所有基模型
    print("训练完整模型（包含所有基模型）...")
    
    full_model = StackingEnsemble(base_models_dict=base_models, meta_model=meta_model)
    full_model.train_base_models(X_train, y_train, X_test)
    full_model.train_meta_model(y_train)
    full_metrics, _, _ = full_model.evaluate(X_test, y_test)
    
    # 记录完整模型的性能
    model_ablation_results.append({
        'model_config': 'full_ensemble',
        'removed_model': None,
        'model_count': len(base_models),
        'accuracy': full_metrics['accuracy'],
        'precision': full_metrics['precision'],
        'recall': full_metrics['recall'],
        'f1': full_metrics['f1'],
        'auc': full_metrics['auc']
    })
    
    # 评估每个基模型的单独性能
    base_model_metrics = full_model.evaluate_base_models(X_test, y_test)
    
    # 2. 逐一移除每个基模型，测试对整体性能的影响
    for model_name in base_models.keys():
        print(f"移除基模型: {model_name}")
        
        # 创建不包含当前模型的模型集合
        reduced_models = {k: v for k, v in base_models.items() if k != model_name}
        
        # 训练减少模型的集成
        reduced_model = StackingEnsemble(base_models_dict=reduced_models, meta_model=meta_model)
        reduced_model.train_base_models(X_train, y_train, X_test)
        reduced_model.train_meta_model(y_train)
        reduced_metrics, _, _ = reduced_model.evaluate(X_test, y_test)
        
        # 记录结果
        model_ablation_results.append({
            'model_config': f'without_{model_name}',
            'removed_model': model_name,
            'model_count': len(reduced_models),
            'accuracy': reduced_metrics['accuracy'],
            'precision': reduced_metrics['precision'],
            'recall': reduced_metrics['recall'],
            'f1': reduced_metrics['f1'],
            'auc': reduced_metrics['auc']
        })
    
    # 创建结果DataFrame
    ablation_df = pd.DataFrame(model_ablation_results)
    
    # 计算每个模型消融后与完整模型的性能差异
    full_model_metrics = ablation_df[ablation_df['model_config'] == 'full_ensemble'].iloc[0]
    ablation_df['auc_diff'] = full_model_metrics['auc'] - ablation_df['auc']
    ablation_df['f1_diff'] = full_model_metrics['f1'] - ablation_df['f1']
    
    # 保存结果 - 确保包含auc_diff和f1_diff列
    ablation_df.to_csv(os.path.join(result_dir, 'model_ablation_results.csv'), index=False)
    print(f"模型消融实验结果已保存，包含列: {', '.join(ablation_df.columns)}")
    
    # 保存基模型单独性能
    base_model_metrics.to_csv(os.path.join(result_dir, 'base_model_performance.csv'), index=False)
    print(f"基模型性能结果已保存，包含列: {', '.join(base_model_metrics.columns)}")
    
    print("模型消融实验完成！")
    
    return ablation_df, base_model_metrics

def run_feature_ablation_experiment(features, labels, result_dir='./results'):
    """
    运行特征消融实验：逐一移除每个特征组，测试对整体性能的影响
    
    参数:
        features: 特征矩阵
        labels: 标签
        result_dir: 结果保存目录
    
    返回:
        feature_ablation_results: DataFrame，包含特征消融实验结果
    """
    print(f"开始特征消融实验... 完整特征维度: {features.shape}")
    print(f"特征组信息:")
    for group_name, feature_slice in FEATURE_GROUPS.items():
        if isinstance(feature_slice, slice):
            group_size = feature_slice.stop - feature_slice.start
            print(f"  - {group_name}: {feature_slice.start}-{feature_slice.stop-1} ({group_size}维)")
        else:
            print(f"  - {group_name}: {len(feature_slice)}维")
    
    # 创建结果目录
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split_data(features, labels)
    print(f"数据集划分完成：训练集：{X_train.shape}, 测试集：{X_test.shape}")
    
    # 获取所有基模型
    base_models = get_base_models()
    print(f"使用 {len(base_models)} 个基模型进行特征消融实验")
    
    # 创建元模型
    meta_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    
    # 记录实验结果
    feature_ablation_results = []
    
    # 1. 首先训练完整模型，包含所有特征
    print("训练完整模型（包含所有特征）...")
    X_train_full, X_test_full = preprocess_features(X_train, X_test)
    
    full_model = StackingEnsemble(base_models_dict=base_models, meta_model=meta_model)
    full_model.train_base_models(X_train_full, y_train, X_test_full)
    full_model.train_meta_model(y_train)
    full_metrics, _, _ = full_model.evaluate(X_test_full, y_test)
    
    # 记录完整模型的性能
    feature_ablation_results.append({
        'feature_group': 'full_features',
        'removed_feature': None,
        'feature_dim': features.shape[1],
        'accuracy': full_metrics['accuracy'],
        'precision': full_metrics['precision'],
        'recall': full_metrics['recall'],
        'f1': full_metrics['f1'],
        'auc': full_metrics['auc']
    })
    
    # 2. 逐一移除每个特征组，测试对整体性能的影响
    for group_name, feature_slice in FEATURE_GROUPS.items():
        print(f"移除特征组: {group_name}")
        
        # 创建特征掩码
        feature_mask = np.ones(features.shape[1], dtype=bool)
        feature_mask[feature_slice] = False
        
        # 移除特征
        X_train_reduced = X_train[:, feature_mask]
        X_test_reduced = X_test[:, feature_mask]
        
        # 预处理特征
        X_train_reduced, X_test_reduced = preprocess_features(X_train_reduced, X_test_reduced)
        
        # 训练减少特征的模型
        reduced_model = StackingEnsemble(base_models_dict=base_models, meta_model=meta_model)
        reduced_model.train_base_models(X_train_reduced, y_train, X_test_reduced)
        reduced_model.train_meta_model(y_train)
        reduced_metrics, _, _ = reduced_model.evaluate(X_test_reduced, y_test)
        
        # 记录结果
        feature_ablation_results.append({
            'feature_group': f'without_{group_name}',
            'removed_feature': group_name,
            'feature_dim': X_train_reduced.shape[1],
            'accuracy': reduced_metrics['accuracy'],
            'precision': reduced_metrics['precision'],
            'recall': reduced_metrics['recall'],
            'f1': reduced_metrics['f1'],
            'auc': reduced_metrics['auc']
        })
    
    # 创建结果DataFrame
    feature_ablation_df = pd.DataFrame(feature_ablation_results)
    
    # 计算每个特征消融后与完整特征的性能差异
    full_feature_metrics = feature_ablation_df[feature_ablation_df['feature_group'] == 'full_features'].iloc[0]
    feature_ablation_df['auc_diff'] = full_feature_metrics['auc'] - feature_ablation_df['auc']
    feature_ablation_df['f1_diff'] = full_feature_metrics['f1'] - feature_ablation_df['f1']
    
    # 保存结果
    feature_ablation_df.to_csv(os.path.join(result_dir, 'feature_ablation_results.csv'), index=False)
    print(f"特征消融实验结果已保存，包含列: {', '.join(feature_ablation_df.columns)}")
    
    print("特征消融实验完成！")
    
    return feature_ablation_df

def plot_model_ablation_results(ablation_df, save_dir):
    """
    绘制模型消融实验结果
    
    参数:
        ablation_df: DataFrame，包含模型消融实验结果
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 过滤掉完整模型的结果
    ablation_subset = ablation_df[ablation_df['model_config'] != 'full_ensemble'].copy()
    
    # 提取出完整模型的性能指标
    full_model_metrics = ablation_df[ablation_df['model_config'] == 'full_ensemble'].iloc[0]
    full_model_auc = full_model_metrics['auc']
    full_model_f1 = full_model_metrics['f1']
    
    # 确保有auc_diff和f1_diff列，如果没有就计算它们
    if 'auc_diff' not in ablation_subset.columns:
        ablation_subset['auc_diff'] = full_model_auc - ablation_subset['auc']
    
    if 'f1_diff' not in ablation_subset.columns:
        ablation_subset['f1_diff'] = full_model_f1 - ablation_subset['f1']
    
    # 排序，使差异大的在前面
    ablation_subset = ablation_subset.sort_values(by='auc_diff', ascending=False)
    
    # 1. 绘制AUC差异图
    plt.figure(figsize=(12, 8))
    bars = plt.bar(ablation_subset['removed_model'], ablation_subset['auc_diff'], color='skyblue')
    
    # 添加完整模型的AUC值的水平线
    plt.axhline(y=0, linestyle='--', color='r', alpha=0.7, label=f'完整模型 AUC={full_model_auc:.3f}')
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', rotation=0)
    
    plt.xlabel('移除的模型')
    plt.ylabel('AUC下降值')
    plt.title('移除不同基模型后的AUC下降量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'model_ablation_auc_diff.png'), dpi=300)
    plt.close()
    
    # 2. 绘制F1分数差异图
    plt.figure(figsize=(12, 8))
    bars = plt.bar(ablation_subset['removed_model'], ablation_subset['f1_diff'], color='lightgreen')
    
    # 添加完整模型的F1值的水平线
    plt.axhline(y=0, linestyle='--', color='r', alpha=0.7, label=f'完整模型 F1={full_model_f1:.3f}')
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', rotation=0)
    
    plt.xlabel('移除的模型')
    plt.ylabel('F1下降值')
    plt.title('移除不同基模型后的F1分数下降量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'model_ablation_f1_diff.png'), dpi=300)
    plt.close()
    
    # 3. 绘制雷达图比较各个模型的性能
    # 准备数据
    model_names = list(ablation_subset['removed_model'])
    model_names.append('full_ensemble')  # 添加完整模型
    
    # 找到完整模型的行
    full_model_row = ablation_df[ablation_df['model_config'] == 'full_ensemble'].iloc[0]
    
    # 合并数据
    ablation_with_full = pd.concat([ablation_subset, pd.DataFrame([full_model_row])])
    
    # 准备雷达图数据
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # 创建一个图表
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 添加轴标签
    plt.xticks(angles[:-1], metrics)
    
    # 绘制每个模型的雷达图
    for idx, model in enumerate(ablation_with_full.iterrows()):
        model_config = model[1]['model_config']
        model_name = model[1]['removed_model'] if not pd.isna(model[1]['removed_model']) else 'full_ensemble'
        values = [model[1][metric] for metric in metrics]
        values += values[:1]  # 闭合图形
        
        if model_config == 'full_ensemble':
            ax.plot(angles, values, 'o-', linewidth=2, label=f'完整模型')
            ax.fill(angles, values, alpha=0.1)
        else:
            ax.plot(angles, values, 'o-', linewidth=1, alpha=0.7, label=f'移除{model_name}')
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('不同模型组合的性能雷达图')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'model_ablation_radar.png'), dpi=300)
    plt.close()

def plot_base_model_performance(base_model_df, save_dir):
    """
    绘制基模型单独性能对比图
    
    参数:
        base_model_df: DataFrame，包含基模型性能
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 按AUC排序
    base_model_df = base_model_df.sort_values(by='auc', ascending=False)
    
    # 1. 绘制AUC和F1分数条形图
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(base_model_df))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, base_model_df['auc'], width, label='AUC', color='skyblue')
    bars2 = plt.bar(x + width/2, base_model_df['f1'], width, label='F1分数', color='lightgreen')
    
    # 添加数据标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.xlabel('基模型')
    plt.ylabel('性能指标')
    plt.title('基模型的AUC和F1分数对比')
    plt.xticks(x, base_model_df['model'])
    plt.ylim(0.75, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'base_model_performance.png'), dpi=300)
    plt.close()
    
    # 2. 绘制综合性能热力图
    plt.figure(figsize=(12, 8))
    
    # 准备热力图数据
    heatmap_data = base_model_df.set_index('model')[['accuracy', 'precision', 'recall', 'f1', 'auc']]
    
    # 创建热力图
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
    
    plt.title('基模型性能指标热力图')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'base_model_heatmap.png'), dpi=300)
    plt.close()

def plot_feature_ablation_results(feature_ablation_df, save_dir):
    """
    绘制特征消融实验结果
    
    参数:
        feature_ablation_df: DataFrame，包含特征消融实验结果
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 过滤掉完整特征的结果
    feature_subset = feature_ablation_df[feature_ablation_df['feature_group'] != 'full_features'].copy()
    
    # 提取出完整特征的性能指标
    full_feature_metrics = feature_ablation_df[feature_ablation_df['feature_group'] == 'full_features'].iloc[0]
    full_feature_auc = full_feature_metrics['auc']
    full_feature_f1 = full_feature_metrics['f1']
    
    # 确保有auc_diff和f1_diff列，如果没有就计算它们
    if 'auc_diff' not in feature_subset.columns:
        feature_subset['auc_diff'] = full_feature_auc - feature_subset['auc']
    
    if 'f1_diff' not in feature_subset.columns:
        feature_subset['f1_diff'] = full_feature_f1 - feature_subset['f1']
    
    # 排序，使差异大的在前面
    feature_subset = feature_subset.sort_values(by='auc_diff', ascending=False)
    
    # 1. 绘制AUC差异图
    plt.figure(figsize=(12, 8))
    
    # 提取特征组名称（去掉"without_"前缀）
    feature_names = [name.replace('without_', '') for name in feature_subset['feature_group']]
    
    bars = plt.bar(feature_names, feature_subset['auc_diff'], color='skyblue')
    
    # 添加完整特征的AUC值的水平线
    plt.axhline(y=0, linestyle='--', color='r', alpha=0.7, label=f'完整特征 AUC={full_feature_auc:.3f}')
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.xlabel('移除的特征组')
    plt.ylabel('AUC下降值')
    plt.title('移除不同特征组后的AUC下降量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'feature_ablation_auc_diff.png'), dpi=300)
    plt.close()
    
    # 2. 绘制F1分数差异图
    plt.figure(figsize=(12, 8))
    bars = plt.bar(feature_names, feature_subset['f1_diff'], color='lightgreen')
    
    # 添加完整特征的F1值的水平线
    plt.axhline(y=0, linestyle='--', color='r', alpha=0.7, label=f'完整特征 F1={full_feature_f1:.3f}')
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.xlabel('移除的特征组')
    plt.ylabel('F1下降值')
    plt.title('移除不同特征组后的F1分数下降量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'feature_ablation_f1_diff.png'), dpi=300)
    plt.close()

def create_all_visualizations(result_dir='./results', save_dir='./figures'):
    """
    创建所有可视化图表
    
    参数:
        result_dir: 结果目录
        save_dir: 图表保存目录
    """
    # 加载模型消融实验结果
    model_ablation_df = pd.read_csv(os.path.join(result_dir, 'model_ablation_results.csv'))
    base_model_df = pd.read_csv(os.path.join(result_dir, 'base_model_performance.csv'))
    feature_ablation_df = pd.read_csv(os.path.join(result_dir, 'feature_ablation_results.csv'))
    
    # 绘制图表
    plot_model_ablation_results(model_ablation_df, save_dir)
    plot_base_model_performance(base_model_df, save_dir)
    plot_feature_ablation_results(feature_ablation_df, save_dir)

def run_all_experiments(result_dir='./results'):
    """
    运行所有消融实验
    
    参数:
        result_dir: 结果保存目录
    
    返回:
        所有实验结果
    """
    # 加载数据
    features, labels = load_data()
    
    # 创建结果目录
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 运行模型消融实验
    model_ablation_results, base_model_performance = run_model_ablation_experiment(features, labels, result_dir)
    
    # 运行特征消融实验
    feature_ablation_results = run_feature_ablation_experiment(features, labels, result_dir)
    
    # 返回所有实验结果
    return {
        'model_ablation_results': model_ablation_results,
        'feature_ablation_results': feature_ablation_results,
        'base_model_performance': base_model_performance
    }

def main(args):
    """
    运行PE恶意软件检测的Stacking模型消融实验
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子
    np.random.seed(42)
    
    # 设置结果目录
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 设置图表保存目录
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # 运行实验
    if not args.skip_experiments:
        print("\n===================== 开始运行消融实验 =====================")
        start_time = time.time()
        all_results = run_all_experiments(result_dir)
        end_time = time.time()
        
        print(f"\n实验完成! 总耗时: {(end_time - start_time)/60:.2f} 分钟")
    
    # 生成可视化
    print("\n===================== 开始生成可视化图表 =====================")
    create_all_visualizations(result_dir, figures_dir)
    print("\n可视化图表生成完成!")
    
    print(f"\n所有结果已保存到: {result_dir}")
    print(f"所有图表已保存到: {figures_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行PE恶意软件检测的Stacking模型消融实验')
    parser.add_argument('--skip-experiments', action='store_true', help='跳过实验，只生成可视化图表')
    args = parser.parse_args()
    
    main(args) 