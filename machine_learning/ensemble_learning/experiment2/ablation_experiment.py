import os
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from stacking_model import (
    StackingEnsemble,
    load_data,
    load_trained_models,
    get_base_models
)

# 特征组定义
FEATURE_GROUPS = {
    'ByteHistogram': slice(0, 256),  # 0-255
    'ByteEntropyHistogram': slice(256, 512),  # 256-511
    'GeneralFileInfo': slice(512, 522),  # 512-521
    'HeaderFileInfo': slice(522, 584),  # 522-583
    'ExportsInfo': slice(584, 712),  # 584-711
    'SectionInfo': slice(712, 967)  # 712-966
}

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
    
    # 计算每个特征组消融后与完整特征的性能差异
    full_feature_metrics = feature_ablation_df[feature_ablation_df['feature_group'] == 'full_features'].iloc[0]
    feature_ablation_df['auc_diff'] = full_feature_metrics['auc'] - feature_ablation_df['auc']
    feature_ablation_df['f1_diff'] = full_feature_metrics['f1'] - feature_ablation_df['f1']
    
    # 保存结果 - 确保包含auc_diff和f1_diff列
    feature_ablation_df.to_csv(os.path.join(result_dir, 'feature_ablation_results.csv'), index=False)
    print(f"特征消融实验结果已保存，包含列: {', '.join(feature_ablation_df.columns)}")
    
    print("特征消融实验完成！")
    
    return feature_ablation_df

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


if __name__ == "__main__":
    # 运行所有实验
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    all_results = run_all_experiments(result_dir) 