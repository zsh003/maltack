import os
import copy
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from constants import (FEATURE_ENGINEERING_PATH, FEATURE_NAMES_PATH, 
                      HASH_LIST_PATH, BLACK_LIST_PATH, 
                      LIGHTGBM_PARAMS)


def load_data(feature_path=FEATURE_ENGINEERING_PATH,
              keys_path=FEATURE_NAMES_PATH,
              hash_list_path=HASH_LIST_PATH,
              black_list_path=BLACK_LIST_PATH):
    """
    加载特征工程生成的数据和标签
    
    参数:
        feature_path: 特征文件路径
        keys_path: 特征名称文件路径
        hash_list_path: 哈希列表文件路径
        black_list_path: 黑名单文件路径
        
    返回:
        features: 特征矩阵
        labels: 标签向量
        feature_names: 特征名称列表
    """
    print("加载数据集...")
    
    # 加载特征向量和特征名
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    
    with open(keys_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    # 加载标签信息
    with open(hash_list_path, "rb") as f:
        hash_list = pickle.load(f)
    
    with open(black_list_path, "rb") as f:
        black_list = pickle.load(f)
    
    # 生成标签
    labels = []
    for ha in hash_list:
        if ha in black_list:
            labels.append(1)  # 恶意样本
        else:
            labels.append(0)  # 良性样本
    
    # 转换为numpy数组
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"数据集加载完成，特征维度：{features.shape}，标签数量：{len(labels)}")
    print(f"标签分布：良性样本 = {np.sum(labels==0)}，恶意样本 = {np.sum(labels==1)}")
    
    return features, labels, feature_names


def train_lightgbm_model(features, labels, feature_names, 
                          n_splits=5, 
                          test_size=0.2,
                          random_state=42,
                          params=None,
                          save_dir='./results'):
    """
    训练LightGBM模型并进行交叉验证
    
    参数:
        features: 特征矩阵
        labels: 标签向量
        feature_names: 特征名称列表
        n_splits: 交叉验证折数
        test_size: 测试集比例
        random_state: 随机种子
        params: LightGBM参数字典
        save_dir: 结果保存目录
        
    返回:
        models: 训练好的模型列表
        feature_importance_df: 特征重要性DataFrame
        metrics_results: 评估指标结果字典
    """
    # 如果没有提供参数，使用默认参数
    if params is None:
        params = LIGHTGBM_PARAMS.copy()
        params['random_state'] = random_state
    
    # 创建结果保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"训练集特征形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集特征形状: {X_test.shape}, 标签形状: {y_test.shape}")
    
    # 创建DataFrame
    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # 交叉验证
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    # 记录模型、特征重要性和预测结果
    models = []
    feature_importance_df = pd.DataFrame()
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    # 开始交叉验证
    print(f"\n开始{n_splits}折交叉验证训练...")
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n训练折 {fold + 1}/{n_splits}...")
        
        # 准备数据
        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
        X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val)
        
        # 训练模型
        print(f"训练LightGBM模型...设置early_stopping_rounds=50")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50), 
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 保存模型
        models.append(copy.deepcopy(model))
        
        # 获取特征重要性
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = feature_names
        fold_importance["importance"] = model.feature_importance(importance_type='gain')
        fold_importance["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
        
        # 预测
        oof_preds[val_idx] = model.predict(X_fold_val, num_iteration=model.best_iteration)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits
    
    # 计算训练时间
    train_time = time.time() - start_time
    print(f"\n模型训练完成！总训练时间: {train_time:.2f}秒 ({train_time/60:.2f}分钟)")
    
    # 评估模型性能
    print("\n评估模型性能...")
    oof_binary_preds = (oof_preds > 0.5).astype(int)
    test_binary_preds = (test_preds > 0.5).astype(int)
    
    # 计算验证集指标
    val_accuracy = accuracy_score(y_train, oof_binary_preds)
    val_precision = precision_score(y_train, oof_binary_preds)
    val_recall = recall_score(y_train, oof_binary_preds)
    val_f1 = f1_score(y_train, oof_binary_preds)
    val_auc = roc_auc_score(y_train, oof_preds)
    
    # 计算测试集指标
    test_accuracy = accuracy_score(y_test, test_binary_preds)
    test_precision = precision_score(y_test, test_binary_preds)
    test_recall = recall_score(y_test, test_binary_preds)
    test_f1 = f1_score(y_test, test_binary_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    # 记录结果
    metrics_results = {
        'validation': {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1,
            'auc': val_auc
        },
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'auc': test_auc
        }
    }
    
    # 打印结果
    print("\n验证集性能指标:")
    print(f"准确率 (Accuracy): {val_accuracy:.4f}")
    print(f"精确率 (Precision): {val_precision:.4f}")
    print(f"召回率 (Recall): {val_recall:.4f}")
    print(f"F1分数: {val_f1:.4f}")
    print(f"AUC: {val_auc:.4f}")
    
    print("\n测试集性能指标:")
    print(f"准确率 (Accuracy): {test_accuracy:.4f}")
    print(f"精确率 (Precision): {test_precision:.4f}")
    print(f"召回率 (Recall): {test_recall:.4f}")
    print(f"F1分数: {test_f1:.4f}")
    print(f"AUC: {test_auc:.4f}")
    
    # 保存详细的分类报告
    test_report = classification_report(y_test, test_binary_preds)
    print("\n测试集详细分类报告:")
    print(test_report)
    
    # 保存模型和结果
    print("\n保存模型和结果...")
    with open(os.path.join(save_dir, 'lgbm_models.pkl'), 'wb') as f:
        pickle.dump(models, f)
    
    feature_importance_df.to_csv(os.path.join(save_dir, 'feature_importance.csv'), index=False)
    
    # 保存预测结果
    pd.DataFrame({
        'true_label': y_test,
        'predicted_prob': test_preds,
        'predicted_label': test_binary_preds
    }).to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)
    
    # 保存评估指标
    pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
        'validation': [val_accuracy, val_precision, val_recall, val_f1, val_auc],
        'test': [test_accuracy, test_precision, test_recall, test_f1, test_auc]
    }).to_csv(os.path.join(save_dir, 'model_metrics.csv'), index=False)
    
    print(f"\n所有结果已保存到: {save_dir}")
    
    return models, feature_importance_df, metrics_results, (X_test, y_test, test_preds)


def train_pca_lightgbm_model(features, labels, feature_names, 
                              n_components=10,
                              n_splits=3, 
                              test_size=0.2,
                              random_state=42,
                              params=None,
                              save_dir='./results'):
    """
    训练基于PCA降维的LightGBM模型
    
    参数:
        features: 特征矩阵
        labels: 标签向量
        feature_names: 特征名称列表
        n_components: PCA保留的主成分数量
        n_splits: 交叉验证折数
        test_size: 测试集比例
        random_state: 随机种子
        params: LightGBM参数字典
        save_dir: 结果保存目录
        
    返回:
        models: 训练好的模型列表
        metrics_results: 评估指标结果字典
    """
    print(f"\n开始训练基于PCA降维(保留{n_components}个主成分)的LightGBM模型...")


    # 处理异常值
    # 1. 检测并打印异常值信息
    has_nan = np.isnan(features).any()
    has_inf = np.isinf(features).any()
    
    print(f"数据中包含NaN: {has_nan}")
    print(f"数据中包含无穷值: {has_inf}")
    
    if has_nan or has_inf:
        print("正在处理异常值...")
        # 创建数据副本以避免修改原始数据
        features_clean = features.copy()
        
        # 替换NaN为0
        if has_nan:
            nan_count = np.isnan(features_clean).sum()
            print(f"替换 {nan_count} 个NaN值为0")
            features_clean = np.nan_to_num(features_clean, nan=0.0)
        
        # 替换无穷值为非常大的有限值
        if has_inf:
            inf_count = np.isinf(features_clean).sum()
            print(f"替换 {inf_count} 个无穷值")
            features_clean = np.nan_to_num(features_clean, posinf=1.0e10, neginf=-1.0e10)
        
        # 检查数据范围，处理超出范围的值
        max_val = np.max(features_clean)
        min_val = np.min(features_clean)
        print(f"清洗后数据范围: [{min_val}, {max_val}]")
        
        # 使用处理过的数据
        features = features_clean
        
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # PCA降维
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_features = pca.fit_transform(scaled_features)
    
    print(f"PCA降维后特征维度: {reduced_features.shape}")
    print(f"累计解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 生成新的特征名
    reduced_feature_names = [f'PC{i+1}' for i in range(n_components)]
    
    # 保存PCA模型
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(os.path.join(save_dir, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    
    # 保存PCA解释方差
    pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(n_components)],
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_explained_variance': np.cumsum(pca.explained_variance_ratio_)
    }).to_csv(os.path.join(save_dir, 'pca_explained_variance.csv'), index=False)
    
    # 保存PCA成分与原始特征的对应关系
    pca_components = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(n_components)]
    )
    pca_components.to_csv(os.path.join(save_dir, 'pca_components.csv'))
    
    # 训练模型
    if params is None:
        params = LIGHTGBM_PARAMS.copy()
        params['random_state'] = random_state
    
    models, _, metrics_results, test_data = train_lightgbm_model(
        reduced_features, labels, reduced_feature_names,
        n_splits=n_splits, 
        test_size=test_size,
        random_state=random_state,
        params=params,
        save_dir=os.path.join(save_dir, 'pca_model')
    )
    
    return models, metrics_results, test_data, pca 