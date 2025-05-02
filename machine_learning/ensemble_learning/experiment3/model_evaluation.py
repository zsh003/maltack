import os
import numpy as np
import pandas as pd
import pickle
import time
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from lgbm_model import train_lightgbm_model, train_pca_lightgbm_model
from constants import LIGHTGBM_PARAMS, XGBOOST_PARAMS, RANDOM_FOREST_PARAMS


def compare_models(features, labels, feature_names, random_state=42, save_dir='./results'):
    """
    比较不同机器学习模型的性能
    
    参数:
        features: 特征矩阵
        labels: 标签向量
        feature_names: 特征名称列表
        random_state: 随机种子
        save_dir: 结果保存目录
        
    返回:
        comparison_df: 模型比较结果DataFrame
    """
    print("\n开始比较不同模型性能...")

    # 首先清洗数据中的异常值
    print("\n清洗数据中的异常值...")
    features = clean_data(features)
    
    # 创建结果保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 模型比较结果
    comparison_results = []
    
    # 测试数据字典，用于后续可视化
    test_data_dict = {}

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state, stratify=labels
    )
    
    # 1. LightGBM模型 - 原始特征
    print("\n1. 训练LightGBM模型（原始特征）")
    lgbm_params = LIGHTGBM_PARAMS.copy()
    lgbm_params['random_state'] = random_state
    lgbm_params['early_stopping_rounds'] = 20
    
    _, _, lgbm_metrics, lgbm_test_data = train_lightgbm_model(
        features, labels, feature_names,
        params=lgbm_params,
        save_dir=os.path.join(save_dir, 'lightgbm')
    )
    test_data_dict['LightGBM'] = lgbm_test_data
    
    lgbm_result = {
        'model': 'LightGBM',
        'accuracy': lgbm_metrics['test']['accuracy'],
        'precision': lgbm_metrics['test']['precision'],
        'recall': lgbm_metrics['test']['recall'],
        'f1': lgbm_metrics['test']['f1'],
        'auc': lgbm_metrics['test']['auc'],
        'training_time': 0  # 已经在train_lightgbm_model中记录
    }
    comparison_results.append(lgbm_result)
    
    # 2. LightGBM模型 - PCA降维
    print("\n2. 训练LightGBM模型（PCA降维，保留10个主成分）")
    _, pca_metrics, pca_test_data, _ = train_pca_lightgbm_model(
        features, labels, feature_names,
        n_components=10,
        params=lgbm_params,
        save_dir=os.path.join(save_dir, 'lightgbm_pca')
    )
    test_data_dict['LightGBM+PCA'] = pca_test_data
    
    pca_result = {
        'model': 'LightGBM+PCA',
        'accuracy': pca_metrics['test']['accuracy'],
        'precision': pca_metrics['test']['precision'],
        'recall': pca_metrics['test']['recall'],
        'f1': pca_metrics['test']['f1'],
        'auc': pca_metrics['test']['auc'],
        'training_time': 0  # 已经在train_pca_lightgbm_model中记录
    }
    comparison_results.append(pca_result)
    
    # 3. XGBoost模型
    print("\n3. 训练XGBoost模型")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state, stratify=labels
    )
    
    # XGBoost参数
    xgb_params = XGBOOST_PARAMS.copy()
    xgb_params['random_state'] = random_state
    
    start_time = time.time()
    
    # 训练XGBoost模型
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=20,
        verbose=True
    )
    
    xgb_train_time = time.time() - start_time
    
    # 预测
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    xgb_binary_preds = (xgb_preds > 0.5).astype(int)
    test_data_dict['XGBoost'] = (X_test, y_test, xgb_preds)
    
    # 评估指标
    xgb_accuracy = accuracy_score(y_test, xgb_binary_preds)
    xgb_precision = precision_score(y_test, xgb_binary_preds)
    xgb_recall = recall_score(y_test, xgb_binary_preds)
    xgb_f1 = f1_score(y_test, xgb_binary_preds)
    xgb_auc = roc_auc_score(y_test, xgb_preds)
    
    # 保存模型和结果
    with open(os.path.join(save_dir, 'xgboost', 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    
    if not os.path.exists(os.path.join(save_dir, 'xgboost')):
        os.makedirs(os.path.join(save_dir, 'xgboost'))
    
    pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
        'value': [xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_auc]
    }).to_csv(os.path.join(save_dir, 'xgboost', 'xgb_metrics.csv'), index=False)
    
    xgb_result = {
        'model': 'XGBoost',
        'accuracy': xgb_accuracy,
        'precision': xgb_precision,
        'recall': xgb_recall,
        'f1': xgb_f1,
        'auc': xgb_auc,
        'training_time': xgb_train_time
    }
    comparison_results.append(xgb_result)
    
    # 4. 随机森林模型
    print("\n4. 训练随机森林模型")
    
    # 随机森林参数
    rf_params = RANDOM_FOREST_PARAMS.copy()
    rf_params['random_state'] = random_state
    
    start_time = time.time()
    
    # 训练随机森林模型
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    
    rf_train_time = time.time() - start_time
    
    # 预测
    rf_preds = rf_model.predict_proba(X_test)[:, 1]
    rf_binary_preds = (rf_preds > 0.5).astype(int)
    test_data_dict['RandomForest'] = (X_test, y_test, rf_preds)
    
    # 评估指标
    rf_accuracy = accuracy_score(y_test, rf_binary_preds)
    rf_precision = precision_score(y_test, rf_binary_preds)
    rf_recall = recall_score(y_test, rf_binary_preds)
    rf_f1 = f1_score(y_test, rf_binary_preds)
    rf_auc = roc_auc_score(y_test, rf_preds)
    
    # 保存模型和结果
    if not os.path.exists(os.path.join(save_dir, 'random_forest')):
        os.makedirs(os.path.join(save_dir, 'random_forest'))
    
    with open(os.path.join(save_dir, 'random_forest', 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)
    
    pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
        'value': [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc]
    }).to_csv(os.path.join(save_dir, 'random_forest', 'rf_metrics.csv'), index=False)
    
    rf_result = {
        'model': 'RandomForest',
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1': rf_f1,
        'auc': rf_auc,
        'training_time': rf_train_time
    }
    comparison_results.append(rf_result)
    
    # 创建比较结果DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # 保存比较结果
    comparison_df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    print("\n模型比较结果:")
    print(comparison_df)
    
    return comparison_df, test_data_dict


def analyze_feature_robustness(model, X, y, feature_names, perturbation_ratio=0.1, 
                              n_runs=5, random_state=42, save_dir='./results'):
    """
    分析特征鲁棒性：在特征扰动下评估模型性能
    
    参数:
        model: 训练好的模型
        X: 特征矩阵
        y: 标签向量
        feature_names: 特征名称列表
        perturbation_ratio: 扰动比例
        n_runs: 运行次数
        random_state: 随机种子
        save_dir: 结果保存目录
        
    返回:
        robustness_df: 鲁棒性分析结果DataFrame
    """
    print("\n开始特征鲁棒性分析...")
    
    # 创建结果保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 检查模型类型并使用适当的预测方法
    if hasattr(model, 'predict_proba'):
        # 标准scikit-learn API模型
        print("使用predict_proba方法预测...")
        y_pred = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'predict'):
        # LightGBM Booster对象
        print("使用predict方法预测...")
        y_pred = model.predict(X)
        # 如果输出不是概率，尝试转换
        if np.max(y_pred) > 1.0 or np.min(y_pred) < 0.0:
            print("预测值不是概率分布，进行归一化处理...")
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid函数
    else:
        raise ValueError("模型没有可用的预测方法")
    
    # 基准性能（无扰动）
    baseline_auc = roc_auc_score(y, y_pred)
    baseline_accuracy = accuracy_score(y, (y_pred > 0.5).astype(int))
    
    print(f"基准性能: AUC = {baseline_auc:.4f}, Accuracy = {baseline_accuracy:.4f}")
    
    # 扰动分析结果
    robustness_results = []
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 对每次运行进行特征扰动分析
    for run in range(n_runs):
        print(f"运行 {run + 1}/{n_runs}...")
        
        # 创建扰动数据
        X_perturbed = X.copy()
        
        # 随机选择特征进行扰动
        n_features_to_perturb = int(X.shape[1] * perturbation_ratio)
        perturb_indices = np.random.choice(X.shape[1], n_features_to_perturb, replace=False)
        
        # 对选定特征进行高斯噪声扰动
        for idx in perturb_indices:
            feature_std = np.std(X[:, idx])
            noise = np.random.normal(0, feature_std * 0.5, X.shape[0])
            X_perturbed[:, idx] += noise
        
        # 预测和评估（使用与模型类型对应的预测方法）
        if hasattr(model, 'predict_proba'):
            y_pred_perturbed = model.predict_proba(X_perturbed)[:, 1]
        elif hasattr(model, 'predict'):
            y_pred_perturbed = model.predict(X_perturbed)
            # 如果输出不是概率，尝试转换
            if np.max(y_pred_perturbed) > 1.0 or np.min(y_pred_perturbed) < 0.0:
                y_pred_perturbed = 1.0 / (1.0 + np.exp(-y_pred_perturbed))  # sigmoid函数
        
        
        perturbed_auc = roc_auc_score(y, y_pred_perturbed)
        perturbed_accuracy = accuracy_score(y, (y_pred_perturbed > 0.5).astype(int))
        
        # 记录结果
        result = {
            'run': run + 1,
            'auc': perturbed_auc,
            'accuracy': perturbed_accuracy,
            'auc_drop': baseline_auc - perturbed_auc,
            'accuracy_drop': baseline_accuracy - perturbed_accuracy
        }
        robustness_results.append(result)
    
    # 创建结果DataFrame
    robustness_df = pd.DataFrame(robustness_results)
    
    # 计算平均值
    avg_result = {
        'run': 'Average',
        'auc': robustness_df['auc'].mean(),
        'accuracy': robustness_df['accuracy'].mean(),
        'auc_drop': robustness_df['auc_drop'].mean(),
        'accuracy_drop': robustness_df['accuracy_drop'].mean()
    }
    robustness_df = pd.concat([robustness_df, pd.DataFrame([avg_result])], ignore_index=True)
    
    # 保存结果
    robustness_df.to_csv(os.path.join(save_dir, 'feature_robustness.csv'), index=False)
    
    print("\n特征鲁棒性分析结果:")
    print(robustness_df)
    print(f"平均AUC下降: {avg_result['auc_drop']:.4f}")
    print(f"平均Accuracy下降: {avg_result['accuracy_drop']:.4f}")
    
    return robustness_df


def measure_inference_time(models_dict, X, n_repeats=100, save_dir='./results'):
    """
    测量模型推理时间
    
    参数:
        models_dict: 模型字典
        X: 特征矩阵
        n_repeats: 重复次数
        save_dir: 结果保存目录
        
    返回:
        inference_time_df: 推理时间结果DataFrame
    """
    print("\n开始测量模型推理时间...")
    
    # 创建结果保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 记录推理时间结果
    inference_time_results = []
    
    # 数据清洗
    print("清洗输入数据...")
    X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    
    # 测量每个模型的推理时间
    for model_name, model in models_dict.items():
        print(f"\n测量 {model_name} 模型的推理时间...")
        
        # 检查模型类型
        has_predict_proba = hasattr(model, 'predict_proba')
        has_predict = hasattr(model, 'predict')
        
        if not (has_predict_proba or has_predict):
            print(f"警告：{model_name} 模型没有predict_proba或predict方法，跳过")
            continue
        
        # 预测函数封装
        def predict_fn(X_data):
            if has_predict_proba:
                return model.predict_proba(X_data)[:, 1]
            else:
                return model.predict(X_data)
        
        # 预热
        _ = predict_fn(X[:10])
        
        # 测量单个样本的推理时间
        single_sample_times = []
        for i in range(min(100, X.shape[0])):
            start_time = time.time()
            _ = predict_fn(X[i:i+1])
            single_sample_times.append((time.time() - start_time) * 1000)  # 转换为毫秒
        
        # 测量批量推理时间
        batch_times = []
        for _ in range(n_repeats):
            start_time = time.time()
            _ = predict_fn(X)
            batch_times.append((time.time() - start_time) * 1000 / X.shape[0])  # 平均到每个样本，转换为毫秒
        
        # 记录结果
        result = {
            'model': model_name,
            'single_sample_avg_ms': np.mean(single_sample_times),
            'single_sample_std_ms': np.std(single_sample_times),
            'batch_avg_ms': np.mean(batch_times),
            'batch_std_ms': np.std(batch_times),
            'model_size_mb': get_model_size(model)
        }
        inference_time_results.append(result)
    
    # 创建推理时间结果DataFrame
    inference_time_df = pd.DataFrame(inference_time_results)
    
    # 保存结果
    inference_time_df.to_csv(os.path.join(save_dir, 'inference_time.csv'), index=False)
    
    print("\n模型推理时间测量完成，结果已保存")
    
    return inference_time_df


def get_model_size(model):
    """
    估算模型大小（MB）
    
    参数:
        model: 训练好的模型
        
    返回:
        size_mb: 模型大小（MB）
    """
    import sys
    import io
    
    model_bytes = io.BytesIO()
    pickle.dump(model, model_bytes)
    size_mb = sys.getsizeof(model_bytes.getvalue()) / (1024 * 1024)
    
    return round(size_mb, 2) 


def clean_data(features, verbose=True):
    """
    清洗数据中的异常值（NaN、无穷大和极端值）
    
    参数:
        features: 特征矩阵
        verbose: 是否打印详细信息
        
    返回:
        clean_features: 清洗后的特征矩阵
    """
    # 检测异常值
    has_nan = np.isnan(features).any()
    has_inf = np.isinf(features).any()
    
    if verbose:
        print(f"数据中包含NaN: {has_nan}")
        print(f"数据中包含无穷值: {has_inf}")
    
    if has_nan or has_inf:
        if verbose:
            print("正在清洗数据中的异常值...")
        
        # 创建数据副本以避免修改原始数据
        features_clean = features.copy()
        
        # 替换NaN为0
        if has_nan:
            nan_count = np.isnan(features_clean).sum()
            if verbose:
                print(f"替换 {nan_count} 个NaN值为0")
            features_clean = np.nan_to_num(features_clean, nan=0.0)
        
        # 替换无穷值
        if has_inf:
            inf_count = np.isinf(features_clean).sum()
            if verbose:
                print(f"替换 {inf_count} 个无穷值")
            features_clean = np.nan_to_num(features_clean, posinf=1.0e10, neginf=-1.0e10)
        
        # 检查是否还有异常值
        if np.isnan(features_clean).any() or np.isinf(features_clean).any():
            if verbose:
                print("警告：清洗后仍存在异常值！")
        
        # 检查数据范围
        if verbose:
            max_val = np.max(features_clean)
            min_val = np.min(features_clean)
            print(f"清洗后数据范围: [{min_val}, {max_val}]")
        
        return features_clean
    else:
        if verbose:
            print("数据中没有异常值，无需清洗")
        return features