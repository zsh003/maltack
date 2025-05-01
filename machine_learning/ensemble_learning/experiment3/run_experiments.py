import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import pickle

from lgbm_model import load_data, train_lightgbm_model, train_pca_lightgbm_model
from feature_analysis import (plot_feature_importance, plot_roc_and_pr_curves, 
                              plot_model_comparison, plot_pca_analysis, 
                              plot_data_distribution)
from model_evaluation import (compare_models, analyze_feature_robustness, 
                             measure_inference_time)
from constants import create_directories


def run_all_experiments(result_dir='./experiment3/results', figure_dir='./experiment3/figures'):
    """
    运行所有实验
    
    参数:
        result_dir: 结果保存目录
        figure_dir: 图表保存目录
    """
    # 开始时间
    start_time = time.time()
    
    # 创建目录
    create_directories()
    for dir_path in [result_dir, figure_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # 1. 加载数据
    print("=" * 80)
    print("1. 加载数据")
    print("=" * 80)
    features, labels, feature_names = load_data()
    
    # 2. 训练基础LightGBM模型
    print("\n" + "=" * 80)
    print("2. 训练基础LightGBM模型")
    print("=" * 80)
    models, feature_importance_df, metrics_results, test_data = train_lightgbm_model(
        features, labels, feature_names, 
        save_dir=os.path.join(result_dir, 'lightgbm')
    )
    X_test, y_test, y_pred = test_data
    
    # 3. 绘制基础模型的特征重要性和性能图表
    print("\n" + "=" * 80)
    print("3. 绘制基础模型的特征重要性和性能图表")
    print("=" * 80)
    # 绘制特征重要性
    plot_feature_importance(
        feature_importance_df, 
        save_dir=figure_dir, 
        top_n=20
    )
    
    # 绘制ROC和PR曲线
    plot_roc_and_pr_curves(
        y_test, y_pred, 
        save_dir=figure_dir
    )
    
    # 绘制数据分布
    plot_data_distribution(
        features, labels, feature_names, 
        save_dir=figure_dir, 
        top_n=5
    )
    
    # 4. 训练PCA降维的LightGBM模型并比较
    print("\n" + "=" * 80)
    print("4. 训练PCA降维的LightGBM模型并比较")
    print("=" * 80)
    pca_models, pca_metrics, pca_test_data, pca = train_pca_lightgbm_model(
        features, labels, feature_names, 
        n_components=10, 
        save_dir=os.path.join(result_dir, 'lightgbm_pca')
    )
    
    # 绘制PCA分析图
    plot_pca_analysis(
        pca, feature_names, 
        save_dir=figure_dir
    )
    
    # 5. 比较不同模型性能
    print("\n" + "=" * 80)
    print("5. 比较不同模型性能")
    print("=" * 80)
    comparison_df, test_data_dict = compare_models(
        features, labels, feature_names, 
        save_dir=result_dir
    )
    
    # 绘制模型比较图
    plot_model_comparison(
        comparison_df, 
        save_dir=figure_dir
    )
    
    # 6. 分析模型鲁棒性和推理时间
    print("\n" + "=" * 80)
    print("6. 分析模型鲁棒性和推理时间")
    print("=" * 80)
    
    # 获取第一个LightGBM模型进行鲁棒性分析
    lgbm_model = models[0]
    
    # 分析特征鲁棒性
    robustness_df = analyze_feature_robustness(
        lgbm_model, X_test, y_test, feature_names, 
        perturbation_ratio=0.1, 
        n_runs=5, 
        save_dir=os.path.join(result_dir, 'robustness')
    )
    
    # 构建模型字典用于推理时间测量
    models_dict = {
        'LightGBM': lgbm_model
    }
    
    # 读取其他模型（如果存在）
    xgb_path = os.path.join(result_dir, 'xgboost', 'xgb_model.pkl')
    rf_path = os.path.join(result_dir, 'random_forest', 'rf_model.pkl')
    
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            models_dict['XGBoost'] = pickle.load(f)
    
    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            models_dict['RandomForest'] = pickle.load(f)
    
    # 测量推理时间
    inference_time_df = measure_inference_time(
        models_dict, X_test, 
        n_repeats=50, 
        save_dir=os.path.join(result_dir, 'inference_time')
    )
    
    # 计算总运行时间
    total_time = time.time() - start_time
    print(f"\n所有实验完成！总运行时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    # 返回结果字典
    return {
        'feature_importance': feature_importance_df,
        'metrics': metrics_results,
        'comparison': comparison_df,
        'robustness': robustness_df,
        'inference_time': inference_time_df
    }


def main(args):
    """
    主函数
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 设置结果目录
    result_dir = args.result_dir
    figure_dir = args.figure_dir
    
    print(f"结果将保存到: {result_dir}")
    print(f"图表将保存到: {figure_dir}")
    
    # 运行所有实验
    if not args.skip_experiments:
        print("\n" + "=" * 80)
        print("开始运行所有实验")
        print("=" * 80)
        results = run_all_experiments(result_dir, figure_dir)
    else:
        print("\n跳过实验，仅生成图表...")
        # 此处可以添加仅生成图表的代码
    
    print("\n所有任务完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行PE恶意软件检测的LightGBM特征工程实验')
    parser.add_argument('--result-dir', type=str, default='./experiment3/results', help='结果保存目录')
    parser.add_argument('--figure-dir', type=str, default='./experiment3/figures', help='图表保存目录')
    parser.add_argument('--skip-experiments', action='store_true', help='跳过实验，仅生成图表')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    main(args) 