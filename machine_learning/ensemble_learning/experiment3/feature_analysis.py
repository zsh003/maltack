import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plot_feature_importance(feature_importance_df, save_dir='./figures', top_n=20):
    """
    绘制特征重要性图
    
    参数:
        feature_importance_df: 特征重要性DataFrame
        save_dir: 保存目录
        top_n: 展示前n个重要特征
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 计算每个特征的平均重要性
    mean_importance = feature_importance_df.groupby('feature')['importance'].mean()
    mean_importance = mean_importance.reset_index()
    mean_importance = mean_importance.sort_values(by='importance', ascending=False)
    
    # 保存特征重要性排名
    mean_importance.to_csv(os.path.join(save_dir, 'mean_feature_importance.csv'), index=False)
    
    # 绘制前N个特征的重要性
    top_features = mean_importance.head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'前{top_n}个重要特征')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'top_{top_n}_feature_importance.png'), dpi=300)
    plt.close()
    
    # 绘制特征重要性分布
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='feature', y='importance', data=feature_importance_df[
        feature_importance_df['feature'].isin(top_features['feature'])
    ])
    plt.xticks(rotation=90)
    plt.title(f'前{top_n}个特征的重要性分布（交叉验证）')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'top_{top_n}_feature_importance_boxplot.png'), dpi=300)
    plt.close()
    
    print(f"Top {top_n} 特征重要性图表已保存")


def plot_roc_and_pr_curves(y_true, y_pred, save_dir='./figures'):
    """
    绘制ROC和PR曲线
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率(FPR)')
    plt.ylabel('真正率(TPR)')
    plt.title('接收者操作特征曲线(ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 绘制PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AUC = {pr_auc:.3f})')
    plt.xlabel('召回率(Recall)')
    plt.ylabel('精确率(Precision)')
    plt.title('精确率-召回率曲线(PR)')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300)
    plt.close()
    
    print("ROC和PR曲线图表已保存")


def plot_model_comparison(metrics_df, save_dir='./figures'):
    """
    绘制不同模型性能比较图
    
    参数:
        metrics_df: 包含不同模型性能指标的DataFrame
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 转换为适合绘图的格式
    plt.figure(figsize=(14, 8))
    
    # 每个指标一个子图
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    bar_width = 0.35
    index = np.arange(len(metrics_df['model']))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        bars = plt.bar(index, metrics_df[metric], bar_width, label=metric)
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', rotation=0)
        
        plt.xlabel('模型')
        plt.ylabel(metric)
        plt.title(f'{metric} 指标比较')
        plt.xticks(index, metrics_df['model'], rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # 绘制雷达图
    plt.figure(figsize=(10, 8))
    
    # 准备雷达图数据
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 添加轴标签
    plt.xticks(angles[:-1], metrics)
    
    # 绘制每个模型的雷达图
    for i, row in metrics_df.iterrows():
        values = [row[metric] for metric in metrics]
        values += values[:1]  # 闭合图形
        
        plt.plot(angles, values, 'o-', linewidth=2, label=row['model'])
        plt.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right')
    plt.title('不同模型性能雷达图')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'model_comparison_radar.png'), dpi=300)
    plt.close()
    
    print("模型比较图表已保存")


def plot_pca_analysis(pca, feature_names, save_dir='./figures'):
    """
    绘制PCA分析图
    
    参数:
        pca: 训练好的PCA模型
        feature_names: 特征名称列表
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制解释方差比
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', label='累计解释方差')
    plt.xlabel('主成分')
    plt.ylabel('解释方差比')
    plt.title('各主成分解释方差')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'pca_explained_variance.png'), dpi=300)
    plt.close()
    
    # 绘制前两个主成分的特征权重
    plt.figure(figsize=(14, 14))
    
    # 选择前10个特征
    top_features = 10
    
    # 选择前2个主成分
    for i in range(2):
        plt.subplot(2, 1, i+1)
        # 获取特征重要性
        importance = pd.Series(np.abs(pca.components_[i]), index=feature_names)
        top_indices = importance.nlargest(top_features).index
        
        # 绘制柱状图
        sns.barplot(x=importance[top_indices], y=top_indices)
        plt.title(f'主成分 {i+1} 特征权重 (Top {top_features})')
        plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'pca_feature_weights.png'), dpi=300)
    plt.close()
    
    print("PCA分析图表已保存")


def plot_data_distribution(X, y, feature_names, save_dir='./figures', top_n=5):
    """
    绘制数据分布图
    
    参数:
        X: 特征矩阵
        y: 标签向量
        feature_names: 特征名称列表
        save_dir: 保存目录
        top_n: 展示前n个重要特征
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 将特征矩阵和标签转换为DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    # 绘制前n个特征的分布
    plt.figure(figsize=(15, top_n * 3))
    
    for i in range(min(top_n, len(feature_names))):
        plt.subplot(top_n, 1, i+1)
        sns.histplot(data=df, x=feature_names[i], hue='label', bins=30, kde=True, 
                    element='step', common_norm=False)
        plt.title(f'特征 "{feature_names[i]}" 在恶意和良性样本中的分布')
        plt.xlabel(feature_names[i])
        plt.ylabel('频率')
        plt.legend(['良性(0)', '恶意(1)'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'top_{top_n}_feature_distribution.png'), dpi=300)
    plt.close()
    
    # 使用t-SNE降维可视化
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], alpha=0.5, label='良性(0)')
    plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], alpha=0.5, label='恶意(1)')
    plt.title('t-SNE降维可视化')
    plt.xlabel('t-SNE特征1')
    plt.ylabel('t-SNE特征2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'tsne_visualization.png'), dpi=300)
    plt.close()
    
    print("数据分布图表已保存") 