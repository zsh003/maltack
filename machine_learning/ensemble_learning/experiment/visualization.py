import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
import tensorflow as tf

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_model_metrics(result_dir):
    """加载模型评估指标"""
    metrics_files = [
        os.path.join(result_dir, 'basic_cnn_model_metrics.pkl'),
        os.path.join(result_dir, 'multibranch_cnn_model_metrics.pkl'),
        os.path.join(result_dir, 'enhanced_cnn_model_metrics.pkl')
    ]
    
    all_metrics = []
    for file_path in metrics_files:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                metrics = pickle.load(f)
                all_metrics.append(metrics)
    
    return all_metrics

def load_models(result_dir):
    """加载已训练的模型"""
    model_files = [
        os.path.join(result_dir, 'basic_cnn_model.h5'),
        os.path.join(result_dir, 'multibranch_cnn_model.h5'),
        os.path.join(result_dir, 'enhanced_cnn_model.h5')
    ]
    
    models = []
    for file_path in model_files:
        if os.path.exists(file_path):
            model = tf.keras.models.load_model(file_path)
            models.append(model)
    
    return models

def plot_radar_chart(all_metrics, model_names, save_path=None):
    """绘制雷达图比较多个模型的性能"""
    # 提取性能指标
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    display_names = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
    
    # 准备数据
    metrics_values = []
    for metrics in all_metrics:
        values = [metrics[name] for name in metrics_names]
        metrics_values.append(values)
    
    # 雷达图设置
    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合雷达图
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 绘制每个模型的雷达图
    for i, values in enumerate(metrics_values):
        values = np.concatenate((values, [values[0]]))  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, label=model_names[i])
        ax.fill(angles, values, alpha=0.1)
    
    # 设置刻度标签
    ax.set_thetagrids(angles[:-1] * 180/np.pi, display_names)
    
    # 设置雷达图范围
    ax.set_ylim(0.8, 1.0)  # 调整范围以更好地显示差异
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('模型性能雷达图对比')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"雷达图已保存到: {save_path}")
    
    plt.close()

def plot_precision_recall_curves(models, test_data, model_names, save_path=None):
    """绘制精确率-召回率曲线"""
    plt.figure(figsize=(10, 8))
    
    test_features, test_label = test_data
    
    # 为每个模型绘制PR曲线
    for i, (model, name) in enumerate(zip(models, model_names)):
        # 获取预测概率
        y_pred_prob = model.predict(test_features).flatten()
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(test_label, y_pred_prob)
        average_precision = average_precision_score(test_label, y_pred_prob)
        
        # 绘制PR曲线
        plt.plot(recall, precision, lw=2, label=f'{name} (AP = {average_precision:.3f})')
    
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线对比')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"PR曲线图已保存到: {save_path}")
    
    plt.close()

def plot_features_tsne(models, test_data, model_names, save_path=None):
    """使用t-SNE可视化模型提取的特征分布"""
    test_features, test_label = test_data
    
    # 创建图形
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    
    if len(models) == 1:
        axes = [axes]
    
    # 为每个模型可视化特征
    for i, (model, name) in enumerate(zip(models, model_names)):
        # 创建一个新的模型，只保留到倒数第二层
        feature_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
        
        # 提取特征
        features = feature_model.predict(test_features)
        
        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        # 绘制散点图
        for class_id in np.unique(test_label):
            idx = test_label == class_id
            label = '恶意样本' if class_id == 1 else '良性样本'
            axes[i].scatter(
                features_tsne[idx, 0], 
                features_tsne[idx, 1], 
                alpha=0.5,
                label=label
            )
        
        axes[i].set_title(f'{name}特征分布')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"特征分布图已保存到: {save_path}")
    
    plt.close()

def plot_learning_curves_detailed(history_files, model_names, save_path=None):
    """绘制详细的学习曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # 加载训练历史
    histories = []
    for file_path in history_files:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                history = pickle.load(f)
                histories.append(history)
    
    # 绘制损失曲线
    for history, name in zip(histories, model_names):
        axes[0, 0].plot(history['loss'], label=f'{name} 训练')
        axes[0, 0].plot(history['val_loss'], '--', label=f'{name} 验证')
    
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 绘制准确率曲线
    for history, name in zip(histories, model_names):
        axes[0, 1].plot(history['accuracy'], label=f'{name} 训练')
        axes[0, 1].plot(history['val_accuracy'], '--', label=f'{name} 验证')
    
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 绘制AUC曲线
    for history, name in zip(histories, model_names):
        if 'auc' in history:
            axes[1, 0].plot(history['auc'], label=f'{name} 训练')
            axes[1, 0].plot(history['val_auc'], '--', label=f'{name} 验证')
    
    axes[1, 0].set_title('AUC曲线')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 绘制学习率曲线
    for history, name in zip(histories, model_names):
        if 'lr' in history:
            axes[1, 1].plot(history['lr'], label=f'{name}')
    
    axes[1, 1].set_title('学习率曲线')
    axes[1, 1].set_ylabel('学习率')
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"详细学习曲线已保存到: {save_path}")
    
    plt.close()

def plot_model_comparison_bar(all_metrics, model_names, save_path=None):
    """绘制模型性能对比条形图"""
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    display_names = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
    
    # 准备数据
    data = []
    for i, metrics in enumerate(all_metrics):
        for metric_name, display_name in zip(metrics_names, display_names):
            data.append({
                '模型': model_names[i],
                '指标': display_name,
                '值': metrics[metric_name]
            })
    
    df = pd.DataFrame(data)
    
    # 绘制条形图
    plt.figure(figsize=(12, 8))
    bar_plot = sns.barplot(x='指标', y='值', hue='模型', data=df)
    
    # 在条形上添加数值标签
    for p in bar_plot.patches:
        bar_plot.annotate(
            f'{p.get_height():.3f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext=(0, 9),
            textcoords='offset points'
        )
    
    plt.title('模型性能对比')
    plt.ylim(0.75, 1.0)  # 调整范围以更好地显示差异
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"模型对比条形图已保存到: {save_path}")
    
    plt.close()

def create_detailed_visualizations(test_data, result_dir):
    """创建详细的可视化分析"""
    # 确保结果目录存在
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 模型名称
    model_names = ['基础CNN模型', '多分支CNN模型', '增强训练CNN模型']
    
    # 加载模型评估指标
    all_metrics = load_model_metrics(result_dir)
    
    # 加载模型
    models = load_models(result_dir)
    
    # 生成雷达图
    if len(all_metrics) > 0:
        plot_radar_chart(
            all_metrics, 
            model_names[:len(all_metrics)], 
            os.path.join(result_dir, 'radar_chart_comparison.png')
        )
    
    # 绘制精确率-召回率曲线
    if len(models) > 0 and test_data is not None:
        plot_precision_recall_curves(
            models, 
            test_data, 
            model_names[:len(models)], 
            os.path.join(result_dir, 'pr_curves_comparison.png')
        )
    
    # 绘制特征分布
    if len(models) > 0 and test_data is not None:
        plot_features_tsne(
            models, 
            test_data, 
            model_names[:len(models)], 
            os.path.join(result_dir, 'feature_distribution_tsne.png')
        )
    
    # 绘制性能对比条形图
    if len(all_metrics) > 0:
        plot_model_comparison_bar(
            all_metrics, 
            model_names[:len(all_metrics)], 
            os.path.join(result_dir, 'performance_bar_comparison.png')
        )
    
    print("详细可视化分析已保存到:", result_dir)

if __name__ == "__main__":
    # 如果需要独立运行visualization.py，需要先导入必要的模块并加载数据
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from enhanced_cnn_model import load_data
    
    # 设置结果目录
    RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    
    # 加载测试数据
    _, _, test_data = load_data()
    
    # 创建详细可视化
    create_detailed_visualizations(test_data, RESULT_DIR) 