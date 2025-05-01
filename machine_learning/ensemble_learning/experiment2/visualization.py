import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
    ablation_subset = ablation_df[ablation_df['model_name'] != 'full_model'].copy()
    
    # 排序，使差异大的在前面
    ablation_subset = ablation_subset.sort_values(by='auc_diff', ascending=False)
    
    # 提取出完整模型的性能指标
    full_model_metrics = ablation_df[ablation_df['model_name'] == 'full_model'].iloc[0]
    full_model_auc = full_model_metrics['auc']
    full_model_f1 = full_model_metrics['f1']
    
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
    model_names.append('full_model')  # 添加完整模型
    
    # 找到完整模型的行
    full_model_row = ablation_df[ablation_df['model_name'] == 'full_model'].iloc[0]
    
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
        model_name = model[1]['removed_model'] if not pd.isna(model[1]['removed_model']) else 'full_model'
        values = [model[1][metric] for metric in metrics]
        values += values[:1]  # 闭合图形
        
        if model_name == 'full_model':
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
    
    # 排序，使差异大的在前面
    feature_subset = feature_subset.sort_values(by='auc_diff', ascending=False)
    
    # 提取出完整特征的性能指标
    full_feature_metrics = feature_ablation_df[feature_ablation_df['feature_group'] == 'full_features'].iloc[0]
    full_feature_auc = full_feature_metrics['auc']
    full_feature_f1 = full_feature_metrics['f1']
    
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
                 ha='center', va='bottom', rotation=0)
    
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
                 ha='center', va='bottom', rotation=0)
    
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
    
    # 3. 绘制特征维度与性能的关系图
    plt.figure(figsize=(10, 6))
    
    full_features_row = feature_ablation_df[feature_ablation_df['feature_group'] == 'full_features'].iloc[0]
    feature_subset_with_full = pd.concat([feature_subset, pd.DataFrame([full_features_row])])
    
    # 按特征维度排序
    feature_subset_with_full = feature_subset_with_full.sort_values(by='feature_dim')
    
    # 提取特征组名称，对于完整特征使用"全部特征"
    feature_names = []
    for name in feature_subset_with_full['feature_group']:
        if name == 'full_features':
            feature_names.append('全部特征')
        else:
            feature_names.append(name.replace('without_', '去除'))
    
    plt.plot(feature_subset_with_full['feature_dim'], feature_subset_with_full['auc'], 'o-', label='AUC')
    plt.plot(feature_subset_with_full['feature_dim'], feature_subset_with_full['f1'], 's-', label='F1分数')
    
    # 添加数据标签
    for i, (dim, auc, f1, name) in enumerate(zip(feature_subset_with_full['feature_dim'], 
                                               feature_subset_with_full['auc'],
                                               feature_subset_with_full['f1'],
                                               feature_names)):
        plt.annotate(f'{name}\n维度:{dim}', 
                   (dim, auc), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center')
    
    plt.xlabel('特征维度')
    plt.ylabel('性能指标')
    plt.title('特征维度与模型性能的关系')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'feature_dimension_performance.png'), dpi=300)
    plt.close()

def plot_stacking_vs_base_models(model_ablation_df, base_model_df, save_dir):
    """
    绘制Stacking模型与基模型性能对比图
    
    参数:
        model_ablation_df: DataFrame，包含模型消融实验结果
        base_model_df: DataFrame，包含基模型性能
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取Stacking模型性能
    stacking_metrics = model_ablation_df[model_ablation_df['model_name'] == 'full_model'].iloc[0]
    
    # 1. 绘制AUC对比图
    plt.figure(figsize=(14, 8))
    
    # 按AUC排序
    base_models_sorted = base_model_df.sort_values(by='auc', ascending=True)
    
    # 提取模型名称和AUC值
    models = list(base_models_sorted['model'])
    models.append('Stacking')
    
    auc_values = list(base_models_sorted['auc'])
    auc_values.append(stacking_metrics['auc'])
    
    # 确定颜色，Stacking模型使用不同颜色
    colors = ['skyblue'] * len(base_models_sorted) + ['red']
    
    # 绘制水平条形图
    y_pos = np.arange(len(models))
    bars = plt.barh(y_pos, auc_values, color=colors)
    
    # 添加数据标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}',
                 ha='left', va='center')
    
    plt.yticks(y_pos, models)
    plt.xlabel('AUC值')
    plt.title('Stacking模型与各基模型的AUC对比')
    plt.xlim(0.75, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'stacking_vs_base_models_auc.png'), dpi=300)
    plt.close()
    
    # 2. 绘制F1分数对比图
    plt.figure(figsize=(14, 8))
    
    # 按F1分数排序
    base_models_sorted = base_model_df.sort_values(by='f1', ascending=True)
    
    # 提取模型名称和F1值
    models = list(base_models_sorted['model'])
    models.append('Stacking')
    
    f1_values = list(base_models_sorted['f1'])
    f1_values.append(stacking_metrics['f1'])
    
    # 绘制水平条形图
    y_pos = np.arange(len(models))
    bars = plt.barh(y_pos, f1_values, color=colors)
    
    # 添加数据标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}',
                 ha='left', va='center')
    
    plt.yticks(y_pos, models)
    plt.xlabel('F1分数')
    plt.title('Stacking模型与各基模型的F1分数对比')
    plt.xlim(0.75, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'stacking_vs_base_models_f1.png'), dpi=300)
    plt.close()
    
    # 3. 绘制综合性能提升图
    plt.figure(figsize=(12, 8))
    
    # 计算Stacking相对于基模型的性能提升
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # 获取基模型的平均性能
    base_avg = {}
    for metric in metrics:
        base_avg[metric] = base_model_df[metric].mean()
    
    # 计算提升比例
    improvements = {}
    for metric in metrics:
        improvements[metric] = (stacking_metrics[metric] - base_avg[metric]) / base_avg[metric] * 100
    
    # 绘制条形图
    x_pos = np.arange(len(metrics))
    bars = plt.bar(x_pos, [improvements[m] for m in metrics], color='lightgreen')
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom')
    
    plt.xticks(x_pos, metrics)
    plt.ylabel('提升百分比 (%)')
    plt.title('Stacking模型相对于基模型平均性能的提升')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'stacking_improvement.png'), dpi=300)
    plt.close()

def create_all_visualizations(result_dir='./results', save_dir='./figures'):
    """
    创建所有可视化图表
    
    参数:
        result_dir: 结果目录，包含消融实验的CSV文件
        save_dir: 保存可视化图表的目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载模型消融实验结果
    model_ablation_path = os.path.join(result_dir, 'model_ablation_results.csv')
    if os.path.exists(model_ablation_path):
        model_ablation_df = pd.read_csv(model_ablation_path)
        plot_model_ablation_results(model_ablation_df, save_dir)
    else:
        print(f"找不到模型消融实验结果: {model_ablation_path}")
    
    # 加载基模型性能结果
    base_model_path = os.path.join(result_dir, 'base_model_performance.csv')
    if os.path.exists(base_model_path):
        base_model_df = pd.read_csv(base_model_path)
        plot_base_model_performance(base_model_df, save_dir)
        
        # 绘制Stacking与基模型的对比图
        if os.path.exists(model_ablation_path):
            plot_stacking_vs_base_models(model_ablation_df, base_model_df, save_dir)
    else:
        print(f"找不到基模型性能结果: {base_model_path}")
    
    # 加载特征消融实验结果
    feature_ablation_path = os.path.join(result_dir, 'feature_ablation_results.csv')
    if os.path.exists(feature_ablation_path):
        feature_ablation_df = pd.read_csv(feature_ablation_path)
        plot_feature_ablation_results(feature_ablation_df, save_dir)
    else:
        print(f"找不到特征消融实验结果: {feature_ablation_path}")


if __name__ == "__main__":
    # 创建所有可视化图表
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    
    create_all_visualizations(result_dir, save_dir) 