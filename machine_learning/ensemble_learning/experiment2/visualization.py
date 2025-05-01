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
    
    # 3. 绘制雷达图比较各个特征组的性能
    # 准备数据
    feature_group_names = list(feature_subset['removed_feature'])
    feature_group_names.append('full_features')  # 添加完整特征
    
    # 找到完整特征的行
    full_feature_row = feature_ablation_df[feature_ablation_df['feature_group'] == 'full_features'].iloc[0]
    
    # 合并数据
    feature_with_full = pd.concat([feature_subset, pd.DataFrame([full_feature_row])])
    
    # 准备雷达图数据
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # 创建一个图表
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    # 添加轴标签
    plt.xticks(angles[:-1], metrics)
    
    # 绘制每个特征组的雷达图
    for idx, feature in enumerate(feature_with_full.iterrows()):
        feature_group = feature[1]['feature_group']
        feature_name = feature[1]['removed_feature'] if not pd.isna(feature[1]['removed_feature']) else 'full_features'
        values = [feature[1][metric] for metric in metrics]
        values += values[:1]  # 闭合图形
        
        if feature_group == 'full_features':
            ax.plot(angles, values, 'o-', linewidth=2, label=f'完整特征')
            ax.fill(angles, values, alpha=0.1)
        else:
            ax.plot(angles, values, 'o-', linewidth=1, alpha=0.7, label=f'移除{feature_name}')
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('不同特征组的性能雷达图')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'feature_ablation_radar.png'), dpi=300)
    plt.close()

def plot_stacking_vs_base_models(model_ablation_df, base_model_df, save_dir):
    """
    绘制Stacking与各个基模型性能对比图
    
    参数:
        model_ablation_df: DataFrame, 包含模型消融实验结果
        base_model_df: DataFrame, 包含基模型性能评估结果
        save_dir: 保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取完整集成模型的性能
    stacking_performance = model_ablation_df[model_ablation_df['model_config'] == 'full_ensemble'].iloc[0]
    
    # 准备比较数据
    models = list(base_model_df['model'])
    models.append('Stacking Ensemble')
    
    # 合并性能指标数据
    performance_data = []
    
    # 添加基模型性能
    for idx, row in base_model_df.iterrows():
        performance_data.append({
            'model': row['model'],
            'accuracy': row['accuracy'],
            'precision': row['precision'],
            'recall': row['recall'],
            'f1': row['f1'],
            'auc': row['auc']
        })
    
    # 添加Stacking模型性能
    performance_data.append({
        'model': 'Stacking Ensemble',
        'accuracy': stacking_performance['accuracy'],
        'precision': stacking_performance['precision'],
        'recall': stacking_performance['recall'],
        'f1': stacking_performance['f1'],
        'auc': stacking_performance['auc']
    })
    
    # 转换为DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # 按AUC降序排序
    performance_df = performance_df.sort_values(by='auc', ascending=False)
    
    # 1. 绘制AUC和F1分数条形图
    plt.figure(figsize=(16, 10))
    
    x = np.arange(len(performance_df))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, performance_df['auc'], width, label='AUC', color='skyblue')
    bars2 = plt.bar(x + width/2, performance_df['f1'], width, label='F1分数', color='lightgreen')
    
    # 高亮Stacking模型
    for i, model in enumerate(performance_df['model']):
        if model == 'Stacking Ensemble':
            bars1[i].set_color('navy')
            bars2[i].set_color('darkgreen')
    
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
    
    plt.xlabel('模型')
    plt.ylabel('性能指标')
    plt.title('Stacking集成与各基模型的AUC和F1分数对比')
    plt.xticks(x, performance_df['model'], rotation=45, ha='right')
    plt.ylim(0.75, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'stacking_vs_base_models.png'), dpi=300)
    plt.close()
    
    # 2. 绘制性能提升条形图
    plt.figure(figsize=(14, 8))
    
    # 计算Stacking相对于各基模型的性能提升
    stacking_auc = stacking_performance['auc']
    stacking_f1 = stacking_performance['f1']
    
    improvement_data = []
    
    for idx, row in base_model_df.iterrows():
        improvement_data.append({
            'model': row['model'],
            'auc_improvement': stacking_auc - row['auc'],
            'f1_improvement': stacking_f1 - row['f1']
        })
    
    # 转换为DataFrame
    improvement_df = pd.DataFrame(improvement_data)
    
    # 按AUC提升降序排序
    improvement_df = improvement_df.sort_values(by='auc_improvement', ascending=False)
    
    x = np.arange(len(improvement_df))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, improvement_df['auc_improvement'], width, label='AUC提升', color='lightcoral')
    bars2 = plt.bar(x + width/2, improvement_df['f1_improvement'], width, label='F1分数提升', color='lightsalmon')
    
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
    plt.ylabel('性能提升')
    plt.title('Stacking集成相对于各基模型的性能提升')
    plt.xticks(x, improvement_df['model'], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'stacking_improvement.png'), dpi=300)
    plt.close()

def create_all_visualizations(result_dir='./results', save_dir='./figures'):
    """
    创建所有可视化图表
    
    参数:
        result_dir: 结果目录
        save_dir: 图表保存目录
    """
    print("开始创建所有可视化图表...")
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 加载和绘制模型消融实验结果
    model_ablation_file = os.path.join(result_dir, 'model_ablation_results.csv')
    try:
        if os.path.exists(model_ablation_file):
            model_ablation_df = pd.read_csv(model_ablation_file)
            print(f"已加载模型消融实验结果，形状: {model_ablation_df.shape}, 列: {', '.join(model_ablation_df.columns)}")
            plot_model_ablation_results(model_ablation_df, save_dir)
            print("模型消融实验图表创建完成")
        else:
            print(f"未找到模型消融实验结果文件: {model_ablation_file}")
    except Exception as e:
        print(f"创建模型消融实验图表时出错: {str(e)}")
    
    # 2. 加载和绘制基模型性能结果
    base_model_file = os.path.join(result_dir, 'base_model_performance.csv')
    try:
        if os.path.exists(base_model_file):
            base_model_df = pd.read_csv(base_model_file)
            print(f"已加载基模型性能结果，形状: {base_model_df.shape}, 列: {', '.join(base_model_df.columns)}")
            plot_base_model_performance(base_model_df, save_dir)
            print("基模型性能图表创建完成")
        else:
            print(f"未找到基模型性能结果文件: {base_model_file}")
    except Exception as e:
        print(f"创建基模型性能图表时出错: {str(e)}")
    
    # 3. 加载和绘制特征消融实验结果
    feature_ablation_file = os.path.join(result_dir, 'feature_ablation_results.csv')
    try:
        if os.path.exists(feature_ablation_file):
            feature_ablation_df = pd.read_csv(feature_ablation_file)
            print(f"已加载特征消融实验结果，形状: {feature_ablation_df.shape}, 列: {', '.join(feature_ablation_df.columns)}")
            plot_feature_ablation_results(feature_ablation_df, save_dir)
            print("特征消融实验图表创建完成")
        else:
            print(f"未找到特征消融实验结果文件: {feature_ablation_file}")
    except Exception as e:
        print(f"创建特征消融实验图表时出错: {str(e)}")
    
    # 4. 绘制Stacking与基模型的对比图
    try:
        if os.path.exists(model_ablation_file) and os.path.exists(base_model_file):
            plot_stacking_vs_base_models(model_ablation_df, base_model_df, save_dir)
            print("Stacking与基模型对比图表创建完成")
        else:
            print("缺少绘制Stacking与基模型对比图所需的文件")
    except Exception as e:
        print(f"创建Stacking与基模型对比图表时出错: {str(e)}")
    
    print(f"所有可视化图表已保存到目录: {save_dir}")
    return True


if __name__ == "__main__":
    # 创建所有可视化图表
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    
    create_all_visualizations(result_dir, save_dir) 