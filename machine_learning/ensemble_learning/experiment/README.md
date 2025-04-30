# 基于多分支CNN的恶意PE识别模型对比实验

本实验基于"基于多分支CNN的恶意PE识别模型构建"文档，实现了三种CNN模型对比：

1. **基础CNN模型**：原有的双卷积层架构
2. **多分支CNN模型**：在基础模型上新增注意力分支和残差分支
3. **增强训练CNN模型**：在多分支模型基础上使用软标签、对抗训练和余弦退火学习率

## 项目结构

```
experiment/
├── enhanced_cnn_model.py  # 优化后的多分支CNN模型实现
├── comparative_experiment.py  # 对比实验实现
├── visualization.py  # 详细可视化分析
├── run_experiments.py  # 主运行脚本
├── results/  # 实验结果保存目录
└── README.md  # 本文件
```

## 实验内容

本实验通过三个模型比较不同设计的效果：

1. 网络结构优化：引入注意力机制(SE模块)和残差连接，提升特征捕获能力
2. 训练策略优化：软标签训练、对抗训练(FGSM)和余弦退火学习率调度
3. 性能对比：通过多种指标(准确率、AUC、F1分数等)对比三种模型的性能差异

## 运行方法

### 环境要求

- Python 3.7.16
- TensorFlow 2.3.0
- scikit-learn 0.23.1
- 详细依赖见项目根目录requirements.txt

### 运行完整实验

```bash
cd ensemble_learning
python experiment/run_experiments.py
```

### 只运行可视化分析(跳过训练)

```bash
cd ensemble_learning
python experiment/run_experiments.py --skip-training
```

### 单独运行各个组件

```bash
# 只运行对比实验
python experiment/comparative_experiment.py

# 只生成可视化分析
python experiment/visualization.py
```

## 实验结果

运行实验后，结果将保存在`experiment/results/`目录下，包括：

- 模型文件: `*.h5`
- 模型指标: `*_metrics.pkl`
- 性能指标表格: `performance_metrics.csv`
- 可视化图表:
  - 训练历史对比图: `training_history_comparison.png`
  - ROC曲线对比图: `roc_curves_comparison.png`
  - 混淆矩阵对比图: `confusion_matrices_comparison.png`
  - 性能雷达图: `radar_chart_comparison.png`
  - 精确率-召回率曲线: `pr_curves_comparison.png`
  - 特征分布t-SNE可视化: `feature_distribution_tsne.png`
  - 性能条形图: `performance_bar_comparison.png`

## 模型性能对比

根据文档中的消融实验结果，预期优化后的模型将展现以下性能提升：

- 多分支模型在测试集上的AUC提升至0.985，F1分数提高到0.961
- 增强训练模型泛化能力显著提升，Recall由0.965增至0.972
- 软标签训练使模型输出概率分布更平滑，增强Stacking集成效果 