# 基于Stacking集成学习的恶意PE检测消融实验

本实验基于"基于基模型堆叠的静态PE特征识别模型构建"文档，实现了Stacking集成学习模型的消融实验，包括基模型消融和特征消融，用于评估每个组件对整体性能的贡献。

## 项目结构

```
experiment2/
├── stacking_model.py      # Stacking集成学习模型实现
├── ablation_experiment.py # 消融实验实现
├── visualization.py       # 可视化工具
├── run_experiments.py     # 主运行脚本
├── results/               # 实验结果目录
├── figures/               # 可视化图表目录
└── README.md              # 本文件
```

## 数据集与特征

- 特征数据：967维PE静态特征，包括：
  - ByteHistogram (256维)
  - ByteEntropyHistogram (256维)
  - GeneralFileInfo (10维)
  - HeaderFileInfo (62维)
  - ExportsInfo (128维)
  - SectionInfo (255维)
- 数据存储：`../pe_raw/pe_raw_vectors.pkl`
- 标签信息：`../models/black_list.pkl`（恶意样本列表）和`../models/hash_list.pkl`（所有样本列表）

## 实验内容

### 1. 基模型消融实验

逐一移除9个基模型，观察对整体性能的影响：
- LogisticRegression (lr)
- GradientBoostingClassifier (gbc)
- BaggingClassifier (bc)
- XGBClassifier (xgb)
- DecisionTreeClassifier (dt)
- LinearSVC (svm)
- RandomForestClassifier (rfc)
- ExtraTreesClassifier (etc)
- AdaBoostClassifier (ada)

### 2. 特征消融实验

逐一移除6类特征组，观察对整体性能的影响：
- ByteHistogram
- ByteEntropyHistogram
- GeneralFileInfo
- HeaderFileInfo
- ExportsInfo
- SectionInfo

### 3. 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- AUC值 (Area Under ROC Curve)

## 运行方法

### 环境要求

- Python 3.7.16
- scikit-learn 0.23.1
- numpy 1.18.5
- pandas 1.1.0
- matplotlib 3.3.0
- seaborn 0.12.2
- xgboost 1.1.1

### 运行完整实验

```bash
cd ensemble_learning
python experiment2/run_experiments.py
```

### 仅生成可视化图表（跳过实验）

```bash
cd ensemble_learning
python experiment2/run_experiments.py --skip-experiments
```

## 实验结果

运行实验后，结果将保存在以下目录：
- 实验数据：`experiment2/results/`
  - `model_ablation_results.csv`：模型消融实验结果
  - `base_model_performance.csv`：基模型单独性能
  - `feature_ablation_results.csv`：特征消融实验结果
  
- 可视化图表：`experiment2/figures/`
  - 模型消融图表：
    - `model_ablation_auc_diff.png`：移除不同模型后的AUC下降量
    - `model_ablation_f1_diff.png`：移除不同模型后的F1下降量
    - `model_ablation_radar.png`：模型组合性能雷达图
  - 基模型性能图表：
    - `base_model_performance.png`：基模型性能条形图
    - `base_model_heatmap.png`：基模型性能热力图
  - 特征消融图表：
    - `feature_ablation_auc_diff.png`：移除不同特征后的AUC下降量
    - `feature_ablation_f1_diff.png`：移除不同特征后的F1下降量
    - `feature_dimension_performance.png`：特征维度与性能关系图
  - Stacking与基模型对比：
    - `stacking_vs_base_models_auc.png`：AUC对比
    - `stacking_vs_base_models_f1.png`：F1分数对比
    - `stacking_improvement.png`：Stacking提升量 