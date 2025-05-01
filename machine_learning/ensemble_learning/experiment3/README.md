# 基于LightGBM特征工程的PE恶意软件识别模型

本实验基于特征工程和LightGBM构建PE恶意软件识别模型，并进行多角度分析。

## 实验内容

1. **模型构建与评估**
   - 基于56维人工选取的特征构建LightGBM模型
   - 模型配置优化：num_leaves=128, max_depth=6, learning_rate=0.05等
   - 5折交叉验证训练以防止单次过拟合

2. **特征重要性分析**
   - 提取并可视化顶部特征：节区熵、节区大小等
   - 特征与标签关联分析

3. **降维与对比**
   - 使用PCA降维（保留10个主成分）
   - 对比原始特征与降维后的模型性能差异

4. **模型对比**
   - 与其他常用模型比较：XGBoost、RandomForest
   - 多维度指标对比：准确率、精确率、召回率、F1、AUC

5. **鲁棒性分析**
   - 特征扰动实验：随机替换10%特征值，评估模型性能变化

6. **推理性能评估**
   - 计算单样本和批量推理时间
   - 模型大小统计

## 项目结构

```
experiment3/
├── constants.py        # 常量定义
├── lgbm_model.py       # LightGBM模型实现
├── feature_analysis.py # 特征分析与可视化
├── model_evaluation.py # 模型评估与比较
├── run_experiments.py  # 主运行脚本
└── README.md           # 说明文档
```

## 特征说明

本实验使用56维人工选取特征，包括：
- 节区熵特征（16维）：反映PE文件执行逻辑与代码分布
- 字符串模式特征（26维）：通过正则匹配恶意代码常用函数名、URL等
- YARA检测特征（2维）：通过规则匹配恶意代码特征字符串
- 关键字计数特征（5维）：计算敏感API调用频率
- 操作码特征（7维）：提取可执行代码中的指令频率

## 运行方法

```bash
# 运行全部实验
python run_experiments.py

# 仅生成图表（假设已有结果）
python run_experiments.py --skip-experiments

# 指定结果和图表保存目录
python run_experiments.py --result-dir ./my_results --figure-dir ./my_figures
```

## 性能指标

在测试集上，LightGBM模型表现：
- AUC：0.971
- 准确率：0.951
- 精确率：0.944
- 召回率：0.959
- F1分数：0.951

推理时间：单样本 < 1ms 