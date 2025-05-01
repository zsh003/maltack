# 第五章 基于Stacking和加权融合的多层集成学习方案

本章在三种子模型输出基础上，构建三级融合框架，通过异构模型互补与层级融合实现性能提升：第一层并行子模型；第二层Stacking元模型；第三层基于逻辑回归与随机森林进行加权融合决策，最终输出二元分类决策并详细描述每一步的实现与优化。

## 5.1 多层集成学习模型框架

### 5.1.1 子模型并行训练  
利用Python多进程或分布式调度，分别加载CNN、Stacking与LightGBM模型，对样本特征并行预测，输出$\mathbf{p}_{\mathrm{CNN}},\mathbf{p}_{\mathrm{Stack}},\mathbf{p}_{\mathrm{LGB}}$三组概率向量。

### 5.1.2 Stacking元模型构建  
将三组概率拼接为$\mathbf{P}=[p_{\mathrm{CNN}},p_{\mathrm{Stack}},p_{\mathrm{LGB}}]\in\mathbb{R}^3$，采用5折CV生成元特征并训练LogisticRegression(L2,C=1.0)与RandomForest(n_estimators=200,max_depth=10)两类元模型。

### 5.1.3 加权融合策略  
基于元模型输出概率$(p_{\mathrm{LR}},p_{\mathrm{RF}})$，应用加权融合：
$$
p_{\mathrm{final}}=\alpha\,p_{\mathrm{LR}}+(1-\alpha)\,p_{\mathrm{RF}},\quad \alpha\in[0,1].
$$
通过网格搜索在$\{0.1,0.2,\dots,0.9\}$区间确定最优$\alpha=0.6$，并以阈值0.5裁定最终标签。



## 5.2 Stacking集成方案细节

### 5.2.1 OOF预测生成策略  
为避免信息泄露，对训练集进行K=5折划分。每折基模型仅在4折训练，向第5折预测并收集概率，直至所有折完成。

### 5.2.2 元特征构造与融合形式  
合并三类子模型在测试集上的平均概率与训练集的OOF概率，构成$(N,3)$矩阵，作为元模型输入。

### 5.2.3 元模型训练与正则化  
LogisticRegression使用L2正则化防止过拟合；RandomForest采用Out‐Of‐Bag评价并调整树深度。

### 5.2.4 验证与对比  
通过ROC曲线(AUC)、Precision‐Recall曲线与混淆矩阵对比单一元模型与Stacking性能，证明Stacking在查全与查准之间取得更优平衡。

## 5.3 加权融合方案细节

### 5.3.1 权重参数优化  
通过在验证集上对$\alpha$进行网格搜索，以最大化F1分数；结果$\alpha=0.6$对应F1最高。

### 5.3.2 决策阈值与不确定样本处理  
默认阈值0.5，对于概率在$(0.45,0.55)$区间的不确定样本可触发二次人工审查流程。

### 5.3.3 实验结果及误差分析  
最终方案在测试集获得F1=0.943，误报率0.037，漏报率0.029。与单一LogisticRegression(F1=0.927)和单一RandomForest(F1=0.934)相比，F1分别提升1.6%与0.9%。

### 5.3.4 与其他融合方法对比  
与简单投票(Voting)和平均融合(Mean)相比，加权融合在权重最优时提高F1约0.8%。此外，使用Stacking+Weighted的组合优于纯Boosting或Bagging集成策略，适应性更强。

