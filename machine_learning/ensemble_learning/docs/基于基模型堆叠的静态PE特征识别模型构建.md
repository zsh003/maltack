## 4.3 基于基模型堆叠的静态PE特征识别模型构建

### 4.3.1 特征准备与标准化

使用967维特征，经标准Scaler零均值单位方差化后输入各基模型。基于树模型的不敏感性，可直接采用原始特征；对于LinearSVC与LogisticRegression，额外进行L2正则化控制。本模块使用967维PE静态结构化特征，来源如下：

- ByteHistogram (256维)、ByteEntropyHistogram (256维)；
- GeneralFileInfo (10维)、HeaderFileInfo (62维)；
- ExportsInfo (128维)、SectionInfo (255维)。

### 4.3.2 异构基模型选型与调优

选用九种互补性强的基模型：逻辑回归、GradientBoosting (GBDT)、袋装法(Bagging)、XGBoost、决策树、线性SVM、随机森林、极端随机树(ExtraTrees)、AdaBoost。模型多样化减少偏差，并利用Boosting与Bagging优势共同提升稳健性（Breiman[2]）。各模型超参数通过网格搜索5折CV确定，如GBDT的learning_rate∈{0.01,0.1}、depth∈{3,5}等。

[2] Breiman, L. “Random Forests.” Machine Learning, 45(1):5–32, 2001.

选用9种异构基模型对967维特征进行分类：

- LogisticRegression (L2, C=1.0)；
- GradientBoostingClassifier (n\_estimators=100, lr=0.1, depth=3)；
- BaggingClassifier (base=DT(max\_depth=5), n\_estimators=10)；
- XGBClassifier (n\_estimators=100, max\_depth=5, eta=0.1)；
- DecisionTreeClassifier (max\_depth=10)；
- LinearSVC (C=1.0)；
- RandomForestClassifier (n\_estimators=100)；
- ExtraTreesClassifier (n\_estimators=100)；
- AdaBoostClassifier (base=DT(max\_depth=3), n\_estimators=50, lr=1.0)。

### 4.3.3 Stacking构建流程

采用5折交叉验证生成OOF(Out-Of-Fold)预测：每轮基模型在4折上训练，并在留出的1折上输出概率，合并后生成$9\times N$维堆叠特征矩阵$Z\in\mathbb{R}^{9\times N}$。元模型选用RandomForest(n_estimators=200, max_depth=10)进行二次学习。最终测试阶段基模型输出取平均概率作为元特征，输入元模型生成最终预测$\hat y_{\mathrm{Stack}}$。

### 4.3.4 消融实验与特征重要性

通过逐一移除单一基模型与单一特征组（如SectionInfo）进行消融，对比堆叠前后AUC变化。结果表明移除随机森林基模型时AUC下降1.2%，移除SectionInfo特征时AUC下降0.9%，凸显其对组合模型贡献最大（图4.2）。

### 4.3.5 实验结果与讨论

在测试集上，Stacking模型AUC=0.975、Accuracy=0.958、Precision=0.952、Recall=0.963、F1=0.957，较基模型平均AUC(0.965)提升1.0%。