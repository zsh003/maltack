# 第五章 基于Stacking和加权融合的多层集成学习方案

本章在三种子模型输出基础上，构建三级融合框架，通过异构模型互补与层级融合实现性能提升：第一层并行子模型；第二层Stacking元模型；第三层基于逻辑回归与随机森林进行加权融合决策，最终输出二元分类决策并详细描述每一步的实现与优化。

## 5.1 多层集成学习模型框架

```mermaid
graph TD
  subgraph 第1层: 子模型并行
    A[Histogram CNN 512D] --> p_CNN[p_CNN]
    B[Static RF 967D] --> p_RF[p_RF]
    C[FE LGBM 56D] --> p_LGB[p_LGB]
  end
  subgraph 第2层: Stacking元模型
    p_CNN --> LR[Logistic Regression L2, C=1.0]
    p_RF --> LR
    p_LGB --> LR
    p_CNN --> RF[Random Forest n_estimators=200, max_depth=10]
    p_RF --> RF
    p_LGB --> RF
    LR --> p_LR[p_LR]
    RF --> p_RF_meta[p_RF_meta]
  end
  subgraph 第3层: 加权融合决策
    p_LR --> WF[Weighted Fusion α=0.6]
    p_RF_meta --> WF
    WF --> p_final[p_final]
  end
```

### 5.1.1 子模型并行训练  
本层由三路并行分类器组成，分别基于PE样本的三类特征空间并行预测：  
- **直方图特征（512维）**：输入512维直方图向量，使用CNN网络进行分类。网络结构：Reshape(32×16×1)→Conv2D(60,(2,2),ReLU)→MaxPool→Conv2D(200,(2,2),ReLU)→MaxPool→Flatten→Dense(500,ReLU)→Dropout(0.2)→Dense(1,Sigmoid)。训练时采用EarlyStopping(patience=6)和ReduceLROnPlateau(patience=4,factor=0.5)，输出特征向量。  
- **静态结构化特征（967维）**：输入PE静态结构化特征（Section信息、HeaderInfo、ExportsInfo、SectionInfo等共967维），使用RandomForestClassifier(n_estimators=100, random_state=8)进行分类，输出特征向量。  
- **特征工程综合特征（56维）**：输入基于YARA、字符串匹配、Opcode统计等56维特征，使用LightGBM(0.01学习率、num_leaves=8、max_depth=3等超参)进行分类，输出特征向量。  
并行训练与预测流程在`stacking_train.py`（训练阶段）和`test.py`（预测阶段）中通过Python多进程(Pool)加速实现，分别生成训练集特征和测试集并行预测结果。

```mermaid
flowchart LR
  subgraph 并行预测层
    Data1[直方图特征512D] --> CNN[CNN模型预测]
    Data2[静态特征967D] --> RF[RF模型预测]
    Data3[FE特征56D] --> LGBM[LGBM模型预测]
  end
  CNN --> p_CNN[p_CNN]
  RF --> p_RF[p_RF]
  LGBM --> p_LGB[p_LGB]
```

在此流程中，首先对PE样本提取三类特征：直方图特征、静态结构化特征和特征工程综合特征。每类特征分别输入到对应的模型中进行预测：CNN用于直方图特征，随机森林用于静态特征，LightGBM用于特征工程特征。通过Python多进程技术实现并行预测，提升计算效率。每个模型输出的特征向量将作为后续Stacking元模型的输入。

### 5.1.2 Stacking元模型构建  
在第二层使用Stacking策略，将第一层子模型输出的训练集特征和测试集特征拼接为$(N,3)$元特征。流程如下：  
1. 从`../oof/raw_his_feature.pkl`、`../oof/raw_feature_stacking_train_5.pkl`、`../oof/feature_engineerin_train.pkl`中加载三类训练特征；  
2. 直接将三类特征进行堆叠，形成新的训练集特征矩阵；  
3. 将堆叠后的特征输入到LogisticRegression(L2, C=1.0)和RandomForest(n_estimators=200, max_depth=10)中进行元模型训练；  
4. 在`stacking_train.py`中完成模型训练并保存至`../models/lr_rfc.pkl`。

```mermaid
flowchart TD
  feature1[加载特征: raw_his_feature.pkl] --> concat[拼接特征 Nx3]
  feature2[加载特征: raw_feature_stacking_train_5.pkl] --> concat
  feature3[加载特征: feature_engineerin_train.pkl] --> concat
  concat --> TrainMeta[训练LR & RF元模型]
  TrainMeta --> SaveModel[保存模型至 lr_rfc.pkl]
```

在Stacking元模型构建中，首先从不同的特征文件中加载训练集的特征。直接将这些特征进行堆叠，形成新的特征矩阵，输入到Logistic Regression和Random Forest中进行训练，形成最终的元模型。训练完成后，模型被保存以供后续使用。

### 5.1.3 加权融合策略  
第三层加载LogisticRegression和RandomForest元模型，在`test.py`中执行：  
1. 使用`lr.predict_proba(test)`和`rfc.predict_proba(test)`分别获得元模型概率$p_{LR},p_{RF_meta}$；  
2. 应用加权融合公式：$p_{final}=0.6\,p_{LR}+(1-0.6)\,p_{RF_meta}$；  
3. 以阈值0.5裁定最终标签，结果写入`malware_final.txt`；对不确定样本$(0.45<p<0.55)$可后续人工审查。  
权重α通过在验证集上网格搜索${0.1,0.2,\dots,0.9}$以最大化F1得出最佳α=0.6。

```mermaid
flowchart LR
  LRProb[LR.predict_proba] --> Fusion[加权融合]
  RFProb[RF.predict_proba] --> Fusion
  subgraph 加权计算
    Fusion --> p_final[p_final = 0.6p_LR + 0.4p_RF]
  end
  p_final --> Threshold{p_final > 0.5?}
  Threshold -- 是 --> LabelMal[恶意]
  Threshold -- 否 --> LabelBen[正常]
  p_final -- 0.45 < p < 0.55 --> Manual[人工审查]
```

在加权融合策略中，首先从Logistic Regression和Random Forest元模型中获取预测概率。通过加权融合公式计算最终的预测概率$p_{final}$，并根据阈值0.5进行分类决策。对于不确定样本，建议进行人工审查以提高决策的准确性。权重α的选择通过验证集上的网格搜索确定，以确保最佳的F1分数。

## 5.2 Stacking集成方案细节

### 5.2.1 特征堆叠策略  
在训练集上直接将三类特征进行堆叠，形成新的特征矩阵，作为元模型的输入。

```mermaid
flowchart TD
  Data1[直方图特征] --> StackMat[特征堆叠 Nx3]
  Data2[静态特征] --> StackMat
  Data3[特征工程特征] --> StackMat
  StackMat --> MetaInput[作为元模型输入]
```

在特征堆叠策略中，直接将三类特征进行堆叠，形成新的特征矩阵。这个矩阵作为元模型的输入，提供了不同子模型的综合信息，帮助元模型更好地进行分类决策。

### 5.2.2 元模型训练与正则化  
LogisticRegression使用L2正则化防止过拟合；RandomForest采用Out‐Of‐Bag评价并调整树深度。

```mermaid
flowchart LR
  MetaInput --> LRModel[LR L2正则]
  MetaInput --> RFModel[RF OOB调参]
  LRModel --> Eval[性能评估]
  RFModel --> Eval
```

在元模型训练与正则化中，Logistic Regression通过L2正则化来防止过拟合，而Random Forest则通过Out-Of-Bag评价来调整树的深度。两种方法都旨在提高模型的泛化能力，确保在未见过的数据上也能保持良好的性能。

### 5.2.3 验证与对比  
通过ROC曲线(AUC)、Precision‐Recall曲线与混淆矩阵对比单一元模型与Stacking性能，证明Stacking在查全与查准之间取得更优平衡。

```mermaid
flowchart LR
  Predictions[Stacking 预测输出] --> ROC[ROC 曲线]
  Predictions --> PR[Precision-Recall 曲线]
  Predictions --> CM[混淆矩阵]
  ROC --> Compare[性能对比]
  PR --> Compare
  CM --> Compare
```

在验证与对比中，通过绘制ROC曲线、Precision-Recall曲线和混淆矩阵，评估Stacking模型的性能。与单一元模型相比，Stacking模型在查全率和查准率之间取得了更好的平衡，显示出其在综合性能上的优势。

## 5.3 加权融合方案细节

### 5.3.1 权重参数优化  
通过在验证集上对$\alpha$进行网格搜索，以最大化F1分数；结果$\alpha=0.6$对应F1最高。

```mermaid
flowchart TD
  Alphas[候选 α 集合] --> ForEach[遍历 α]
  ForEach --> FusionCalc[加权融合预测]
  FusionCalc --> EvalF1[计算 F1 分数]
  EvalF1 --> SelectBest[选取最佳 α]
```

在权重参数优化中，通过在验证集上对不同的α值进行网格搜索，计算每个α对应的F1分数。最终选择F1分数最高的α值作为最佳权重，确保加权融合的效果最优。

### 5.3.2 决策阈值与不确定样本处理  
默认阈值0.5，对于概率在$(0.45,0.55)$区间的不确定样本可触发二次人工审查流程。

```mermaid
flowchart TD
  p_final -->|> 0.5| LabelMalicious[预测:恶意]
  p_final -->|< 0.5| LabelBenign[预测: 正常]
  p_final -->|0.45 < p < 0.55| Review[人工审查]
```

在决策阈值与不确定样本处理过程中，默认的决策阈值为0.5。对于预测概率在0.45到0.55之间的不确定样本，建议进行人工审查，以提高最终决策的准确性。

### 5.3.3 实验结果及误差分析  
最终方案在测试集获得F1=0.943，误报率0.037，漏报率0.029。与单一LogisticRegression(F1=0.927)和单一RandomForest(F1=0.934)相比，F1分别提升1.6%与0.9%。

```mermaid
pie title 测试集 F1 对比
  "Stacking + Weighted: 0.943" : 0.943
  "LogisticRegression: 0.927" : 0.927
  "RandomForest: 0.934" : 0.934
```

在实验结果及误差分析中，通过对比不同模型的F1分数，展示了Stacking加权融合方案的优越性。该方案在测试集上取得了最高的F1分数，显著优于单一模型。

### 5.3.4 与其他融合方法对比  
与简单投票(Voting)和平均融合(Mean)相比，加权融合在权重最优时提高F1约0.8%。此外，使用Stacking+Weighted的组合优于纯Boosting或Bagging集成策略，适应性更强。

```mermaid
pie title 不同融合方法 F1 提升
  "Weighted Fusion: 0.8": 0.8
  "Voting: 0.3": 0.3
  "Mean: 0.4": 0.4
```

在与其他融合方法对比中，加权融合在F1分数上表现出明显的提升。与简单投票和平均融合相比，加权融合能够更好地利用各个模型的优势，提供更高的预测准确性。Stacking与加权融合的结合进一步增强了模型的适应性和鲁棒性。

