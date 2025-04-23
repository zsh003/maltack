#!/bin/bash

start_time=$(date +%s)

python run.py $1   # 训练集预处理，PE头修复，直方图、PE静态、特征工程等所有特征提取
python train_histogram.py  # 直方图特征模型训练
python train_pe_raw.py  # PE静态特征模型训练
python train_feature_engineering.py  # 特征工程模型训练
python stacking_train.py  # 集成学习模型训练
python test.py  # 测试模型，预测结果

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Program end. Time cost: $(($cost_time/60))min $(($cost_time%60))s"
