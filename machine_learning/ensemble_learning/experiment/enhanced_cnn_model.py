import os
import time
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 超参数
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 4396

LENGTH = 512
WIDTH, HEIGHT = 32, 16
BATCH_SIZE = 128
EPOCH = 50
SHUFFLE = True
CLASSES = 2

LR = 0.001

# 注意力模块(SE模块)
def squeeze_excite_block(input_tensor, ratio=16):
    """实现Squeeze-and-Excitation注意力机制"""
    init = input_tensor
    filters = init.shape[-1]
    
    # Squeeze操作：全局平均池化
    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape((1, 1, filters))(se)
    
    # Excitation操作：两个全连接层
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    # 与原特征图相乘，应用通道注意力
    x = layers.Multiply()([init, se])
    return x

# 余弦退火学习率调度
def cosine_annealing_schedule(epoch, lr):
    """实现余弦退火学习率调度"""
    max_lr = LR
    min_lr = LR * 0.01
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(epoch / EPOCH * np.pi))

# FGSM生成对抗样本
def generate_adversarial_examples(model, x, y, epsilon=0.01):
    """使用FGSM方法生成对抗样本"""
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor)
        loss = tf.keras.losses.BinaryCrossentropy()(y, predictions)
    
    gradients = tape.gradient(loss, x_tensor)
    signed_grad = tf.sign(gradients)
    perturbed_x = x_tensor + epsilon * signed_grad
    perturbed_x = tf.clip_by_value(perturbed_x, 0, 1)
    
    return perturbed_x.numpy()

# 软标签生成函数
def generate_soft_labels(features, labels, k=5):
    """使用KNN生成软标签"""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    
    # 获取每个样本的k个最近邻的标签
    _, indices = knn.kneighbors(features)
    
    # 计算软标签
    soft_labels = np.zeros_like(labels, dtype=np.float32)
    for i, idx in enumerate(indices):
        neighbors_labels = labels[idx]
        soft_labels[i] = np.mean(neighbors_labels)
    
    return soft_labels

# 创建多分支CNN模型
def create_enhanced_model():
    """创建优化后的多分支CNN模型"""
    # 输入层：接收512维直方图特征
    inputs = layers.Input(shape=(LENGTH, 1), dtype='float32')
    
    # 特征重塑：将1D特征转换为2D伪图像
    re_inputs = tf.reshape(inputs, [-1, WIDTH, HEIGHT, 1])
    
    # 主分支：原有的双卷积层设计
    # 卷积层1
    Conv_1 = layers.Conv2D(60, (2, 2), padding='same', activation='relu')(re_inputs)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(Conv_1)
    
    # 卷积层2
    Conv_2 = layers.Conv2D(200, (2, 2), padding='same', activation='relu')(pool_1)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(Conv_2)
    
    # 注意力分支
    attn_branch = squeeze_excite_block(pool_2)
    
    # 残差分支：添加跳跃连接
    res_branch = layers.Add()([pool_1, layers.Conv2D(200, (1, 1), padding='same')(pool_1)])  # 调整通道数
    res_branch = layers.Conv2D(200, (2, 2), padding='same', activation='relu')(res_branch)
    res_branch = layers.MaxPooling2D(pool_size=(2, 2))(res_branch)
    
    # 合并多分支特征
    merged = layers.Concatenate()([pool_2, attn_branch, res_branch])
    
    # 全连接层
    Flat = layers.Flatten()(merged)
    Dense_1 = layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.001))(Flat)
    dropout = layers.Dropout(0.2)(Dense_1)
    outputs = layers.Dense(1, activation='sigmoid')(dropout)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model

# 加载数据集
def load_data():
    """加载并预处理直方图特征数据集"""
    # 加载样本哈希列表
    with open("../models/hash_list.pkl", "rb") as f:
        hash_list = pickle.load(f)
    test_num = len(hash_list)
    
    # 加载黑样本列表
    with open("../models/black_list.pkl", "rb") as f:
        black_list = pickle.load(f)
    
    # 收集直方图特征路径
    hisdatapath = "../histogram"
    his_black_path = []
    his_white_path = []
    for parent, _, filenames in os.walk(hisdatapath):
        for filename in filenames:
            fp = os.path.join(parent, filename)
            if filename[:-4] in black_list:
                his_black_path.append(fp)
            else:
                his_white_path.append(fp)
    
    # 加载特征和标签
    raw_his_feature, raw_his_labels = [], []
    
    with tqdm(total=test_num, ncols=80, desc="读取直方图特征") as pbar:
        for fp in his_black_path:
            with open(fp, 'r') as f:
                feature = f.readlines()
            feature = [float(his.strip()) for his in feature]
            raw_his_feature.append(feature)
            raw_his_labels.append(1)
            pbar.update(1)
        for fp in his_white_path:
            with open(fp, 'r') as f:
                feature = f.readlines()
            feature = [float(his.strip()) for his in feature]
            raw_his_feature.append(feature)
            raw_his_labels.append(0)
            pbar.update(1)
    
    # 转换为numpy数组
    features = np.array(raw_his_feature, dtype=np.float32)
    labels = np.array(raw_his_labels, dtype=np.int32)
    
    # 数据打乱
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    index = list(range(len(labels)))
    np.random.shuffle(index)
    
    features = features[index]
    labels = labels[index]
    
    # 数据集划分
    train_features, test_features, train_label, test_label = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=SEED)
    
    train_features, valid_features, train_label, valid_label = train_test_split(
        train_features,
        train_label,
        test_size=VAL_SIZE,
        stratify=train_label,
        random_state=SEED)
    
    return (train_features, train_label), (valid_features, valid_label), (test_features, test_label)

# 增强模型训练函数
def train_enhanced_model(model, train_data, valid_data, use_soft_labels=False, use_adversarial=False, use_cosine_lr=False):
    """训练增强型CNN模型，支持软标签、对抗训练和余弦退火学习率"""
    (train_features, train_label), (valid_features, valid_label) = train_data, valid_data
    
    # 回调函数列表
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, min_delta=1e-4, mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, verbose=1)
    ]
    
    # 如果使用余弦退火学习率
    if use_cosine_lr:
        callbacks.append(LearningRateScheduler(cosine_annealing_schedule))
    
    # 如果使用软标签
    if use_soft_labels:
        print("使用软标签训练...")
        soft_train_label = generate_soft_labels(train_features, train_label, k=5)
        train_label = soft_train_label
    
    # 创建数据集
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_label)) \
                           .batch(BATCH_SIZE) \
                           .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_features, valid_label)) \
                           .batch(BATCH_SIZE) \
                           .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # 对抗训练
    if use_adversarial:
        print("使用对抗训练...")
        history = []
        for epoch in range(EPOCH):
            # 常规训练
            h = model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=1,
                verbose=1,
                callbacks=callbacks if epoch == 0 else None  # 只在第一个epoch时使用回调
            )
            
            if 'history' not in locals():
                history = h.history
            else:
                for k, v in h.history.items():
                    history[k].extend(v)
            
            # 提前终止检查
            if hasattr(model, 'stop_training') and model.stop_training:
                break
                
            # 生成对抗样本并训练
            print(f"Epoch {epoch+1}/{EPOCH}: 生成对抗样本...")
            
            # 批次处理以避免内存问题
            batch_size = 256
            for i in range(0, len(train_features), batch_size):
                batch_x = train_features[i:i+batch_size]
                batch_y = train_label[i:i+batch_size]
                
                # 生成对抗样本
                batch_x_adv = generate_adversarial_examples(model, batch_x, batch_y)
                
                # 训练对抗样本
                model.train_on_batch(batch_x_adv, batch_y)
                
    else:
        # 常规训练
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCH,
            verbose=1,
            callbacks=callbacks
        )
    
    return model, history

# 评估模型
def evaluate_model(model, test_data):
    """评估模型性能"""
    test_features, test_label = test_data
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_features, test_label)) \
                          .batch(BATCH_SIZE) \
                          .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # 评估模型
    results = model.evaluate(test_ds)
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'auc': results[2],
        'recall': results[3],
        'precision': results[4]
    }
    
    # 计算F1分数
    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-10)
    
    # 获取预测结果
    predictions = model.predict(test_ds)
    pred_labels = (predictions > 0.5).astype(int).flatten()
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_label, pred_labels)
    
    return metrics, cm, predictions

# 保存模型和结果
def save_model_and_results(model, metrics, model_name, result_dir='../models'):
    """保存模型和评估结果"""
    # 创建保存目录（如果不存在）
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 保存模型
    model_path = os.path.join(result_dir, f'{model_name}.h5')
    model.save(model_path, save_format="tf")
    
    # 保存指标结果
    metrics_path = os.path.join(result_dir, f'{model_name}_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"模型和评估结果已保存到：{result_dir}")
    return model_path, metrics_path 