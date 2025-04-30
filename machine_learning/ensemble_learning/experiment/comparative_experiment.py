import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle
import sys
import pandas as pd

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入优化模型相关函数
from enhanced_cnn_model import (
    load_data, 
    create_enhanced_model, 
    train_enhanced_model, 
    evaluate_model, 
    save_model_and_results
)

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

# 设置随机种子
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 创建结果目录
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def create_basic_model():
    """创建基础CNN模型，与原始train_histogram.py中的模型结构相同"""
    # 输入层：接收512维直方图特征
    inputs = layers.Input(shape=(LENGTH, 1), dtype='float32')
    
    # 特征重塑：将1D特征转换为2D伪图像
    re_inputs = tf.reshape(inputs, [-1, WIDTH, HEIGHT, 1])
    
    # 双卷积层设计
    # 卷积层1
    Conv_1 = layers.Conv2D(60, (2, 2), padding='same', activation='relu')(re_inputs)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(Conv_1)
    
    # 卷积层2
    Conv_2 = layers.Conv2D(200, (2, 2), padding='same', activation='relu')(pool_1)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(Conv_2)
    
    # 全连接层
    Flat = layers.Flatten()(pool_2)
    Dense_1 = layers.Dense(500, activation='relu')(Flat)
    dropout = layers.Dropout(0.2)(Dense_1)
    outputs = layers.Dense(1, activation='sigmoid')(dropout)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model

def create_multibranch_model():
    """创建多分支CNN模型，但不包含软标签和对抗训练"""
    # 使用优化后的模型结构
    return create_enhanced_model()

def train_basic_model(model, train_data, valid_data):
    """训练基础CNN模型"""
    (train_features, train_label), (valid_features, valid_label) = train_data, valid_data
    
    # 创建数据集
    train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_label)) \
                           .batch(BATCH_SIZE) \
                           .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_features, valid_label)) \
                           .batch(BATCH_SIZE) \
                           .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # 回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, min_delta=1e-4, mode='min', restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, verbose=1)
    ]
    
    # 训练模型
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCH,
        verbose=1,
        callbacks=callbacks
    )
    
    return model, history

def plot_training_history(histories, model_names, save_path=None):
    """绘制多个模型的训练历史对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # 绘制训练和验证损失
    for history, name in zip(histories, model_names):
        ax1.plot(history.history['loss'], label=f'{name} 训练')
        ax1.plot(history.history['val_loss'], label=f'{name} 验证')
    
    ax1.set_title('模型损失对比')
    ax1.set_ylabel('损失')
    ax1.set_xlabel('轮次')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制训练和验证准确率
    for history, name in zip(histories, model_names):
        ax2.plot(history.history['accuracy'], label=f'{name} 训练')
        ax2.plot(history.history['val_accuracy'], label=f'{name} 验证')
    
    ax2.set_title('模型准确率对比')
    ax2.set_ylabel('准确率')
    ax2.set_xlabel('轮次')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"训练历史图已保存到: {save_path}")
    
    plt.close()

def plot_roc_curves(models, test_data, model_names, save_path=None):
    """绘制多个模型的ROC曲线对比图"""
    plt.figure(figsize=(10, 8))
    
    test_features, test_label = test_data
    
    # 为每个模型绘制ROC曲线
    for model, name in zip(models, model_names):
        # 获取预测概率
        y_pred_prob = model.predict(test_features).flatten()
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(test_label, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC曲线对比')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC曲线图已保存到: {save_path}")
    
    plt.close()

def plot_confusion_matrices(cms, model_names, save_path=None):
    """绘制多个模型的混淆矩阵对比图"""
    fig, axes = plt.subplots(1, len(cms), figsize=(6*len(cms), 5))
    
    if len(cms) == 1:
        axes = [axes]
    
    classes = ['良性', '恶意']
    
    for i, (cm, name) in enumerate(zip(cms, model_names)):
        # 计算混淆矩阵中的各项指标
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 绘制混淆矩阵热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[i])
        axes[i].set_title(f'{name}\nAcc: {accuracy:.3f}, F1: {f1:.3f}\nPrecision: {precision:.3f}, Recall: {recall:.3f}')
        axes[i].set_xlabel('预测标签')
        axes[i].set_ylabel('真实标签')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵图已保存到: {save_path}")
    
    plt.close()

def create_metrics_table(all_metrics, model_names, save_path=None):
    """创建指标对比表格"""
    # 构建性能指标表格数据
    table_data = []
    for metrics, name in zip(all_metrics, model_names):
        row = {
            '模型': name,
            '准确率': f"{metrics['accuracy']:.3f}",
            'AUC': f"{metrics['auc']:.3f}",
            '精确率': f"{metrics['precision']:.3f}",
            '召回率': f"{metrics['recall']:.3f}",
            'F1分数': f"{metrics['f1']:.3f}"
        }
        table_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存为CSV
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"性能指标表已保存到: {save_path}")
    
    return df

def run_experiments():
    """运行对比实验"""
    print("开始加载数据...")
    train_data, valid_data, test_data = load_data()
    print("数据加载完成！")
    
    # 所有模型、历史记录、指标和混淆矩阵的列表
    models = []
    histories = []
    all_metrics = []
    cms = []
    model_names = ['基础CNN模型', '多分支CNN模型', '增强训练CNN模型']
    
    # 1. 训练基础CNN模型
    print("\n开始训练基础CNN模型...")
    basic_model = create_basic_model()
    basic_model, basic_history = train_basic_model(basic_model, train_data, valid_data)
    basic_metrics, basic_cm, _ = evaluate_model(basic_model, test_data)
    save_model_and_results(basic_model, basic_metrics, 'basic_cnn_model', RESULT_DIR)
    
    print("基础CNN模型训练完成！")
    print(f"测试集性能指标: 准确率={basic_metrics['accuracy']:.3f}, F1分数={basic_metrics['f1']:.3f}")
    
    models.append(basic_model)
    histories.append(basic_history)
    all_metrics.append(basic_metrics)
    cms.append(basic_cm)
    
    # 2. 训练多分支CNN模型(不使用软标签和对抗训练)
    print("\n开始训练多分支CNN模型...")
    multibranch_model = create_multibranch_model()
    multibranch_model, multibranch_history = train_enhanced_model(
        multibranch_model, 
        train_data, 
        valid_data,
        use_soft_labels=False,
        use_adversarial=False,
        use_cosine_lr=False
    )
    multibranch_metrics, multibranch_cm, _ = evaluate_model(multibranch_model, test_data)
    save_model_and_results(multibranch_model, multibranch_metrics, 'multibranch_cnn_model', RESULT_DIR)
    
    print("多分支CNN模型训练完成！")
    print(f"测试集性能指标: 准确率={multibranch_metrics['accuracy']:.3f}, F1分数={multibranch_metrics['f1']:.3f}")
    
    models.append(multibranch_model)
    histories.append(multibranch_history)
    all_metrics.append(multibranch_metrics)
    cms.append(multibranch_cm)
    
    # 3. 训练增强训练CNN模型(使用软标签和对抗训练)
    print("\n开始训练增强训练CNN模型...")
    enhanced_model = create_multibranch_model()
    enhanced_model, enhanced_history = train_enhanced_model(
        enhanced_model, 
        train_data, 
        valid_data,
        use_soft_labels=True,
        use_adversarial=True,
        use_cosine_lr=True
    )
    enhanced_metrics, enhanced_cm, _ = evaluate_model(enhanced_model, test_data)
    save_model_and_results(enhanced_model, enhanced_metrics, 'enhanced_cnn_model', RESULT_DIR)
    
    print("增强训练CNN模型训练完成！")
    print(f"测试集性能指标: 准确率={enhanced_metrics['accuracy']:.3f}, F1分数={enhanced_metrics['f1']:.3f}")
    
    models.append(enhanced_model)
    histories.append(enhanced_history)
    all_metrics.append(enhanced_metrics)
    cms.append(enhanced_cm)
    
    # 生成对比可视化
    print("\n生成对比可视化...")
    
    # 绘制训练历史
    plot_training_history(
        histories, 
        model_names, 
        os.path.join(RESULT_DIR, 'training_history_comparison.png')
    )
    
    # 绘制ROC曲线
    plot_roc_curves(
        models, 
        test_data, 
        model_names, 
        os.path.join(RESULT_DIR, 'roc_curves_comparison.png')
    )
    
    # 绘制混淆矩阵
    plot_confusion_matrices(
        cms, 
        model_names, 
        os.path.join(RESULT_DIR, 'confusion_matrices_comparison.png')
    )
    
    # 创建性能指标表格
    metrics_df = create_metrics_table(
        all_metrics, 
        model_names, 
        os.path.join(RESULT_DIR, 'performance_metrics.csv')
    )
    
    print("\n对比实验完成！结果已保存到:", RESULT_DIR)
    
    return models, histories, all_metrics, cms, metrics_df

if __name__ == "__main__":
    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"找到 {len(gpus)} 个 GPU 设备")
        except RuntimeError as e:
            print(e)
    
    # 运行对比实验
    run_experiments() 