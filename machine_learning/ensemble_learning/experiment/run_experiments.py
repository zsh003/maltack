import os
import sys
import tensorflow as tf
import numpy as np
import time
import argparse

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入实验和可视化模块
from comparative_experiment import run_experiments
from visualization import create_detailed_visualizations
from enhanced_cnn_model import load_data

def main(args):
    """运行PE恶意软件识别的对比实验"""
    # 设置随机种子
    np.random.seed(4396)
    tf.random.set_seed(4396)
    
    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"找到 {len(gpus)} 个 GPU 设备")
        except RuntimeError as e:
            print(e)
    
    # 创建结果目录
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 加载数据
    print("加载数据集...")
    train_data, valid_data, test_data = load_data()
    print("数据加载完成！")
    
    # 运行对比实验
    if not args.skip_training:
        print("\n开始运行对比实验...")
        start_time = time.time()
        models, histories, all_metrics, cms, metrics_df = run_experiments()
        end_time = time.time()
        print(f"对比实验完成！总耗时: {(end_time - start_time)/60:.2f} 分钟")
        
        # 显示性能指标表格
        print("\n模型性能指标对比:")
        print(metrics_df.to_string(index=False))
    
    # 创建详细可视化
    print("\n创建详细可视化分析...")
    create_detailed_visualizations(test_data, result_dir)
    print("可视化分析完成！")
    
    print(f"\n所有结果已保存到: {result_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行PE恶意软件识别的对比实验')
    parser.add_argument('--skip-training', action='store_true', help='跳过训练过程，只生成可视化分析')
    args = parser.parse_args()
    
    main(args) 