import os
import sys
import argparse
import time
import numpy as np

from ablation_experiment import run_all_experiments
from visualization import create_all_visualizations

def main(args):
    """
    运行PE恶意软件检测的Stacking模型消融实验
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子
    np.random.seed(42)
    
    # 设置结果目录
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 设置图表保存目录
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # 运行实验
    if not args.skip_experiments:
        print("\n===================== 开始运行消融实验 =====================")
        start_time = time.time()
        all_results = run_all_experiments(result_dir)
        end_time = time.time()
        
        print(f"\n实验完成! 总耗时: {(end_time - start_time)/60:.2f} 分钟")
    
    # 生成可视化
    print("\n===================== 开始生成可视化图表 =====================")
    create_all_visualizations(result_dir, figures_dir)
    print("\n可视化图表生成完成!")
    
    print(f"\n所有结果已保存到: {result_dir}")
    print(f"所有图表已保存到: {figures_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行PE恶意软件检测的Stacking模型消融实验')
    parser.add_argument('--skip-experiments', action='store_true', help='跳过实验，只生成可视化图表')
    args = parser.parse_args()
    
    main(args) 