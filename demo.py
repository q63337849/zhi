#!/usr/bin/env python3
"""
RDPG 快速演示脚本
运行少量回合的训练来演示系统功能
"""

import subprocess
import sys

def main():
    print("="*70)
    print(" RDPG 快速演示")
    print("="*70)
    print()
    print("这个脚本将运行一个快速演示：")
    print("  1. 训练 50 个回合")
    print("  2. 测试训练好的模型 5 次")
    print()
    input("按 Enter 键开始...")
    
    # 训练
    print("\n" + "="*70)
    print(" 步骤 1: 训练模型（50回合）")
    print("="*70)
    try:
        subprocess.run([
            sys.executable, "train_rdpg.py",
            "--train",
            "--episodes", "50",
            "--max_steps", "100"
        ], check=True)
    except subprocess.CalledProcessError:
        print("训练失败!")
        return
    
    # 测试
    print("\n" + "="*70)
    print(" 步骤 2: 测试模型（5回合）")
    print("="*70)
    try:
        subprocess.run([
            sys.executable, "train_rdpg.py",
            "--test",
            "--test_episodes", "5"
        ], check=True)
    except subprocess.CalledProcessError:
        print("测试失败!")
        return
    
    print("\n" + "="*70)
    print(" 演示完成!")
    print("="*70)
    print()
    print("查看结果：")
    print("  - 训练曲线: logs/final_training_progress.png")
    print("  - 模型文件: model/rdpg_*")
    print()
    print("运行完整训练：")
    print("  python train_rdpg.py --train --episodes 500")
    print()

if __name__ == '__main__':
    main()
