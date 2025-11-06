#!/usr/bin/env python3
"""
可视化演示脚本
展示训练过程的可视化效果（使用模拟数据）
修复了PyCharm兼容性问题
"""

import numpy as np
import os

# 修复PyCharm兼容性 - 必须在import pyplot之前
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def simulate_training_data(num_episodes=100):
    """生成模拟的训练数据"""
    episodes = np.arange(num_episodes)

    # 模拟奖励提升过程
    rewards = -1500 + 1300 * (1 - np.exp(-episodes / 30))
    rewards += np.random.normal(0, 100, num_episodes)

    # 模拟Q损失下降
    q_losses = 10 * np.exp(-episodes / 20) + np.random.uniform(0, 0.5, num_episodes)

    # 模拟策略损失
    policy_losses = -20 + 10 * np.exp(-episodes / 25) + np.random.uniform(-2, 2, num_episodes)

    return episodes, rewards, q_losses, policy_losses


def create_visualization():
    """创建训练可视化"""
    print("=" * 70)
    print(" RDPG 训练过程可视化演示")
    print("=" * 70)
    print()
    print("这个脚本展示训练过程中的实时可视化效果")
    print("使用模拟数据来演示三个关键指标：")
    print("  1. Episode Reward - 每回合获得的总奖励")
    print("  2. Q-Network Loss - Q网络的训练损失")
    print("  3. Policy Network Loss - 策略网络的训练损失")
    print()

    try:
        # 生成模拟数据
        episodes, rewards, q_losses, policy_losses = simulate_training_data(100)

        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('RDPG Training Progress (Simulated)', fontsize=16, fontweight='bold')

        # 奖励曲线
        axes[0].plot(episodes, rewards, 'b-', alpha=0.6, label='Episode Reward', linewidth=1.5)
        window = 20
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(episodes[window - 1:], smoothed, 'r-', linewidth=2.5, label=f'Moving Avg ({window})')
        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel('Total Reward', fontsize=12)
        axes[0].set_title('Training Rewards', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=-200, color='g', linestyle='--', alpha=0.5, label='Target')

        # Q损失
        axes[1].plot(episodes, q_losses, 'g-', alpha=0.7, linewidth=1.5)
        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Q Loss', fontsize=12)
        axes[1].set_title('Q-Network Loss', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)

        # 策略损失
        axes[2].plot(episodes, policy_losses, 'm-', alpha=0.7, linewidth=1.5)
        axes[2].set_xlabel('Episode', fontsize=12)
        axes[2].set_ylabel('Policy Loss', fontsize=12)
        axes[2].set_title('Policy Network Loss', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        os.makedirs('./logs', exist_ok=True)
        save_path = './logs/visualization_demo.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 关闭图表释放内存

        print(f"✓ 可视化图表已保存到: {save_path}")
        print(f"✓ 请打开该文件查看训练可视化效果")

    except Exception as e:
        print(f"✗ 生成图表时出错: {e}")
        print("  请检查matplotlib是否正确安装")
        return

    print()
    print("=" * 70)
    print(" 可视化说明")
    print("=" * 70)
    print()
    print("1. Episode Reward (蓝色曲线)")
    print("   - 显示每个训练回合获得的总奖励")
    print("   - 红色曲线是移动平均，用于观察总体趋势")
    print("   - 理想情况下应该逐渐上升并趋于稳定")
    print()
    print("2. Q-Network Loss (绿色曲线)")
    print("   - Q网络的训练损失")
    print("   - 应该逐渐下降并趋于稳定")
    print("   - 表示价值估计的准确性在提高")
    print()
    print("3. Policy Network Loss (紫色曲线)")
    print("   - 策略网络的损失")
    print("   - 负值表示Q值在增加（策略在改进）")
    print("   - 训练过程中会有波动是正常的")
    print()
    print("✓ 运行成功！图表已保存，可以打开查看")
    print()


if __name__ == '__main__':
    try:
        create_visualization()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()