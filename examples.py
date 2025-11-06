#!/usr/bin/env python3
"""
RDPG 使用示例脚本
展示如何在代码中使用RDPG算法
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def example_basic_usage():
    """示例1: 基本使用"""
    print("="*70)
    print(" 示例1: 基本使用")
    print("="*70)
    print()
    
    code = """
# 导入必要的模块
import gymnasium as gym
from common.buffers import ReplayBufferLSTM2
from common.value_networks import QNetworkLSTM2
from common.policy_networks import DPG_PolicyNetworkLSTM2

# 创建环境
env = gym.make("Pendulum-v1")

# 创建经验回放缓冲区
replay_buffer = ReplayBufferLSTM2(capacity=100000)

# 创建RDPG算法
from train_rdpg import RDPG
alg = RDPG(replay_buffer, env.observation_space, env.action_space, hidden_dim=64)

# 训练循环
for episode in range(100):
    state, _ = env.reset()
    # ... 训练代码 ...
    """
    
    print("基本代码结构:")
    print(code)
    print()


def example_custom_environment():
    """示例2: 使用自定义环境"""
    print("="*70)
    print(" 示例2: 使用自定义环境")
    print("="*70)
    print()
    
    code = """
# 使用项目包含的路径规划环境
from common.path_env_gpt import RlGame

# 创建环境
# n: 场景ID (1-4), m: 难度, l: 地图大小, render: 是否渲染
env = RlGame(n=2, m=3, l=10, render=True)

# 获取环境信息
state_space = env.observation_space
action_space = env.action_space

# 创建RDPG算法
from train_rdpg import RDPG
from common.buffers import ReplayBufferLSTM2

replay_buffer = ReplayBufferLSTM2(capacity=100000)
alg = RDPG(replay_buffer, state_space, action_space, hidden_dim=64)

# 训练
state = env.reset()
# ... 训练循环 ...
    """
    
    print("自定义环境代码:")
    print(code)
    print()


def example_visualization():
    """示例3: 添加自定义可视化"""
    print("="*70)
    print(" 示例3: 自定义可视化")
    print("="*70)
    print()
    
    code = """
from train_rdpg import TrainingVisualizer

# 创建可视化器
visualizer = TrainingVisualizer(save_dir='./my_logs')

# 训练循环中更新数据
for episode in range(num_episodes):
    # ... 训练代码 ...
    total_reward = sum(episode_rewards)
    avg_q_loss = np.mean(q_losses)
    avg_policy_loss = np.mean(policy_losses)
    
    # 更新可视化
    visualizer.update(episode, total_reward, avg_q_loss, avg_policy_loss)
    
    # 每10回合更新图表
    if episode % 10 == 0:
        visualizer.plot()

# 保存最终结果
visualizer.save('final_results.png')
visualizer.close()
    """
    
    print("可视化代码:")
    print(code)
    print()


def example_model_operations():
    """示例4: 模型保存和加载"""
    print("="*70)
    print(" 示例4: 模型保存和加载")
    print("="*70)
    print()
    
    save_code = """
# 保存模型
alg.save_model('./model/my_rdpg_model')

# 这会创建三个文件:
# - my_rdpg_model_q          (Q网络)
# - my_rdpg_model_target_q   (目标Q网络)
# - my_rdpg_model_policy     (策略网络)
    """
    
    load_code = """
# 加载模型
alg.load_model('./model/my_rdpg_model')

# 设置为评估模式（测试时）
alg.policy_net.eval()
alg.qnet.eval()

# 使用加载的模型
state, _ = env.reset()
action, hidden_out = alg.policy_net.get_action(state, last_action, hidden_in, noise_scale=0.0)
    """
    
    print("保存模型:")
    print(save_code)
    print()
    print("加载模型:")
    print(load_code)
    print()


def example_hyperparameter_tuning():
    """示例5: 超参数调优"""
    print("="*70)
    print(" 示例5: 超参数调优")
    print("="*70)
    print()
    
    code = """
# 在RDPG类初始化时调整学习率
class RDPG():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim):
        # ... 其他初始化代码 ...
        
        # 调整学习率
        q_lr = 1e-4         # Q网络学习率（降低）
        policy_lr = 5e-4    # 策略网络学习率（降低）
        
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

# 在update方法中调整其他参数
def train_with_custom_params():
    for episode in range(num_episodes):
        # ... 收集经验 ...
        
        # 调整更新参数
        q_loss, policy_loss = alg.update(
            batch_size=5,              # 增加batch size
            gamma=0.95,                # 调整折扣因子
            soft_tau=5e-3,             # 调整软更新率
            target_update_delay=5      # 调整目标网络更新频率
        )
    """
    
    print("超参数调优:")
    print(code)
    print()
    print("常用超参数:")
    print("  - learning_rate: 1e-4 到 1e-3")
    print("  - hidden_dim: 32, 64, 128, 256")
    print("  - batch_size: 3, 5, 10")
    print("  - gamma: 0.95 到 0.99")
    print("  - soft_tau: 1e-3 到 1e-2")
    print()


def main():
    """主函数"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "RDPG 使用示例和最佳实践" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    examples = [
        ("1", "基本使用", example_basic_usage),
        ("2", "自定义环境", example_custom_environment),
        ("3", "可视化", example_visualization),
        ("4", "模型操作", example_model_operations),
        ("5", "超参数调优", example_hyperparameter_tuning),
    ]
    
    print("可用示例:")
    for num, name, _ in examples:
        print(f"  [{num}] {name}")
    print("  [0] 全部显示")
    print()
    
    try:
        choice = input("选择示例 (0-5): ").strip()
        print()
        
        if choice == "0":
            for _, _, func in examples:
                func()
        else:
            for num, _, func in examples:
                if choice == num:
                    func()
                    break
            else:
                print("无效选择")
        
        print()
        print("="*70)
        print(" 更多信息")
        print("="*70)
        print()
        print("查看完整文档:")
        print("  - README.md: 详细说明")
        print("  - QUICKSTART.md: 快速入门")
        print("  - PROJECT_SUMMARY.md: 项目总结")
        print()
        print("运行示例:")
        print("  python train_rdpg.py --train --episodes 100")
        print("  python visualize_demo.py")
        print("  python demo.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n错误: {e}")


if __name__ == '__main__':
    main()
