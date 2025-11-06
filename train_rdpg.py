#!/usr/bin/env python3
"""
RDPG (Recurrent Deterministic Policy Gradient) 训练和测试程序
包含实时可视化功能
"""

import os
import sys
import argparse
import numpy as np

# 修复matplotlib在PyCharm中的兼容性问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from datetime import datetime
import time

# 添加路径
sys.path.append(os.path.dirname(__file__))

from common.buffers import ReplayBufferLSTM2
from common.value_networks import QNetworkLSTM2
from common.policy_networks import DPG_PolicyNetworkLSTM2
from common.path_env_gpt import CurriculumRobotEnv
from gymnasium import spaces

# GPU设置
GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if GPU else "cpu")
print(f"使用设备: {device}")


class RDPG():
    """RDPG算法类"""
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim):
        self.replay_buffer = replay_buffer
        self.hidden_dim = hidden_dim
        
        self.qnet = QNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.target_qnet = QNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.policy_net = DPG_PolicyNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.target_policy_net = DPG_PolicyNetworkLSTM2(state_space, action_space, hidden_dim).to(device)

        print('Q network: ', self.qnet)
        print('Policy network: ', self.policy_net)

        for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)
        
        self.q_criterion = nn.MSELoss()
        q_lr = 1e-3
        policy_lr = 1e-3
        self.update_cnt = 0

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    def target_soft_update(self, net, target_net, soft_tau):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return target_net

    def update(self, batch_size, gamma=0.99, soft_tau=1e-2, target_update_delay=3):
        self.update_cnt += 1
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = \
            self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
    
        # Critic update
        with torch.no_grad():
            new_next_action, _ = self.target_policy_net.evaluate(next_state, action, hidden_out)
            target_q, _ = self.target_qnet(next_state, new_next_action, action, hidden_out)
            target_q = reward + (1 - done) * gamma * target_q
    
        predict_q, _ = self.qnet(state, action, last_action, hidden_in)
        q_loss = self.q_criterion(predict_q, target_q)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
    
        # Actor update
        for p in self.qnet.parameters():
            p.requires_grad_(False)
    
        new_action, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        actor_q, _ = self.qnet(state, new_action, last_action, hidden_in)
        policy_loss = -actor_q.mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
    
        for p in self.qnet.parameters():
            p.requires_grad_(True)
    
        # Soft update targets
        if self.update_cnt % target_update_delay == 0:
            self.target_qnet = self.target_soft_update(self.qnet, self.target_qnet, soft_tau)
            self.target_policy_net = self.target_soft_update(
                self.policy_net, self.target_policy_net, soft_tau
            )
    
        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.qnet.state_dict(), path + '_q')
        torch.save(self.target_qnet.state_dict(), path + '_target_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')
        print(f"模型已保存到: {path}")

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path + '_q', map_location=device))
        self.target_qnet.load_state_dict(torch.load(path + '_target_q', map_location=device))
        self.policy_net.load_state_dict(torch.load(path + '_policy', map_location=device))
        self.qnet.eval()
        self.target_qnet.eval()
        self.policy_net.eval()
        print(f"模型已加载: {path}")


class NormalizedActions(gym.ActionWrapper):
    """动作归一化包装器"""

    def __init__(self, env):
        super().__init__(env)
        self._orig_low = self.env.action_space.low
        self._orig_high = self.env.action_space.high
        self.action_space = spaces.Box(
            low=-np.ones_like(self._orig_low, dtype=np.float32),
            high=np.ones_like(self._orig_high, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        low = self._orig_low
        high = self._orig_high
        scaled_action = low + (action + 1.0) * 0.5 * (high - low)
        return np.clip(scaled_action, low, high)

    def reverse_action(self, action):
        low = self._orig_low
        high = self._orig_high
        normalized_action = 2 * (action - low) / (high - low) - 1
        return np.clip(normalized_action, self.action_space.low, self.action_space.high)


class TrainingVisualizer:
    """训练过程可视化类（非交互模式，兼容PyCharm）"""
    def __init__(self, save_dir='./logs', enable_plot=True):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.rewards = []
        self.q_losses = []
        self.policy_losses = []
        self.episodes = []
        self.enable_plot = enable_plot
        
    def update(self, episode, reward, q_loss, policy_loss):
        """更新数据"""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.q_losses.append(q_loss if q_loss is not None else 0)
        self.policy_losses.append(policy_loss if policy_loss is not None else 0)
        
    def plot(self):
        """绘制图表（非交互模式）"""
        if not self.enable_plot or len(self.episodes) == 0:
            return
        
        try:
            # 创建新图表
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle('RDPG Training Progress', fontsize=16, fontweight='bold')
            
            # 奖励曲线
            axes[0].plot(self.episodes, self.rewards, 'b-', alpha=0.6, label='Episode Reward')
            if len(self.rewards) > 10:
                window = min(20, len(self.rewards) // 5)
                if window > 0:
                    smoothed = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
                    axes[0].plot(self.episodes[window-1:], smoothed, 'r-', 
                                    linewidth=2, label=f'Moving Avg ({window})')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Total Reward')
            axes[0].set_title('Training Rewards')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Q损失
            if len(self.q_losses) > 0:
                axes[1].plot(self.episodes, self.q_losses, 'g-', alpha=0.6)
                axes[1].set_xlabel('Episode')
                axes[1].set_ylabel('Q Loss')
                axes[1].set_title('Q-Network Loss')
                axes[1].grid(True, alpha=0.3)
            
            # 策略损失
            if len(self.policy_losses) > 0:
                axes[2].plot(self.episodes, self.policy_losses, 'm-', alpha=0.6)
                axes[2].set_xlabel('Episode')
                axes[2].set_ylabel('Policy Loss')
                axes[2].set_title('Policy Network Loss')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存当前图表
            save_path = os.path.join(self.save_dir, 'current_progress.png')
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)  # 关闭图表释放内存
            
        except Exception as e:
            print(f"警告: 绘图时出错 - {e}")
            self.enable_plot = False  # 禁用后续绘图
        
    def save(self, filename='training_progress.png'):
        """保存最终图表"""
        if not self.enable_plot or len(self.episodes) == 0:
            print("跳过图表保存（无数据或绘图已禁用）")
            return
        
        try:
            # 创建最终图表
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle('RDPG Training Progress - Final', fontsize=16, fontweight='bold')
            
            # 奖励曲线
            axes[0].plot(self.episodes, self.rewards, 'b-', alpha=0.6, label='Episode Reward')
            if len(self.rewards) > 10:
                window = min(20, len(self.rewards) // 5)
                if window > 0:
                    smoothed = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
                    axes[0].plot(self.episodes[window-1:], smoothed, 'r-', 
                                    linewidth=2, label=f'Moving Avg ({window})')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Total Reward')
            axes[0].set_title('Training Rewards')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Q损失
            if len(self.q_losses) > 0:
                axes[1].plot(self.episodes, self.q_losses, 'g-', alpha=0.6)
                axes[1].set_xlabel('Episode')
                axes[1].set_ylabel('Q Loss')
                axes[1].set_title('Q-Network Loss')
                axes[1].grid(True, alpha=0.3)
            
            # 策略损失
            if len(self.policy_losses) > 0:
                axes[2].plot(self.episodes, self.policy_losses, 'm-', alpha=0.6)
                axes[2].set_xlabel('Episode')
                axes[2].set_ylabel('Policy Loss')
                axes[2].set_title('Policy Network Loss')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(self.save_dir, filename)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"图表已保存到: {save_path}")
            
        except Exception as e:
            print(f"警告: 保存图表时出错 - {e}")
        
        # 保存数据（始终尝试）
        try:
            data_path = os.path.join(self.save_dir, 'training_data.npz')
            np.savez(data_path, 
                    episodes=self.episodes,
                    rewards=self.rewards,
                    q_losses=self.q_losses,
                    policy_losses=self.policy_losses)
            print(f"训练数据已保存到: {data_path}")
        except Exception as e:
            print(f"警告: 保存数据时出错 - {e}")
    
    def close(self):
        """清理资源"""
        pass  # 非交互模式无需清理


def train(args):
    """训练函数"""
    print("\n" + "="*60)
    print("开始训练RDPG算法")
    print("="*60)
    
    # 创建环境（动态场景路径规划）
    env_base = CurriculumRobotEnv(
        scene_id=args.scene_id,
        max_steps=args.max_steps,
        map_size=args.map_size,
    )
    env = NormalizedActions(env_base)
    action_space = env.action_space
    state_space = env.observation_space

    print(f"状态空间: {state_space.shape}")
    print(f"动作空间: {action_space.shape}")
    
    # 超参数
    hidden_dim = 64
    batch_size = 3
    update_itr = 1
    replay_buffer_size = int(1e6)
    max_episodes = args.episodes
    max_steps = args.max_steps
    
    # 创建replay buffer和算法
    replay_buffer = ReplayBufferLSTM2(replay_buffer_size)
    model_path = './model/rdpg'
    alg = RDPG(replay_buffer, state_space, action_space, hidden_dim)
    
    # 创建可视化器
    enable_plot = not getattr(args, 'no_plot', False)  # 默认启用绘图
    visualizer = TrainingVisualizer(save_dir='./logs', enable_plot=enable_plot)
    
    print(f"\n训练参数:")
    print(f"  - 最大回合数: {max_episodes}")
    print(f"  - 每回合最大步数: {max_steps}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Replay buffer size: {replay_buffer_size}")
    print(f"  - 场景编号: {args.scene_id}")
    print(f"  - 地图尺寸: {args.map_size}")
    print(f"  - 可视化: {'禁用' if not enable_plot else '启用'}")
    print()
    
    # 训练循环
    start_time = time.time()
    
    for i_episode in range(max_episodes):
        q_loss_list = []
        policy_loss_list = []
        
        state, _ = env.reset()
        last_action = np.zeros(env.action_space.shape, dtype=np.float32)
        
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        
        hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                     torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))
        
        for step in range(max_steps):
            hidden_in = hidden_out
            action, hidden_out = alg.policy_net.get_action(state, last_action, hidden_in)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if step == 0:
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out
            
            episode_state.append(state)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_done.append(done)
            
            state = next_state
            last_action = action
            
            if len(replay_buffer) > batch_size:
                for _ in range(update_itr):
                    q_loss, policy_loss = alg.update(batch_size)
                    q_loss_list.append(q_loss)
                    policy_loss_list.append(policy_loss)
            
            if done:
                break
        
        # 保存episode到replay buffer
        replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action,
                          episode_last_action, episode_reward, episode_next_state, episode_done)
        
        # 计算统计信息
        total_reward = np.sum(episode_reward)
        avg_q_loss = np.average(q_loss_list) if len(q_loss_list) > 0 else 0
        avg_policy_loss = np.average(policy_loss_list) if len(policy_loss_list) > 0 else 0
        
        # 更新可视化
        visualizer.update(i_episode, total_reward, avg_q_loss, avg_policy_loss)
        
        # 打印进度
        if i_episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {i_episode:4d} | Reward: {total_reward:8.2f} | "
                  f"Q Loss: {avg_q_loss:7.4f} | Policy Loss: {avg_policy_loss:7.4f} | "
                  f"Buffer: {len(replay_buffer):5d} | Time: {elapsed_time:.1f}s")
            visualizer.plot()
        
        # 定期保存模型
        if i_episode % 50 == 0 and i_episode > 0:
            alg.save_model(model_path)
            visualizer.save(f'training_progress_ep{i_episode}.png')
    
    # 训练结束
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    alg.save_model(model_path)
    visualizer.save('final_training_progress.png')
    visualizer.close()
    
    total_time = time.time() - start_time
    print(f"\n总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"最终平均奖励 (最后10回合): {np.mean(visualizer.rewards[-10:]):.2f}")


def test(args):
    """测试函数"""
    print("\n" + "="*60)
    print("开始测试RDPG算法")
    print("="*60)
    
    # 创建环境
    env_base = CurriculumRobotEnv(
        scene_id=args.scene_id,
        max_steps=args.max_steps,
        map_size=args.map_size,
        render_mode="human" if args.render else None,
    )
    env = NormalizedActions(env_base)
    action_space = env.action_space
    state_space = env.observation_space
    print(f"测试环境: 场景 {args.scene_id}, 地图尺寸 {args.map_size}, 渲染={'开启' if args.render else '关闭'}")
    
    # 超参数
    hidden_dim = 64
    test_episodes = args.test_episodes
    max_steps = args.max_steps
    
    # 创建replay buffer和算法（只用于加载模型）
    replay_buffer = ReplayBufferLSTM2(100)
    model_path = './model/rdpg'
    alg = RDPG(replay_buffer, state_space, action_space, hidden_dim)
    
    # 加载模型
    try:
        alg.load_model(model_path)
    except Exception as e:
        print(f"错误: 无法加载模型 - {e}")
        print("请先运行训练: python train_rdpg.py --train")
        return
    
    # 测试循环
    test_rewards = []
    
    for i_episode in range(test_episodes):
        state, _ = env.reset()
        last_action = np.zeros(action_space.shape, dtype=np.float32)
        hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                     torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))
        
        episode_reward = 0
        
        for step in range(max_steps):
            hidden_in = hidden_out
            action, hidden_out = alg.policy_net.get_action(state, last_action, hidden_in, noise_scale=0.0)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            last_action = action
            state = next_state
            episode_reward += reward
            
            if args.render:
                env.render()
                time.sleep(0.02)
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {i_episode:3d} | Reward: {episode_reward:8.2f}")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print(f"平均奖励: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"最好奖励: {np.max(test_rewards):.2f}")
    print(f"最差奖励: {np.min(test_rewards):.2f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='RDPG训练和测试程序')
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--test_episodes', type=int, default=10, help='测试回合数')
    parser.add_argument('--max_steps', type=int, default=150, help='每回合最大步数')
    parser.add_argument('--scene-id', type=int, default=4, help='路径规划场景编号（4 为动态障碍场景）')
    parser.add_argument('--map_size', type=float, default=10.0, help='地图大小')
    parser.add_argument('--render', action='store_true', help='测试时渲染环境')
    parser.add_argument('--no-plot', action='store_true', help='禁用训练可视化（加快速度）')
    
    args = parser.parse_args()
    
    if not args.train and not args.test:
        print("请指定模式: --train 或 --test")
        parser.print_help()
        return
    
    if args.train:
        train(args)
    
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
