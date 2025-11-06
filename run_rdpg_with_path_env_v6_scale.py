# -*- coding: utf-8 -*-
"""
RDPG + 路径环境(v6，带动作规范化/最小速度约束/可配置参数)
用法示例：
  训练：
  python run_rdpg_with_path_env_v6_scale.py --train --scene_id 1 --max_episodes 500 --model_path ./model/rdpg_path --min_speed 0.2

  测试：
  python run_rdpg_with_path_env_v6_scale.py --test --model_path ./model/rdpg_path
"""
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# === 环境：使用你上传的 path_env_gpt.RlGame 兼容包装器 ===
# RlGame 包装了 CurriculumRobotEnv，提供 observation_space / action_space / reset / step 等接口
from path_env_gpt import RlGame  # 已存在于你的 path_env_gpt.py 中
# 说明见：动作空间 Box(low=[0,-1.5], high=[1,1.5])，状态维度10。  # :contentReference[oaicite:4]{index=4}

# === 算法：使用你上传的 rdpg.RDPG 类与序列回放缓冲 ===
from common.buffers import ReplayBufferLSTM2     # 与 rdpg.py 中的使用一致  # :contentReference[oaicite:5]{index=5}
from rdpg import RDPG                            # 你上传的 RDPG 实现（含 update/save/load）  # :contentReference[oaicite:6]{index=6}

# -----------------------------
# 工具函数
# -----------------------------
def device_select(idx: int = 0):
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{idx}")
        cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
    print(f"使用设备: {dev}")
    return dev

def ensure_dir(path: str):
    if path and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

def map_policy_action_to_env(raw_action: np.ndarray,
                             min_speed: float = 0.0,
                             max_speed: float = 1.0) -> np.ndarray:
    """
    策略网络输出 raw_action ∈ [-1,1]^2
    线速度: 映射到 [0,1] 再做下限约束；角速度: 映射到 [-1.5,1.5]
    """
    raw_action = np.asarray(raw_action, dtype=np.float32)
    # 线速度：[-1,1] → [0,1]
    v = (raw_action[0] + 1.0) * 0.5
    v = float(np.clip(v, min_speed, max_speed))
    # 角速度：[-1,1] → [-1.5, 1.5]
    w = float(np.clip(raw_action[1] * 1.5, -1.5, 1.5))
    return np.array([v, w], dtype=np.float32)

# -----------------------------
# 训练与测试流程
# -----------------------------
def train(args):
    device = device_select(args.gpu)
    # 环境构造（RlGame 的 n/m/l 分别对应 scene_id / 占位 / 地图尺寸）
    env = RlGame(n=args.scene_id, m=3, l=args.map_size, render=args.render)  # :contentReference[oaicite:7]{index=7}
    state_space, action_space = env.observation_space, env.action_space

    # 回放缓冲区（序列版）
    replay_buffer = ReplayBufferLSTM2(max_size=int(args.replay_size))  # 与 rdpg.py 使用一致  # :contentReference[oaicite:8]{index=8}

    # 算法
    alg = RDPG(replay_buffer, state_space, action_space, hidden_dim=args.hidden_dim)  # 构造签名同 rdpg.py  # :contentReference[oaicite:9]{index=9}

    # 训练超参
    max_episodes = args.max_episodes
    max_steps = args.max_steps
    batch_size = args.batch_size
    update_itr = args.update_itr
    warmup_episodes = args.warmup_episodes
    save_interval = args.save_interval
    min_speed = args.min_speed
    ensure_dir(args.model_path)

    rewards_hist = []

    for ep in range(max_episodes):
        # 每回合缓存（序列）
        ep_states, ep_actions_raw, ep_last_actions_raw = [], [], []
        ep_rewards, ep_next_states, ep_dones = [], [], []

        state = env.reset()
        # 上一步动作（raw，保持 [-1,1] 空间以匹配网络拼接尺度）
        last_action_raw = np.zeros(action_space.shape[0], dtype=np.float32)

        # 初始化 LSTM 隐状态（注意：rdpg.RDPG 内部 update 里会用传入的 hidden_in/out）
        hdim = args.hidden_dim
        hidden_out = (torch.zeros([1, 1, hdim], dtype=torch.float32, device=device),
                      torch.zeros([1, 1, hdim], dtype=torch.float32, device=device))

        q_loss_list, p_loss_list = [], []

        for t in range(max_steps):
            hidden_in = hidden_out
            # 策略网络取动作（raw_action ∈ [-1,1]^2）
            raw_action, hidden_out = alg.policy_net.get_action(state, last_action_raw, hidden_in)

            # 供环境执行的动作（做映射与最小速度约束）
            env_action = map_policy_action_to_env(raw_action, min_speed=min_speed, max_speed=1.0)

            # 与环境交互
            next_state, reward, done, _ = env.step(env_action)
            if args.render:
                env.render()

            # 记录序列
            if t == 0:
                ini_hidden_in, ini_hidden_out = hidden_in, hidden_out
            ep_states.append(state)
            ep_actions_raw.append(np.array(raw_action, dtype=np.float32))          # 存 raw
            ep_last_actions_raw.append(np.array(last_action_raw, dtype=np.float32))# 存 raw
            ep_rewards.append(float(reward))
            ep_next_states.append(next_state)
            ep_dones.append(bool(done))

            # 滚动
            state = next_state
            last_action_raw = raw_action

            # 训练更新（在有足够回合后进行）
            if (len(replay_buffer) > batch_size) and (ep >= warmup_episodes):
                for _ in range(update_itr):
                    q_loss, p_loss = alg.update(batch_size)
                    q_loss_list.append(q_loss)
                    p_loss_list.append(p_loss)

            if done:
                break

        # 把完整回合推入序列回放缓冲
        replay_buffer.push(ini_hidden_in, ini_hidden_out,
                           ep_states, ep_actions_raw, ep_last_actions_raw,
                           ep_rewards, ep_next_states, ep_dones)

        ep_return = float(np.sum(ep_rewards))
        rewards_hist.append(ep_return)

        # 日志
        ql = np.mean(q_loss_list) if q_loss_list else np.nan
        pl = np.mean(p_loss_list) if p_loss_list else np.nan
        print(f"[Train] Ep {ep:4d} | Return={ep_return:8.2f} | QL={ql:.5f} | PL={pl:.5f} | Buffer={len(replay_buffer)}")

        # 保存
        if (ep + 1) % save_interval == 0:
            alg.save_model(args.model_path)
            print(f"[SAVE] 已保存模型到前缀: {args.model_path}")

    # 训练结束再存一次
    alg.save_model(args.model_path)
    print(f"[FINAL SAVE] 已保存模型到前缀: {args.model_path}")


def test(args):
    device = device_select(args.gpu)
    env = RlGame(n=args.scene_id, m=3, l=args.map_size, render=True)  # 测试默认渲染
    state_space, action_space = env.observation_space, env.action_space

    # 构造一个“空”回放，仅满足 RDPG 构造签名
    replay_buffer = ReplayBufferLSTM2(max_size=1)
    alg = RDPG(replay_buffer, state_space, action_space, hidden_dim=args.hidden_dim)
    # 读取模型
    alg.load_model(args.model_path)
    print(f"[LOAD] 已加载模型前缀: {args.model_path}")

    test_episodes = args.test_episodes
    max_steps = args.max_steps
    min_speed = args.min_speed

    for ep in range(test_episodes):
        state = env.reset()
        last_action_raw = np.zeros(action_space.shape[0], dtype=np.float32)
        hdim = args.hidden_dim
        hidden_out = (torch.zeros([1, 1, hdim], dtype=torch.float32, device=device),
                      torch.zeros([1, 1, hdim], dtype=torch.float32, device=device))

        ep_ret = 0.0
        for t in range(max_steps):
            hidden_in = hidden_out
            raw_action, hidden_out = alg.policy_net.get_action(state, last_action_raw, hidden_in)
            env_action = map_policy_action_to_env(raw_action, min_speed=min_speed, max_speed=1.0)
            next_state, reward, done, info = env.step(env_action)
            env.render()

            ep_ret += float(reward)
            state = next_state
            last_action_raw = raw_action
            if done:
                break
        print(f"[Test] Ep {ep:3d} | Return={ep_ret:8.2f}")


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="训练模式")
    mode.add_argument("--test", action="store_true", help="测试模式")

    # 环境相关
    p.add_argument("--scene_id", type=int, default=1, help="场景编号（传给 RlGame 的 n）")
    p.add_argument("--map_size", type=float, default=10.0, help="地图尺寸（传给 RlGame 的 l）")
    p.add_argument("--render", action="store_true", help="渲染")

    # 训练相关
    p.add_argument("--max_episodes", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--update_itr", type=int, default=1)
    p.add_argument("--warmup_episodes", type=int, default=5)
    p.add_argument("--replay_size", type=float, default=1e6)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--save_interval", type=int, default=20)

    # 模型路径/设备
    p.add_argument("--model_path", type=str, default="./model/rdpg_path", help="模型文件前缀，不含后缀")
    p.add_argument("--gpu", type=int, default=0)

    # 动作约束
    p.add_argument("--min_speed", type=float, default=0.0, help="线速度最小值（映射后下限）")

    # 测试
    p.add_argument("--test_episodes", type=int, default=10)

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.train:
        train(args)
    elif args.test:
        test(args)
