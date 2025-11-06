# RDPG (Recurrent Deterministic Policy Gradient) 训练系统

基于LSTM的循环确定性策略梯度算法实现，用于强化学习任务。

## 项目结构

```
rdpg_project/
├── common/                  # 公共模块
│   ├── buffers.py          # 经验回放缓冲区
│   ├── value_networks.py   # 价值网络（Q网络）
│   ├── policy_networks.py  # 策略网络
│   ├── utils.py            # 工具函数
│   └── initialize.py       # 网络初始化
├── model/                   # 模型保存目录
├── logs/                    # 训练日志和可视化
├── train_rdpg.py           # 主训练和测试程序
├── reacher.py              # Reacher环境
└── README.md               # 本文件
```

## 功能特点

1. **LSTM网络结构**: 使用LSTM处理时序信息
2. **完整的RDPG算法**: Actor-Critic架构，带目标网络
3. **实时可视化**: 训练过程中实时显示奖励和损失曲线
4. **模型保存和加载**: 定期保存模型检查点
5. **测试模式**: 评估训练好的策略

## 安装依赖

```bash
pip install torch numpy matplotlib gymnasium
```

## 使用方法

### 1. 训练模型

```bash
# 基础训练（500回合）
python train_rdpg.py --train

# 自定义训练回合数
python train_rdpg.py --train --episodes 1000

# 调整每回合步数
python train_rdpg.py --train --episodes 500 --max_steps 200
```

### 2. 测试模型

```bash
# 测试模型
python train_rdpg.py --test

# 测试并渲染环境
python train_rdpg.py --test --render

# 自定义测试回合数
python train_rdpg.py --test --test_episodes 20
```

### 3. 训练并测试

```bash
python train_rdpg.py --train --test --episodes 300 --test_episodes 10
```

## 参数说明

- `--train`: 训练模式
- `--test`: 测试模式
- `--episodes`: 训练回合数（默认: 500）
- `--test_episodes`: 测试回合数（默认: 10）
- `--max_steps`: 每回合最大步数（默认: 100）
- `--render`: 测试时渲染环境

## 训练过程可视化

训练期间会实时显示三个图表：

1. **Episode Reward**: 每回合的总奖励和移动平均
2. **Q-Network Loss**: Q网络的损失曲线
3. **Policy Network Loss**: 策略网络的损失曲线

可视化图表会自动保存到 `logs/` 目录。

## 模型文件

训练好的模型保存在 `model/` 目录：

- `rdpg_q`: Q网络权重
- `rdpg_target_q`: 目标Q网络权重
- `rdpg_policy`: 策略网络权重

## 算法说明

RDPG (Recurrent Deterministic Policy Gradient) 是DDPG算法的循环神经网络版本：

- **Actor**: 使用LSTM的确定性策略网络
- **Critic**: 使用LSTM的Q值网络
- **目标网络**: 软更新的目标网络，提高训练稳定性
- **经验回放**: 存储整个episode用于批量训练

## 环境

默认使用 Gymnasium 的 Pendulum-v1 环境：

- **状态空间**: 3维（角度的sin, cos和角速度）
- **动作空间**: 1维连续动作（力矩）
- **目标**: 将摆杆倒立起来

## 注意事项

1. 训练需要一定时间，建议先用较少回合数测试
2. GPU加速会显著提升训练速度
3. 可根据任务调整超参数（学习率、batch size等）
4. 首次运行会自动创建必要的目录

## 示例输出

```
使用设备: cuda:0
状态空间: (3,)
动作空间: (1,)

训练参数:
  - 最大回合数: 500
  - 每回合最大步数: 100
  - Batch size: 3
  - Hidden dim: 64
  - Replay buffer size: 1000000

Episode    0 | Reward:  -1234.56 | Q Loss:  0.1234 | Policy Loss: -12.3456 | Buffer:    1 | Time: 1.2s
Episode   10 | Reward:   -987.65 | Q Loss:  0.0987 | Policy Loss:  -9.8765 | Buffer:   11 | Time: 12.3s
...
```

## 许可

MIT License
