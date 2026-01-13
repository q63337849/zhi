"""
path_env_sim2real_v3.py

Sim-to-Real 最终版本 - 无GPS依赖

修复内容 (v3 - 2024-12-01):
===========================
1. 起点和目标点附近 1m 范围内不生成障碍物
2. 起点和目标点距离保证 >= 6m
3. 静态障碍物之间保持 >= 0.8m 的间距，确保无人机可安全通过
4. 动态障碍物触墙反弹改进：增加边界缓冲区，避免连续触墙和沿墙滑行

状态空间设计 (9 + N_beams 维):
====================
  [0-1]   cos/sin(θ)           机器人朝向      IMU
  [2]     v_lin_norm           线速度          编码器
  [3]     v_ang_norm           角速度          陀螺仪
  [4-5]   cos/sin(θ_rel)       目标相对方向    里程计+任务系统
  [6]     d_goal_norm          目标距离        里程计+任务系统
  [7-8]   prev_action          上一动作        自身记录
  [9-24]  lidar[16]            激光雷达        LiDAR传感器

动作空间:
- v_lin ∈ [0, 1] (归一化线速度，只能前进)
- v_ang ∈ [-0.8, 0.8] (rad/s)
"""

import math
from typing import List, Dict, Optional, Tuple
import numpy as np

# 兼容 gym 和 gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class SmoothDynamicObstacle:
    """
    平滑曲线运动的动态障碍物

    改进 (v3):
    - 增加边界缓冲区 (boundary_buffer)
    - 改进反弹逻辑，避免连续触墙
    - 反弹后随机偏转，避免沿墙滑行
    """

    def __init__(
        self,
        init_pos: np.ndarray,
        speed: float = 0.4,
        radius: float = 0.25,
        map_size: float = 10.0,
        wall_boundary: float = 0.3,
        turn_prob: float = 0.05,
        max_turn_rate: float = math.radians(45.0),
        turn_smooth: float = 0.1,
        boundary_buffer: float = 0.5,
        variable_speed: bool = False,
        speed_min: Optional[float] = None,
        speed_max: Optional[float] = None,
        accel_std: float = 1.0,
        accel_prob: float = 0.3,
    ):
        self.init_pos = np.array(init_pos, dtype=np.float32)
        self.pos = self.init_pos.copy()
        self.speed = float(speed)
        self.radius = float(radius)
        self.map_size = float(map_size)
        self.wall_boundary = float(wall_boundary)
        self.turn_prob = float(turn_prob)
        self.max_turn_rate = float(max_turn_rate)
        self.turn_smooth = float(turn_smooth)
        self.boundary_buffer = float(boundary_buffer)  # ✅ 边界缓冲区

        self.heading = np.random.uniform(-math.pi, math.pi)
        self.target_heading = self.heading
        self.vel = np.zeros(2, dtype=np.float32)

        self.bounce_cooldown = 0

        # ✅ 新增: 变速配置
        self.variable_speed = bool(variable_speed)
        # 若未显式给出，就围绕初始 speed 做一个合理区间
        self.speed_min = float(self.speed * 0.5) if speed_min is None else float(speed_min)
        self.speed_max = float(self.speed * 1.5) if speed_max is None else float(speed_max)
        self.accel_std = float(accel_std)
        self.accel_prob = float(accel_prob)
        self.accel = 0.0  # 当前加速度

    def reset(self):
        self.pos = self.init_pos.copy()
        self.heading = np.random.uniform(-math.pi, math.pi)
        self.target_heading = self.heading
        self.vel[:] = 0.0
        self.bounce_cooldown = 0

    def update(self, dt: float = 0.1):
        # 冷却计时
        if self.bounce_cooldown > 0:
            self.bounce_cooldown -= 1

        # 随机改变目标朝向
        if np.random.rand() < self.turn_prob:
            delta = np.random.uniform(-self.max_turn_rate, self.max_turn_rate)
            self.target_heading = self.heading + float(delta)

        # 平滑转向
        self.heading = (1.0 - self.turn_smooth) * self.heading + self.turn_smooth * self.target_heading
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))

        if self.variable_speed:
            # 随机更新一次加速度（类似随机游走）
            if np.random.rand() < self.accel_prob:
                self.accel = np.random.normal(0.0, self.accel_std)

            # 根据加速度更新速度，并裁剪到安全区间
            self.speed += self.accel * dt
            self.speed = float(np.clip(self.speed, self.speed_min, self.speed_max))

        # 移动
        dx = self.speed * math.cos(self.heading) * dt
        dy = self.speed * math.sin(self.heading) * dt
        new_pos = self.pos + np.array([dx, dy], dtype=np.float32)

        # ✅ 改进: 使用更大的缓冲区检测边界
        min_coord = self.wall_boundary + self.radius + self.boundary_buffer
        max_coord = self.map_size - self.wall_boundary - self.radius - self.boundary_buffer

        # ✅ 改进: 只在冷却结束后才反弹，避免连续反弹
        if self.bounce_cooldown == 0:
            bounced = False

            if new_pos[0] < min_coord or new_pos[0] > max_coord:
                # 水平方向反弹 + 随机偏转
                self.heading = math.pi - self.heading
                # ✅ 新增: 随机偏转 ±30°，避免沿墙滑行
                self.heading += np.random.uniform(-math.radians(30), math.radians(30))
                bounced = True

            if new_pos[1] < min_coord or new_pos[1] > max_coord:
                # 垂直方向反弹 + 随机偏转
                self.heading = -self.heading
                self.heading += np.random.uniform(-math.radians(30), math.radians(30))
                bounced = True

            if bounced:
                # 归一化角度
                self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
                self.target_heading = self.heading
                # ✅ 设置冷却时间 (约0.5秒)
                self.bounce_cooldown = 5

                # 反弹后重新计算位移
                dx = self.speed * math.cos(self.heading) * dt
                dy = self.speed * math.sin(self.heading) * dt
                new_pos = self.pos + np.array([dx, dy], dtype=np.float32)

        # 硬边界限制 (使用原始边界，不含缓冲区)
        hard_min = self.wall_boundary + self.radius
        hard_max = self.map_size - self.wall_boundary - self.radius
        new_pos[0] = np.clip(new_pos[0], hard_min, hard_max)
        new_pos[1] = np.clip(new_pos[1], hard_min, hard_max)

        self.pos = new_pos
        # 使用 heading 计算速度向量
        self.vel = np.array(
            [self.speed * math.cos(self.heading),
             self.speed * math.sin(self.heading)],
            dtype=np.float32
        )


class Sim2RealReward:
    """
    成功率优先的稳健版奖励（dt=0.1, v<=1.0, max_steps=400）
    - 不显式加入预测安全项（TTC/rollout）
    - 用连续势能 + closing趋势惩罚提升动态环境鲁棒性
    - 量级对齐：goal_bonus不过大；collision_penalty足够大；每步进度与安全同阶
    """

    def __init__(self, dt: float = 0.1, v_max: float = 1.0, max_steps: int = 400):
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.max_steps = int(max_steps)

        # ===== 终止项（成功率优先但不赌博）=====
        self.goal_bonus = 260.0
        self.collision_penalty = -750.0

        # ===== 进度与时间 =====
        self.w_progress = 1.2
        self.step_penalty = -0.03

        # ===== 障碍势能（连续软斥力 + 内圈强barrier）=====
        self.d_soft = 3.2        # 软斥力“提前给梯度”
        self.sigma_soft = 1.1
        self.w_soft = 0.40       # 成功率优先：软斥力不要太大

        self.d_safe = 1.05       # 内圈强惩罚
        self.w_barrier = 24.0

        # ===== closing趋势惩罚（动态逼近敏感，但不算TTC）=====
        self.w_closing = 0.9
        self.closing_clip = 3.0  # m/s

        self.prev_min_dist = None

    def reset(self):
        self.prev_min_dist = None

    def compute(
        self,
        robot_pos: np.ndarray,
        goal_pos: np.ndarray,
        prev_goal_dist: float,
        static_obstacles: List[Dict],
        dynamic_obstacles: List,
        robot_radius: float,
        collision: bool,
        success: bool,
    ) -> float:
        if success:
            return float(self.goal_bonus)
        if collision:
            return float(self.collision_penalty)

        # 1) 进度（按 v_max*dt 归一化，尺度稳定）
        dist_to_goal = float(np.linalg.norm(goal_pos - robot_pos))
        progress = float(prev_goal_dist - dist_to_goal)  # m/step
        denom = self.v_max * self.dt + 1e-6
        progress_norm = np.clip(progress / denom, -1.0, 1.0)
        reward = self.w_progress * float(progress_norm) + self.step_penalty

        # 2) 最近障碍表面距离 min_dist（静态+动态）
        min_dist = float('inf')
        for obs in static_obstacles:
            d = float(np.linalg.norm(robot_pos - obs['pos'])) - float(obs['radius']) - float(robot_radius)
            min_dist = min(min_dist, d)
        for dyn in dynamic_obstacles:
            d = float(np.linalg.norm(robot_pos - dyn.pos)) - float(dyn.radius) - float(robot_radius)
            min_dist = min(min_dist, d)

        min_dist_safe = max(float(min_dist), 0.0)

        # 3) 连续软斥力（不分段，提前提供梯度）
        #    值域大致在 [-w_soft, 0]
        reward -= self.w_soft * float(np.exp(-min_dist_safe / self.sigma_soft))

        # 4) 内圈强barrier（接近d_safe快速变陡）
        if min_dist < self.d_safe:
            x = (self.d_safe - min_dist_safe) / max(self.d_safe, 1e-6)
            reward -= self.w_barrier * float(x * x)

        # 5) closing趋势惩罚：若最近距离在变小，则惩罚（动态横穿更稳）
        if self.prev_min_dist is not None and np.isfinite(self.prev_min_dist) and np.isfinite(min_dist):
            closing = (float(self.prev_min_dist) - float(min_dist)) / max(self.dt, 1e-6)  # m/s
            if closing > 0.0:
                reward -= self.w_closing * float(np.clip(closing, 0.0, self.closing_clip))

        self.prev_min_dist = float(min_dist)
        return float(reward)



class Sim2RealEnv(gym.Env):
    """
    Sim-to-Real 友好的UAV避障环境 (无GPS依赖)

    状态空间 (9 + N_beams):
        [0-1]   cos/sin(θ)        机器人朝向 (IMU)
        [2]     v_lin_norm        归一化线速度 (编码器)
        [3]     v_ang_norm        归一化角速度 (陀螺仪)
        [4-5]   cos/sin(θ_rel)    目标相对方向 (里程计计算)
        [6]     d_goal_norm       归一化目标距离 (里程计计算)
        [7-8]   prev_action       上一动作 (自身记录)
        [9: ]   lidar[N_beams]    N束激光雷达 (LiDAR)

    动作空间:
        [v_lin, v_ang] ∈ [0,1] × [-0.8, 0.8]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # ✅ 配置参数
    MIN_LINEAR_SPEED = 0.0          # 最小线速度
    START_GOAL_CLEAR_RADIUS = 1.0   # 起点/目标周围无障碍物的半径
    MIN_START_GOAL_DIST = 6.0       # 起点到目标的最小距离
    MIN_OBSTACLE_CLEARANCE = 0.8    # 静态障碍物之间的最小间距（表面到表面）

    def __init__(
        self,
        scene_id: int = 4,
        max_steps: int = 400,
        map_size: float = 10.0,
        render_mode: Optional[str] = None,
        num_static: Optional[int] = None,
        num_dynamic: Optional[int] = None,
        num_lidar_beams: int = 16,
        lidar_max_range: float = 5.0,
        lidar_fov_deg: float = 360.0,
        lidar_noise_std: float = 0.0,
    ):
        super().__init__()

        # 环境参数
        self.scene_id = scene_id
        self.max_steps = max_steps
        self.map_size = float(map_size)
        self.wall_boundary = 0.3
        self.dt = 0.1

        # 机器人参数
        self.robot_radius = 0.15
        self.max_linear_speed = 1.0
        self.max_angular_speed = 0.8
        self.goal_radius = 0.3

        # LiDAR参数
        self.num_lidar_beams = num_lidar_beams
        self.lidar_max_range = lidar_max_range
        self.lidar_fov_deg = float(lidar_fov_deg)
        self.lidar_noise_std = float(lidar_noise_std)
        fov_rad = math.radians(self.lidar_fov_deg)
        # 相对机器人朝向的角度，均匀分布在 [-fov/2, fov/2]
        self.lidar_angles = np.linspace(-fov_rad / 2.0, fov_rad / 2.0, num_lidar_beams, endpoint=False)

        # 状态维度: 2(朝向) + 2(速度) + 3(目标) + 2(prev_action) + N_beams(lidar)
        self.state_dim = 9 + num_lidar_beams

        # 空间定义 (v_lin ∈ [0, 1]，只能前进)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.8], dtype=np.float32),
            high=np.array([1.0, 0.8], dtype=np.float32),
            dtype=np.float32
        )

        # ============ 仿真用的绝对坐标 (真实部署时不存在) ============
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.goal_pos = np.zeros(2, dtype=np.float32)

        # ============ Sim2Real: 里程计坐标系 ============
        self.odom_pos = np.zeros(2, dtype=np.float32)
        self.goal_rel = np.zeros(2, dtype=np.float32)
        self.start_pos = np.zeros(2, dtype=np.float32)

        # 其他状态
        self.robot_angle = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = 0.0
        self.current_step = 0

        # 障碍物
        self.static_obstacles: List[Dict] = []
        self.dynamic_obstacles: List[SmoothDynamicObstacle] = []
        self.num_static_override = num_static
        self.num_dynamic_override = num_dynamic

        # 奖励和轨迹
        self.reward_fn = Sim2RealReward(dt=self.dt, v_max=self.max_linear_speed, max_steps=self.max_steps)
        self.trajectory: List[np.ndarray] = []
        self.dynamic_trajs: List[List[np.ndarray]] = []

        # 渲染
        self.render_mode = render_mode
        self.fig = None
        self.ax = None

        print(f"✓ Sim2RealEnv 初始化: 状态{self.state_dim}D, LiDAR{num_lidar_beams}束({self.lidar_fov_deg:.0f}°), 噪声σ={self.lidar_noise_std:.3f}, 无GPS依赖")

    def _sample_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样起点和目标点

        保证:
        1. 起点和目标距离 >= MIN_START_GOAL_DIST (6m)
        2. 两点都在有效区域内
        """
        margin = self.wall_boundary + self.robot_radius + 0.5
        low, high = margin, self.map_size - margin

        for _ in range(500):
            start = np.array([np.random.uniform(low, high),
                             np.random.uniform(low, high)], dtype=np.float32)
            goal = np.array([np.random.uniform(low, high),
                            np.random.uniform(low, high)], dtype=np.float32)

            dist = np.linalg.norm(goal - start)
            if dist >= self.MIN_START_GOAL_DIST:
                return start, goal

        # 备用: 对角线位置
        return (np.array([1.5, 1.5], dtype=np.float32),
                np.array([8.5, 8.5], dtype=np.float32))

    def _build_scene(self, start_pos: np.ndarray, goal_pos: np.ndarray):
        """
        构建场景障碍物

        保证:
        1. 障碍物不在起点/目标 START_GOAL_CLEAR_RADIUS (1.5m) 范围内
        2. 静态障碍物之间间距 >= MIN_OBSTACLE_CLEARANCE (0.8m)
        """
        self.static_obstacles = []
        self.dynamic_obstacles = []

        defaults = {
            1: (0, 1), 2: (2, 0), 3: (3, 1), 4: (4, 2), 6: (6, 4),
        }
        n_static, n_dynamic = defaults.get(self.scene_id, (4, 2))

        if self.num_static_override is not None:
            n_static = self.num_static_override
        if self.num_dynamic_override is not None:
            n_dynamic = self.num_dynamic_override

        # 生成静态障碍物
        for _ in range(n_static):
            radius = float(np.random.uniform(0.3, 0.45))  # 稍微减小最大半径
            pos = self._sample_obstacle_position(radius, start_pos, goal_pos)
            if pos is not None:
                self.static_obstacles.append({'pos': pos, 'radius': radius})

        # 生成动态障碍物 (优先放在起点-目标连线附近，制造“横穿人群”场景)
        speed_min = 0.8  # 略快一些，提高难度
        speed_max = 1.6
        variable_speed = False
        accel_std = 0.0
        accel_prob = 0.0

        # ✅ 场景 6：6 静 + 4 动，动态障碍物在给定速度区间内随机加速/减速
        if self.scene_id == 6:
            speed_min = 0.6  # 允许稍微慢一点
            speed_max = 1.6  # 上限不变，仍然是“行人中有快跑的”
            variable_speed = True  # 开启变速
            accel_std = 2.0  # 加速度标准差（越大速度变化越剧烈）
            accel_prob = 0.4  # 每一步 40% 概率更新一次加速度

        # 起点 → 目标方向及其垂直方向
        path_vec = goal_pos - start_pos
        path_len = np.linalg.norm(path_vec) + 1e-6
        path_dir = path_vec / path_len
        perp_dir = np.array([-path_dir[1], path_dir[0]], dtype=np.float32)

        for _ in range(n_dynamic):
            radius = 0.25

            # 优先在主航道附近采样
            pos = None
            for _try in range(30):
                # 在起点→目标中段 [0.2, 0.8] 之间选一个位置
                t = np.random.uniform(0.2, 0.8)
                center = start_pos + t * path_vec
                # 沿垂直方向偏移，形成走廊两侧的动态障碍
                offset = np.random.uniform(-1.2, 1.2)
                cand = center + offset * perp_dir

                # 起终点清空区域
                if np.linalg.norm(cand - start_pos) < (self.START_GOAL_CLEAR_RADIUS + radius):
                    continue
                if np.linalg.norm(cand - goal_pos) < (self.START_GOAL_CLEAR_RADIUS + radius):
                    continue

                # 与静态障碍保持足够间距
                ok = True
                for obs in self.static_obstacles:
                    surface_dist = np.linalg.norm(cand - obs['pos']) - radius - obs['radius']
                    if surface_dist < self.MIN_OBSTACLE_CLEARANCE:
                        ok = False
                        break
                if not ok:
                    continue

                # 边界缓冲，避免一出生就贴墙
                if (cand[0] < self.wall_boundary + radius + 0.3 or
                    cand[0] > self.map_size - self.wall_boundary - radius - 0.3 or
                    cand[1] < self.wall_boundary + radius + 0.3 or
                    cand[1] > self.map_size - self.wall_boundary - radius - 0.3):
                    continue

                pos = cand
                break

            # 如果在主航道附近多次尝试失败，退回通用采样逻辑
            if pos is None:
                pos = self._sample_obstacle_position(radius, start_pos, goal_pos, is_dynamic=True)

            if pos is None:
                continue

            # 每个障碍物速度在 [speed_min, speed_max] 范围内随机
            speed = np.random.uniform(speed_min, speed_max)

            # 让障碍物大致沿垂直于起点→目标的方向运动（横穿）
            heading_sign = np.random.choice([-1.0, 1.0])
            heading_vec = heading_sign * perp_dir
            heading = math.atan2(heading_vec[1], heading_vec[0])

            dyn = SmoothDynamicObstacle(
                init_pos=pos, speed=speed, radius=radius,
                map_size=self.map_size, wall_boundary=self.wall_boundary,
                turn_prob=0.01, max_turn_rate=math.radians(25.0), turn_smooth=0.05,
                boundary_buffer=0.5,
                # ✅ 变速相关参数
                variable_speed = variable_speed,
                speed_min = speed_min,
                speed_max = speed_max,
                accel_std = accel_std,
                accel_prob = accel_prob,
            )
            # 初始化朝向，避免一开始乱转
            dyn.heading = heading
            dyn.target_heading = heading

            self.dynamic_obstacles.append(dyn)

    def _sample_obstacle_position(
        self,
        radius: float,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        is_dynamic: bool = False,
        max_tries: int = 200
    ) -> Optional[np.ndarray]:
        """
        采样障碍物位置

        保证:
        1. 不在起点/目标 START_GOAL_CLEAR_RADIUS 范围内
        2. 与其他静态障碍物间距 >= MIN_OBSTACLE_CLEARANCE
        3. 动态障碍物需要额外的边界缓冲
        """
        margin = self.wall_boundary + radius + 0.3
        if is_dynamic:
            margin += 0.5  # 动态障碍物需要更大的边界margin
        low, high = margin, self.map_size - margin

        for _ in range(max_tries):
            pos = np.array([np.random.uniform(low, high),
                           np.random.uniform(low, high)], dtype=np.float32)

            # ✅ 检查1: 不在起点附近
            if np.linalg.norm(pos - start_pos) < (self.START_GOAL_CLEAR_RADIUS + radius):
                continue

            # ✅ 检查2: 不在目标附近
            if np.linalg.norm(pos - goal_pos) < (self.START_GOAL_CLEAR_RADIUS + radius):
                continue

            # ✅ 检查3: 与其他静态障碍物保持足够间距
            ok = True
            for obs in self.static_obstacles:
                surface_dist = np.linalg.norm(pos - obs['pos']) - radius - obs['radius']
                if surface_dist < self.MIN_OBSTACLE_CLEARANCE:
                    ok = False
                    break
            if not ok:
                continue

            # ✅ 检查4: 与动态障碍物保持间距
            for dyn in self.dynamic_obstacles:
                if np.linalg.norm(pos - dyn.pos) < (radius + dyn.radius + 0.5):
                    ok = False
                    break
            if not ok:
                continue

            return pos

        return None  # 采样失败

    def _reset_headon_test(self):
        """
        固定起点 / 朝向 / 目标点，并放置 1 个在 2m 外正向接近无人机的动态障碍物
        场景示意：
            Robot(3,5) --2m--> Obstacle(5,5) --朝左--> Robot
                             Goal(8,5)
        """
        # 清空障碍物
        self.static_obstacles = []
        self.dynamic_obstacles = []

        # 固定起点和目标点
        self.robot_pos = np.array([1.0, 5.0], dtype=np.float32)
        self.goal_pos = np.array([8.0, 5.0], dtype=np.float32)

        # 里程计坐标系初始化
        self.start_pos = self.robot_pos.copy()
        self.odom_pos = np.zeros(2, dtype=np.float32)
        self.goal_rel = self.goal_pos - self.start_pos

        # 固定朝向：朝 +x 方向
        self.robot_angle = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)

        # 目标距离
        self.prev_goal_dist = float(np.linalg.norm(self.goal_rel - self.odom_pos))
        self.trajectory = [self.robot_pos.copy()]

        # ===== 动态障碍物配置 =====
        # 在无人机前方 2m 处
        obstacle_distance = 2.5
        obs_pos = self.robot_pos + np.array([obstacle_distance, 0.0], dtype=np.float32)

        # 障碍物速度（可以根据需要调整）
        dyn_speed = 1.2  # m/s，略快于无人机 1m/s

        dyn = SmoothDynamicObstacle(
            init_pos=obs_pos,
            speed=dyn_speed,
            radius=0.25,
            map_size=self.map_size,
            wall_boundary=self.wall_boundary,
            # 关闭随机转向，保持直线朝向无人机
            turn_prob=0.0,
            max_turn_rate=0.0,
            turn_smooth=1.0,
            boundary_buffer=0.5,
        )

        # 让障碍物朝“机器人当前位置”运动
        heading_vec = self.robot_pos - obs_pos
        dyn.heading = math.atan2(heading_vec[1], heading_vec[0])
        dyn.target_heading = dyn.heading
        dyn.vel = np.array(
            [dyn.speed * math.cos(dyn.heading),
             dyn.speed * math.sin(dyn.heading)],
            dtype=np.float32
        )

        self.dynamic_obstacles = [dyn]
        self.dynamic_trajs = [[dyn.pos.copy()]]


    def _check_collision(self, pos: np.ndarray) -> bool:
        """碰撞检测"""
        # 边界
        if (pos[0] < self.wall_boundary + self.robot_radius or
            pos[0] > self.map_size - self.wall_boundary - self.robot_radius or
            pos[1] < self.wall_boundary + self.robot_radius or
            pos[1] > self.map_size - self.wall_boundary - self.robot_radius):
            return True
        # 静态障碍物
        for obs in self.static_obstacles:
            if np.linalg.norm(pos - obs['pos']) < (self.robot_radius + obs['radius']):
                return True
        # 动态障碍物
        for dyn in self.dynamic_obstacles:
            if np.linalg.norm(pos - dyn.pos) < (self.robot_radius + dyn.radius):
                return True
        return False

    def _compute_lidar(self) -> np.ndarray:
        """计算LiDAR读数"""
        lidar = np.full(self.num_lidar_beams, self.lidar_max_range, dtype=np.float32)

        x, y = float(self.robot_pos[0]), float(self.robot_pos[1])
        xmin = self.wall_boundary + self.robot_radius
        xmax = self.map_size - self.wall_boundary - self.robot_radius
        ymin, ymax = xmin, xmax

        for i, rel_ang in enumerate(self.lidar_angles):
            theta = self.robot_angle + rel_ang
            dir_x, dir_y = math.cos(theta), math.sin(theta)
            direction = np.array([dir_x, dir_y], dtype=np.float32)

            # 墙面交点
            t_candidates = []
            eps = 1e-6
            if abs(dir_x) > eps:
                t = (xmax - x) / dir_x if dir_x > 0 else (xmin - x) / dir_x
                if t > 0:
                    t_candidates.append(t)
            if abs(dir_y) > eps:
                t = (ymax - y) / dir_y if dir_y > 0 else (ymin - y) / dir_y
                if t > 0:
                    t_candidates.append(t)

            t_min = min(t_candidates) if t_candidates else self.lidar_max_range
            t_min = min(t_min, self.lidar_max_range)

            # 障碍物交点
            origin = self.robot_pos
            for obs in self.static_obstacles:
                t_hit = self._ray_circle_intersect(origin, direction, obs['pos'], obs['radius'] + self.robot_radius)
                if t_hit is not None and 0 < t_hit < t_min:
                    t_min = t_hit

            for dyn in self.dynamic_obstacles:
                t_hit = self._ray_circle_intersect(origin, direction, dyn.pos, dyn.radius + self.robot_radius)
                if t_hit is not None and 0 < t_hit < t_min:
                    t_min = t_hit

            lidar[i] = t_min

        # 归一化到 [0,1]
        lidar = np.clip(lidar / self.lidar_max_range, 0.0, 1.0)

        # 加一点测量噪声，考验鲁棒性
        if getattr(self, "lidar_noise_std", 0.0) > 0.0:
            noise = np.random.normal(0.0, self.lidar_noise_std, size=lidar.shape).astype(np.float32)
            lidar = np.clip(lidar + noise, 0.0, 1.0)

        return lidar

    @staticmethod
    def _ray_circle_intersect(origin: np.ndarray, direction: np.ndarray,
                               center: np.ndarray, radius: float) -> Optional[float]:
        """射线与圆交点"""
        oc = center - origin
        t_proj = float(np.dot(oc, direction))
        if t_proj < 0:
            return None

        dist2 = float(np.dot(oc, oc)) - t_proj * t_proj
        r2 = radius * radius
        if dist2 > r2:
            return None

        thc = math.sqrt(max(r2 - dist2, 0.0))
        t0 = t_proj - thc
        return t0 if t0 > 0 else (t_proj + thc if t_proj + thc > 0 else None)

    def _get_state(self) -> np.ndarray:
        """获取状态向量 (9 + N_beams)D - 无GPS依赖"""
        # 1. 机器人朝向 (IMU)
        cos_theta = math.cos(self.robot_angle)
        sin_theta = math.sin(self.robot_angle)

        # 2. 速度 (编码器/陀螺仪)
        v_lin_norm = np.clip(self.linear_speed / self.max_linear_speed, -1.0, 1.0)
        v_ang_norm = np.clip(self.angular_speed / self.max_angular_speed, -1.0, 1.0)

        # 3. 目标信息 (里程计坐标系)
        to_goal_odom = self.goal_rel - self.odom_pos
        dist_to_goal = float(np.linalg.norm(to_goal_odom))
        angle_to_goal = math.atan2(to_goal_odom[1], to_goal_odom[0])

        # 目标相对于机器人朝向的角度
        rel_angle = angle_to_goal - self.robot_angle
        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))

        # 归一化距离
        max_dist = self.map_size * math.sqrt(2.0)
        d_goal_norm = np.clip(dist_to_goal / max_dist, 0.0, 1.0)

        # 4. LiDAR
        lidar = self._compute_lidar()

        # 组装状态
        state = np.array([
            cos_theta, sin_theta,
            v_lin_norm, v_ang_norm,
            math.cos(rel_angle), math.sin(rel_angle), d_goal_norm,
            self.prev_action[0], self.prev_action[1],
        ], dtype=np.float32)

        state = np.concatenate([state, lidar])
        return np.clip(state, -1.0, 1.0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        # 特殊测试场景：scene_id == 5
        if self.scene_id == 5:
            self._reset_headon_test()
            return self._get_state(), {}

        # ✅ 先采样起点和目标 (保证距离 >= 6m)
        self.robot_pos, self.goal_pos = self._sample_start_goal()

        # ✅ 然后生成障碍物 (避开起点和目标)
        self._build_scene(self.robot_pos, self.goal_pos)

        # 重置动态障碍物
        for dyn in self.dynamic_obstacles:
            dyn.reset()

        # ============ 初始化里程计坐标系 ============
        self.start_pos = self.robot_pos.copy()
        self.odom_pos = np.zeros(2, dtype=np.float32)
        self.goal_rel = self.goal_pos - self.start_pos

        # 重置其他状态
        self.robot_angle = np.random.uniform(-math.pi, math.pi)
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = float(np.linalg.norm(self.goal_rel - self.odom_pos))
        self.trajectory = [self.robot_pos.copy()]

        # 初始化动态障碍物轨迹记录
        self.dynamic_trajs = []
        for dyn in self.dynamic_obstacles:
            self.dynamic_trajs.append([dyn.pos.copy()])

        self.reward_fn.reset()
        return self._get_state(), {}

    def step(self, action: np.ndarray):
        """执行动作"""
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 强制最小速度
        if action[0] < self.MIN_LINEAR_SPEED:
            action[0] = self.MIN_LINEAR_SPEED

        # 更新速度
        self.linear_speed = float(action[0]) * self.max_linear_speed
        self.angular_speed = float(action[1])

        # 检查动态障碍物轨迹列表
        if len(self.dynamic_trajs) != len(self.dynamic_obstacles):
            self.dynamic_trajs = []
            for dyn in self.dynamic_obstacles:
                self.dynamic_trajs.append([dyn.pos.copy()])

        # 更新动态障碍物
        for i, dyn in enumerate(self.dynamic_obstacles):
            dyn.update(self.dt)
            if i < len(self.dynamic_trajs):
                self.dynamic_trajs[i].append(dyn.pos.copy())

        # 更新机器人位姿
        self.robot_angle += self.angular_speed * self.dt
        self.robot_angle = math.atan2(math.sin(self.robot_angle), math.cos(self.robot_angle))

        dx = self.linear_speed * math.cos(self.robot_angle) * self.dt
        dy = self.linear_speed * math.sin(self.robot_angle) * self.dt
        delta = np.array([dx, dy], dtype=np.float32)

        new_pos = self.robot_pos + delta

        # 碰撞检测
        collision = self._check_collision(new_pos)
        if not collision:
            self.robot_pos = new_pos
            self.odom_pos += delta

        self.trajectory.append(self.robot_pos.copy())

        # 终止条件
        to_goal_odom = self.goal_rel - self.odom_pos
        dist_to_goal = float(np.linalg.norm(to_goal_odom))
        success = dist_to_goal < self.goal_radius
        timeout = self.current_step >= self.max_steps
        terminated = collision or success
        truncated = timeout and not terminated

        # 奖励
        reward = self.reward_fn.compute(
            self.robot_pos, self.goal_pos, self.prev_goal_dist,
            self.static_obstacles, self.dynamic_obstacles,
            self.robot_radius, collision, success
        )

        # 更新状态
        self.prev_goal_dist = dist_to_goal
        self.prev_action = action.copy()

        info = {
            'success': success,
            'collision': collision,
            'timeout': timeout and not terminated,
            'dist_to_goal': dist_to_goal,
            'odom_pos': self.odom_pos.copy(),
        }

        return self._get_state(), float(reward), terminated, truncated, info

    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return

        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            if self.render_mode == 'human':
                plt.ion()
                plt.show(block=False)

        self.ax.clear()
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Sim2Real Scene {self.scene_id} | Step {self.current_step} | No GPS')
        self.ax.grid(True, alpha=0.3)

        # 边界
        rect = Rectangle((self.wall_boundary, self.wall_boundary),
                         self.map_size - 2 * self.wall_boundary,
                         self.map_size - 2 * self.wall_boundary,
                         fill=False, edgecolor='black', linewidth=2)
        self.ax.add_patch(rect)

        # 静态障碍物
        for obs in self.static_obstacles:
            circle = Circle(obs['pos'], obs['radius'], color='gray', alpha=0.7)
            self.ax.add_patch(circle)

        # 动态障碍物
        for i, dyn in enumerate(self.dynamic_obstacles):
            circle = Circle(dyn.pos, dyn.radius, color='red', alpha=0.7)
            self.ax.add_patch(circle)

            # 绘制轨迹
            if i < len(self.dynamic_trajs) and len(self.dynamic_trajs[i]) > 1:
                dyn_traj = np.array(self.dynamic_trajs[i])
                self.ax.plot(
                    dyn_traj[:, 0], dyn_traj[:, 1],
                    linestyle='--', linewidth=1.5,
                    color='orange', alpha=0.6
                )

            # 运动方向箭头
            arrow_len = 0.5
            arrow_dx = arrow_len * math.cos(dyn.heading)
            arrow_dy = arrow_len * math.sin(dyn.heading)
            self.ax.arrow(
                dyn.pos[0], dyn.pos[1],
                arrow_dx, arrow_dy,
                head_width=0.15, head_length=0.12,
                fc='darkred', ec='darkred', alpha=0.9,
                length_includes_head=True
            )

        # 机器人
        robot = Circle(self.robot_pos, self.robot_radius, color='blue', alpha=0.8)
        self.ax.add_patch(robot)

        # 机器人朝向箭头
        arrow_length = 0.5
        robot_dx = arrow_length * math.cos(self.robot_angle)
        robot_dy = arrow_length * math.sin(self.robot_angle)
        self.ax.arrow(
            self.robot_pos[0], self.robot_pos[1],
            robot_dx, robot_dy,
            head_width=0.15, head_length=0.12,
            fc='darkblue', ec='darkblue', alpha=0.9,
            length_includes_head=True
        )

        # 起点标记
        self.ax.plot(self.start_pos[0], self.start_pos[1], 'bs', markersize=10, label='Start')

        # 目标
        goal = Circle(self.goal_pos, self.goal_radius, color='green', alpha=0.3)
        self.ax.add_patch(goal)
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], 'g*', markersize=15, label='Goal')

        # 机器人轨迹
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=2)

        # LiDAR可视化
        lidar = self._compute_lidar() * self.lidar_max_range
        for i, (ang, dist) in enumerate(zip(self.lidar_angles, lidar)):
            theta = self.robot_angle + ang
            end_x = self.robot_pos[0] + dist * math.cos(theta)
            end_y = self.robot_pos[1] + dist * math.sin(theta)
            color = 'orange' if dist < 1.0 else 'yellow'
            self.ax.plot([self.robot_pos[0], end_x], [self.robot_pos[1], end_y],
                        color=color, alpha=0.3, linewidth=0.5)

        self.ax.legend(loc='upper right')

        if self.render_mode == 'human':
            try:
                plt.pause(0.01)
                self.fig.canvas.flush_events()
            except Exception:
                pass

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# ==================== 兼容性包装类 ==================== #

class RlGameSim2Real:
    """兼容原有接口的包装类"""

    def __init__(
        self,
        n: int = 4,
        l: float = 10.0,
        render: bool = False,
        num_static: Optional[int] = None,
        num_dynamic: Optional[int] = None,
    ):
        render_mode = "human" if render else None
        self.env = Sim2RealEnv(
            scene_id=n,
            map_size=l,
            render_mode=render_mode,
            num_static=num_static,
            num_dynamic=num_dynamic,
            # 为了凸显注意力+LSTM 的优势，这里默认使用更高维的 LiDAR 配置
            num_lidar_beams=32,
            lidar_max_range=5.0,
            lidar_fov_deg=240.0,
            lidar_noise_std=0.03,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_dim = self.env.state_dim
        self.action_dim = 2

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def trajectory(self):
        return self.env.trajectory

    @property
    def dynamic_trajs(self):
        return self.env.dynamic_trajs

    def get_trajectory(self):
        return np.array(self.env.trajectory)


# ==================== 测试代码 ==================== #

if __name__ == "__main__":
    print("=" * 60)
    print("Sim2Real V3 环境测试 - 全面修复版本")
    print("=" * 60)

    env = RlGameSim2Real(n=4, render=False)

    print(f"\n📊 状态空间: {env.state_dim}D")
    print(f"📊 动作空间: {env.action_dim}D")

    print("\n修复内容 (v3):")
    print(f"  1. 起点/目标周围 {Sim2RealEnv.START_GOAL_CLEAR_RADIUS}m 无障碍物")
    print(f"  2. 起点到目标距离 >= {Sim2RealEnv.MIN_START_GOAL_DIST}m")
    print(f"  3. 静态障碍物间距 >= {Sim2RealEnv.MIN_OBSTACLE_CLEARANCE}m")
    print("  4. 动态障碍物触墙反弹改进 (缓冲区+冷却+随机偏转)")

    # 验证配置
    print("\n--- 验证 Scene 4 配置 ---")

    # 多次测试
    valid_count = 0
    for i in range(10):
        state = env.reset()

        # 检查起点目标距离
        start_goal_dist = np.linalg.norm(env.env.goal_pos - env.env.robot_pos)

        # 检查障碍物与起点/目标的距离
        min_start_dist = float('inf')
        min_goal_dist = float('inf')
        for obs in env.env.static_obstacles:
            d_start = np.linalg.norm(obs['pos'] - env.env.robot_pos) - obs['radius']
            d_goal = np.linalg.norm(obs['pos'] - env.env.goal_pos) - obs['radius']
            min_start_dist = min(min_start_dist, d_start)
            min_goal_dist = min(min_goal_dist, d_goal)

        if (start_goal_dist >= 6.0 and
            min_start_dist >= 1.3 and
            min_goal_dist >= 1.3):
            valid_count += 1

    print(f"  配置验证通过率: {valid_count}/10")
    print(f"  静态障碍物数量: {len(env.env.static_obstacles)}")
    print(f"  动态障碍物数量: {len(env.env.dynamic_obstacles)}")

    # 运行测试
    print("\n--- 运行100个episode测试 ---")
    successes, collisions, timeouts = 0, 0, 0
    early_collisions = 0  # 10步内碰撞

    for ep in range(100):
        state = env.reset()

        for step in range(400):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            if done:
                if info['success']:
                    successes += 1
                elif info['collision']:
                    collisions += 1
                    if step < 10:
                        early_collisions += 1
                else:
                    timeouts += 1
                break

    print(f"\n📈 结果统计 (随机策略):")
    print(f"  成功: {successes}%")
    print(f"  碰撞: {collisions}% (其中早期碰撞: {early_collisions})")
    print(f"  超时: {timeouts}%")

    env.close()

    print("\n" + "=" * 60)
    print("✓ 测试完成")
    print("=" * 60)