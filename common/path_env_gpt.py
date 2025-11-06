"""
path_env_gpt.py
机器人路径规划环境 + 兼容旧接口 + S2补丁（清障奖励/停滞检测）
"""

import numpy as np
import collections
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import gymnasium as gym
from gymnasium import spaces


class DynamicObstacle:
    """动态障碍物"""

    def __init__(self, start_pos, end_pos, speed=0.3, radius=0.25):
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.end_pos = np.array(end_pos, dtype=np.float32)
        self.pos = self.start_pos.copy()
        self.speed = speed
        self.radius = radius
        self.direction = 1

    def update(self, dt=0.1):
        target = self.end_pos if self.direction == 1 else self.start_pos
        direction_vec = target - self.pos
        distance = np.linalg.norm(direction_vec)

        if distance < 0.1:
            self.direction *= -1
            return

        move = direction_vec / max(distance, 1e-6) * self.speed * dt
        self.pos += move

    def reset(self):
        self.pos = self.start_pos.copy()
        self.direction = 1


class CurriculumRobotEnv(gym.Env):
    """主环境类"""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, scene_id=1, max_steps=150, map_size=10.0, render_mode=None):
        super(CurriculumRobotEnv, self).__init__()

        self.scene_id = scene_id
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        self.map_size = float(map_size)
        self.wall_boundary = 0.3

        self.robot_radius = 0.25
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_pos = np.array([9.0, 9.0], dtype=np.float32)
        self.robot_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_angle = 0.0

        self.state_dim = 10
        self.action_dim = 2

        self.observation_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(self.state_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.5], dtype=np.float32),
            high=np.array([1.0, 1.5], dtype=np.float32),
            dtype=np.float32
        )

        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.trajectory = []
        self.prev_goal_dist = 0.0

        # S2补丁参数
        self.clear_bonus = 15.0        # 清障一次性奖励
        self.step_penalty = -0.03      # 时间代价
        self.progress_k = 2.0          # 距离进度系数
        self.heading_k = 0.5           # 朝向奖励系数
        self.stuck_K = 25              # 停滞检测窗口
        self.stuck_tol = 0.15          # 窗口内总进展阈值（米）
        self.stuck_penalty = -5.0      # 停滞惩罚并提前终止
        self._d_hist = collections.deque(maxlen=self.stuck_K)
        self._los_blocked_prev = None

        self.fig = None
        self.ax = None

        self._setup_scene()

    # --------------- 工具与场景 ---------------
    def _setup_scene(self):
        """设置场景"""
        self.static_obstacles = []
        self.dynamic_obstacles = []

        if self.scene_id == 1:
            pass
        elif self.scene_id == 2:
            self.static_obstacles = [
                {'pos': np.array([5.0, 5.0], dtype=np.float32), 'radius': 0.3},
            ]
        elif self.scene_id == 3:
            self.static_obstacles = [
                {'pos': np.array([3.0, 3.0], dtype=np.float32), 'radius': 0.5},
                {'pos': np.array([6.0, 5.0], dtype=np.float32), 'radius': 0.4},
                {'pos': np.array([5.0, 7.0], dtype=np.float32), 'radius': 0.45},
            ]
        elif self.scene_id == 4:
            self.static_obstacles = [
                {'pos': np.array([3.0, 3.0], dtype=np.float32), 'radius': 0.5},
                {'pos': np.array([6.0, 5.0], dtype=np.float32), 'radius': 0.4},
                {'pos': np.array([5.0, 7.0], dtype=np.float32), 'radius': 0.45},
            ]
            self.dynamic_obstacles = [
                DynamicObstacle([2.0, 5.0], [7.0, 5.0], speed=0.4),
            ]

    def _check_collision(self, pos, radius=None):
        if radius is None:
            radius = self.robot_radius
        # 墙
        if (pos[0] < self.wall_boundary or
                pos[0] > self.map_size - self.wall_boundary or
                pos[1] < self.wall_boundary or
                pos[1] > self.map_size - self.wall_boundary):
            return True
        # 静态障碍
        for obs in self.static_obstacles:
            dist = np.linalg.norm(pos - obs['pos'])
            if dist < (radius + obs['radius']):
                return True
        # 动态障碍
        for dyn_obs in self.dynamic_obstacles:
            dist = np.linalg.norm(pos - dyn_obs.pos)
            if dist < (radius + dyn_obs.radius):
                return True
        return False

    def _get_min_obstacle_distance(self):
        min_dist = float('inf')
        wall_dists = [
            self.robot_pos[0] - self.wall_boundary,
            self.map_size - self.wall_boundary - self.robot_pos[0],
            self.robot_pos[1] - self.wall_boundary,
            self.map_size - self.wall_boundary - self.robot_pos[1]
        ]
        min_dist = min(min_dist, min(wall_dists))
        for obs in self.static_obstacles:
            dist = np.linalg.norm(self.robot_pos - obs['pos']) - obs['radius']
            min_dist = min(min_dist, dist)
        for dyn_obs in self.dynamic_obstacles:
            dist = np.linalg.norm(self.robot_pos - dyn_obs.pos) - dyn_obs.radius
            min_dist = min(min_dist, dist)
        return max(0.0, min_dist)

    def _goal_angle(self):
        return np.arctan2(
            self.goal_pos[1] - self.robot_pos[1],
            self.goal_pos[0] - self.robot_pos[0]
        )

    def _cos_to_goal(self):
        # 当前朝向与目标方向的余弦
        g_ang = self._goal_angle()
        ang_diff = g_ang - self.robot_angle
        ang_diff = np.arctan2(np.sin(ang_diff), np.cos(ang_diff))
        return np.cos(ang_diff)

    @staticmethod
    def _los_blocked_any(p, g, obstacles):
        # 线段与任一圆是否相交
        px, py = p; gx, gy = g
        vx, vy = gx - px, gy - py
        vv = vx*vx + vy*vy
        for obs in obstacles:
            cx, cy = obs['pos']; r = obs['radius']
            wx, wy = cx - px, cy - py
            t = 0.0 if vv == 0 else max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
            qx, qy = px + t*vx, py + t*vy
            if (qx - cx)**2 + (qy - cy)**2 <= (r + 1e-6)**2:
                return True
        return False

    # --------------- 接口 ---------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.trajectory = []

        # 采样起点与终点（避开障碍）
        valid_start = False
        for _ in range(100):
            self.robot_pos = np.random.uniform(0.5, 2.0, size=2).astype(np.float32)
            if not self._check_collision(self.robot_pos):
                valid_start = True
                break
        if not valid_start:
            self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)

        valid_goal = False
        for _ in range(100):
            self.goal_pos = np.random.uniform(8.0, 9.5, size=2).astype(np.float32)
            if not self._check_collision(self.goal_pos):
                valid_goal = True
                break
        if not valid_goal:
            self.goal_pos = np.array([9.0, 9.0], dtype=np.float32)

        self.robot_angle = self._goal_angle()
        self.robot_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_goal_dist = np.linalg.norm(self.goal_pos - self.robot_pos)

        for dyn_obs in self.dynamic_obstacles:
            dyn_obs.reset()

        self.trajectory.append(self.robot_pos.copy())

        # S2 补丁状态重置
        self._d_hist.clear()
        self._d_hist.append(self.prev_goal_dist)
        self._los_blocked_prev = self._los_blocked_any(self.robot_pos, self.goal_pos, self.static_obstacles)

        state = self._get_state()
        info = {}
        return state, info

    def _get_state(self):
        goal_dist = np.linalg.norm(self.goal_pos - self.robot_pos)
        goal_angle = self._goal_angle()
        angle_diff = goal_angle - self.robot_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        min_obs_dist = self._get_min_obstacle_distance()
        state = np.array([
            self.robot_pos[0] / self.map_size,
            self.robot_pos[1] / self.map_size,
            self.robot_vel[0],
            self.robot_vel[1],
            goal_dist / self.map_size,
            np.sin(angle_diff),
            np.cos(angle_diff),
            min_obs_dist / self.map_size,
            np.sin(self.robot_angle),
            np.cos(self.robot_angle)
        ], dtype=np.float32)
        return state

    def step(self, action):
        self.current_step += 1
        for dyn_obs in self.dynamic_obstacles:
            dyn_obs.update()

        linear_vel = float(action[0])
        angular_vel = float(action[1])

        dt = 0.1
        self.robot_angle += angular_vel * dt
        self.robot_angle = np.arctan2(np.sin(self.robot_angle), np.cos(self.robot_angle))
        self.robot_vel[0] = linear_vel * np.cos(self.robot_angle)
        self.robot_vel[1] = linear_vel * np.sin(self.robot_angle)
        self.robot_pos = self.robot_pos + self.robot_vel * dt
        self.trajectory.append(self.robot_pos.copy())

        goal_dist = np.linalg.norm(self.goal_pos - self.robot_pos)

        reward = 0.0
        terminated = False
        truncated = False
        info = {'success': False, 'collision': False, 'timeout': False}

        # 终止判定
        if goal_dist < 0.4:
            reward = 100.0
            terminated = True
            info['success'] = True
        elif self._check_collision(self.robot_pos):
            reward = -50.0
            terminated = True
            info['collision'] = True
        elif self.current_step >= self.max_steps:
            reward = -60.0
            truncated = True
            info['timeout'] = True
        else:
            # ---- 基础奖励：时间成本 + 进度 + 朝向 ----
            d_prev = self._d_hist[-1]
            d_now = goal_dist
            progress = (d_prev - d_now)
            cos_heading = self._cos_to_goal()
            reward = self.step_penalty + self.progress_k * progress + self.heading_k * cos_heading

            # ---- 原先的细化项 ----
            # 线速度鼓励
            reward += linear_vel * 1.0
            # 障碍距离调节
            min_obs_dist = self._get_min_obstacle_distance()
            if len(self.static_obstacles) > 0 or len(self.dynamic_obstacles) > 0:
                if min_obs_dist < 0.3:
                    reward -= (0.3 - min_obs_dist) * 20.0
                elif 0.3 <= min_obs_dist < 0.8 and linear_vel > 0.3:
                    reward += 3.0
                elif min_obs_dist >= 0.8:
                    reward += 0.5
            # 接近终点奖励
            if goal_dist < 3.0: reward += 2.0
            if goal_dist < 2.0: reward += 3.0
            if goal_dist < 1.0: reward += 5.0

            # ---- 清障奖励：从被阻 -> 不被阻 ----
            los_now = self._los_blocked_any(self.robot_pos, self.goal_pos, self.static_obstacles)
            if (self._los_blocked_prev is True) and (los_now is False):
                reward += self.clear_bonus
            self._los_blocked_prev = los_now

            # ---- 停滞检测 ----
            self._d_hist.append(d_now)
            done_stuck = False
            if len(self._d_hist) == self.stuck_K:
                if (max(self._d_hist) - min(self._d_hist)) < self.stuck_tol and d_now > 0.6:
                    reward += self.stuck_penalty
                    done_stuck = True
            if done_stuck:
                truncated = True
                info['timeout'] = True

        self.prev_goal_dist = goal_dist
        next_state = self._get_state()
        return next_state, reward, terminated, truncated, info

    # --------------- 渲染 ---------------
    def render(self):
        if self.render_mode is None:
            return
        if self.fig is None:
            try:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                if self.render_mode == 'human':
                    plt.ion(); plt.show(block=False)
            except Exception:
                return
        self.ax.clear()
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        boundary = Rectangle(
            (self.wall_boundary, self.wall_boundary),
            self.map_size - 2 * self.wall_boundary,
            self.map_size - 2 * self.wall_boundary,
            fill=False, edgecolor='black', linewidth=2
        )
        self.ax.add_patch(boundary)

        for obs in self.static_obstacles:
            self.ax.add_patch(Circle(obs['pos'], obs['radius'], color='gray', alpha=0.7))
        for dyn_obs in self.dynamic_obstacles:
            self.ax.add_patch(Circle(dyn_obs.pos, dyn_obs.radius, color='orange', alpha=0.7))

        self.ax.add_patch(Circle(self.goal_pos, 0.4, color='green', alpha=0.5))
        self.ax.add_patch(Circle(self.robot_pos, self.robot_radius, color='blue', alpha=0.7))

        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=2)

        if self.render_mode == 'human':
            try:
                plt.pause(0.001)
                self.fig.canvas.flush_events()
            except Exception:
                pass

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    # --------------- 兼容层 ---------------
    def get_trajectory(self):
        return np.array(self.trajectory)

    def get_path_length(self):
        if len(self.trajectory) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(self.trajectory)):
            length += np.linalg.norm(self.trajectory[i] - self.trajectory[i - 1])
        return length


# ==================== 兼容层（关键！）====================

class RlGame:
    """
    兼容旧接口的包装器
    旧代码: RlGame(n=1, m=3, l=10, render=False)
    """

    def __init__(self, n=1, m=3, l=10, render=False):
        scene_id = n
        render_mode = 'human' if render else None
        map_size = float(l)
        self.env = CurriculumRobotEnv(
            scene_id=scene_id,
            map_size=map_size,
            render_mode=render_mode
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return state, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def get_trajectory(self):
        return self.env.get_trajectory()

    def get_path_length(self):
        return self.env.get_path_length()


if __name__ == "__main__":
    print("=" * 60)
    print("测试兼容接口")
    print("=" * 60)
    env = RlGame(n=2, m=3, l=10, render=False)
    state = env.reset()
    print(f"Reset成功，状态: {state.shape}")
    for i in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"Step {i + 1}: reward={reward:.2f}, done={done}")
        if done:
            break
    env.close()
    print("\n✅ 兼容层测试成功！旧代码可以直接运行了")
