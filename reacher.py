"""
Simple 2-DOF Reacher Environment (Simplified version without pygame)
"""
import numpy as np


class Reacher:
    """2-DOF planar robot arm reaching task"""
    
    def __init__(self, screen_size=1000, num_joints=2, link_lengths=None, 
                 ini_joint_angles=None, target_pos=None, render=False):
        self.screen_size = screen_size
        self.num_joints = num_joints
        self.link_lengths = link_lengths if link_lengths else [200, 140]
        self.ini_joint_angles = ini_joint_angles if ini_joint_angles else [0.1, 0.1]
        self.joint_angles = np.array(self.ini_joint_angles, dtype=np.float32)
        self.target_pos = np.array(target_pos if target_pos else [369, 430], dtype=np.float32)
        
        self.num_actions = num_joints
        self.num_observations = 2 * num_joints + 2
        
        self.dt = 0.05
        self.joint_velocities = np.zeros(num_joints, dtype=np.float32)
            
    def reset(self):
        """Reset the environment"""
        self.joint_angles = np.array(self.ini_joint_angles, dtype=np.float32)
        self.joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        # Randomize target position
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(150, 300)
        self.target_pos = np.array([
            self.screen_size // 2 + radius * np.cos(angle),
            self.screen_size // 2 + radius * np.sin(angle)
        ], dtype=np.float32)
        return self._get_observation()
    
    def step(self, action):
        """Execute one step"""
        action = np.clip(action, -1, 1)
        
        # Simple dynamics
        angular_acceleration = action * 5.0
        self.joint_velocities += angular_acceleration * self.dt
        self.joint_velocities = np.clip(self.joint_velocities, -5, 5)
        
        self.joint_angles += self.joint_velocities * self.dt
        self.joint_angles = np.clip(self.joint_angles, -np.pi, np.pi)
        
        # Calculate end effector position
        end_effector_pos = self._forward_kinematics()
        
        # Calculate reward
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance * 0.01
        
        done = distance < 20
        if done:
            reward += 10.0
        
        reward -= 0.001 * np.sum(np.square(action))
        
        return self._get_observation(), reward, done, {}
    
    def _forward_kinematics(self):
        """Calculate end effector position"""
        x = self.screen_size // 2
        y = self.screen_size // 2
        angle_sum = 0
        
        for i in range(self.num_joints):
            angle_sum += self.joint_angles[i]
            x += self.link_lengths[i] * np.cos(angle_sum)
            y += self.link_lengths[i] * np.sin(angle_sum)
        
        return np.array([x, y], dtype=np.float32)
    
    def _get_observation(self):
        """Get current observation"""
        end_effector_pos = self._forward_kinematics()
        target_relative = (self.target_pos - end_effector_pos) / self.screen_size
        
        obs = np.concatenate([
            np.sin(self.joint_angles),
            np.cos(self.joint_angles),
            self.joint_velocities / 5.0,
            target_relative
        ])
        return obs.astype(np.float32)
    
    def render(self):
        """Render placeholder"""
        pass
    
    def close(self):
        """Close placeholder"""
        pass
