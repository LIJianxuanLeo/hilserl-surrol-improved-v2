"""
Staged Dense Reward Wrapper for Pick-and-Place Task

Provides a multi-stage reward signal that guides the policy through:
  Stage 1 (Approach):  Move TCP close to the cube
  Stage 2 (Grasp):     Close gripper when near the cube
  Stage 3 (Lift):      Lift the cube off the table

Each stage builds on the previous one, providing continuous gradient
signal that dramatically speeds up SAC convergence compared to sparse
or simple dense rewards.

IMPORTANT: Per-step rewards are scaled by 1/max_episode_steps so that
the episode cumulative reward stays in [0, ~1] range, compatible with
SAC temperature=1.0 and grad_clip_norm=1.0.

Designed for RTX 3060 (12GB) with batch_size=128.
"""

import gymnasium as gym
import numpy as np


class StagedRewardWrapper(gym.Wrapper):
    """Multi-stage dense reward for pick tasks.

    Per-step reward (before scaling):
      - reach:   0.25 * exp(-10 * dist_xy) * exp(-10 * dist_z_above)
      - grasp:   0.25 * (gripper_closed & near_cube)
      - lift:    0.50 * clamp(lift_height / target_height)

    Per-step reward is then multiplied by (1 / max_episode_steps) so that
    the entire episode's cumulative reward stays in [0, ~1] range.

    Success step overrides with reward = 1.0 (unscaled).

    Expected episode rewards:
      - Random policy:  ~0.02 - 0.08
      - Near cube:      ~0.15 - 0.25
      - Grasping cube:  ~0.35 - 0.50
      - Full success:   ~1.0  - 1.3  (shaping + 1.0 success bonus)
    """

    def __init__(self, env: gym.Env, lift_target: float = 0.1, max_episode_steps: int = 100):
        super().__init__(env)
        self._lift_target = lift_target
        self._z_init = None
        # Scale per-step reward so episode sum ≈ [0, 1]
        self._reward_scale = 1.0 / max_episode_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Cache initial block height
        self._z_init = self.unwrapped._data.sensor("block_pos").data[2]
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        # Read state from MuJoCo sensors
        data = self.unwrapped._data
        block_pos = data.sensor("block_pos").data
        tcp_pos = data.sensor("2f85/pinch_pos").data

        # ---- Stage 1: Approach (0 ~ 0.25) ----
        # Reward for XY proximity (horizontal approach)
        dist_xy = np.linalg.norm(block_pos[:2] - tcp_pos[:2])
        # Reward for being slightly above the cube (good pre-grasp height)
        dz = tcp_pos[2] - block_pos[2]
        # Optimal height: 0 to 0.03m above the cube
        dist_z = max(0, abs(dz) - 0.03)
        r_reach = 0.25 * np.exp(-10 * dist_xy) * np.exp(-10 * dist_z)

        # ---- Stage 2: Grasp (0 ~ 0.25) ----
        dist_3d = np.linalg.norm(block_pos - tcp_pos)
        is_near = dist_3d < 0.05  # within 5cm

        # Check if gripper is applying force (cube is grasped)
        # Use lift as proxy for successful grasp
        lift = block_pos[2] - self._z_init if self._z_init else 0
        is_grasped = is_near and lift > 0.005  # slight lift = cube in gripper

        if is_grasped:
            r_grasp = 0.25
        elif is_near:
            r_grasp = 0.10  # partial reward for being close
        else:
            r_grasp = 0.0

        # ---- Stage 3: Lift (0 ~ 0.50) ----
        lift_ratio = max(0, lift) / self._lift_target if self._lift_target > 0 else 0
        r_lift = 0.50 * min(lift_ratio, 1.0)

        # ---- Total reward (scaled) ----
        # Scale per-step reward so episode cumulative stays in [0, ~1]
        reward = (r_reach + r_grasp + r_lift) * self._reward_scale

        # Success: override with unscaled 1.0 as the dominant learning signal
        if info.get("succeed", False):
            reward = 1.0

        return obs, reward, terminated, truncated, info
