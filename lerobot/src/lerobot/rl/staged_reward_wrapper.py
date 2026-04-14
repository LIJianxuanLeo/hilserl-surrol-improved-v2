"""
Staged Dense Reward Wrapper v2 for Pick-and-Place Task

设计原则：per-step reward [0, 10] 使 Q ≈ 15-40，远大于 SAC 熵项 ~6，
确保 actor 梯度由任务奖励主导而非熵探索。

三阶段密集奖励：
  Stage 1 (Approach):  引导 TCP 靠近方块（逆距离核，远处仍有梯度）
  Stage 2 (Grasp):     平滑距离渐变 + 抬升确认（无离散跳跃）
  Stage 3 (Lift):      与抬起高度成正比

成功奖励为加法 += 10.0（非覆盖），保持 Q 函数连续。

v1→v2 关键变更：
  1. 移除 1/max_episode_steps 缩放（v1 根因：Q ≈ 0.3 << 熵项 6）
  2. exp(-10d) → 1/(1+5d) 逆距离核（远处仍有梯度）
  3. 离散 0/0.10/0.25 → 平滑线性渐变（消除 Q 不连续）
  4. succeed 时 reward = 1.0 → reward += 10.0（加法，Q 平滑递增）

适用于 RTX 3060 (12GB)，配合 SAC temperature=1.0, discount=0.99。
目标：1-2.5 小时收敛。
"""

import gymnasium as gym
import numpy as np


class StagedRewardWrapper(gym.Wrapper):
    """Multi-stage dense reward for pick tasks (v2).

    Per-step reward (NO scaling):
      - reach:   0-3.0   逆距离核 1/(1+5d)，远处仍有梯度
      - grasp:   0-3.0   平滑渐变 8cm→1cm + 抬升确认
      - lift:    0-4.0   与抬起高度线性正比
      - success: +10.0   加法叠加

    Expected episode rewards:
      - Random policy:   ~15-30
      - Near cube:       ~25-40
      - Grasping cube:   ~35-50
      - Full success:    ~40-60 (shaping + success bonus)
    """

    def __init__(self, env: gym.Env, lift_target: float = 0.1,
                 max_episode_steps: int = 100):
        super().__init__(env)
        self._lift_target = lift_target
        self._z_init = None
        # v2: 不再缩放。这是 v1 收敛慢的根因。

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Cache initial block height
        self._z_init = self.unwrapped._data.sensor("block_pos").data[2]
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        # Read state from MuJoCo sensors
        data = self.unwrapped._data
        block_pos = data.sensor("block_pos").data.copy()
        tcp_pos = data.sensor("2f85/pinch_pos").data.copy()

        # ── Stage 1: Approach (0 ~ 3.0) ────────────────────────
        # Inverse-distance kernel: 1/(1+5d)
        #   20cm → 0.50 (vs exp: 0.14) — 远处仍有学习梯度
        #   40cm → 0.33 (vs exp: 0.02)
        dist_xy = np.linalg.norm(block_pos[:2] - tcp_pos[:2])
        dz = tcp_pos[2] - block_pos[2]
        # Allow 0-3cm above cube as good pre-grasp height
        dist_z = max(0.0, abs(dz) - 0.03)

        reach_xy = 1.0 / (1.0 + 5.0 * dist_xy)
        reach_z = 1.0 / (1.0 + 5.0 * dist_z)
        r_reach = 3.0 * reach_xy * reach_z

        # ── Stage 2: Grasp (0 ~ 3.0) ───────────────────────────
        # Smooth linear ramp: 0 at 8cm → 1.0 at 1cm (no discrete jumps)
        dist_3d = np.linalg.norm(block_pos - tcp_pos)
        grasp_proximity = np.clip((0.08 - dist_3d) / 0.07, 0.0, 1.0)

        # Lift confirmation: extra bonus when cube is actually lifted
        lift = (block_pos[2] - self._z_init) if self._z_init is not None else 0.0
        lift_grasp_bonus = 0.5 * float(lift > 0.005) * float(dist_3d < 0.05)

        # Normalize: max of (proximity + bonus) = 1.0 + 0.5 = 1.5
        r_grasp = 3.0 * min(grasp_proximity + lift_grasp_bonus, 1.5) / 1.5

        # ── Stage 3: Lift (0 ~ 4.0) ────────────────────────────
        # Largest weight: hardest sub-skill, needs strongest signal
        lift_ratio = np.clip(lift / self._lift_target, 0.0, 1.0) \
            if self._lift_target > 0 else 0.0
        r_lift = 4.0 * lift_ratio

        # ── Total (0 ~ 10.0 per step, NOT scaled) ──────────────
        reward = r_reach + r_grasp + r_lift

        # ── Success bonus: additive +10 (not override) ─────────
        # Additive preserves Q-function continuity at success boundary
        if info.get("succeed", False):
            reward += 10.0

        return obs, reward, terminated, truncated, info
