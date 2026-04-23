"""
Training CSV Logger for HIL-SERL SurRoL

将训练指标保存为 CSV 文件，方便离线查看训练进度，无需 WandB。

日志文件结构：
    output_dir/training_logs/
    ├── training_metrics.csv       # 训练指标（loss, temperature, grad_norm, Q-value 等）
    ├── episode_metrics.csv        # 每 episode 指标（reward, intervention_rate, rolling success 等）
    ├── eval_metrics.csv           # 周期性策略评估（无干预成功率）
    ├── training_summary.json      # 训练总结（最终指标、最佳指标、训练时长等）
    └── experiment_metadata.json   # 实验元数据（seed, git hash, config dump, hardware）
"""

import csv
import json
import logging
import os
import platform
import subprocess
import time
from collections import deque
from pathlib import Path


class TrainingCSVLogger:
    """CSV-based training logger that stores metrics to local files.

    Automatically creates CSV files with headers on first write,
    and appends rows as training progresses.
    """

    TRAINING_FIELDS = [
        "timestamp",
        "optimization_step",
        "loss_critic",
        "loss_actor",
        "loss_temperature",
        "loss_discrete_critic",
        "temperature",
        "critic_grad_norm",
        "actor_grad_norm",
        "temperature_grad_norm",
        "discrete_critic_grad_norm",
        "replay_buffer_size",
        "offline_replay_buffer_size",
        "optimization_freq_hz",
        # F3: Q-value statistics for V2 design verification (Q ∈ [15, 40])
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
        "entropy_term",  # = temperature * |log_prob|
    ]

    EPISODE_FIELDS = [
        "timestamp",
        "interaction_step",
        "episodic_reward",
        "episode_intervention",
        "intervention_rate",
        "policy_freq_hz",
        "policy_freq_90p_hz",
        # F2: rolling-window success metrics (last 50 episodes overall)
        "is_success",
        "rolling_success_rate_50",
        "rolling_intervention_rate_50",
        # F1-proxy: rolling success rate over last 20 NO-INTERVENTION episodes
        # Approximates frozen-policy eval without separate process protocol changes.
        "rolling_policy_only_success_20",
    ]

    EVAL_FIELDS = [
        "timestamp",
        "optimization_step",
        "eval_episodes",
        "eval_success_rate",
        "eval_mean_reward",
        "eval_mean_episode_length",
    ]

    # Threshold above which an episodic_reward counts as success.
    # For V1 sparse reward: episode reward = 1.0 on success, 0.0 otherwise (use 0.5)
    # For V2 dense reward: success bonus +10 → episode reward typically > 10 on success (use 10.0)
    # We use 0.5 as a generic threshold that works for sparse reward.
    # Override by setting env var HILSERL_SUCCESS_THRESHOLD.
    DEFAULT_SUCCESS_THRESHOLD = 0.5

    def __init__(self, output_dir: str, job_name: str = "", success_threshold: float | None = None):
        self.log_dir = os.path.join(output_dir, "training_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.job_name = job_name
        self.training_csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        self.episode_csv_path = os.path.join(self.log_dir, "episode_metrics.csv")
        self.eval_csv_path = os.path.join(self.log_dir, "eval_metrics.csv")
        self.summary_path = os.path.join(self.log_dir, "training_summary.json")
        self.metadata_path = os.path.join(self.log_dir, "experiment_metadata.json")

        self.start_time = time.time()

        # Success threshold: env var > param > default
        env_threshold = os.environ.get("HILSERL_SUCCESS_THRESHOLD")
        if env_threshold is not None:
            self.success_threshold = float(env_threshold)
        elif success_threshold is not None:
            self.success_threshold = success_threshold
        else:
            self.success_threshold = self.DEFAULT_SUCCESS_THRESHOLD

        # Tracking best metrics
        self._best_reward = float("-inf")
        self._best_reward_step = 0
        self._lowest_loss = float("inf")
        self._total_episodes = 0
        self._total_interventions = 0
        self._total_successes = 0
        self._last_optimization_step = 0
        self._best_eval_success_rate = 0.0
        self._best_eval_step = 0

        # F2: Rolling windows for last-50-episode statistics
        self._success_window = deque(maxlen=50)
        self._intervention_window = deque(maxlen=50)
        # F1-proxy: Rolling success window over only NON-INTERVENTION episodes (policy-only signal)
        self._policy_only_success_window = deque(maxlen=20)

        # Initialize CSV files with headers
        self._init_csv(self.training_csv_path, self.TRAINING_FIELDS)
        self._init_csv(self.episode_csv_path, self.EPISODE_FIELDS)
        self._init_csv(self.eval_csv_path, self.EVAL_FIELDS)

        logging.info(f"[TrainingCSVLogger] Logging to: {self.log_dir}")
        logging.info(f"[TrainingCSVLogger] Success threshold: {self.success_threshold}")

    def _init_csv(self, filepath: str, fields: list):
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()

    def save_metadata(self, cfg_dict: dict | None = None, extra: dict | None = None):
        """F4: Save experiment metadata for reproducibility.

        Captures git hash, hostname, hardware info, full config snapshot, and start time.
        Called once at training start. Safe to call again to overwrite.
        """
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).parent,
            ).decode().strip()
        except Exception:
            git_hash = "unknown"

        try:
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
            ).decode().strip()
        except Exception:
            git_branch = "unknown"

        try:
            git_dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
            ).decode().strip())
        except Exception:
            git_dirty = False

        # Try to get GPU info
        gpu_info = "unknown"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
        except Exception:
            pass

        metadata = {
            "job_name": self.job_name,
            "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.start_time)),
            "start_time_unix": int(self.start_time),
            "git_commit": git_hash,
            "git_branch": git_branch,
            "git_dirty": git_dirty,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "gpu": gpu_info,
            "success_threshold": self.success_threshold,
            "cfg_snapshot": cfg_dict or {},
            "extra": extra or {},
        }

        try:
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            logging.info(f"[TrainingCSVLogger] Metadata saved: {self.metadata_path}")
        except Exception as e:
            logging.warning(f"[TrainingCSVLogger] Failed to write metadata: {e}")

    def log_training(self, training_infos: dict, optimization_step: int):
        """Log training metrics (loss, temperature, grad_norm, Q stats, etc.)

        Called every log_freq optimization steps from the learner process.
        """
        row = {
            "timestamp": f"{time.time() - self.start_time:.1f}",
            "optimization_step": optimization_step,
            "loss_critic": self._fmt(training_infos.get("loss_critic")),
            "loss_actor": self._fmt(training_infos.get("loss_actor")),
            "loss_temperature": self._fmt(training_infos.get("loss_temperature")),
            "loss_discrete_critic": self._fmt(training_infos.get("loss_discrete_critic")),
            "temperature": self._fmt(training_infos.get("temperature")),
            "critic_grad_norm": self._fmt(training_infos.get("critic_grad_norm")),
            "actor_grad_norm": self._fmt(training_infos.get("actor_grad_norm")),
            "temperature_grad_norm": self._fmt(training_infos.get("temperature_grad_norm")),
            "discrete_critic_grad_norm": self._fmt(training_infos.get("discrete_critic_grad_norm")),
            "replay_buffer_size": training_infos.get("replay_buffer_size", ""),
            "offline_replay_buffer_size": training_infos.get("offline_replay_buffer_size", ""),
            "optimization_freq_hz": self._fmt(training_infos.get("Optimization frequency loop [Hz]")),
            # F3: Q-value statistics
            "q_mean": self._fmt(training_infos.get("q_mean")),
            "q_std": self._fmt(training_infos.get("q_std")),
            "q_min": self._fmt(training_infos.get("q_min")),
            "q_max": self._fmt(training_infos.get("q_max")),
            "entropy_term": self._fmt(training_infos.get("entropy_term")),
        }

        # Track best loss
        loss = training_infos.get("loss_critic")
        if loss is not None and loss < self._lowest_loss:
            self._lowest_loss = loss

        self._last_optimization_step = optimization_step

        self._append_csv(self.training_csv_path, row, self.TRAINING_FIELDS)

    def log_episode(self, interaction_message: dict):
        """Log episode-level metrics (reward, intervention rate, rolling success, etc.)

        Called when the learner receives an interaction message from the actor.
        """
        episodic_reward = interaction_message.get("Episodic reward")
        intervention_int = int(interaction_message.get("Episode intervention", 0) or 0)

        # F2: derive success and update rolling windows
        is_success = 0
        if episodic_reward is not None:
            is_success = 1 if episodic_reward >= self.success_threshold else 0
            self._success_window.append(is_success)
            self._intervention_window.append(intervention_int)
            # F1-proxy: only count policy-only episodes (no human intervention)
            if intervention_int == 0:
                self._policy_only_success_window.append(is_success)

        rolling_success = (sum(self._success_window) / len(self._success_window)) if self._success_window else 0.0
        rolling_intervention = (
            sum(self._intervention_window) / len(self._intervention_window)
        ) if self._intervention_window else 0.0
        rolling_policy_only_success = (
            sum(self._policy_only_success_window) / len(self._policy_only_success_window)
        ) if self._policy_only_success_window else 0.0

        row = {
            "timestamp": f"{time.time() - self.start_time:.1f}",
            "interaction_step": interaction_message.get("Interaction step", ""),
            "episodic_reward": self._fmt(episodic_reward),
            "episode_intervention": intervention_int,
            "intervention_rate": self._fmt(interaction_message.get("Intervention rate", 0.0)),
            "policy_freq_hz": self._fmt(interaction_message.get("Policy frequency [Hz]")),
            "policy_freq_90p_hz": self._fmt(interaction_message.get("Policy frequency 90th-p [Hz]")),
            "is_success": is_success,
            "rolling_success_rate_50": self._fmt(rolling_success),
            "rolling_intervention_rate_50": self._fmt(rolling_intervention),
            "rolling_policy_only_success_20": self._fmt(rolling_policy_only_success),
        }

        # Track best reward
        if episodic_reward is not None:
            if episodic_reward > self._best_reward:
                self._best_reward = episodic_reward
                self._best_reward_step = interaction_message.get("Interaction step", 0)
            self._total_episodes += 1
            if is_success:
                self._total_successes += 1

        if intervention_int == 1:
            self._total_interventions += 1

        self._append_csv(self.episode_csv_path, row, self.EPISODE_FIELDS)

    def log_eval(self, eval_results: dict, optimization_step: int):
        """F1: Log periodic frozen-policy evaluation results.

        eval_results expected keys:
            - num_episodes (int)
            - success_rate (float in [0, 1])
            - mean_reward (float)
            - mean_episode_length (float)
        """
        success_rate = float(eval_results.get("success_rate", 0.0))
        row = {
            "timestamp": f"{time.time() - self.start_time:.1f}",
            "optimization_step": optimization_step,
            "eval_episodes": eval_results.get("num_episodes", 0),
            "eval_success_rate": self._fmt(success_rate),
            "eval_mean_reward": self._fmt(eval_results.get("mean_reward")),
            "eval_mean_episode_length": self._fmt(eval_results.get("mean_episode_length")),
        }

        if success_rate > self._best_eval_success_rate:
            self._best_eval_success_rate = success_rate
            self._best_eval_step = optimization_step

        self._append_csv(self.eval_csv_path, row, self.EVAL_FIELDS)
        logging.info(
            f"[TrainingCSVLogger] Eval @ step {optimization_step}: "
            f"success_rate={success_rate:.2%} "
            f"mean_reward={eval_results.get('mean_reward', 0.0):.3f}"
        )

    def save_summary(self):
        """Save training summary as JSON. Called at end of training or on checkpoint."""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        train_success_rate = (
            self._total_successes / self._total_episodes if self._total_episodes > 0 else 0.0
        )
        summary = {
            "job_name": self.job_name,
            "training_duration_s": round(elapsed, 1),
            "training_duration_h": round(hours, 2),
            "total_optimization_steps": self._last_optimization_step,
            "total_episodes": self._total_episodes,
            "total_interventions": self._total_interventions,
            "total_successes": self._total_successes,
            "training_success_rate": round(train_success_rate, 3),
            "intervention_episode_ratio": (
                round(self._total_interventions / max(self._total_episodes, 1), 3)
            ),
            "best_episodic_reward": round(self._best_reward, 4) if self._best_reward > float("-inf") else None,
            "best_reward_at_step": self._best_reward_step,
            "best_eval_success_rate": round(self._best_eval_success_rate, 4),
            "best_eval_at_step": self._best_eval_step,
            "lowest_critic_loss": round(self._lowest_loss, 6) if self._lowest_loss < float("inf") else None,
            "success_threshold": self.success_threshold,
            "log_files": {
                "training_metrics": self.training_csv_path,
                "episode_metrics": self.episode_csv_path,
                "eval_metrics": self.eval_csv_path,
                "metadata": self.metadata_path,
            },
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logging.info(f"[TrainingCSVLogger] Summary saved: {self.summary_path}")

    def _append_csv(self, filepath: str, row: dict, fields: list):
        """Append a single row to CSV file."""
        try:
            with open(filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writerow(row)
        except Exception as e:
            logging.warning(f"[TrainingCSVLogger] Failed to write to {filepath}: {e}")

    @staticmethod
    def _fmt(value):
        """Format numeric value for CSV output."""
        if value is None:
            return ""
        if isinstance(value, float):
            if abs(value) < 0.001:
                return f"{value:.6f}"
            return f"{value:.4f}"
        return value
