"""
Training CSV Logger for HIL-SERL SurRoL

将训练指标保存为 CSV 文件，方便离线查看训练进度，无需 WandB。

日志文件结构：
    output_dir/training_logs/
    ├── training_metrics.csv     # 训练指标（loss, temperature, grad_norm 等）
    ├── episode_metrics.csv      # 每 episode 指标（reward, intervention_rate 等）
    └── training_summary.json    # 训练总结（最终指标、最佳指标、训练时长等）
"""

import csv
import json
import logging
import os
import time
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
    ]

    EPISODE_FIELDS = [
        "timestamp",
        "interaction_step",
        "episodic_reward",
        "episode_intervention",
        "intervention_rate",
        "policy_freq_hz",
        "policy_freq_90p_hz",
    ]

    def __init__(self, output_dir: str, job_name: str = ""):
        self.log_dir = os.path.join(output_dir, "training_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.job_name = job_name
        self.training_csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        self.episode_csv_path = os.path.join(self.log_dir, "episode_metrics.csv")
        self.summary_path = os.path.join(self.log_dir, "training_summary.json")

        self.start_time = time.time()

        # Tracking best metrics
        self._best_reward = float("-inf")
        self._best_reward_step = 0
        self._lowest_loss = float("inf")
        self._total_episodes = 0
        self._total_interventions = 0
        self._last_optimization_step = 0

        # Initialize CSV files with headers
        self._init_csv(self.training_csv_path, self.TRAINING_FIELDS)
        self._init_csv(self.episode_csv_path, self.EPISODE_FIELDS)

        logging.info(f"[TrainingCSVLogger] Logging to: {self.log_dir}")

    def _init_csv(self, filepath: str, fields: list):
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()

    def log_training(self, training_infos: dict, optimization_step: int):
        """Log training metrics (loss, temperature, grad_norm, etc.)

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
        }

        # Track best loss
        loss = training_infos.get("loss_critic")
        if loss is not None and loss < self._lowest_loss:
            self._lowest_loss = loss

        self._last_optimization_step = optimization_step

        self._append_csv(self.training_csv_path, row, self.TRAINING_FIELDS)

    def log_episode(self, interaction_message: dict):
        """Log episode-level metrics (reward, intervention rate, etc.)

        Called when the learner receives an interaction message from the actor.
        """
        row = {
            "timestamp": f"{time.time() - self.start_time:.1f}",
            "interaction_step": interaction_message.get("Interaction step", ""),
            "episodic_reward": self._fmt(interaction_message.get("Episodic reward")),
            "episode_intervention": int(interaction_message.get("Episode intervention", 0)),
            "intervention_rate": self._fmt(interaction_message.get("Intervention rate", 0.0)),
            "policy_freq_hz": self._fmt(interaction_message.get("Policy frequency [Hz]")),
            "policy_freq_90p_hz": self._fmt(interaction_message.get("Policy frequency 90th-p [Hz]")),
        }

        # Track best reward
        reward = interaction_message.get("Episodic reward")
        if reward is not None:
            if reward > self._best_reward:
                self._best_reward = reward
                self._best_reward_step = interaction_message.get("Interaction step", 0)
            self._total_episodes += 1

        intervention = interaction_message.get("Episode intervention")
        if intervention is not None and int(intervention) == 1:
            self._total_interventions += 1

        self._append_csv(self.episode_csv_path, row, self.EPISODE_FIELDS)

    def save_summary(self):
        """Save training summary as JSON. Called at end of training or on checkpoint."""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        summary = {
            "job_name": self.job_name,
            "training_duration_s": round(elapsed, 1),
            "training_duration_h": round(hours, 2),
            "total_optimization_steps": self._last_optimization_step,
            "total_episodes": self._total_episodes,
            "total_interventions": self._total_interventions,
            "intervention_episode_ratio": (
                round(self._total_interventions / max(self._total_episodes, 1), 3)
            ),
            "best_episodic_reward": round(self._best_reward, 4) if self._best_reward > float("-inf") else None,
            "best_reward_at_step": self._best_reward_step,
            "lowest_critic_loss": round(self._lowest_loss, 6) if self._lowest_loss < float("inf") else None,
            "log_files": {
                "training_metrics": self.training_csv_path,
                "episode_metrics": self.episode_csv_path,
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
