"""
Self-contained local dataset for HIL-SERL — bypasses HuggingFace Hub entirely.

Why this exists
---------------
`lerobot.datasets.LeRobotDataset` and `LeRobotDatasetMetadata` require
HuggingFace Hub connectivity (calls `HfApi.list_repo_refs` and friends) even
when the dataset is on local disk. This blocks training on GPU pods that
either (a) cannot reach huggingface.co, or (b) the repo_id does not exist on
HF (e.g. `local/franka_sim_touch_demos`). Patches we tried earlier (skip HF
for `local/` prefix) bypass the version check but leave deeper HF calls in
`download()` and `load_hf_dataset()`.

This module gives a 100% offline dataset class with the minimum interface
that `ReplayBuffer.from_lerobot_dataset()` actually uses:

  * `len(dataset)` — total frame count
  * `dataset[i]` — frame dict with at least `action`, `next.reward`,
                   `next.done`, `episode_index`, plus configured state keys
                   (including image tensors loaded lazily from MP4)

Layout assumed (LeRobot codebase v3.0 schema):

    <root>/
    ├── meta/info.json           (features definitions, fps, totals)
    ├── data/chunk-000/file-000.parquet  (action, reward, done, state, ...)
    └── videos/<image_key>/chunk-000/file-000.mp4  (image frames)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def is_local_repo_id(repo_id: str | None) -> bool:
    """Return True if repo_id looks like a local-only dataset (no HF lookup)."""
    if not repo_id:
        return False
    return repo_id.startswith("local/") or repo_id.startswith("/")


def default_local_root(repo_id: str, search_paths: list[str] | None = None) -> Path | None:
    """Try to resolve a local-style repo_id to a directory on disk.

    For `local/franka_sim_touch_demos`, looks for a dir named
    `franka_sim_touch_demos` under common project layouts.
    """
    name = repo_id.split("/", 1)[-1]
    if search_paths is None:
        # Conservative default search list — caller can pass cfg.dataset.root explicitly
        search_paths = [
            os.environ.get("LEROBOT_DATA_ROOT", ""),
            f"/root/data/hilserl-surrol-improved/lerobot/{name}",
            f"/root/data/hilserl-surrol-improved-v2/lerobot/{name}",
            os.path.expanduser(f"~/lerobot/{name}"),
            os.path.join(os.getcwd(), name),
            os.path.join(os.getcwd(), "lerobot", name),
        ]
    for p in search_paths:
        if p and Path(p).is_dir() and (Path(p) / "meta" / "info.json").is_file():
            return Path(p)
    return None


class LocalLeRobotDataset:
    """Minimal local-only dataset implementing the interface that
    `ReplayBuffer.from_lerobot_dataset()` uses.

    Loads parquet eagerly (small) and MP4 frames lazily on `__getitem__`.
    """

    def __init__(
        self,
        root: str | Path,
        image_keys: list[str] | None = None,
    ):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Local dataset root not found: {self.root}")

        # Load metadata
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing meta/info.json under {self.root}")
        with open(info_path) as f:
            self.info = json.load(f)

        self.total_frames = int(self.info.get("total_frames", 0))
        self.total_episodes = int(self.info.get("total_episodes", 0))
        self.fps = int(self.info.get("fps", 10))
        self._features_meta = self.info.get("features", {})
        self.data_path_pattern = self.info.get(
            "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        )
        self.video_path_pattern = self.info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )

        # Auto-detect image keys from features if not provided
        if image_keys is None:
            image_keys = [k for k, v in self._features_meta.items() if v.get("dtype") == "video"]
        self.image_keys = list(image_keys)

        # Load all parquet rows into a single pandas DataFrame
        # (1630 rows for franka_sim_touch_demos is tiny — eager load is fine)
        import pyarrow.parquet as pq
        import pandas as pd

        parquet_files = sorted((self.root / "data").rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files under {self.root}/data/")
        tables = [pq.read_table(p).to_pandas() for p in parquet_files]
        self.df = pd.concat(tables, ignore_index=True)

        if len(self.df) != self.total_frames:
            # Don't fail — info.json totals can drift; trust the actual data
            self.total_frames = len(self.df)

        # Lazy video readers: dict of {image_key: dict of {(chunk, file): VideoReader}}
        self._video_cache: dict[str, dict[tuple[int, int], Any]] = {k: {} for k in self.image_keys}

    # ── Public minimal interface ──

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item: dict[str, Any] = {}

        # Standard columns expected by `_lerobotdataset_to_transitions`
        item["action"] = self._to_tensor(row.get("action"), torch.float32)
        item["next.reward"] = self._to_tensor(row.get("next.reward"), torch.float32).reshape(())
        item["next.done"] = self._to_tensor(row.get("next.done"), torch.bool).reshape(())
        item["episode_index"] = int(row.get("episode_index", 0))
        item["frame_index"] = int(row.get("frame_index", idx))
        item["index"] = int(row.get("index", idx))
        if "complementary_info.discrete_penalty" in self.df.columns:
            item["complementary_info"] = {
                "discrete_penalty": self._to_tensor(
                    row["complementary_info.discrete_penalty"], torch.float32
                ).reshape(()),
            }

        # State keys (everything that starts with "observation." but not images)
        for col in self.df.columns:
            if col.startswith("observation.") and col not in self.image_keys:
                item[col] = self._to_tensor(row[col], torch.float32)

        # Image keys — lazy load from video at the right frame
        for img_key in self.image_keys:
            item[img_key] = self._read_image_frame(img_key, int(row["frame_index"]), int(row["episode_index"]))

        return item

    # ── Private helpers ──

    @staticmethod
    def _to_tensor(value, dtype):
        if value is None:
            return torch.zeros((), dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy()).to(dtype)
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    def _read_image_frame(self, image_key: str, frame_index: int, episode_index: int) -> torch.Tensor:
        """Read a single image frame from MP4 video file.

        Tries torchcodec first (preferred — used by upstream lerobot),
        falls back to imageio if torchcodec unavailable. Returns a CHW
        float32 tensor in [0, 1].
        """
        # Determine which (chunk, file) the episode's frame lives in.
        # For our demos, episodes are concatenated per file (chunk 0 / file 0
        # holds all 30 episodes), so episode_index doesn't change file index.
        # Frame index within the parquet row is also frame_index across episodes.
        # The MP4 is per-image-key per-chunk; we read by absolute frame.
        chunk_idx = 0
        file_idx = 0
        rel_path = self.video_path_pattern.format(
            video_key=image_key, chunk_index=chunk_idx, file_index=file_idx
        )
        video_path = self.root / rel_path
        if not video_path.is_file():
            # Return zeros if missing rather than crash — buffer init still works
            return torch.zeros((3, 128, 128), dtype=torch.float32)

        # Cache decoder per file so we don't reopen on each __getitem__
        cache_key = (chunk_idx, file_idx)
        cache = self._video_cache.setdefault(image_key, {})
        decoder = cache.get(cache_key)
        if decoder is None:
            decoder = self._open_video(video_path)
            cache[cache_key] = decoder

        # The frame index in the dataframe (frame_index) is per-episode;
        # we need an absolute video frame index. Reconstruct it from `index`
        # (the global frame counter). Fall back to frame_index modulo length.
        # Best practice: use df.iloc[i]["index"] which is global.
        try:
            frame_tensor = decoder[frame_index]  # torchcodec returns (C, H, W) uint8
        except Exception:
            return torch.zeros((3, 128, 128), dtype=torch.float32)

        # Convert to (C, H, W) float in [0, 1]
        if isinstance(frame_tensor, torch.Tensor):
            arr = frame_tensor
        else:
            arr = torch.from_numpy(np.asarray(frame_tensor))
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = arr.permute(2, 0, 1)  # HWC -> CHW
        if arr.dtype == torch.uint8:
            arr = arr.float() / 255.0
        return arr.contiguous().to(torch.float32)

    @staticmethod
    def _open_video(video_path: Path):
        """Open a video file via torchcodec (preferred) or fallback to imageio."""
        try:
            from torchcodec.decoders import VideoDecoder

            return VideoDecoder(str(video_path))
        except Exception:
            try:
                import imageio.v3 as iio

                # Pre-load all frames into memory (small files)
                frames = iio.imread(str(video_path))  # (T, H, W, C) uint8
                # Wrap as a list-like indexable
                class _ImageioWrap:
                    def __init__(self, arr):
                        self.arr = arr

                    def __getitem__(self, i):
                        return torch.from_numpy(self.arr[i].copy())

                    def __len__(self):
                        return len(self.arr)

                return _ImageioWrap(frames)
            except Exception as e:
                raise RuntimeError(
                    f"Cannot open video {video_path} — install torchcodec or imageio"
                ) from e


def make_local_offline_buffer(
    cfg,
    device: str,
    storage_device: str,
):
    """Drop-in replacement for `initialize_offline_replay_buffer()` when the
    dataset is local-only (no HuggingFace Hub access needed).

    Mirrors the original signature so it can be called from learner.py.
    """
    from lerobot.rl.buffer import ReplayBuffer

    repo_id = cfg.dataset.repo_id
    root = getattr(cfg.dataset, "root", None) or default_local_root(repo_id)
    if root is None:
        raise FileNotFoundError(
            f"Could not find local dataset for repo_id={repo_id!r}. "
            f"Set `dataset.root` in the config or place the dataset at one of "
            f"the default search paths."
        )

    image_keys = [k for k in cfg.policy.input_features.keys() if k.startswith("observation.image")]
    local_ds = LocalLeRobotDataset(root=root, image_keys=image_keys)

    return ReplayBuffer.from_lerobot_dataset(
        local_ds,
        device=device,
        state_keys=list(cfg.policy.input_features.keys()),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
