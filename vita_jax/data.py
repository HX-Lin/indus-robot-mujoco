"""LeRobot frame/episode-aligned data adapter for the JAX VITA port.

The adapter intentionally mirrors the existing PyTorch ``EECacheDataset``
contract without importing PyTorch or copying datasets:

* sidecar ``.npz`` caches supply absolute 16-D EE state, 14-D residual actions,
  absolute future EE rows, optional action-aligned stage labels, and optional
  colour grounding masks;
* the base LeRobot dataset is asked for the observation history and one extra
  action-aligned future RGB frame per camera;
* the future frame is split off before the policy sees the observation batch,
  and a per-sample ``future_visual_valid`` mask records whether ``t+4`` is
  inside the episode.

``lerobot`` is imported lazily so the rest of the package (and its tests) can
run without the dataset stack installed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import DataConfig, ModelConfig


# ---------------------------------------------------------------------------
# NumPy pose helpers (dataset cache construction, no scipy dependency)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rotvec_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q * np.where(q[:, 0:1] < 0, -1.0, 1.0)
    w, v = q[:, 0], q[:, 1:4]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(v_norm, w[:, None])
    scale = np.where(v_norm > 1e-8, angle / np.maximum(v_norm, 1e-8), 0.0)
    return scale * v


def ee_pose14(state_ee: np.ndarray) -> np.ndarray:
    """[N,16] xyz/quat_wxyz/grip x2 -> [N,14] xyz/rotvec/grip x2."""
    a = np.asarray(state_ee, dtype=np.float64)
    if a.shape[-1] != 16:
        raise ValueError("state_ee must have 16 columns")
    parts = []
    for arm in range(2):
        base = arm * 8
        q = a[:, base + 3:base + 7]
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        parts.extend([a[:, base:base + 3], _quat_wxyz_to_rotvec_np(q),
                      a[:, base + 7:base + 8]])
    return np.concatenate(parts, axis=-1)


def stats_dict(a: np.ndarray) -> Dict[str, object]:
    a = np.asarray(a, dtype=np.float64)

    def q(p):
        return np.quantile(a, p, axis=0).tolist()

    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "count": [len(a)],
        "q01": q(0.01), "q10": q(0.10), "q50": q(0.50),
        "q90": q(0.90), "q99": q(0.99),
    }


def future_image_key(image_key: str) -> str:
    suffix = image_key.split("observation.images.", 1)[-1]
    return "observation.future_images." + suffix


# ---------------------------------------------------------------------------
# Sidecars and episode bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLayout:
    starts: np.ndarray
    ends: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray

    def __post_init__(self):
        if not (len(self.starts) == len(self.ends) == len(self.episode_index)
                == len(self.frame_index)):
            raise ValueError("episode layout arrays must have equal length")


def episode_layout(episode_index: Sequence[int], frame_index: Sequence[int]) -> EpisodeLayout:
    ep = np.asarray(episode_index, dtype=np.int64)
    fr = np.asarray(frame_index, dtype=np.int64)
    if ep.ndim != 1 or fr.ndim != 1 or len(ep) != len(fr):
        raise ValueError("episode_index and frame_index must be 1-D and equal length")
    if len(ep) == 0:
        return EpisodeLayout(np.zeros(0, np.int64), np.zeros(0, np.int64), ep, fr)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:] - 1, len(ep) - 1]
    lengths = ends - starts + 1
    return EpisodeLayout(
        starts=np.repeat(starts, lengths),
        ends=np.repeat(ends, lengths),
        episode_index=ep,
        frame_index=fr,
    )


class SidecarData:
    """Validated frame-aligned sidecars for one task."""

    def __init__(self, config: DataConfig, chunk_size: int = 16):
        self.config = config
        z = np.load(config.ee_cache)
        for key in ("state_ee", "episode_index", "frame_index"):
            if key not in z:
                raise ValueError(f"EE cache is missing {key!r}")
        self.state_ee = np.asarray(z["state_ee"], dtype=np.float32)
        if self.state_ee.ndim != 2 or self.state_ee.shape[1] != 16:
            raise ValueError("EE cache state_ee must have shape (N, 16)")
        self.action_ee = (np.asarray(z["action_ee"], dtype=np.float32)
                          if "action_ee" in z else None)
        self.action_res = (np.asarray(z["action_res"], dtype=np.float32)
                           if "action_res" in z else None)
        self.layout = episode_layout(z["episode_index"], z["frame_index"])
        n = len(self.state_ee)
        self.chunk_size = int(chunk_size)
        self.action_ee_chunk = None
        if self.action_ee is not None:
            if len(self.action_ee) != n:
                raise ValueError("action_ee must align with state_ee")
            ids = np.clip(
                np.arange(n)[:, None] + np.arange(self.chunk_size)[None],
                0, self.layout.ends[:, None])
            self.action_ee_chunk = self.action_ee[ids].astype(np.float32)

        self.stage = None
        if config.stage_cache:
            st = np.load(config.stage_cache)
            if "stage" not in st:
                raise ValueError("stage cache is missing 'stage'")
            self.stage = np.asarray(st["stage"], dtype=np.int64)
            if len(self.stage) != n:
                raise ValueError("stage cache length does not match EE cache")
            for key in ("episode_index", "frame_index"):
                if key not in st or not np.array_equal(st[key], z[key]):
                    raise ValueError(f"stage cache {key} does not match EE rows")

        self.grounding_masks = None
        self.grounding_image_keys: List[str] = []
        if config.grounding_cache:
            gr = np.load(config.grounding_cache)
            for key in ("index", "episode_index", "frame_index"):
                if key not in gr:
                    raise ValueError(f"grounding cache is missing {key!r}")
            if not np.array_equal(gr["episode_index"], z["episode_index"]) or not np.array_equal(
                    gr["frame_index"], z["frame_index"]):
                raise ValueError("grounding cache frame rows do not match EE rows")
            if "colors" not in gr or gr["colors"].tolist() != ["red", "green", "blue"]:
                raise ValueError("grounding cache must label red/green/blue in order")
            self.grounding_image_keys = [str(k) for k in gr["image_keys"].tolist()]
            self.grounding_masks = np.asarray(gr["masks"], dtype=np.uint8)
            if self.grounding_masks.shape[:3] != (n, len(self.grounding_image_keys), 3):
                raise ValueError("invalid colour grounding cache shape")

        self.stats = {
            "observation.state": stats_dict(self.state_ee),
            "action_ee_chunk": stats_dict(ee_pose14(self.state_ee)),
        }
        if self.action_res is not None:
            self.stats["action"] = stats_dict(self.action_res)
        if self.action_ee_chunk is not None:
            self.stats["action_ee_chunk_rows"] = stats_dict(self.action_ee_chunk.reshape(-1, 16))

    def __len__(self):
        return len(self.state_ee)

    def chunk_for(self, index: int) -> np.ndarray:
        if self.action_ee_chunk is None:
            raise ValueError("EE cache has no action_ee; absolute ODE targets require it")
        return self.action_ee_chunk[int(index)]

    def future_valid(self, index: int, horizon: int) -> bool:
        return int(index) + int(horizon) <= int(self.layout.ends[int(index)])


# ---------------------------------------------------------------------------
# Base dataset construction (lazy import)
# ---------------------------------------------------------------------------

def build_delta_timestamps(fps: float, image_keys: Sequence[str], obs_horizon: int,
                           chunk_size: int, future_visual_horizon: int) -> Dict[str, list]:
    """Mirror the active policy's delta-timestamp layout."""
    if obs_horizon < 1:
        raise ValueError("obs_horizon must be positive")
    timestamps: Dict[str, list] = {
        "action": [i / fps for i in range(chunk_size)],
    }
    if obs_horizon > 1:
        obs_ts = [-(obs_horizon - 1 - i) / fps for i in range(obs_horizon)]
        timestamps["observation.state"] = list(obs_ts)
        for key in image_keys:
            ts = list(obs_ts)
            if future_visual_horizon > 0:
                ts.append(future_visual_horizon / fps)
            timestamps[key] = ts
    return timestamps


def open_lerobot_dataset(config: DataConfig, model: ModelConfig, fps: float):
    """Open the base LeRobot dataset, importing the dependency only here."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover - dependency specific
        raise ImportError(
            "The LeRobot dataset stack is required for training. "
            "Install lerobot in the runtime environment. The JAX unit tests "
            "use in-memory sources and do not need it."
        ) from exc
    delta = build_delta_timestamps(
        fps=fps,
        image_keys=list(model.image_keys),
        obs_horizon=model.obs_horizon,
        chunk_size=model.chunk_size,
        future_visual_horizon=config.future_visual_horizon,
    )
    return LeRobotDataset(
        repo_id=config.repo_id,
        root=config.data_dir,
        video_backend=config.video_backend,
        delta_timestamps=delta,
    )


# ---------------------------------------------------------------------------
# JAX-facing dataset
# ---------------------------------------------------------------------------

class VITADataset:
    """Frame-aligned dataset returning the batch keys used by the policy."""

    def __init__(self, base, sidecar: SidecarData, model: ModelConfig,
                 future_visual_horizon: int = 4):
        self.base = base
        self.sidecar = sidecar
        self.model = model
        self.future_visual_horizon = int(future_visual_horizon)
        if len(sidecar) != len(base):
            raise ValueError(
                f"sidecar rows ({len(sidecar)}) do not match base frames ({len(base)})")
        self.future_image_keys = [future_image_key(k) for k in model.image_keys]
        self.image_keys = list(model.image_keys)
        self.episode_starts = sidecar.layout.starts
        self.episode_ends = sidecar.layout.ends
        self.episode_index = sidecar.layout.episode_index

    def __len__(self):
        return len(self.base)

    def future_valid(self, index: int) -> bool:
        return self.sidecar.future_valid(index, self.future_visual_horizon)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        idx = int(index)
        item = self.base[idx]
        g = int(item.get("index", idx))
        if g != idx:
            raise ValueError(
                f"base dataset row {idx} returned global index {g}; "
                "the JAX sidecars require the original frame ordering")
        for key in ("episode_index", "frame_index"):
            if key in item and int(np.asarray(item[key])) != int(
                    getattr(self.sidecar.layout, key)[g]):
                raise ValueError(
                    f"base dataset {key} does not align with EE sidecar row {g}")
        if self.future_visual_horizon > 0:
            for key in self.image_keys:
                frames = np.asarray(item[key])
                if frames.ndim != 4 or frames.shape[0] < 2:
                    raise ValueError(
                        "future_visual_horizon requires the base dataset to "
                        "append the future frame to each image key")
                item[future_image_key(key)] = frames[-1]
                item[key] = frames[:-1]
            item["observation.future_visual_valid"] = np.asarray(
                self.future_valid(g), dtype=np.bool_)

        # Replace the base state/action rows with the sidecar's absolute EE
        # rows.  The base loader still supplies RGB and episode boundaries.
        state = np.asarray(item["observation.state"])
        if state.ndim == 2:
            s = state.shape[0]
            ids = np.clip(np.arange(g - s + 1, g + 1),
                          self.episode_starts[g], g)
        else:
            ids = np.asarray([g])
        item["observation.state"] = self.sidecar.state_ee[ids]
        item["action_ee_chunk"] = self.sidecar.chunk_for(g)
        if self.sidecar.action_res is not None:
            t = np.asarray(item["action"]).shape[0]
            aid = np.clip(np.arange(g, g + t), g, self.episode_ends[g])
            item["action"] = self.sidecar.action_res[aid]

        if self.sidecar.stage is not None:
            item["observation.stage"] = np.asarray(self.sidecar.stage[g], dtype=np.int32)
        if self.sidecar.grounding_masks is not None:
            for camera, key in enumerate(self.sidecar.grounding_image_keys):
                item["grounding." + key] = (
                    self.sidecar.grounding_masks[g, camera].astype(np.float32) / 255.0)
        return item


def collate(samples: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stack a list of samples (numpy only; JAX conversion happens later)."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = samples[0].keys()
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        out[key] = np.stack([np.asarray(s[key]) for s in samples], axis=0)
    return out


class EpisodeSampler:
    """Episode-aware frame sampler with the active run's tail drop."""

    def __init__(self, dataset: VITADataset, episodes: Iterable[int],
                 drop_n_last_frames: int, seed: int = 42, shuffle: bool = True):
        self.dataset = dataset
        self.episodes = sorted(int(e) for e in episodes)
        self.drop_n = max(0, int(drop_n_last_frames))
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        ep = dataset.episode_index
        self.indices = []
        for e in self.episodes:
            rows = np.flatnonzero(ep == e)
            if len(rows) == 0:
                continue
            valid = rows[:max(1, len(rows) - self.drop_n)]
            self.indices.append(valid)
        self.indices = np.concatenate(self.indices) if self.indices else np.zeros(0, np.int64)

    def __len__(self):
        return len(self.indices)

    def epoch(self) -> np.ndarray:
        order = self.indices.copy()
        if self.shuffle:
            self.rng.shuffle(order)
        return order

    def batches(self, batch_size: int) -> Iterator[List[int]]:
        order = self.epoch()
        for start in range(0, len(order), batch_size):
            yield order[start:start + batch_size].tolist()


def split_episodes(episode_index: np.ndarray, val_split: float, seed: int
                   ) -> Tuple[List[int], List[int]]:
    episodes = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * float(val_split))) if val_split > 0 else 0
    val = sorted(int(e) for e in shuffled[:n_val])
    train = sorted(int(e) for e in shuffled[n_val:])
    return train, val


def build_datasets(config: DataConfig, model: ModelConfig, fps: float
                   ) -> Tuple[VITADataset, Optional[VITADataset]]:
    """Build train/val datasets from runtime paths; does not copy any data."""
    sidecar = SidecarData(config, chunk_size=model.chunk_size)
    base = open_lerobot_dataset(config, model, fps)
    dataset = VITADataset(base, sidecar, model,
                          future_visual_horizon=config.future_visual_horizon)
    train_eps, val_eps = split_episodes(sidecar.layout.episode_index,
                                        config.val_split, config.seed)
    train_ds = _EpisodeSubset(dataset, train_eps)
    val_ds = _EpisodeSubset(dataset, val_eps) if val_eps else None
    return train_ds, val_ds


class _EpisodeSubset:
    """A lightweight view that exposes a shared VITADataset and episode list."""

    def __init__(self, dataset: VITADataset, episodes: Sequence[int]):
        self.dataset = dataset
        self.episodes = list(episodes)
        self.episode_index = dataset.episode_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def sampler(self, drop_n_last_frames: int, seed: int = 42, shuffle: bool = True):
        return EpisodeSampler(self.dataset, self.episodes,
                              drop_n_last_frames, seed=seed, shuffle=shuffle)

    @property
    def image_keys(self):
        return self.dataset.image_keys


def normalization_arrays(sidecar: SidecarData) -> Dict[str, Dict[str, np.ndarray]]:
    """Small normalized arrays used by the ODE head and input normalization."""
    return {
        "observation.state": {
            "mean": np.asarray(sidecar.stats["observation.state"]["mean"], np.float32),
            "std": np.asarray(sidecar.stats["observation.state"]["std"], np.float32),
        },
        "action_ee_chunk": {
            "mean": np.asarray(sidecar.stats["action_ee_chunk"]["mean"], np.float32),
            "std": np.asarray(sidecar.stats["action_ee_chunk"]["std"], np.float32),
        },
    }
