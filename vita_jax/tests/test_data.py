import numpy as np

from vita_jax.config import DataConfig, ModelConfig
from vita_jax.data import (SidecarData, VITADataset, build_delta_timestamps,
                           future_image_key, split_episodes)


class FakeBase:
    def __init__(self, n=6, frames=2):
        self.n = n
        self.frames = frames

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        item = {
            "index": np.asarray(idx),
            "observation.state": np.zeros((self.frames, 16), np.float32),
            "action": np.zeros((3, 14), np.float32),
            "observation.images.cam": np.zeros(
                (self.frames + 1, 3, 8, 8), np.float32),
        }
        return item


def _write_cache(tmp_path):
    n = 6
    state = np.zeros((n, 16), np.float32)
    state[:, 0] = np.arange(n)
    state[:, 3] = 1.0
    state[:, 8] = np.arange(n)
    state[:, 11] = 1.0
    np.savez(tmp_path / "ee.npz", state_ee=state, action_ee=state.copy(),
             episode_index=np.asarray([0, 0, 0, 1, 1, 1]),
             frame_index=np.asarray([0, 1, 2, 0, 1, 2]))
    np.savez(tmp_path / "stage.npz", stage=np.zeros(n, np.int64),
             episode_index=np.asarray([0, 0, 0, 1, 1, 1]),
             frame_index=np.asarray([0, 1, 2, 0, 1, 2]))


def test_timestamps_use_dataset_fps_for_t_plus_four():
    timestamps = build_delta_timestamps(
        fps=15.0, image_keys=("observation.images.cam",),
        obs_horizon=2, chunk_size=16, future_visual_horizon=4)
    assert timestamps["observation.images.cam"] == [
        -1 / 15.0, 0.0, 4 / 15.0]
    assert timestamps["action"][:4] == [i / 15.0 for i in range(4)]


def test_zero_validation_split_keeps_all_episodes_in_training():
    train, val = split_episodes(np.repeat(np.arange(50), 2), 0.0, seed=42)
    assert val == []
    assert train == list(range(50))


def test_future_frame_split_and_invalid_horizon_mask(tmp_path):
    _write_cache(tmp_path)
    data_cfg = DataConfig(data_dir="x", repo_id="x",
                          ee_cache=str(tmp_path / "ee.npz"),
                          stage_cache=str(tmp_path / "stage.npz"),
                          future_visual_horizon=4)
    model = ModelConfig(image_keys=("observation.images.cam",))
    sidecar = SidecarData(data_cfg, chunk_size=4)
    ds = VITADataset(FakeBase(), sidecar, model, future_visual_horizon=1)
    first = ds[0]
    assert first["observation.images.cam"].shape[0] == 2
    assert first[future_image_key("observation.images.cam")].shape[0] == 3
    assert bool(first["observation.future_visual_valid"]) is True
    last_episode0 = ds[2]
    assert bool(last_episode0["observation.future_visual_valid"]) is False
    assert last_episode0["action_ee_chunk"].shape == (4, 16)
