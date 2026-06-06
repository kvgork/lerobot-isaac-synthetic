"""
test_obs_transform.py
=====================
Unit tests for the env-obs → LeRobot-row transform used by the DR replay path
(B.1 writer mapping). Requires torch — skipped where torch is absent.

Canonical SO-101 contract (matches robot_data_recorder):
  - observation.images.<cam>  HWC uint8, default column ``overhead``
  - observation.state         (12,) float32 = joint_pos[6] + joint_vel[6]
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from lerobot_isaac_synthetic.isaac_dr.replay_runner import (  # noqa: E402
    _env_obs_to_lerobot_row,
    _to_action_vec,
    _to_hwc_uint8,
    _to_state_vec,
)


def _fake_obs():
    # The env source camera term is named ``d435_rgb`` (CHW); the adapter must
    # auto-detect it and re-export under the canonical ``overhead`` column.
    return {
        "policy": {
            "d435_rgb": torch.rand(1, 3, 480, 640),  # float CHW batched
            "joint_pos": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
            "joint_vel": torch.zeros(1, 6),
            "last_action": torch.zeros(1, 6),
        }
    }


def test_camera_to_hwc_uint8_float_chw_input():
    img = _to_hwc_uint8(torch.rand(1, 3, 480, 640))
    assert img.shape == (480, 640, 3)  # CHW → HWC
    assert img.dtype == np.uint8
    assert 0 <= int(img.min()) and int(img.max()) <= 255


def test_camera_to_hwc_uint8_already_uint8_chw():
    raw = (torch.rand(3, 480, 640) * 255).to(torch.uint8)
    img = _to_hwc_uint8(raw)
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8


def test_camera_to_hwc_uint8_already_hwc():
    raw = (torch.rand(480, 640, 3) * 255).to(torch.uint8)
    img = _to_hwc_uint8(raw)
    assert img.shape == (480, 640, 3)


def test_state_vec_pads_velocity_to_12():
    # joint_pos only → velocity zero-filled, 12-dim canonical state.
    st = _to_state_vec(torch.tensor([[1.0, 2, 3, 4, 5, 6]]))
    assert st.shape == (12,)
    assert st.dtype == np.float32
    assert np.allclose(st[6:], 0.0)


def test_state_vec_uses_velocity_when_present():
    st = _to_state_vec(
        torch.tensor([[1.0, 2, 3, 4, 5, 6]]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
    )
    assert st.shape == (12,)
    assert np.allclose(st[:6], [1, 2, 3, 4, 5, 6])
    assert np.allclose(st[6:], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_action_vec():
    assert _to_action_vec(torch.tensor([1.0, 2, 3, 4, 5, 6])).shape == (6,)
    assert _to_action_vec(torch.tensor([[1.0, 2, 3, 4, 5, 6]])).shape == (6,)


def test_full_row_with_overhead_camera():
    # Default canonical column is ``overhead``; the env's d435_rgb tensor is
    # auto-detected and re-exported under it, in HWC.
    row = _env_obs_to_lerobot_row(_fake_obs(), "overhead")
    assert set(row) == {"observation.images.overhead", "observation.state"}
    assert row["observation.images.overhead"].shape == (480, 640, 3)
    assert row["observation.state"].shape == (12,)


def test_full_row_state_only_when_camera_none():
    row = _env_obs_to_lerobot_row(_fake_obs(), None)
    assert set(row) == {"observation.state"}
    assert row["observation.state"].shape == (12,)
