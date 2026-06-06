"""
test_mimicgen_bridge.py
=======================
Tests for the un-stubbed MimicGen bridge (Bundle D landed 2026-05-21).

These tests do NOT require MimicGen / robosuite / MuJoCo to be installed.
They verify:
- The gate behavior (NotImplementedError when disabled).
- The ImportError surfacing when the skill module is unavailable.
- The delegation wiring (with the skill module mocked).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Activation gate
# ---------------------------------------------------------------------------


def test_run_mimicgen_disabled_raises(monkeypatch):
    monkeypatch.delenv("LEROBOT_MIMICGEN_ENABLED", raising=False)
    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen

    with pytest.raises(NotImplementedError, match="explicit activation"):
        run_mimicgen(
            real_dataset_path="/tmp/in",
            n_synthetic_demos=10,
            task_config="pick_and_place",
            output_path="/tmp/out",
        )


def test_convert_real_to_mimicgen_disabled_raises(monkeypatch):
    monkeypatch.delenv("LEROBOT_MIMICGEN_ENABLED", raising=False)
    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import (
        convert_real_to_mimicgen_hdf5,
    )

    with pytest.raises(NotImplementedError, match="explicit activation"):
        convert_real_to_mimicgen_hdf5("/tmp/in", "/tmp/out.hdf5")


def test_convert_mimicgen_to_lerobot_disabled_raises(monkeypatch):
    monkeypatch.delenv("LEROBOT_MIMICGEN_ENABLED", raising=False)
    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import (
        convert_mimicgen_hdf5_to_lerobot,
    )

    with pytest.raises(NotImplementedError, match="explicit activation"):
        convert_mimicgen_hdf5_to_lerobot("/tmp/in.hdf5", "/tmp/out")


def test_env_var_enables_path(monkeypatch, tmp_path):
    """When env var is set but skill is unavailable, get a clear ImportError."""
    monkeypatch.setenv("LEROBOT_MIMICGEN_ENABLED", "1")
    # Ensure skill module is not importable
    monkeypatch.setitem(sys.modules, "lerobot_mimicgen_bridge.operations", None)
    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import (
        convert_real_to_mimicgen_hdf5,
    )

    with pytest.raises((ImportError, AttributeError, TypeError)):
        # AttributeError / TypeError covers the None-module edge case
        convert_real_to_mimicgen_hdf5(
            real_dataset_path=str(tmp_path / "in"),
            output_hdf5_path=str(tmp_path / "out.hdf5"),
        )


# ---------------------------------------------------------------------------
# Delegation wiring (skill module mocked)
# ---------------------------------------------------------------------------


def _install_mock_skill(monkeypatch):
    """Install a mock skill operations module in sys.modules."""
    mod = types.ModuleType("lerobot_mimicgen_bridge.operations")
    ok = MagicMock()
    ok.success = True
    ok.message = "ok"
    mod.parquet_to_mimicgen = MagicMock(return_value=ok)
    mod.mimicgen_to_parquet = MagicMock(return_value=ok)
    mod.run_mimicgen = MagicMock(return_value=ok)
    monkeypatch.setitem(sys.modules, "lerobot_mimicgen_bridge", types.ModuleType("lerobot_mimicgen_bridge"))
    monkeypatch.setitem(sys.modules, "lerobot_mimicgen_bridge.operations", mod)
    return mod


def test_convert_real_to_mimicgen_delegates(monkeypatch, tmp_path):
    monkeypatch.setenv("LEROBOT_MIMICGEN_ENABLED", "1")
    mock_ops = _install_mock_skill(monkeypatch)

    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import (
        convert_real_to_mimicgen_hdf5,
    )

    out_hdf5 = tmp_path / "out.hdf5"
    result = convert_real_to_mimicgen_hdf5(
        real_dataset_path=str(tmp_path / "real"),
        output_hdf5_path=str(out_hdf5),
        n_source_demos=5,
    )

    assert result == out_hdf5.resolve()
    mock_ops.parquet_to_mimicgen.assert_called_once()
    call_kwargs = mock_ops.parquet_to_mimicgen.call_args.kwargs
    assert call_kwargs["n_source_demos"] == 5


def test_convert_mimicgen_to_lerobot_delegates(monkeypatch, tmp_path):
    monkeypatch.setenv("LEROBOT_MIMICGEN_ENABLED", "1")
    mock_ops = _install_mock_skill(monkeypatch)

    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import (
        convert_mimicgen_hdf5_to_lerobot,
    )

    out = tmp_path / "out"
    result = convert_mimicgen_hdf5_to_lerobot(
        hdf5_path=str(tmp_path / "in.hdf5"),
        output_dataset_path=str(out),
        task_name="my_task",
        fps=20,
        camera_name="front",
    )

    assert result == out.resolve()
    mock_ops.mimicgen_to_parquet.assert_called_once()
    call_kwargs = mock_ops.mimicgen_to_parquet.call_args.kwargs
    assert call_kwargs["task_name"] == "my_task"
    assert call_kwargs["fps"] == 20
    assert call_kwargs["camera_name"] == "front"


def test_run_mimicgen_full_pipeline_delegates(monkeypatch, tmp_path):
    monkeypatch.setenv("LEROBOT_MIMICGEN_ENABLED", "1")
    mock_ops = _install_mock_skill(monkeypatch)

    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen

    out = tmp_path / "augmented"
    result = run_mimicgen(
        real_dataset_path=str(tmp_path / "real"),
        n_synthetic_demos=42,
        task_config="my_task",
        output_path=str(out),
        n_source_demos=8,
        fps=30,
        camera_name="agentview",
    )

    assert result == out.resolve()
    mock_ops.parquet_to_mimicgen.assert_called_once()
    mock_ops.run_mimicgen.assert_called_once()
    mock_ops.mimicgen_to_parquet.assert_called_once()

    run_kwargs = mock_ops.run_mimicgen.call_args.kwargs
    assert run_kwargs["n_demos"] == 42
    assert run_kwargs["task_config"] == "my_task"


def test_skill_failure_surfaces_as_runtime_error(monkeypatch, tmp_path):
    monkeypatch.setenv("LEROBOT_MIMICGEN_ENABLED", "1")
    mod = _install_mock_skill(monkeypatch)
    fail = MagicMock()
    fail.success = False
    fail.message = "synthetic failure"
    mod.parquet_to_mimicgen = MagicMock(return_value=fail)

    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import (
        convert_real_to_mimicgen_hdf5,
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        convert_real_to_mimicgen_hdf5(
            real_dataset_path=str(tmp_path / "real"),
            output_hdf5_path=str(tmp_path / "out.hdf5"),
        )
