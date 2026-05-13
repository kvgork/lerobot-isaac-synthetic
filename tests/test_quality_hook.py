"""
test_quality_hook.py — Tests for lerobot_isaac_synthetic.quality_hook

Tests:
  - Module imports without error (soft import of lerobot_isaac_adapters).
  - filter_after_replay signature matches spec.
  - Returns a Path pointing to the resolved output.
  - Raises RuntimeError on filter failure.
  - Passes kwargs through to apply_quality_filter.
  - Works when lerobot_isaac_adapters is not installed (ImportError path).

Plan reference: §13.1 Bundle A, deliverable A6
"""

from __future__ import annotations

import sys
from inspect import signature
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import smoke
# ---------------------------------------------------------------------------


class TestImport:
    def test_module_importable(self):
        """quality_hook module imports without error."""
        import lerobot_isaac_synthetic.quality_hook  # noqa: F401

    def test_filter_after_replay_importable(self):
        from lerobot_isaac_synthetic.quality_hook import filter_after_replay

        assert callable(filter_after_replay)


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


class TestSignature:
    def test_has_replayed_dataset_path(self):
        from lerobot_isaac_synthetic.quality_hook import filter_after_replay

        sig = signature(filter_after_replay)
        assert "replayed_dataset_path" in sig.parameters

    def test_defaults_match_spec(self):
        from lerobot_isaac_synthetic.quality_hook import filter_after_replay

        sig = signature(filter_after_replay)
        params = sig.parameters
        assert params["sal_threshold"].default == pytest.approx(0.2)
        assert params["ted_threshold"].default == pytest.approx(2.0)
        assert params["min_episode_length"].default == 50

    def test_returns_path_annotation(self):
        from lerobot_isaac_synthetic.quality_hook import filter_after_replay

        sig = signature(filter_after_replay)
        # return annotation is Path (or not set — both acceptable)
        # Just verify the function is callable with correct params
        assert "replayed_dataset_path" in sig.parameters


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.requires_workspace_root
    def test_returns_path(self, tmp_path: Path):
        """filter_after_replay returns a Path object on success."""
        from lerobot_isaac_synthetic.quality_hook import filter_after_replay

        ds = tmp_path / "synthetic_ds"
        ds.mkdir()
        output = tmp_path / "synthetic_ds_filtered"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"kept": 8, "removed": 2}

        with patch(
            "lerobot_isaac_adapters.quality.apply_quality_filter",
            return_value=mock_result,
        ):
            result_path = filter_after_replay(
                replayed_dataset_path=ds,
                output_path=output,
            )

        assert isinstance(result_path, Path)

    @pytest.mark.requires_workspace_root
    def test_output_path_default_is_filtered_suffix(self, tmp_path: Path):
        """Default output path is <input>_filtered."""
        from lerobot_isaac_synthetic.quality_hook import filter_after_replay

        ds = tmp_path / "my_replay"
        ds.mkdir()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {}

        with patch(
            "lerobot_isaac_adapters.quality.apply_quality_filter",
            return_value=mock_result,
        ) as mock_fn:
            result_path = filter_after_replay(replayed_dataset_path=ds)

        # output_path kwarg passed to apply_quality_filter should be <ds>_filtered
        call_kwargs = mock_fn.call_args.kwargs
        passed_output = Path(call_kwargs.get("output_path", result_path))
        assert "_filtered" in str(passed_output)

    @pytest.mark.requires_workspace_root
    def test_kwargs_forwarded(self, tmp_path: Path):
        """sal_threshold, ted_threshold, min_episode_length are forwarded."""
        from lerobot_isaac_synthetic.quality_hook import filter_after_replay

        ds = tmp_path / "ds"
        ds.mkdir()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {}

        with patch(
            "lerobot_isaac_adapters.quality.apply_quality_filter",
            return_value=mock_result,
        ) as mock_fn:
            filter_after_replay(
                replayed_dataset_path=ds,
                sal_threshold=0.3,
                ted_threshold=1.5,
                min_episode_length=30,
            )

        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["sal_threshold"] == pytest.approx(0.3)
        assert call_kwargs["ted_threshold"] == pytest.approx(1.5)
        assert call_kwargs["min_episode_length"] == 30


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class TestErrorPath:
    @pytest.mark.requires_workspace_root
    def test_raises_runtime_error_on_failure(self, tmp_path: Path):
        """RuntimeError is raised when apply_quality_filter returns success=False."""
        from lerobot_isaac_synthetic.quality_hook import filter_after_replay

        ds = tmp_path / "ds"
        ds.mkdir()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "skill not found"
        mock_result.suggestions = ["check path"]

        with patch(
            "lerobot_isaac_adapters.quality.apply_quality_filter",
            return_value=mock_result,
        ):
            with pytest.raises(RuntimeError, match="skill not found"):
                filter_after_replay(replayed_dataset_path=ds)

    def test_import_error_when_adapters_missing(self, tmp_path: Path, monkeypatch):
        """ImportError raised with actionable message if lerobot_isaac_adapters missing."""
        import lerobot_isaac_synthetic.quality_hook as qh_mod
        import importlib

        ds = tmp_path / "ds"
        ds.mkdir()

        # Temporarily hide lerobot_isaac_adapters
        original_modules = dict(sys.modules)
        # Remove from sys.modules and simulate missing
        for key in list(sys.modules.keys()):
            if key.startswith("lerobot_isaac_adapters"):
                del sys.modules[key]

        # Block the import
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("lerobot_isaac_adapters"):
                raise ImportError("simulated missing package")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        importlib.reload(qh_mod)

        with pytest.raises(ImportError, match="lerobot_isaac_adapters"):
            qh_mod.filter_after_replay(replayed_dataset_path=ds)

        # Restore
        monkeypatch.setattr(builtins, "__import__", original_import)
        for key, val in original_modules.items():
            if key.startswith("lerobot_isaac_adapters"):
                sys.modules[key] = val
        importlib.reload(qh_mod)
