"""
test_dry_run.py
===============
Tests for the ``replay_runner`` CLI ``--dry_run`` flag.

The dry-run path short-circuits before any lazy imports (lerobot,
lerobot_isaac_env, gymnasium) are attempted, so these tests work
without Isaac Lab or lerobot installed.
"""

import sys
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_main(argv: list, replay_runner_mod) -> str:
    """Invoke replay_runner.main() with patched sys.argv; return captured stdout."""
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        replay_runner_mod.main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# --dry_run tests (no lerobot/isaaclab needed)
# ---------------------------------------------------------------------------


def test_dry_run_returns_without_error(tmp_path, capsys):
    """--dry_run exits cleanly (return code 0 = no exception)."""
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "real_dataset"
    _run_main(
        [
            "replay_runner",
            "--source_dataset",
            str(fake_src),
            "--dry_run",
        ],
        replay_runner,
    )
    # If we reach here, no exception was raised — test passes


def test_dry_run_prints_source_dataset(tmp_path, capsys):
    """--dry_run output includes the resolved source_dataset path."""
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "real_dataset"
    _run_main(
        [
            "replay_runner",
            "--source_dataset",
            str(fake_src),
            "--dry_run",
        ],
        replay_runner,
    )
    out = capsys.readouterr().out
    assert str(fake_src) in out


def test_dry_run_prints_n_variants(tmp_path, capsys):
    """--dry_run output includes n_variants."""
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "real_dataset"
    _run_main(
        [
            "replay_runner",
            "--source_dataset",
            str(fake_src),
            "--n_variants",
            "7",
            "--dry_run",
        ],
        replay_runner,
    )
    out = capsys.readouterr().out
    assert "7" in out


def test_dry_run_prints_task(tmp_path, capsys):
    """--dry_run output includes the task name."""
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "real_dataset"
    _run_main(
        [
            "replay_runner",
            "--source_dataset",
            str(fake_src),
            "--task",
            "stack",
            "--dry_run",
        ],
        replay_runner,
    )
    out = capsys.readouterr().out
    assert "stack" in out


def test_dry_run_default_output_path_contains_timestamp(tmp_path, capsys):
    """When --output_path is omitted, --dry_run shows a timestamped default path."""
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "real_dataset"
    _run_main(
        [
            "replay_runner",
            "--source_dataset",
            str(fake_src),
            "--dry_run",
        ],
        replay_runner,
    )
    out = capsys.readouterr().out
    assert "dr_replay_" in out


def test_dry_run_explicit_output_path(tmp_path, capsys):
    """When --output_path is given, --dry_run echoes it back."""
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "real_dataset"
    fake_out = tmp_path / "synthetic_out"
    _run_main(
        [
            "replay_runner",
            "--source_dataset",
            str(fake_src),
            "--output_path",
            str(fake_out),
            "--dry_run",
        ],
        replay_runner,
    )
    out = capsys.readouterr().out
    assert str(fake_out) in out


def test_dry_run_seed_propagated(tmp_path, capsys):
    """--seed value appears in the dry-run output."""
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "real_dataset"
    _run_main(
        [
            "replay_runner",
            "--source_dataset",
            str(fake_src),
            "--seed",
            "42",
            "--dry_run",
        ],
        replay_runner,
    )
    out = capsys.readouterr().out
    assert "42" in out


def test_dry_run_does_not_import_lerobot(tmp_path, monkeypatch):
    """--dry_run must NOT trigger lerobot/gymnasium imports."""
    import sys as _sys

    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    # Track import attempts for lerobot / gymnasium / lerobot_isaac_env
    imported = []

    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ("lerobot", "gymnasium", "lerobot_isaac_env"):
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    fake_src = tmp_path / "real_dataset"
    old_argv = _sys.argv[:]
    _sys.argv = [
        "replay_runner",
        "--source_dataset",
        str(fake_src),
        "--dry_run",
    ]
    try:
        replay_runner.main()
    finally:
        _sys.argv = old_argv

    assert imported == [], (
        f"dry_run imported heavy deps that should be deferred: {imported}"
    )


# ---------------------------------------------------------------------------
# Non-dry-run guard: ensures ImportError (not NIE) when lerobot missing
# ---------------------------------------------------------------------------


def test_full_run_raises_import_error_not_not_implemented(tmp_path):
    """Without --dry_run, main() triggers replay which raises ImportError."""
    import sys as _sys
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "real_dataset"
    fake_out = tmp_path / "out"

    old_argv = _sys.argv[:]
    _sys.argv = [
        "replay_runner",
        "--source_dataset",
        str(fake_src),
        "--output_path",
        str(fake_out),
    ]
    try:
        with pytest.raises(ImportError):
            replay_runner.main()
    finally:
        _sys.argv = old_argv
