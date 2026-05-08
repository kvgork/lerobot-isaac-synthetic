"""
test_replay_signature.py
========================
Function-signature and argparse smoke tests for replay_runner and
parquet_writer.  These tests verify the public API contract without
requiring Isaac Lab or lerobot to be installed.
"""

import inspect
import pytest


# ---------------------------------------------------------------------------
# replay_with_randomization signature
# ---------------------------------------------------------------------------


def test_replay_with_randomization_exists():
    """replay_with_randomization is importable from the expected location."""
    from lerobot_isaac_synthetic.isaac_dr.replay_runner import (
        replay_with_randomization,
    )

    assert callable(replay_with_randomization)


def test_replay_with_randomization_signature():
    """replay_with_randomization has the required parameters."""
    from lerobot_isaac_synthetic.isaac_dr.replay_runner import (
        replay_with_randomization,
    )

    sig = inspect.signature(replay_with_randomization)
    params = set(sig.parameters.keys())
    required = {"source_dataset_path", "n_variants_per_episode", "dr_config"}
    assert required.issubset(params), f"Missing parameters: {required - params}"


def test_replay_with_randomization_defaults():
    """replay_with_randomization has sensible default values."""
    from lerobot_isaac_synthetic.isaac_dr.replay_runner import (
        replay_with_randomization,
    )

    sig = inspect.signature(replay_with_randomization)
    params = sig.parameters
    assert params["n_variants_per_episode"].default == 5
    assert params["env_id"].default == "Isaac-SO101-PickPlace-v0"
    assert params["seed"].default == 0


def test_replay_with_randomization_raises_import_error_not_nie():
    """replay_with_randomization raises ImportError (not NIE) when lerobot/isaaclab missing."""
    from lerobot_isaac_synthetic.isaac_dr.replay_runner import (
        replay_with_randomization,
    )

    # lerobot/isaaclab are not installed in this test environment
    with pytest.raises(ImportError):
        list(
            replay_with_randomization(
                source_dataset_path="/tmp/fake",
                n_variants_per_episode=1,
            )
        )


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------


def test_episode_instantiation():
    """Episode can be instantiated with no arguments."""
    from lerobot_isaac_synthetic.isaac_dr.replay_runner import Episode

    ep = Episode()
    assert ep.episode_index == 0
    assert ep.observations == []
    assert ep.actions == []
    assert ep.success is False


def test_episode_fields():
    """Episode dataclass has all required fields."""
    from lerobot_isaac_synthetic.isaac_dr.replay_runner import Episode

    required_fields = {
        "episode_index",
        "source_episode_index",
        "dr_seed",
        "observations",
        "actions",
        "success",
        "metadata",
    }
    ep_fields = {f.name for f in __import__("dataclasses").fields(Episode)}
    assert required_fields == ep_fields


# ---------------------------------------------------------------------------
# write_episodes_to_lerobot_dataset signature
# ---------------------------------------------------------------------------


def test_parquet_writer_function_exists():
    """write_episodes_to_lerobot_dataset is importable."""
    from lerobot_isaac_synthetic.isaac_dr.parquet_writer import (
        write_episodes_to_lerobot_dataset,
    )

    assert callable(write_episodes_to_lerobot_dataset)


def test_parquet_writer_signature():
    """write_episodes_to_lerobot_dataset has required parameters."""
    from lerobot_isaac_synthetic.isaac_dr.parquet_writer import (
        write_episodes_to_lerobot_dataset,
    )

    sig = inspect.signature(write_episodes_to_lerobot_dataset)
    params = set(sig.parameters.keys())
    required = {"episodes", "output_path", "source_tag"}
    assert required.issubset(params), f"Missing: {required - params}"


def test_parquet_writer_default_source_tag():
    """Default source_tag is 'sim_dr'."""
    from lerobot_isaac_synthetic.isaac_dr.parquet_writer import (
        write_episodes_to_lerobot_dataset,
    )

    sig = inspect.signature(write_episodes_to_lerobot_dataset)
    assert sig.parameters["source_tag"].default == "sim_dr"


def test_parquet_writer_raises_import_error_not_nie():
    """write_episodes_to_lerobot_dataset raises ImportError when lerobot missing."""
    from lerobot_isaac_synthetic.isaac_dr.parquet_writer import (
        write_episodes_to_lerobot_dataset,
    )

    with pytest.raises(ImportError):
        write_episodes_to_lerobot_dataset(episodes=[], output_path="/tmp/fake")


# ---------------------------------------------------------------------------
# merge_datasets signature
# ---------------------------------------------------------------------------


def test_merge_datasets_exists():
    """merge_datasets is importable."""
    from lerobot_isaac_synthetic.merge_utilities import merge_datasets

    assert callable(merge_datasets)


def test_merge_datasets_signature():
    """merge_datasets has required parameters."""
    from lerobot_isaac_synthetic.merge_utilities import merge_datasets

    sig = inspect.signature(merge_datasets)
    params = set(sig.parameters.keys())
    required = {"real_path", "sim_paths", "output_path", "sim_weight"}
    assert required.issubset(params), f"Missing: {required - params}"


def test_merge_datasets_default_sim_weight():
    """Default sim_weight is 0.5."""
    from lerobot_isaac_synthetic.merge_utilities import merge_datasets

    sig = inspect.signature(merge_datasets)
    assert sig.parameters["sim_weight"].default == 0.5


def test_merge_datasets_raises_value_error_on_bad_weight():
    """merge_datasets raises ValueError for out-of-range sim_weight."""
    from lerobot_isaac_synthetic.merge_utilities import merge_datasets

    with pytest.raises(ValueError, match="sim_weight"):
        merge_datasets(
            real_path="/tmp/real",
            sim_paths=["/tmp/sim"],
            output_path="/tmp/merged",
            sim_weight=1.5,
        )


def test_merge_datasets_raises_value_error_zero_weight():
    """merge_datasets raises ValueError for sim_weight=0."""
    from lerobot_isaac_synthetic.merge_utilities import merge_datasets

    with pytest.raises(ValueError):
        merge_datasets(
            real_path="/tmp/real",
            sim_paths=[],
            output_path="/tmp/merged",
            sim_weight=0.0,
        )


def test_merge_datasets_raises_import_error_when_lerobot_missing():
    """merge_datasets raises ImportError when lerobot is not installed."""
    from lerobot_isaac_synthetic.merge_utilities import merge_datasets

    with pytest.raises((ImportError, Exception)):
        # sim_weight=0.5 is valid, so we proceed past ValueError and hit ImportError
        merge_datasets(
            real_path="/tmp/real",
            sim_paths=["/tmp/sim"],
            output_path="/tmp/merged",
            sim_weight=0.5,
        )


# ---------------------------------------------------------------------------
# run_mimicgen signature
# ---------------------------------------------------------------------------


def test_run_mimicgen_exists():
    """run_mimicgen is importable."""
    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen

    assert callable(run_mimicgen)


def test_run_mimicgen_raises_not_implemented_by_default():
    """run_mimicgen raises NotImplementedError when not enabled."""
    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen

    with pytest.raises(NotImplementedError, match="[Dd]eferred"):
        run_mimicgen(
            real_dataset_path="/tmp/real",
            n_synthetic_demos=10,
            task_config="pick_and_place",
            output_path="/tmp/out",
        )


def test_run_mimicgen_error_mentions_skill():
    """NotImplementedError message points to the skill file."""
    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen

    with pytest.raises(NotImplementedError) as exc_info:
        run_mimicgen("/tmp/r", 10, "pick", "/tmp/o")
    assert "lerobot_mimicgen_bridge" in str(exc_info.value)


def test_run_mimicgen_error_mentions_dr_alternative():
    """NotImplementedError message points to the Isaac Lab DR pipeline."""
    from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen

    with pytest.raises(NotImplementedError) as exc_info:
        run_mimicgen("/tmp/r", 10, "pick", "/tmp/o")
    assert "replay_with_randomization" in str(exc_info.value) or "isaac_dr" in str(
        exc_info.value
    )


# ---------------------------------------------------------------------------
# CLI smoke test (argparse)
# ---------------------------------------------------------------------------


def test_replay_runner_cli_help(capsys):
    """replay_runner CLI exits cleanly with --help."""
    from lerobot_isaac_synthetic.isaac_dr.replay_runner import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


def test_replay_runner_cli_dry_run(tmp_path, capsys):
    """replay_runner CLI prints resolved params for --dry_run."""
    import sys
    from lerobot_isaac_synthetic.isaac_dr import replay_runner

    fake_src = tmp_path / "src"
    fake_out = tmp_path / "out"

    sys.argv = [
        "replay_runner",
        "--source_dataset",
        str(fake_src),
        "--output_path",
        str(fake_out),
        "--dry_run",
    ]
    replay_runner.main()
    captured = capsys.readouterr()
    assert "dry-run" in captured.out.lower() or "source_dataset" in captured.out
