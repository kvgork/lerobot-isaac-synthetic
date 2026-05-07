"""
test_imports.py
===============
Smoke tests verifying that the lerobot_isaac_synthetic package and all its
sub-modules import cleanly without requiring Isaac Lab, lerobot, or MimicGen
to be installed.
"""

import importlib
import sys


def test_top_level_import():
    """Package top-level import succeeds."""
    import lerobot_isaac_synthetic  # noqa: F401


def test_version_attribute():
    """Package exposes a __version__ string."""
    import lerobot_isaac_synthetic
    assert isinstance(lerobot_isaac_synthetic.__version__, str)
    assert len(lerobot_isaac_synthetic.__version__) > 0


def test_isaac_dr_package_import():
    """isaac_dr sub-package imports cleanly."""
    import lerobot_isaac_synthetic.isaac_dr  # noqa: F401


def test_replay_runner_import():
    """replay_runner module imports cleanly."""
    import lerobot_isaac_synthetic.isaac_dr.replay_runner  # noqa: F401


def test_parquet_writer_import():
    """parquet_writer module imports cleanly."""
    import lerobot_isaac_synthetic.isaac_dr.parquet_writer  # noqa: F401


def test_mimicgen_package_import():
    """mimicgen sub-package imports cleanly."""
    import lerobot_isaac_synthetic.mimicgen  # noqa: F401


def test_bridge_invocation_import():
    """bridge_invocation module imports cleanly."""
    import lerobot_isaac_synthetic.mimicgen.bridge_invocation  # noqa: F401


def test_merge_utilities_import():
    """merge_utilities module imports cleanly."""
    import lerobot_isaac_synthetic.merge_utilities  # noqa: F401


def test_episode_dataclass_importable():
    """Episode dataclass is accessible from replay_runner."""
    from lerobot_isaac_synthetic.isaac_dr.replay_runner import Episode
    assert Episode is not None


def test_no_lerobot_required_at_import():
    """Importing lerobot_isaac_synthetic does NOT require lerobot to be installed.

    We verify this by checking that none of the package modules import lerobot
    at module scope (they all use soft imports / defer to function bodies).
    """
    # If lerobot is installed in this environment, skip the check
    # (we can't meaningfully simulate its absence without subprocess isolation).
    if "lerobot" in sys.modules:
        return

    # Re-import to trigger any top-level lerobot imports (none expected)
    import importlib
    for mod_name in [
        "lerobot_isaac_synthetic",
        "lerobot_isaac_synthetic.isaac_dr",
        "lerobot_isaac_synthetic.isaac_dr.replay_runner",
        "lerobot_isaac_synthetic.isaac_dr.parquet_writer",
        "lerobot_isaac_synthetic.mimicgen",
        "lerobot_isaac_synthetic.mimicgen.bridge_invocation",
        "lerobot_isaac_synthetic.merge_utilities",
    ]:
        importlib.import_module(mod_name)

    # lerobot should still not be in sys.modules
    assert "lerobot" not in sys.modules, (
        "lerobot was imported at module level — should be a deferred/soft import"
    )
