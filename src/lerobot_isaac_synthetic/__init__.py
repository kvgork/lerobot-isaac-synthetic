"""
lerobot_isaac_synthetic
=======================
Synthetic data generation package for the LeRobot + Isaac Lab training workspace.

Two paths are provided:

1. **Isaac Lab Domain Randomization (DR) replay** — priority path.
   Load real teleoperated episodes, replay the action sequence through an Isaac Lab
   environment whose domain-randomization parameters are re-sampled each trial, and
   capture the resulting observations as new synthetic episodes.
   Entry-points: ``isaac_dr.replay_runner``, ``isaac_dr.parquet_writer``.

2. **MimicGen bridge** — deferred path.
   A thin stub that delegates to the ``lerobot_mimicgen_bridge`` skill and the
   ``lerobot-sim-augmentation-agent``.  Raises ``NotImplementedError`` until
   explicitly enabled via ``LEROBOT_MIMICGEN_ENABLED=1``.
   Entry-point: ``mimicgen.bridge_invocation``.

Merge utilities for combining real + DR + MimicGen episodes into a single
``LeRobotDataset`` are in ``merge_utilities``.

Quick-start (Isaac Lab DR path)
--------------------------------
>>> from lerobot_isaac_synthetic import (
...     replay_with_randomization,
...     write_episodes_to_lerobot_dataset,
...     merge_datasets,
...     Episode,
... )
>>> episodes = replay_with_randomization(
...     source_dataset_path="/data/real",
...     n_variants_per_episode=5,
... )
>>> write_episodes_to_lerobot_dataset(episodes, output_path="/data/synthetic_dr")

Skill reference:
  ${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md
Agent reference:
  ${CLAUDE_CODE_ROOT}/agents/workers/lerobot-sim-augmentation-agent.md
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

from lerobot_isaac_synthetic.isaac_dr.replay_runner import (
    Episode,
    replay_with_randomization,
)
from lerobot_isaac_synthetic.isaac_dr.parquet_writer import (
    write_episodes_to_lerobot_dataset,
)
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

# MimicGen — deferred path; exported so callers can reference it but it raises
# NotImplementedError unless LEROBOT_MIMICGEN_ENABLED=1.
from lerobot_isaac_synthetic.mimicgen.bridge_invocation import (
    run_mimicgen,
    convert_real_to_mimicgen_hdf5,
    convert_mimicgen_hdf5_to_lerobot,
)

__all__ = [
    # Core DR pipeline
    "Episode",
    "replay_with_randomization",
    "write_episodes_to_lerobot_dataset",
    "merge_datasets",
    # MimicGen deferred path
    "run_mimicgen",
    "convert_real_to_mimicgen_hdf5",
    "convert_mimicgen_hdf5_to_lerobot",
]
