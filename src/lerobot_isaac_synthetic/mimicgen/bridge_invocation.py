"""
bridge_invocation
=================
Thin wrapper that invokes the ``lerobot_mimicgen_bridge`` skill to run
MimicGen-based data augmentation.

This module is the **deferred path** described in Phase 4b of the build plan.
All public functions raise ``NotImplementedError`` until the path is activated.

Priority path
-------------
The **Isaac Lab DR replay** pipeline is the recommended way to generate
synthetic data:

    from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization
    from lerobot_isaac_synthetic.isaac_dr.parquet_writer import write_episodes_to_lerobot_dataset

    episodes = replay_with_randomization(source_dataset_path=..., n_variants_per_episode=5)
    write_episodes_to_lerobot_dataset(episodes, output_path=...)

Use the MimicGen path only when the Isaac Lab DR path is unavailable or when
MuJoCo-based augmentation is explicitly required.

Activation
----------
Enable by setting the environment variable ``LEROBOT_MIMICGEN_ENABLED=1``
**and** ensuring MimicGen + robosuite are installed in the current environment.

Recommended invocation (via agent)
------------------------------------
Rather than calling this module directly, invoke the dedicated orchestration
agent, which handles the full pipeline including error recovery and dataset
validation:

    Task(lerobot-sim-augmentation-agent, {
        "real_dataset_path": "/data/real",
        "task_config": "pick_and_place",
        "n_synthetic_demos": 200,
        "output_path": "/data/synthetic_mimicgen"
    })

Skill reference (conversion only — no MimicGen orchestration):
  ${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md

Usage (deferred stub)
---------------------
>>> from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen
>>> run_mimicgen(
...     real_dataset_path="/data/real",
...     n_synthetic_demos=200,
...     task_config="pick_and_place",
...     output_path="/data/synthetic_mimicgen",
... )
NotImplementedError: MimicGen bridge path is deferred.  Use the Isaac Lab DR
  replay pipeline (isaac_dr.replay_runner) as the priority alternative, or
  invoke via `lerobot-sim-augmentation-agent` / skill ...
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENABLED_ENV_VAR = "LEROBOT_MIMICGEN_ENABLED"
_SKILL_PATH = os.path.expandvars(
    "${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md"
)
_AGENT_PATH = os.path.expandvars(
    "${CLAUDE_CODE_ROOT}/agents/workers/lerobot-sim-augmentation-agent.md"
)
_DR_MODULE = "lerobot_isaac_synthetic.isaac_dr.replay_runner"


def _check_enabled() -> bool:
    """Return True if the MimicGen path has been explicitly activated."""
    return os.environ.get(_ENABLED_ENV_VAR, "0").strip() in ("1", "true", "yes")


def run_mimicgen(
    real_dataset_path: str | Path,
    n_synthetic_demos: int,
    task_config: str | dict[str, Any],
    output_path: str | Path,
    enabled: bool = False,
) -> Path:
    """Invoke the MimicGen bridge to augment a real LeRobotDataset.

    This is a **deferred stub**.  The Isaac Lab DR replay pipeline is the
    recommended alternative for synthetic data generation.

    Priority alternative
    --------------------
    Use the now-real Isaac Lab DR pipeline instead:

        from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization
        from lerobot_isaac_synthetic.isaac_dr.parquet_writer import write_episodes_to_lerobot_dataset

        episodes = replay_with_randomization(
            source_dataset_path=real_dataset_path,
            n_variants_per_episode=10,
        )
        write_episodes_to_lerobot_dataset(episodes, output_path=output_path)

    When enabled (``enabled=True`` OR ``LEROBOT_MIMICGEN_ENABLED=1``), the full
    MimicGen implementation would:

    1. Call ``lerobot_mimicgen_bridge.operations.convert_to_mimicgen`` to convert
       the real LeRobotDataset Parquet at ``real_dataset_path`` into MimicGen HDF5
       format.  See ``SKILL_PATH`` for the exact API.
    2. Run MimicGen augmentation (via ``subprocess.run`` or the
       ``lerobot-sim-augmentation-agent``) to generate ``n_synthetic_demos``
       demonstrations.  MimicGen requires robosuite + MuJoCo.
    3. Call ``lerobot_mimicgen_bridge.operations.convert_from_mimicgen`` to
       convert the MimicGen output HDF5 back to LeRobot Parquet format.
    4. Tag all rows with ``source="mimicgen"`` in ``meta/tasks.parquet``.
    5. Return ``Path(output_path)``.

    Parameters
    ----------
    real_dataset_path:
        Path to a real LeRobotDataset directory (Parquet + MP4 format).
    n_synthetic_demos:
        Number of synthetic demonstrations to generate.
    task_config:
        Task name (str) or dict with MimicGen task definition overrides.
    output_path:
        Destination directory for the resulting synthetic LeRobotDataset.
    enabled:
        Explicit activation flag.  If False (default) AND the env var
        ``LEROBOT_MIMICGEN_ENABLED`` is not set, raises ``NotImplementedError``.

    Returns
    -------
    Path
        Absolute path to the created synthetic dataset directory.

    Raises
    ------
    NotImplementedError
        When the path is not enabled.  Points to the Isaac Lab DR pipeline,
        the skill, and the agent that should be used instead.
    ImportError
        (Future) raised if MimicGen / robosuite are not installed.
    """
    if not (enabled or _check_enabled()):
        raise NotImplementedError(
            "MimicGen bridge path is deferred.\n"
            "\n"
            "PRIORITY ALTERNATIVE — use the Isaac Lab DR replay pipeline:\n"
            f"  from {_DR_MODULE} import replay_with_randomization\n"
            "  episodes = replay_with_randomization(source_dataset_path=..., n_variants_per_episode=10)\n"
            "\n"
            f"To enable MimicGen path: set env var {_ENABLED_ENV_VAR}=1 "
            "and install MimicGen + robosuite.\n"
            "\n"
            "Recommended: invoke via the dedicated orchestration agent "
            "(full pipeline with error recovery and dataset validation):\n"
            f"  Task(lerobot-sim-augmentation-agent, {{...}})\n"
            f"  Agent spec: {_AGENT_PATH}\n"
            "\n"
            "For the Parquet <-> MimicGen HDF5 conversion step only:\n"
            f"  Skill spec: {_SKILL_PATH}\n"
        )

    # --- Implementation placeholder (replace when enabling) ---
    raise NotImplementedError(
        "run_mimicgen implementation is not yet complete.  "
        "Follow the docstring steps above to implement."
    )


def convert_real_to_mimicgen_hdf5(
    real_dataset_path: str | Path,
    output_hdf5_path: str | Path,
    enabled: bool = False,
) -> Path:
    """Convert a real LeRobotDataset to MimicGen HDF5 format.

    Thin delegation to the ``lerobot_mimicgen_bridge`` skill.  See skill docs:
    ``${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md``

    Priority alternative: use ``isaac_dr.replay_runner.replay_with_randomization``
    for Isaac Lab DR-based augmentation, which does not require MimicGen or MuJoCo.

    This is a stub — raises ``NotImplementedError`` unless enabled via
    ``LEROBOT_MIMICGEN_ENABLED=1`` or ``enabled=True``.
    """
    if not (enabled or _check_enabled()):
        raise NotImplementedError(
            f"Deferred path — invoke via skill: {_SKILL_PATH}\n"
            f"Priority alternative: {_DR_MODULE}.replay_with_randomization"
        )
    raise NotImplementedError(
        "Delegate to lerobot_mimicgen_bridge.operations.convert_to_mimicgen"
    )


def convert_mimicgen_hdf5_to_lerobot(
    hdf5_path: str | Path,
    output_dataset_path: str | Path,
    source_tag: str = "mimicgen",
    enabled: bool = False,
) -> Path:
    """Convert MimicGen output HDF5 back to LeRobotDataset Parquet format.

    Thin delegation to the ``lerobot_mimicgen_bridge`` skill.  See skill docs:
    ``${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md``

    Priority alternative: use ``isaac_dr.parquet_writer.write_episodes_to_lerobot_dataset``
    for writing episodes produced by the Isaac Lab DR pipeline.

    This is a stub — raises ``NotImplementedError`` unless enabled via
    ``LEROBOT_MIMICGEN_ENABLED=1`` or ``enabled=True``.
    """
    if not (enabled or _check_enabled()):
        raise NotImplementedError(
            f"Deferred path — invoke via skill: {_SKILL_PATH}\n"
            f"Priority alternative: isaac_dr.parquet_writer.write_episodes_to_lerobot_dataset"
        )
    raise NotImplementedError(
        "Delegate to lerobot_mimicgen_bridge.operations.convert_from_mimicgen"
    )
