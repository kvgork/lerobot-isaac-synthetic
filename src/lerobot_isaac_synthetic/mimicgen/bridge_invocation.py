"""
bridge_invocation
=================
Thin wrapper that invokes the ``lerobot_mimicgen_bridge`` skill to run
MimicGen-based data augmentation.

Status (2026-05-21): **un-stubbed**. Bundle D of the deferred-bundle plan
landed; this module now delegates to the canonical skill operations rather
than raising ``NotImplementedError``.

Priority path
-------------
The **Isaac Lab DR replay** pipeline is still the recommended default for
synthetic data generation:

    from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization
    from lerobot_isaac_synthetic.isaac_dr.parquet_writer import write_episodes_to_lerobot_dataset

    episodes = replay_with_randomization(source_dataset_path=..., n_variants_per_episode=5)
    write_episodes_to_lerobot_dataset(episodes, output_path=...)

Use the MimicGen path when:
- The Isaac Lab DR path is unavailable.
- MuJoCo / robosuite-based augmentation is explicitly required.
- A specific MimicGen task config exists for the workload.

Activation gate
---------------
The functions remain gated behind ``LEROBOT_MIMICGEN_ENABLED=1`` or
``enabled=True`` to prevent accidental activation in environments that do not
have MimicGen / robosuite installed. The gate **also** verifies the skill
operations module can be imported; if not, a clear ImportError is raised.

Skill reference:
  ${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md
  Functions: parquet_to_mimicgen, mimicgen_to_parquet, run_mimicgen,
             merge_datasets, validate_conversion
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENABLED_ENV_VAR = "LEROBOT_MIMICGEN_ENABLED"
_SKILL_MODULE = "lerobot_mimicgen_bridge.operations"
_SKILL_PATH = os.path.expandvars(
    "${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/SKILL.md"
)
_AGENT_PATH = os.path.expandvars(
    "${CLAUDE_CODE_ROOT}/agents/workers/lerobot-sim-augmentation-agent.md"
)
_DR_MODULE = "lerobot_isaac_synthetic.isaac_dr.replay_runner"


def _check_enabled() -> bool:
    return os.environ.get(_ENABLED_ENV_VAR, "0").strip() in ("1", "true", "yes")


def _require_active(enabled: bool) -> None:
    if not (enabled or _check_enabled()):
        raise NotImplementedError(
            "MimicGen bridge path requires explicit activation.\n"
            f"  Set {_ENABLED_ENV_VAR}=1 or pass enabled=True.\n"
            f"  Priority alternative: {_DR_MODULE}.replay_with_randomization\n"
            f"  Skill spec: {_SKILL_PATH}\n"
            f"  Agent spec: {_AGENT_PATH}"
        )


def _load_skill_ops():
    try:
        return importlib.import_module(_SKILL_MODULE)
    except ImportError as e:
        raise ImportError(
            f"Cannot import {_SKILL_MODULE}. The MimicGen bridge skill must be "
            f"installed in this env.\n"
            f"Install via: pip install -e ${{CLAUDE_CODE_ROOT}}/skills/lerobot_mimicgen_bridge\n"
            f"Skill spec: {_SKILL_PATH}"
        ) from e


def _unwrap(result: Any, expected_path: Path) -> Path:
    """Skill ops return OperationResult; surface as Path or raise on failure."""
    if hasattr(result, "success") and not result.success:
        raise RuntimeError(
            f"Skill operation failed: {getattr(result, 'message', 'unknown error')}"
        )
    return expected_path


def run_mimicgen(
    real_dataset_path: str | Path,
    n_synthetic_demos: int,
    task_config: str | dict[str, Any],
    output_path: str | Path,
    enabled: bool = False,
    n_source_demos: int = 10,
    fps: int = 30,
    camera_name: str = "agentview",
) -> Path:
    """End-to-end MimicGen augmentation pipeline.

    Steps:
    1. Convert real LeRobotDataset Parquet → MimicGen HDF5 (skill.parquet_to_mimicgen)
    2. Run MimicGen subprocess to generate `n_synthetic_demos` (skill.run_mimicgen)
    3. Convert MimicGen output HDF5 → LeRobotDataset Parquet (skill.mimicgen_to_parquet)
    4. Return output_path

    Parameters
    ----------
    real_dataset_path : path to a real LeRobotDataset directory (Parquet + MP4).
    n_synthetic_demos : count of demos to generate.
    task_config : task name (str) or dict with MimicGen task definition overrides.
    output_path : destination directory for the augmented LeRobotDataset.
    enabled : explicit activation flag (overrides env var check).
    n_source_demos : how many source demos to seed MimicGen with.
    fps : output dataset frame rate.
    camera_name : MimicGen camera key to extract.

    Returns
    -------
    Path to the created synthetic LeRobotDataset directory.
    """
    _require_active(enabled)
    ops = _load_skill_ops()

    real_path = Path(real_dataset_path).resolve()
    out_path = Path(output_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Work-area for intermediate HDF5 files
    work = out_path / ".mimicgen-work"
    work.mkdir(exist_ok=True)
    source_hdf5 = work / "source.hdf5"
    augmented_hdf5 = work / "augmented.hdf5"

    task_name = task_config if isinstance(task_config, str) else task_config.get("name", "pick_and_place")

    logger.info("Step 1/3: parquet → mimicgen hdf5 (source)")
    r = ops.parquet_to_mimicgen(
        dataset_path=str(real_path),
        output_hdf5=str(source_hdf5),
        n_source_demos=n_source_demos,
    )
    _unwrap(r, source_hdf5)

    logger.info("Step 2/3: run mimicgen subprocess (%d demos)", n_synthetic_demos)
    r = ops.run_mimicgen(
        source_hdf5=str(source_hdf5),
        output_hdf5=str(augmented_hdf5),
        n_demos=n_synthetic_demos,
        task_config=task_config if isinstance(task_config, str) else None,
    )
    _unwrap(r, augmented_hdf5)

    logger.info("Step 3/3: mimicgen hdf5 → parquet (augmented dataset)")
    r = ops.mimicgen_to_parquet(
        input_hdf5=str(augmented_hdf5),
        output_path=str(out_path),
        task_name=task_name,
        fps=fps,
        camera_name=camera_name,
    )
    _unwrap(r, out_path)

    logger.info("MimicGen pipeline complete: %s", out_path)
    return out_path


def convert_real_to_mimicgen_hdf5(
    real_dataset_path: str | Path,
    output_hdf5_path: str | Path,
    enabled: bool = False,
    n_source_demos: int = 10,
) -> Path:
    """Convert a real LeRobotDataset (Parquet + MP4) to MimicGen HDF5 format.

    Thin delegation to ``lerobot_mimicgen_bridge.operations.parquet_to_mimicgen``.
    """
    _require_active(enabled)
    ops = _load_skill_ops()

    out = Path(output_hdf5_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    r = ops.parquet_to_mimicgen(
        dataset_path=str(Path(real_dataset_path).resolve()),
        output_hdf5=str(out),
        n_source_demos=n_source_demos,
    )
    return _unwrap(r, out)


def convert_mimicgen_hdf5_to_lerobot(
    hdf5_path: str | Path,
    output_dataset_path: str | Path,
    source_tag: str = "mimicgen",
    enabled: bool = False,
    task_name: str = "pick_and_place",
    fps: int = 30,
    camera_name: str = "agentview",
) -> Path:
    """Convert MimicGen output HDF5 back to LeRobotDataset Parquet format.

    Thin delegation to ``lerobot_mimicgen_bridge.operations.mimicgen_to_parquet``.
    The ``source_tag`` is preserved for caller-side bookkeeping; the skill itself
    tags rows via its own ``task_name`` mechanism.
    """
    _require_active(enabled)
    ops = _load_skill_ops()

    out = Path(output_dataset_path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    r = ops.mimicgen_to_parquet(
        input_hdf5=str(Path(hdf5_path).resolve()),
        output_path=str(out),
        task_name=task_name,
        fps=fps,
        camera_name=camera_name,
    )
    logger.debug("source_tag=%s applied at caller layer", source_tag)
    return _unwrap(r, out)
