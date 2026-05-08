"""
replay_runner
=============
Replay recorded teleoperation episodes through an Isaac Lab environment with
domain-randomization applied on each reset, producing synthetic ``Episode``
objects ready for ``parquet_writer``.

Design notes
------------
- The function does **not** import Isaac Lab at module load time; the import is
  deferred so that ``lerobot_isaac_synthetic`` can be imported on machines where
  Isaac Lab is not installed.
- ``LeRobotDataset`` is also soft-imported for the same reason.
- DR is applied by calling ``env.reset()`` before each replay trial; Isaac Lab's
  ``EventManager`` applies all registered DR terms (object pose, lighting,
  friction, …) automatically at each reset when ``cfg.events.<term>.enabled=True``.
- Action sequences are replayed **open-loop**: the recorded joint-position targets
  from the source dataset are fed step-by-step; no controller correction is applied.
  This keeps the trajectories physically grounded in human demonstrations while
  exploring the DR distribution.

Usage (no Isaac Lab required to import)
----------------------------------------
>>> from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization
>>> episodes = list(replay_with_randomization(
...     source_dataset_path="/data/real_dataset",
...     n_variants_per_episode=5,
...     task="pick",
...     output_path="datasets/dr_replay/",
... ))
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterator

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A single synthetic episode produced by DR-randomized replay.

    Attributes
    ----------
    episode_index:
        Zero-based index within the synthetic batch (assigned by the caller or
        ``parquet_writer``).
    source_episode_index:
        Index of the original episode in the source ``LeRobotDataset``.
    dr_seed:
        Random seed used for this DR variant (for reproducibility).
    observations:
        List of observation dicts, one per timestep.  Keys mirror the
        ``LeRobotDataset`` column convention:
        ``"observation.state"`` (ndarray, shape [12]),
        ``"observation.images.wrist"`` (ndarray uint8 H×W×3),
        ``"observation.images.overhead"`` (ndarray uint8 H×W×3).
    actions:
        List of action arrays (ndarray, shape [6]) — raw radians, LeRobot
        convention, NOT normalised.
    success:
        Whether the episode reached the task success termination condition.
    metadata:
        Arbitrary key/value store for additional per-episode annotations.
    """

    episode_index: int = 0
    source_episode_index: int = 0
    dr_seed: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def replay_with_randomization(
    source_dataset_path: str | Path,
    n_variants_per_episode: int = 5,
    dr_config: dict[str, Any] | None = None,
    task: str = "pick",
    output_path: str | Path | None = None,
    seed: int = 0,
    # Legacy / compatibility aliases kept for back-compat with old callers
    env_id: str = "Isaac-SO101-PickPlace-v0",
    max_episodes: int | None = None,
    base_seed: int | None = None,
) -> Iterator[Episode]:
    """Replay source episodes through an Isaac Lab DR environment.

    Lazily imports ``lerobot`` and ``lerobot_isaac_env``; raises ``ImportError``
    with an actionable message if either is missing.

    Algorithm
    ---------
    1. Soft-import ``lerobot.common.datasets.lerobot_dataset.LeRobotDataset``
       and load ``source_dataset_path``.
    2. Soft-import ``gymnasium`` and call ``gym.make(env_id)`` (headless).
       The env is registered by the ``lerobot_isaac_env`` package.
    3. Apply ``dr_config`` overrides to ``env.cfg.events.*`` before the first
       reset.  Isaac Lab's ``EventManager`` re-reads cfg values on each
       ``env.reset()`` call.
    4. For each source episode ``ep_idx`` (up to ``max_episodes`` if set),
       for each variant ``v`` in ``range(n_variants_per_episode)``:
       a. ``env.reset(seed=variant_seed)`` — DR is applied automatically.
       b. Step through the recorded action sequence frame-by-frame:
          ``obs, _, done, _, info = env.step(action)``
       c. Collect ``(obs, action)`` pairs into an ``Episode``.
       d. ``episode.success = info.get("episode", {}).get("is_success", False)``.
       e. ``yield episode``.
    5. Close the env after all episodes are processed.

    Parameters
    ----------
    source_dataset_path:
        Path to a ``LeRobotDataset`` directory containing real teleoperated data,
        OR a HuggingFace repo_id string (e.g. ``"lerobot/aloha_mobile_shrimp"``).
    n_variants_per_episode:
        Number of DR-randomized replays to generate per source episode.
    dr_config:
        Dict of DR parameter overrides applied to the env's ``EventManager``.
        ``None`` uses env defaults (DR enabled for all registered terms).
    task:
        Short task name passed through to episode metadata and the env
        (default: ``"pick"``).
    output_path:
        Optional destination path.  Not used by this function directly; callers
        (e.g. the CLI ``main()``) pipe the yielded episodes to ``parquet_writer``.
        Provided here so the function signature is self-documenting.
    seed:
        Base random seed.  Seed for variant ``v`` of episode ``i`` =
        ``seed + i * 1000 + v``.
    env_id:
        Gymnasium ID for the Isaac Lab environment (registered by
        ``lerobot_isaac_env``).  Default: ``"Isaac-SO101-PickPlace-v0"``.
    max_episodes:
        If set, only process the first ``max_episodes`` source episodes.
    base_seed:
        Deprecated alias for ``seed``.  If both are provided, ``seed`` wins.

    Yields
    ------
    Episode
        One ``Episode`` per (source_episode, variant) pair.

    Raises
    ------
    ImportError
        If ``lerobot`` or ``lerobot_isaac_env`` / ``gymnasium`` is not installed.
    """
    # Resolve legacy alias
    effective_seed = seed if base_seed is None else base_seed

    # --- Lazy imports — raise ImportError with actionable messages ----------
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lerobot is required to load source datasets.  "
            "Install it with:  pip install lerobot\n"
            "or activate the workspace pixi env:  pixi shell"
        ) from exc

    try:
        import gymnasium as gym  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required to create the Isaac Lab env.  "
            "Install it with:  pip install gymnasium"
        ) from exc

    try:
        import lerobot_isaac_env  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lerobot_isaac_env is required to register Isaac Lab environments.  "
            "Follow the Isaac Lab + lerobot_isaac_env installation guide in "
            "packages/lerobot-isaac-env/README.md"
        ) from exc

    # All imports succeeded — run the actual replay loop.
    source_dataset_path = Path(source_dataset_path)
    logger.info("Loading source dataset from %s", source_dataset_path)
    dataset = LeRobotDataset(str(source_dataset_path))

    n_source = len(dataset.episode_data_index["from"])
    if max_episodes is not None:
        n_source = min(n_source, max_episodes)

    logger.info("Creating env %s (headless)", env_id)
    env = gym.make(env_id, headless=True)

    # Apply DR config overrides before first reset
    if dr_config:
        _apply_dr_config(env, dr_config)

    episode_counter = 0
    try:
        for ep_idx in range(n_source):
            # Extract action sequence for this source episode
            ep_from = dataset.episode_data_index["from"][ep_idx].item()
            ep_to = dataset.episode_data_index["to"][ep_idx].item()
            actions_seq = [
                dataset[frame_idx]["action"] for frame_idx in range(ep_from, ep_to)
            ]

            for variant in range(n_variants_per_episode):
                variant_seed = effective_seed + ep_idx * 1000 + variant
                obs_list, action_list = [], []

                obs, _info = env.reset(seed=variant_seed)

                done = False
                step_idx = 0
                while step_idx < len(actions_seq):
                    action = actions_seq[step_idx]
                    obs, _reward, terminated, truncated, info = env.step(action)
                    obs_list.append(dict(obs) if not isinstance(obs, dict) else obs)
                    action_list.append(action)
                    done = terminated or truncated
                    step_idx += 1
                    if done:
                        break

                success = bool(info.get("episode", {}).get("is_success", False))
                yield Episode(
                    episode_index=episode_counter,
                    source_episode_index=ep_idx,
                    dr_seed=variant_seed,
                    observations=obs_list,
                    actions=action_list,
                    success=success,
                    metadata={"task": task, "env_id": env_id},
                )
                episode_counter += 1
    finally:
        env.close()


def _apply_dr_config(env: Any, dr_config: dict[str, Any]) -> None:
    """Apply DR parameter overrides to ``env.cfg.events``.

    Isaac Lab's ``EventManager`` reads configuration values from
    ``env.cfg.events.<term>.*`` on every ``env.reset()`` call.  This helper
    patches the cfg attrs in-place before the first reset so all subsequent
    resets use the overrides.

    Parameters
    ----------
    env:
        A Gymnasium-wrapped Isaac Lab environment that exposes ``env.cfg``.
    dr_config:
        Dict mapping DR parameter names to values.  Supported keys:
        - ``object_pose_noise_m`` (float)   — positional noise in metres
        - ``lighting_variant`` (bool)       — enable lighting randomisation
        - ``table_friction_range`` (tuple)  — (min, max) friction coefficients
        - ``camera_fov_jitter_deg`` (float) — FOV jitter in degrees
    """
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        logger.warning(
            "env.cfg not found — DR config overrides not applied.  "
            "Ensure the env exposes env.cfg (Isaac Lab standard)."
        )
        return

    events = getattr(cfg, "events", None)
    if events is None:
        logger.warning("env.cfg.events not found — skipping DR config overrides.")
        return

    _PARAM_MAP = {
        "object_pose_noise_m": ("object_pose", "pose_range", "x"),
        "lighting_variant": ("lighting", "enabled"),
        "camera_fov_jitter_deg": ("camera_fov", "jitter_deg"),
    }

    for key, value in dr_config.items():
        if key == "table_friction_range":
            term = getattr(events, "table_friction", None)
            if term is not None:
                term.friction_range = value
            continue
        mapping = _PARAM_MAP.get(key)
        if mapping is None:
            logger.debug("Unknown DR config key %r — ignored.", key)
            continue
        # Navigate the attribute chain
        target = events
        for attr in mapping[:-1]:
            target = getattr(target, attr, None)
            if target is None:
                break
        if target is not None:
            try:
                setattr(target, mapping[-1], value)
            except AttributeError:
                logger.debug(
                    "Could not set DR param %r on env.cfg.events — skipped.", key
                )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_runner",
        description=(
            "Replay teleoperated episodes through Isaac Lab with domain "
            "randomization to produce synthetic LeRobot episodes."
        ),
    )
    p.add_argument(
        "--source_dataset",
        dest="source_dataset",
        required=True,
        help="Path to a real LeRobotDataset directory, or a HuggingFace repo_id.",
    )
    p.add_argument(
        "--n_variants",
        type=int,
        default=5,
        metavar="N",
        help="DR variants per source episode (default: 5).",
    )
    p.add_argument(
        "--task",
        default="pick",
        help="Task name stored in episode metadata (default: pick).",
    )
    p.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help=(
            "Destination path for the synthetic LeRobotDataset.  "
            "Defaults to datasets/dr_replay_<timestamp>/."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for DR variants (default: 0).",
    )
    p.add_argument(
        "--env_id",
        default="Isaac-SO101-PickPlace-v0",
        help="Gymnasium env ID (registered by lerobot_isaac_env).",
    )
    p.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        metavar="M",
        help="Limit to first M source episodes (default: all).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved parameters without running replay.",
    )
    # Keep legacy flag for backward compat with existing tests
    p.add_argument(
        "--source_dataset_path",
        dest="source_dataset_path_legacy",
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--base_seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return p


def _resolve_output_path(output_path: Path | None) -> Path:
    """Return resolved output path, generating a timestamped default if needed."""
    if output_path is not None:
        return output_path
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"datasets/dr_replay_{ts}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve legacy --source_dataset_path alias
    source = args.source_dataset or args.source_dataset_path_legacy
    if source is None:
        parser.error("--source_dataset is required")

    effective_seed = args.base_seed if args.base_seed is not None else args.seed
    resolved_output = _resolve_output_path(args.output_path)

    if args.dry_run:
        print("replay_runner dry-run — resolved parameters:")
        print(f"  source_dataset      : {source}")
        print(f"  output_path         : {resolved_output}")
        print(f"  n_variants          : {args.n_variants}")
        print(f"  task                : {args.task}")
        print(f"  env_id              : {args.env_id}")
        print(f"  max_episodes        : {args.max_episodes}")
        print(f"  seed                : {effective_seed}")
        return

    from lerobot_isaac_synthetic.isaac_dr.parquet_writer import (
        write_episodes_to_lerobot_dataset,
    )

    episodes = replay_with_randomization(
        source_dataset_path=source,
        n_variants_per_episode=args.n_variants,
        task=args.task,
        seed=effective_seed,
        env_id=args.env_id,
        max_episodes=args.max_episodes,
    )
    write_episodes_to_lerobot_dataset(
        episodes=episodes,
        output_path=resolved_output,
    )


if __name__ == "__main__":
    main()
