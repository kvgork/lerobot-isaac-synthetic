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


def _to_hwc_uint8(cam: Any) -> Any:
    """Convert an Isaac camera obs tensor → HWC uint8 numpy.

    Output is channel-LAST (H, W, C) to match the canonical SO-101 schema the
    ``robot_data_recorder`` emits (``observation.images.<cam>`` declared
    ``(480, 640, 3)`` HWC — the layout lerobot's ``dtype: image`` stores).
    Accepts (N,C,H,W) / (C,H,W) / (N,H,W,C) / (H,W,C), float [0,1] or uint8;
    drops the batch dim, transposes CHW→HWC, and scales floats to uint8.
    """
    import numpy as np
    import torch

    t = cam.detach().cpu() if isinstance(cam, torch.Tensor) else torch.as_tensor(cam)
    if t.ndim == 4:  # drop batch dim: (N,C,H,W)/(N,H,W,C) → 3-D
        t = t[0]
    arr = t.numpy()
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        # Channel-first (C,H,W) → channel-last (H,W,C)
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_state_vec(jp: Any, jv: Any = None) -> Any:
    """Convert joint_pos (+ optional joint_vel) obs tensors → (12,) float32.

    The canonical SO-101 ``observation.state`` is 12-dim =
    joint_pos[6] + joint_vel[6], matching ``robot_data_recorder`` output and the
    trainer / world-model bridge expectation. The Isaac policy obs group exposes
    joint_pos only (joint_vel is privileged / usually absent), so velocity is
    zero-filled when not provided — exactly matching the real follower arm, which
    also reports zero velocity. Keeping the 12-dim layout makes synthetic rows
    merge-compatible with real recordings.
    """
    import numpy as np
    import torch

    def _vec6(x: Any) -> Any:
        t = x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if t.ndim == 2:  # (N,6) → (6,)
            t = t[0]
        return t.numpy().astype(np.float32)

    pos = _vec6(jp)
    vel = _vec6(jv) if jv is not None else np.zeros_like(pos)
    return np.concatenate([pos, vel]).astype(np.float32)


def _find_image_obs(pol: dict[str, Any]) -> Any:
    """Return the first obs value that looks like an image tensor (rank ≥ 3).

    Lets the synthetic adapter locate the env camera term regardless of its name
    (e.g. ``d435_rgb``) so it can be re-exported under the dataset's canonical
    camera column. Non-image terms (joint_pos, joint_vel, last_action) are 1-2D
    and skipped. Returns ``None`` if no image-like term is present.
    """
    import numpy as np

    for value in pol.values():
        if value is None:
            continue
        try:
            arr = (
                value.detach().cpu().numpy()
                if hasattr(value, "detach")
                else np.asarray(value)
            )
        except Exception:
            continue
        if arr.ndim >= 3:
            return value
    return None


def _to_action_vec(action: Any) -> Any:
    """Convert a recorded action → (6,) float32 numpy (drop batch dim)."""
    import numpy as np
    import torch

    t = (
        action.detach().cpu()
        if isinstance(action, torch.Tensor)
        else torch.as_tensor(np.asarray(action))
    )
    if t.ndim == 2:
        t = t[0]
    return t.numpy().astype(np.float32)


def _env_obs_to_lerobot_row(obs: Any, camera_key: str | None) -> dict[str, Any]:
    """Flatten one Isaac env obs dict → a canonical SO-101 LeRobot frame row.

    Produces the schema ``robot_data_recorder`` emits, so real + synthetic data
    share one feature contract and can be merged / trained together:
      - image → ``observation.images.<camera_key>`` (HWC uint8). ``camera_key``
        is the *output* dataset column (default ``overhead``). The env's source
        camera term may be named differently (e.g. ``d435_rgb``); it is
        auto-detected and re-exported under ``camera_key``, decoupling the
        dataset column name from the env obs-term name.
      - state → ``observation.state`` (12,) float32 = joint_pos[6]+joint_vel[6]
        (velocity zero-filled when the policy obs group omits it).
    ``camera_key=None`` yields state-only rows.
    """
    pol = obs.get("policy", obs) if isinstance(obs, dict) else obs
    row: dict[str, Any] = {}
    if isinstance(pol, dict):
        if pol.get("joint_pos") is not None:
            row["observation.state"] = _to_state_vec(
                pol["joint_pos"], pol.get("joint_vel")
            )
        if camera_key:
            img = pol.get(camera_key)
            if img is None:
                img = _find_image_obs(pol)
            if img is not None:
                row[f"observation.images.{camera_key}"] = _to_hwc_uint8(img)
    return row

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
    camera_key: str | None = None,
) -> Iterator[Episode]:
    """Replay source episodes through an Isaac Lab DR environment.

    Lazily imports ``lerobot`` and ``lerobot_isaac_env``; raises ``ImportError``
    with an actionable message if either is missing.

    Algorithm
    ---------
    1. Soft-import ``lerobot.datasets.lerobot_dataset.LeRobotDataset``
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
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lerobot is required to load source datasets.  "
            "Install it with:  pip install lerobot\n"
            "or activate the workspace pixi env:  pixi shell"
        ) from exc

    # Boot Isaac Sim FIRST. Importing isaaclab.envs before the app is alive
    # triggers `ModuleNotFoundError: No module named 'omni'`. Order matters.
    #
    # Use isaaclab's AppLauncher (NOT the raw isaacsim.SimulationApp): only
    # AppLauncher sets the internal carb flag that isaaclab's Camera spawn
    # checks. A raw SimulationApp({"enable_cameras": True}) still raises
    # "A camera was spawned without the --enable_cameras flag" at env reset.
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise ImportError(
            "isaaclab is required for Isaac Lab DR replay.  "
            "Run `bash scripts/install_isaac_lab.sh` inside the sim pixi env."
        ) from exc

    # enable_cameras MUST be set at app launch to capture d435_rgb frames.
    _enable_cameras = camera_key is not None
    # AppLauncher inspects sys.argv; strip leftover CLI flags so Kit doesn't
    # choke on them.
    import sys as _sys

    _saved_argv = _sys.argv
    _sys.argv = _sys.argv[:1]
    try:
        _sim_app = AppLauncher(  # noqa: F841 — keep alive
            headless=True, enable_cameras=_enable_cameras
        ).app
    finally:
        _sys.argv = _saved_argv

    # Configure the Isaac Sim asset CDN BEFORE any isaaclab.* import. Isaac
    # Lab evaluates `NUCLEUS_ASSET_ROOT_DIR` at module load time, so if we
    # only set this after `import isaaclab.envs` the constant is already
    # frozen as None and downstream USD lookups fail with
    # `Unable to open the usd file at path: None/Isaac/...`.
    # The path used here is NVIDIA's public S3 mirror; override with the env
    # var `ISAAC_ASSET_ROOT_CLOUD` if you have a local Nucleus server.
    import os as _os
    import carb  # available after SimulationApp init
    _asset_root = _os.environ.get(
        "ISAAC_ASSET_ROOT_CLOUD",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5",
    )
    carb.settings.get_settings().set(
        "/persistent/isaac/asset_root/cloud", _asset_root
    )

    try:
        import gymnasium as gym  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required to create the Isaac Lab env.  "
            "Install it with:  pip install gymnasium"
        ) from exc

    try:
        import lerobot_isaac_env  # noqa: F401
        # Env registration happens lazily after SimulationApp init, since
        # `gym.register(..., entry_point="isaaclab.envs:ManagerBasedRLEnv")`
        # depends on the Kit extension framework being alive.
        from lerobot_isaac_env.tasks import _register_envs as _reg
        _reg()
    except ImportError as exc:
        raise ImportError(
            "lerobot_isaac_env is required to register Isaac Lab environments.  "
            "Follow the Isaac Lab + lerobot_isaac_env installation guide in "
            "packages/lerobot-isaac-env/README.md"
        ) from exc

    # All imports succeeded — run the actual replay loop.
    source_dataset_path = Path(source_dataset_path)
    logger.info("Loading source dataset from %s", source_dataset_path)
    # lerobot 0.5+ requires `repo_id` + `root` separately and treats a positional
    # path as a repo_id (which it tries to hub-resolve, producing
    # HFValidationError). For a local LeRobotDataset, derive a repo_id from the
    # last two path components.
    if source_dataset_path.is_dir():
        parts = source_dataset_path.resolve().parts
        repo_id = (
            "/".join(parts[-2:]) if len(parts) >= 2 else source_dataset_path.name
        )
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(source_dataset_path),
            video_backend="pyav",
        )
    else:
        dataset = LeRobotDataset(repo_id=str(source_dataset_path))

    # lerobot 0.5 dropped `dataset.episode_data_index`. The per-episode
    # (from, to) bounds now live in `dataset.meta.episodes` (a HF Dataset)
    # under columns `dataset_from_index` and `dataset_to_index`.
    ep_meta = dataset.meta.episodes
    if hasattr(ep_meta, "to_pandas"):
        ep_meta = ep_meta.to_pandas()
    n_source = len(ep_meta)
    if max_episodes is not None:
        n_source = min(n_source, max_episodes)

    logger.info("Creating env %s (headless, cameras=%s)", env_id, _enable_cameras)
    if _enable_cameras:
        # Build a cameras-enabled cfg so the env wires the d435_rgb obs term —
        # enabling cameras on the app alone is NOT enough (see
        # so101_env_cfg._wire_cameras). Map env_id -> task cfg class.
        import lerobot_isaac_env.tasks as _tasks

        _CFG_BY_ENV = {
            "Isaac-SO101-PickPlace-v0": "PickAndPlaceEnvCfg",
            "Isaac-SO101-Pick-v0": "PickEnvCfg",
        }
        _cfg_cls_name = _CFG_BY_ENV.get(env_id, "PickAndPlaceEnvCfg")
        _cfg = getattr(_tasks, _cfg_cls_name)(enable_cameras=True)
        env = gym.make(env_id, cfg=_cfg, headless=True)
    else:
        env = gym.make(env_id, headless=True)

    # Apply DR config overrides before first reset
    if dr_config:
        _apply_dr_config(env, dr_config)

    episode_counter = 0
    try:
        for ep_idx in range(n_source):
            # Extract action sequence for this source episode
            ep_from = int(ep_meta["dataset_from_index"].iloc[ep_idx])
            ep_to = int(ep_meta["dataset_to_index"].iloc[ep_idx])
            actions_seq = [
                dataset[frame_idx]["action"] for frame_idx in range(ep_from, ep_to)
            ]

            for variant in range(n_variants_per_episode):
                variant_seed = effective_seed + ep_idx * 1000 + variant
                obs_list, action_list = [], []

                obs, _info = env.reset(seed=variant_seed)

                done = False
                step_idx = 0
                # Isaac Lab env.step expects (num_envs, action_dim) — our
                # source dataset rows are 1-D (action_dim,). Add a batch dim
                # at step time and use torch tensors on the env device.
                import torch as _torch

                while step_idx < len(actions_seq):
                    raw_action = actions_seq[step_idx]
                    if isinstance(raw_action, _torch.Tensor):
                        action_t = raw_action.detach().clone()
                    else:
                        action_t = _torch.as_tensor(raw_action)
                    if action_t.ndim == 1:
                        action_t = action_t.unsqueeze(0)
                    action_t = action_t.to(
                        getattr(env, "device", _torch.device("cpu")),
                        dtype=_torch.float32,
                    )
                    obs, _reward, terminated, truncated, info = env.step(action_t)
                    # Flatten env obs (nested policy group) → LeRobot row so the
                    # parquet writer emits observation.images.<cam> (PNG, HWC) +
                    # observation.state (12,). camera_key=None → state-only rows.
                    obs_list.append(_env_obs_to_lerobot_row(obs, camera_key))
                    action_list.append(_to_action_vec(raw_action))
                    # Isaac Lab returns (num_envs,)-shaped done flags.
                    done_flag = bool(
                        (terminated[0] if hasattr(terminated, "__getitem__")
                         else terminated)
                        or (truncated[0] if hasattr(truncated, "__getitem__")
                            else truncated)
                    )
                    done = done_flag
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
        "--camera_key",
        default="overhead",
        metavar="KEY",
        help=(
            "Dataset camera column name; the frame is stored as "
            "'observation.images.<KEY>'. The env's source camera term (e.g. "
            "d435_rgb) is auto-detected and re-exported under this name, so it "
            "matches the robot_data_recorder canonical 'overhead' column and "
            "real + synthetic data merge cleanly. Default: %(default)s."
        ),
    )
    p.add_argument(
        "--source_tag",
        default="sim_dr",
        metavar="TAG",
        help=(
            "Value written to the 'source' column of meta/episodes.parquet "
            "so training code can filter synthetic vs real. Default: %(default)s."
        ),
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


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

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
        print(f"  camera_key          : {args.camera_key}")
        print(f"  source_tag          : {args.source_tag}")
        print(f"  obs_image_column    : observation.images.{args.camera_key}")
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
        camera_key=args.camera_key,
    )
    # Emit the canonical 2026-06-06 schema (12-dim observation.state =
    # joint_pos[6]+joint_vel[6], single `<camera_key>` PNG column) by letting
    # the writer derive features from the generated episodes themselves
    # (`features=None`). This deliberately does NOT inherit the source
    # dataset's schema: the source (so101-pickplace1) is the legacy
    # 6-dim/d435_rgb layout, which the migration supersedes so synthetic +
    # robot_data_recorder share one feature contract for merge / co-train.
    write_episodes_to_lerobot_dataset(
        episodes=episodes,
        output_path=resolved_output,
        source_tag=args.source_tag,
        task_name=args.task,
        features=None,
    )


def _load_source_features(source: str | Path) -> dict[str, Any] | None:
    """Load a source LeRobotDataset's feature schema (shapes coerced to tuples).

    lerobot's ``add_frame`` compares array shapes (tuples) to feature shapes;
    JSON loads them as lists, so ``(6,) != [6]`` and every add_frame fails.
    Coerce to tuples here. Returns None for non-local sources (HF repo ids) so
    the writer falls back to deriving features from the first episode.
    """
    import json

    info = Path(source) / "meta" / "info.json"
    if not info.is_file():
        return None
    feats = json.loads(info.read_text()).get("features", {})
    meta_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    out: dict[str, Any] = {}
    for k, v in feats.items():
        if k in meta_keys:
            continue
        v = dict(v)
        if v.get("shape") is not None:
            v["shape"] = tuple(v["shape"])
        out[k] = v
    return out or None


if __name__ == "__main__":
    main()
