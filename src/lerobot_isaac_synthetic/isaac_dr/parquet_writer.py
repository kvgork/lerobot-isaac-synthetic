"""
parquet_writer
==============
Write synthetic ``Episode`` objects produced by ``replay_runner`` into a
``LeRobotDataset``-compatible Parquet directory, tagging every row with
``source="sim_dr"``.

Design notes
------------
- All output uses the same Parquet column schema as real SO-101 teleoperation
  data so that the resulting directory can be opened with
  ``LeRobotDataset.from_pretrained(output_path)`` without extra conversion.
- The ``source`` tag is stored in ``meta/episodes.parquet`` (``source`` column).
  This lets policy training code filter by source using standard Pandas/Arrow
  predicates.
- ``LeRobotDataset`` is soft-imported; the function raises ``ImportError`` with
  a helpful message if lerobot is not installed.

LeRobotDataset v3.0 schema reference
--------------------------------------
``data/chunk-{n:03d}/episode_{i:06d}.parquet`` — per-episode frame table:
  - ``frame_index`` (int64)
  - ``episode_index`` (int64)
  - ``timestamp`` (float64)  — seconds since episode start
  - ``observation.state`` (list<float64>, length 12)
  - ``observation.images.wrist`` (binary)  — JPEG-encoded bytes
  - ``observation.images.overhead`` (binary)  — JPEG-encoded bytes
  - ``action`` (list<float64>, length 6)
  - ``next.done`` (bool)

``meta/episodes.parquet`` columns:
  - ``episode_index``, ``length``, ``tasks_index``, ``source``

``meta/tasks.parquet`` columns:
  - ``tasks_index``, ``task``, ``source``

Usage
-----
>>> from lerobot_isaac_synthetic.isaac_dr.parquet_writer import (
...     write_episodes_to_lerobot_dataset,
... )
>>> write_episodes_to_lerobot_dataset(
...     episodes=iter([ep1, ep2]),
...     output_path="/data/synthetic_dr",
...     source_tag="sim_dr",
... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Standard SO-101 feature definition used when no features dict is provided.
_DEFAULT_SO101_FEATURES: dict[str, Any] = {
    "observation.state": {
        "dtype": "float32",
        "shape": (12,),
        "names": [
            "joint_pos_0",
            "joint_pos_1",
            "joint_pos_2",
            "joint_pos_3",
            "joint_pos_4",
            "joint_pos_5",
            "joint_vel_0",
            "joint_vel_1",
            "joint_vel_2",
            "joint_vel_3",
            "joint_vel_4",
            "joint_vel_5",
        ],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.overhead": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    },
    "next.done": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
}


def _derive_features_from_episode(episode: Any) -> dict[str, Any]:
    """Attempt to derive a features dict from the first episode's shapes.

    Falls back to the standard SO-101 feature dict if shapes cannot be
    determined (e.g. episode has no observations).
    """
    if not episode.observations:
        logger.warning(
            "First episode has no observations; using default SO-101 features."
        )
        return dict(_DEFAULT_SO101_FEATURES)

    first_obs = episode.observations[0]
    features: dict[str, Any] = {}

    for key, value in first_obs.items():
        try:
            import numpy as np

            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                # Image — record as video feature
                h, w, c = arr.shape
                feat_name = key
                features[feat_name] = {
                    "dtype": "video",
                    "shape": (h, w, c),
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": 30,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }
            else:
                features[key] = {
                    "dtype": str(arr.dtype),
                    "shape": tuple(arr.shape),
                }
        except Exception:
            pass  # Skip keys that can't be introspected

    # Always add action from the first recorded action
    if episode.actions:
        try:
            import numpy as np

            action_arr = np.asarray(episode.actions[0])
            features["action"] = {
                "dtype": str(action_arr.dtype),
                "shape": tuple(action_arr.shape),
            }
        except Exception:
            features["action"] = _DEFAULT_SO101_FEATURES["action"]
    else:
        features["action"] = _DEFAULT_SO101_FEATURES["action"]

    features["next.done"] = _DEFAULT_SO101_FEATURES["next.done"]

    return features if features else dict(_DEFAULT_SO101_FEATURES)


def write_episodes_to_lerobot_dataset(
    episodes: Iterable,
    output_path: str | Path,
    source_tag: str = "sim_dr",
    task_name: str = "pick_and_place",
    fps: int = 30,
    features: dict[str, Any] | None = None,
    image_writer_threads: int = 4,
) -> Path:
    """Write synthetic episodes to a LeRobotDataset-compatible Parquet directory.

    Parameters
    ----------
    episodes:
        Iterable of ``Episode`` objects from ``replay_runner``.
    output_path:
        Root directory for the new ``LeRobotDataset``.  Created if absent.
    source_tag:
        Value written into the ``source`` column of ``meta/episodes.parquet``.
        Use ``"sim_dr"`` for DR episodes, ``"real"`` for real teleop.
    task_name:
        Descriptive task string stored in ``meta/tasks.parquet``.
    fps:
        Frame rate of the source dataset (default 30, matching SO-101 cameras).
    features:
        LeRobotDataset features dict.  If ``None``, derived from the first
        episode's observation/action shapes (falls back to standard SO-101
        features if the episode is empty).
    image_writer_threads:
        Number of threads for JPEG-encoding camera frames.

    Returns
    -------
    Path
        Absolute path to the created dataset directory.

    Raises
    ------
    ImportError
        If lerobot is not installed.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "LeRobot required to write Parquet — `pip install lerobot`\n"
            "or activate the workspace pixi env:  pixi shell"
        ) from exc

    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = f"local/{output_path.name}"
    episode_list = list(episodes)  # materialise so we can peek at first element

    if features is None:
        if episode_list:
            features = _derive_features_from_episode(episode_list[0])
        else:
            features = dict(_DEFAULT_SO101_FEATURES)
            logger.warning("No episodes provided; dataset will be empty.")

    logger.info(
        "Creating LeRobotDataset at %s (repo_id=%s, fps=%d)",
        output_path,
        repo_id,
        fps,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_path,
        image_writer_threads=image_writer_threads,
    )

    # dataset.meta.tasks (mapping task_name -> tasks_index, v3.0 API) is read
    # automatically by add_frame() based on the task argument.

    for ep in episode_list:
        n_frames = len(ep.actions)
        for t in range(n_frames):
            obs = ep.observations[t] if t < len(ep.observations) else {}
            action = ep.actions[t]
            done = t == n_frames - 1

            frame: dict[str, Any] = {}
            for key, value in obs.items():
                frame[key] = value
            frame["action"] = action
            frame["next.done"] = [done]
            dataset.add_frame(frame)

        # save_episode commits the buffered frames and writes episode metadata
        dataset.save_episode(task=task_name)

    # Append source tag to meta/episodes.parquet
    _tag_source_column(output_path, source_tag)

    logger.info(
        "Wrote %d episodes (%d total frames) to %s",
        len(episode_list),
        sum(len(ep.actions) for ep in episode_list),
        output_path,
    )
    return output_path


def _tag_source_column(output_path: Path, source_tag: str) -> None:
    """Append or update the ``source`` column in meta/episodes.parquet.

    Called after ``dataset.save_episode()`` completes so we don't interfere
    with the LeRobotDataset write path.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning(
            "pandas not installed — skipping source column tagging in "
            "meta/episodes.parquet.  Install with:  pip install pandas"
        )
        return

    episodes_path = output_path / "meta" / "episodes.parquet"
    if not episodes_path.exists():
        logger.warning(
            "meta/episodes.parquet not found at %s — source tag not written.",
            output_path,
        )
        return

    df = pd.read_parquet(episodes_path)
    df["source"] = source_tag
    df.to_parquet(episodes_path, index=False)

    tasks_path = output_path / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        tasks_df["source"] = source_tag
        tasks_df.to_parquet(tasks_path, index=False)
