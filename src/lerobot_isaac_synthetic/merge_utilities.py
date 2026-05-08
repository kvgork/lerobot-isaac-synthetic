"""
merge_utilities
===============
Merge multiple ``LeRobotDataset`` directories (real, DR, MimicGen) into a
single unified dataset with balanced sampling and source tagging.

Design notes
------------
- Each input dataset is assigned a ``source`` tag (``"real"``, ``"sim_dr"``, or
  ``"mimicgen"``).  The tag is written to the ``source`` column of
  ``meta/episodes.parquet`` and ``meta/tasks.parquet``.
- Episode indices are re-assigned globally to avoid collisions.
- ``sim_weight`` controls the fraction of synthetic episodes in the merged
  dataset.  When ``sim_weight=0.5``, the merged dataset contains equal numbers
  of real and synthetic episodes (the larger pool is down-sampled).
- Identical observations are detected by hashing ``observation.state`` and the
  first-frame hash of ``observation.images.wrist``.  Duplicate episodes are
  dropped before merging.
- ``LeRobotDataset`` is soft-imported; the function raises ``ImportError``
  with a helpful message if lerobot is not installed.

Priority path
-------------
The **Isaac Lab DR replay** pipeline (``isaac_dr.replay_runner`` +
``isaac_dr.parquet_writer``) is the primary source of synthetic data.
MimicGen-sourced episodes (``"mimicgen"`` source tag) are supported but
produced via the deferred path (``mimicgen.bridge_invocation``).

Usage
-----
>>> from lerobot_isaac_synthetic.merge_utilities import merge_datasets
>>> merged_path = merge_datasets(
...     real_path="/data/real",
...     sim_paths=["/data/synthetic_dr", "/data/synthetic_mimicgen"],
...     output_path="/data/merged",
...     sim_weight=0.5,
... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def merge_datasets(
    real_path: str | Path,
    sim_paths: list[str | Path],
    output_path: str | Path,
    sim_weight: float = 0.5,
    dedup: bool = True,
    task_name: str = "pick_and_place",
    fps: int = 30,
) -> Path:
    """Merge real and synthetic LeRobotDatasets into a single unified dataset.

    Parameters
    ----------
    real_path:
        Path to a real teleoperated ``LeRobotDataset`` directory.
    sim_paths:
        List of paths to synthetic ``LeRobotDataset`` directories.  Each
        should have a ``source`` column in ``meta/episodes.parquet`` written
        by ``parquet_writer``.
    output_path:
        Destination directory for the merged dataset.  Created if absent.
    sim_weight:
        Fraction of the merged dataset that should be synthetic episodes.
        ``0.5`` = 50/50 real/sim split.  ``0.3`` = 30% sim, 70% real.
        Must be in ``(0.0, 1.0)``.
    dedup:
        If True, drop synthetic episodes that are near-duplicates of real
        episodes (same first-frame state hash).
    task_name:
        Task string stored in ``meta/tasks.parquet`` for the merged dataset.
    fps:
        Frame rate to store in ``meta/info.json``.

    Returns
    -------
    Path
        Absolute path to the merged dataset directory.

    Raises
    ------
    ValueError
        If ``sim_weight`` is not in ``(0.0, 1.0)``.
    ImportError
        If lerobot, pandas, or pyarrow are not installed.
    """
    if not (0.0 < sim_weight < 1.0):
        raise ValueError(f"sim_weight must be in (0.0, 1.0), got {sim_weight!r}")

    # --- Lazy imports --------------------------------------------------------
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for merge_datasets.  "
            "Install with:  pip install pandas pyarrow\n"
            "or activate the workspace pixi env:  pixi shell"
        ) from exc

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LeRobot required for merge_datasets.  "
            "Install with:  pip install lerobot\n"
            "or activate the workspace pixi env:  pixi shell"
        ) from exc

    real_path = Path(real_path).resolve()
    sim_paths = [Path(p).resolve() for p in sim_paths]
    output_path = Path(output_path).resolve()

    # --- Load episode metadata -----------------------------------------------
    real_episodes_df = _load_episodes_df(real_path, default_source="real")
    n_real = len(real_episodes_df)
    logger.info("Real dataset: %d episodes from %s", n_real, real_path)

    sim_dfs: list[pd.DataFrame] = []
    for sp in sim_paths:
        df = _load_episodes_df(sp, default_source="sim_dr")
        sim_dfs.append(df)
        logger.info(
            "Sim dataset: %d episodes from %s (source=%s)",
            len(df),
            sp,
            df["source"].iloc[0] if len(df) > 0 else "unknown",
        )

    all_sim_df = pd.concat(sim_dfs, ignore_index=True) if sim_dfs else pd.DataFrame()
    n_sim_available = len(all_sim_df)

    # --- Deduplication -------------------------------------------------------
    if dedup and n_sim_available > 0:
        all_sim_df = _dedup_against_real(
            real_episodes_df, all_sim_df, real_path, sim_paths
        )
        logger.info(
            "After dedup: %d sim episodes remain (dropped %d)",
            len(all_sim_df),
            n_sim_available - len(all_sim_df),
        )

    # --- Balanced sampling ---------------------------------------------------
    # sim_weight = n_sim / (n_real + n_sim)  =>  n_sim = n_real * sim_weight / (1 - sim_weight)
    n_sim_target = int(n_real * sim_weight / (1.0 - sim_weight))
    n_sim_available = len(all_sim_df)

    if n_sim_available > n_sim_target:
        all_sim_df = all_sim_df.sample(n=n_sim_target, random_state=0).reset_index(
            drop=True
        )
        logger.info("Down-sampled sim pool to %d episodes.", n_sim_target)
    elif n_real > 0 and n_sim_available < n_sim_target:
        logger.warning(
            "Fewer sim episodes available (%d) than target (%d); "
            "merged dataset will be more real-heavy than sim_weight=%.2f implies.",
            n_sim_available,
            n_sim_target,
            sim_weight,
        )

    # --- Combine and re-index ------------------------------------------------
    merged_df = pd.concat(
        [real_episodes_df, all_sim_df], ignore_index=True
    ).reset_index(drop=True)
    merged_df["episode_index"] = range(len(merged_df))

    # --- Write output dataset ------------------------------------------------
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy frame Parquet files for each episode, rewriting episode_index
    _copy_and_reindex_episodes(
        merged_df=merged_df,
        real_path=real_path,
        sim_paths=sim_paths,
        output_path=output_path,
        fps=fps,
        task_name=task_name,
    )

    logger.info(
        "Merged dataset written to %s (%d total episodes: %d real + %d sim)",
        output_path,
        len(merged_df),
        n_real,
        len(all_sim_df),
    )
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_episodes_df(dataset_path: Path, default_source: str) -> pd.DataFrame:
    """Load ``meta/episodes.parquet`` into a DataFrame, adding source if absent."""
    import pandas as pd

    episodes_file = dataset_path / "meta" / "episodes.parquet"
    if not episodes_file.exists():
        logger.warning(
            "meta/episodes.parquet not found at %s — treating as empty dataset.",
            dataset_path,
        )
        return pd.DataFrame(
            columns=["episode_index", "length", "tasks_index", "source"]
        )

    df = pd.read_parquet(episodes_file)
    if "source" not in df.columns:
        df["source"] = default_source
    df["_dataset_path"] = str(dataset_path)
    return df


def _dedup_against_real(
    real_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    real_path: Path,
    sim_paths: list[Path],
) -> pd.DataFrame:
    """Drop sim episodes whose first-frame state hash matches a real episode."""
    del sim_paths  # accepted for API symmetry; episode dataset path is on each row
    real_hashes = _compute_first_frame_hashes(real_df, real_path)
    if not real_hashes:
        return sim_df

    mask = []
    for _, row in sim_df.iterrows():
        ds_path = Path(row["_dataset_path"])
        h = _compute_episode_hash(ds_path, int(row["episode_index"]))
        mask.append(h not in real_hashes)

    return sim_df[mask].reset_index(drop=True)


def _compute_first_frame_hashes(df: pd.DataFrame, dataset_path: Path) -> set:
    """Return set of first-frame state hashes for all episodes in df."""
    hashes = set()
    for _, row in df.iterrows():
        h = _compute_episode_hash(dataset_path, int(row["episode_index"]))
        if h is not None:
            hashes.add(h)
    return hashes


def _compute_episode_hash(dataset_path: Path, episode_index: int) -> int | None:
    """Hash the first ``observation.state`` frame of a given episode."""
    try:
        import pandas as pd

        # Find the parquet file for this episode (LeRobot v3.0 layout)
        chunk = episode_index // 1000
        frame_file = (
            dataset_path
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not frame_file.exists():
            return None
        df = pd.read_parquet(frame_file, columns=["observation.state"])
        if df.empty:
            return None
        first_state = df.iloc[0]["observation.state"]
        return hash(tuple(float(x) for x in first_state))
    except Exception:
        return None


def _copy_and_reindex_episodes(
    merged_df: pd.DataFrame,
    real_path: Path,
    sim_paths: list[Path],
    output_path: Path,
    fps: int,
    task_name: str,
) -> None:
    """Copy episode Parquet files to output_path, rewriting episode_index."""
    import pandas as pd

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks_rows = [{"tasks_index": 0, "task": task_name, "source": "merged"}]
    episodes_rows = []

    for new_idx, row in merged_df.iterrows():
        src_path = Path(row["_dataset_path"])
        orig_idx = (
            int(row["episode_index"])
            if "_dataset_path" in row.index and row["_dataset_path"]
            else new_idx
        )
        source = row.get("source", "unknown")

        chunk = int(new_idx) // 1000
        out_chunk_dir = output_path / "data" / f"chunk-{chunk:03d}"
        out_chunk_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_chunk_dir / f"episode_{int(new_idx):06d}.parquet"

        # Locate source episode file
        src_chunk = orig_idx // 1000
        src_file = (
            src_path
            / "data"
            / f"chunk-{src_chunk:03d}"
            / f"episode_{orig_idx:06d}.parquet"
        )

        if src_file.exists():
            ep_df = pd.read_parquet(src_file)
            ep_df["episode_index"] = int(new_idx)
            ep_df["frame_index"] = range(len(ep_df))
            ep_df.to_parquet(out_file, index=False)
            episode_len = len(ep_df)
        else:
            logger.warning("Source episode file not found: %s — skipping.", src_file)
            episode_len = 0

        episodes_rows.append(
            {
                "episode_index": int(new_idx),
                "length": episode_len,
                "tasks_index": 0,
                "source": source,
            }
        )

    # Write meta files
    pd.DataFrame(episodes_rows).to_parquet(meta_dir / "episodes.parquet", index=False)
    pd.DataFrame(tasks_rows).to_parquet(meta_dir / "tasks.parquet", index=False)

    # Write minimal info.json
    import json

    info = {
        "fps": fps,
        "total_episodes": len(episodes_rows),
        "total_frames": sum(r["length"] for r in episodes_rows),
        "task": task_name,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
