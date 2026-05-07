# lerobot-isaac-synthetic — Public API Reference

---

## Module: `lerobot_isaac_synthetic.isaac_dr.replay_runner`

### `Episode` (dataclass)

A single synthetic episode produced by DR-randomized replay.

| Field | Type | Description |
|-------|------|-------------|
| `episode_index` | `int` | Zero-based index within the synthetic batch. |
| `source_episode_index` | `int` | Index in the original source `LeRobotDataset`. |
| `dr_seed` | `int` | Random seed for this DR variant (for reproducibility). |
| `observations` | `list[dict[str, Any]]` | One obs dict per timestep. Keys: `"observation.state"` (ndarray shape [12]), `"observation.images.wrist"` (ndarray uint8 H×W×3), `"observation.images.overhead"` (ndarray uint8 H×W×3). |
| `actions` | `list[Any]` | Action arrays (ndarray shape [6]) per timestep, radians. |
| `success` | `bool` | Whether episode reached task success termination. |
| `metadata` | `dict[str, Any]` | Arbitrary per-episode annotations (e.g. `{"task": "pick", "env_id": "..."}`) |

---

### `replay_with_randomization(...) -> Iterator[Episode]`

Replay source episodes through Isaac Lab with domain randomization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_dataset_path` | `str \| Path` | required | Path to a `LeRobotDataset` directory or HuggingFace repo_id. |
| `n_variants_per_episode` | `int` | `5` | DR-randomized replays per source episode. |
| `dr_config` | `dict \| None` | `None` | DR parameter overrides. See DR config keys below. |
| `task` | `str` | `"pick"` | Task name stored in episode metadata. |
| `output_path` | `str \| Path \| None` | `None` | Destination (not used directly; for documentation). |
| `seed` | `int` | `0` | Base random seed. Variant seed = `seed + ep_idx * 1000 + variant`. |
| `env_id` | `str` | `"Isaac-SO101-PickPlace-v0"` | Gymnasium env ID (registered by `lerobot_isaac_env`). |
| `max_episodes` | `int \| None` | `None` | Limit to first N source episodes. |
| `base_seed` | `int \| None` | `None` | Deprecated alias for `seed`. |

**Yields:** `Episode` objects — one per (source_episode, variant) pair.

**Raises:**
- `ImportError` — if `lerobot`, `gymnasium`, or `lerobot_isaac_env` not installed.

**DR config keys:**

| Key | Type | Effect |
|-----|------|--------|
| `object_pose_noise_m` | `float` | Object positional noise in metres |
| `lighting_variant` | `bool` | Enable lighting randomisation |
| `table_friction_range` | `tuple` | (min, max) friction coefficients |
| `camera_fov_jitter_deg` | `float` | Camera FOV jitter in degrees |

**Algorithm:** Loads source dataset → creates env → applies DR config → for each
episode × variant: `env.reset(seed=variant_seed)` → step action sequence open-loop →
yield `Episode`.

**Example:**
```python
from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization

episodes = list(replay_with_randomization(
    source_dataset_path="/data/real",
    n_variants_per_episode=5,
    dr_config={"object_pose_noise_m": 0.03},
))
print(f"Generated {len(episodes)} episodes")
```

---

### CLI: `python -m lerobot_isaac_synthetic.isaac_dr.replay_runner`

| Flag | Default | Description |
|------|---------|-------------|
| `--source_dataset` | required | Dataset path or repo_id |
| `--n_variants` | `5` | Variants per source episode |
| `--task` | `"pick"` | Task name in metadata |
| `--output_path` | auto-timestamped | Output directory |
| `--seed` | `0` | Base seed |
| `--env_id` | `"Isaac-SO101-PickPlace-v0"` | Gym env ID |
| `--max_episodes` | `None` | Limit source episodes |
| `--dry_run` | `False` | Print params without running |

---

## Module: `lerobot_isaac_synthetic.isaac_dr.parquet_writer`

### `write_episodes_to_lerobot_dataset(episodes, output_path, source_tag, task_name, fps, features, image_writer_threads) -> Path`

Write synthetic episodes to a LeRobotDataset-compatible Parquet directory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `episodes` | `Iterable[Episode]` | required | Episode objects from `replay_with_randomization`. |
| `output_path` | `str \| Path` | required | Root directory; created if absent. |
| `source_tag` | `str` | `"sim_dr"` | Written to `source` column in `meta/episodes.parquet`. |
| `task_name` | `str` | `"pick_and_place"` | Task string in `meta/tasks.parquet`. |
| `fps` | `int` | `30` | Frame rate stored in `meta/info.json`. |
| `features` | `dict \| None` | `None` | LeRobotDataset features dict. Auto-derived if `None`. |
| `image_writer_threads` | `int` | `4` | Threads for JPEG-encoding camera frames. |

**Returns:** `Path` — absolute path to the created dataset directory.

**Raises:** `ImportError` if lerobot not installed.

---

## Module: `lerobot_isaac_synthetic.merge_utilities`

### `merge_datasets(real_path, sim_paths, output_path, sim_weight, dedup, task_name, fps) -> Path`

Merge real and synthetic `LeRobotDataset` directories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `real_path` | `str \| Path` | required | Real teleoperated dataset. |
| `sim_paths` | `list[str \| Path]` | required | Synthetic dataset directories. |
| `output_path` | `str \| Path` | required | Merged dataset destination. |
| `sim_weight` | `float` | `0.5` | Fraction of merged dataset that is synthetic. Must be in (0, 1). |
| `dedup` | `bool` | `True` | Drop sim episodes with same first-frame state hash as real. |
| `task_name` | `str` | `"pick_and_place"` | Task string in merged metadata. |
| `fps` | `int` | `30` | Frame rate for merged `meta/info.json`. |

**Returns:** `Path` — absolute path to merged dataset directory.

**Raises:**
- `ValueError` — `sim_weight` not in (0, 1).
- `ImportError` — pandas, pyarrow, or lerobot not installed.

---

## Module: `lerobot_isaac_synthetic.mimicgen.bridge_invocation`

### `run_mimicgen(real_dataset_path, n_synthetic_demos, task_config, output_path, enabled=False) -> Path`

**Status: deferred stub — see plan §4b.**

**Raises:** `NotImplementedError` unless `enabled=True` or `LEROBOT_MIMICGEN_ENABLED=1`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `real_dataset_path` | `str \| Path` | Real LeRobotDataset directory. |
| `n_synthetic_demos` | `int` | Synthetic demonstrations to generate. |
| `task_config` | `str \| dict` | Task name or MimicGen task definition. |
| `output_path` | `str \| Path` | Destination for synthetic dataset. |
| `enabled` | `bool` | Explicit activation flag (default `False`). |

Priority alternative: use `replay_with_randomization()` from `isaac_dr.replay_runner`.

---

## Cross-Package References

- `replay_with_randomization()` requires `lerobot_isaac_env` gym registration:
  see `../../lerobot-isaac-env/docs/API.md`
- `write_episodes_to_lerobot_dataset()` and `merge_datasets()` produce datasets
  consumed by `../../lerobot-isaac-adapters/docs/API.md` — `record_episodes()`
