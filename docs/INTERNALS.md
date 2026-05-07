# lerobot-isaac-synthetic — Internals

---

## File Structure Walk-through

```
packages/lerobot-isaac-synthetic/
├── pyproject.toml           — deps: pyarrow>=14, pandas>=2, numpy>=1.24
├── pixi.toml
├── README.md / CLAUDE.md / docs/
├── src/
│   └── lerobot_isaac_synthetic/
│       ├── __init__.py
│       ├── isaac_dr/
│       │   ├── __init__.py
│       │   ├── replay_runner.py    — replay_with_randomization(), Episode, CLI
│       │   └── parquet_writer.py   — write_episodes_to_lerobot_dataset(), _tag_source_column()
│       ├── mimicgen/
│       │   ├── __init__.py
│       │   └── bridge_invocation.py — run_mimicgen() stub, _check_enabled()
│       └── merge_utilities.py      — merge_datasets() + internal helpers
└── tests/
    ├── test_imports.py             — import smoke test
    ├── test_replay_signature.py    — function/dataclass introspection
    └── test_dry_run.py             — CLI --dry_run output
```

---

## Key Data Structures

### `Episode` dataclass

```python
@dataclass
class Episode:
    episode_index: int = 0
    source_episode_index: int = 0
    dr_seed: int = 0
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

`observations` is a list of dicts with LeRobot v3.0 column names as keys.
`actions` is a list of arrays (shape [6], radians).

### `_DEFAULT_SO101_FEATURES` (parquet_writer)

Standard feature definition for the SO-101 column schema. Used as fallback when
`features=None` in `write_episodes_to_lerobot_dataset()` and when the first episode
has no observations to derive shapes from.

### Source column propagation

`parquet_writer._tag_source_column()` appends/updates the `source` column in
`meta/episodes.parquet` and `meta/tasks.parquet` after `LeRobotDataset.save_episode()`
completes. This is done by reading the Parquet file, adding the column, and rewriting it.
This is a post-write patch to avoid interfering with the LeRobot write path.

---

## DR Replay Loop (replay_runner)

```
load LeRobotDataset
create gymnasium env (headless)
apply dr_config overrides to env.cfg.events.*
for ep_idx in range(n_source):
    for variant in range(n_variants_per_episode):
        variant_seed = seed + ep_idx * 1000 + variant
        obs, _ = env.reset(seed=variant_seed)   # DR applied here by EventManager
        for action in actions_seq:
            obs, reward, terminated, truncated, info = env.step(action)
            collect (obs, action)
        yield Episode(...)
env.close()   # always, via try/finally
```

Action replay is **open-loop**: the recorded joint-position targets from the source
dataset are fed step-by-step without any feedback correction. This keeps trajectories
physically grounded in human demonstrations while exploring the DR distribution.

DR is applied automatically by Isaac Lab's `EventManager` at each `env.reset()` call
because the DR event terms are registered with `mode="reset"`.

---

## `_apply_dr_config` (replay_runner)

Maps string keys to attribute chains on `env.cfg.events.*`:

```python
_PARAM_MAP = {
    "object_pose_noise_m": ("object_pose", "pose_range", "x"),
    "lighting_variant":    ("lighting", "enabled"),
    "camera_fov_jitter_deg": ("camera_fov", "jitter_deg"),
}
```

`table_friction_range` is handled separately (sets `term.friction_range`).
Unknown keys are logged at DEBUG level and silently ignored.

---

## Merge Pipeline (merge_utilities)

```
load real meta/episodes.parquet → real_episodes_df
for each sim_path: load meta/episodes.parquet → sim_dfs
concat all sim → all_sim_df
if dedup: hash first-frame observation.state; drop sim matching real hashes
compute n_sim_target = n_real * sim_weight / (1 - sim_weight)
if n_sim_available > n_sim_target: downsample (random_state=0)
concat real + sampled_sim → merged_df
re-index: merged_df["episode_index"] = range(len(merged_df))
copy episode Parquet files, rewriting episode_index column
write meta/episodes.parquet, meta/tasks.parquet, meta/info.json
```

Deduplication uses first-frame `observation.state` hash:
`hash(tuple(float(x) for x in first_state))`.
This is a best-effort near-duplicate filter; collision probability is extremely low
for real robotics data.

---

## Soft-Import Strategy

No Isaac Lab, lerobot, or MimicGen imports at module level anywhere in this package.

| Module | Deferred imports |
|--------|-----------------|
| `replay_runner.py` | `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`, `gymnasium`, `lerobot_isaac_env` |
| `parquet_writer.py` | `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`, `pandas`, `numpy` |
| `merge_utilities.py` | `pandas`, `pyarrow.parquet`, `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` |
| `bridge_invocation.py` | Nothing — stub always raises before any import |

`pyarrow`, `pandas`, `numpy` are declared as hard deps in `pyproject.toml` because they
are lightweight and required for the core Parquet I/O path.

---

## Test Architecture

Three test files, no external deps:

- `test_imports.py` — verifies all sub-modules import cleanly
- `test_replay_signature.py` — inspects `replay_with_randomization` signature using
  `inspect.signature()`; checks `Episode` has required fields; runs `--help` via
  `subprocess.run`
- `test_dry_run.py` — calls `replay_runner.main()` with `["--source_dataset", "x", "--dry_run"]`
  and checks stdout for the "dry-run" header

---

## Known Limitations

1. **MimicGen path is a stub** — all three functions in `bridge_invocation.py` raise
   `NotImplementedError`. The Isaac Lab DR path is the recommended alternative.

2. **Open-loop replay** — the DR replay uses recorded actions without correction.
   Episodes that relied on visual feedback may fail in the DR environment (different
   object pose). Consider using a learned policy for replay once pc_success >= 0.6.

3. **Dedup uses first-frame hash only** — two episodes with the same initial state but
   different trajectories could be incorrectly flagged as duplicates. In practice this
   is very rare for real robotics data.

4. **`_copy_and_reindex_episodes` reads full Parquet files** — for large datasets
   (>10K episodes), this may be slow. Consider batching or using Arrow streaming.

---

## Future Un-stubbing Plan

| Stub | Plan |
|------|------|
| `run_mimicgen()` | Implement following docstring steps; requires MimicGen + robosuite install |
| `convert_real_to_mimicgen_hdf5()` | Delegate to `lerobot_mimicgen_bridge.operations.convert_to_mimicgen` |
| `convert_mimicgen_hdf5_to_lerobot()` | Delegate to `lerobot_mimicgen_bridge.operations.convert_from_mimicgen` |
