# lerobot-isaac-synthetic — Usage Examples

Examples 1–3 run without any external dependencies. Examples 4–8 require Isaac Lab
and/or lerobot.

---

## Example 1 — Import without external deps

```python
import lerobot_isaac_synthetic
from lerobot_isaac_synthetic.isaac_dr import replay_runner, parquet_writer
from lerobot_isaac_synthetic.mimicgen import bridge_invocation
from lerobot_isaac_synthetic import merge_utilities
print("import ok")
```

Expected output:
```
import ok
```

---

## Example 2 — Inspect Episode dataclass

```python
from lerobot_isaac_synthetic.isaac_dr.replay_runner import Episode

ep = Episode(
    episode_index=0,
    source_episode_index=2,
    dr_seed=1042,
    observations=[{"observation.state": [0.0] * 12}],
    actions=[[0.0] * 6],
    success=True,
    metadata={"task": "pick"},
)
print(ep.episode_index)         # 0
print(ep.dr_seed)               # 1042
print(len(ep.observations))     # 1
print(ep.success)               # True
```

---

## Example 3 — CLI dry-run

```bash
python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
  --source_dataset /data/real \
  --output_path /data/synthetic_dr \
  --n_variants 5 \
  --env_id Isaac-SO101-PickPlace-v0 \
  --dry_run
```

Expected output:
```
replay_runner dry-run — resolved parameters:
  source_dataset      : /data/real
  output_path         : /data/synthetic_dr
  n_variants          : 5
  task                : pick
  env_id              : Isaac-SO101-PickPlace-v0
  max_episodes        : None
  seed                : 0
```

---

## Example 4 — Generate 5 DR variants per real episode

Requires: Isaac Lab, lerobot, lerobot-isaac-env.

```python
from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization
from lerobot_isaac_synthetic.isaac_dr.parquet_writer import write_episodes_to_lerobot_dataset

episodes = replay_with_randomization(
    source_dataset_path="/data/real",
    n_variants_per_episode=5,
    dr_config={"object_pose_noise_m": 0.03},
    env_id="Isaac-SO101-PickPlace-v0",
    seed=0,
    max_episodes=10,  # process only first 10 source episodes
)

output = write_episodes_to_lerobot_dataset(
    episodes=episodes,
    output_path="/data/synthetic_dr",
    source_tag="sim_dr",
)
print(f"Wrote synthetic dataset to: {output}")
```

Expected: 50 synthetic episodes (10 source × 5 variants), tagged `source="sim_dr"`.

---

## Example 5 — Custom DR config

```python
from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization

dr_config = {
    "object_pose_noise_m": 0.05,     # 5 cm positional noise
    "lighting_variant": True,         # enable lighting randomisation
    "table_friction_range": (0.3, 0.8),  # friction in [0.3, 0.8]
    "camera_fov_jitter_deg": 5.0,     # ±5° FOV jitter
}

episodes = list(replay_with_randomization(
    source_dataset_path="/data/real",
    n_variants_per_episode=3,
    dr_config=dr_config,
    seed=42,
))
print(f"{len(episodes)} episodes with custom DR")
```

---

## Example 6 — Merge real and synthetic datasets

Requires: lerobot, pandas, pyarrow.

```python
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

merged_path = merge_datasets(
    real_path="/data/real",
    sim_paths=["/data/synthetic_dr"],
    output_path="/data/merged",
    sim_weight=0.5,   # 50% real, 50% synthetic
    dedup=True,       # drop near-duplicate sim episodes
)
print(f"Merged dataset at: {merged_path}")
```

Expected: merged dataset with equal real/sim split, deduplicated, re-indexed.

---

## Example 7 — Inspect MimicGen stub (deferred)

```python
from lerobot_isaac_synthetic.mimicgen.bridge_invocation import run_mimicgen

try:
    run_mimicgen(
        real_dataset_path="/data/real",
        n_synthetic_demos=200,
        task_config="pick_and_place",
        output_path="/data/mimicgen_out",
    )
except NotImplementedError as e:
    print("Expected:", str(e)[:80])
```

Expected output:
```
Expected: MimicGen bridge path is deferred.
PRIORITY ALTERNATIVE — use the Isaac Lab DR replay pipeline:
  from lerobot...
```

---

## Example 8 — Full pipeline: DR replay + merge + write

Requires: Isaac Lab, lerobot, lerobot-isaac-env, pandas, pyarrow.

```python
from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization
from lerobot_isaac_synthetic.isaac_dr.parquet_writer import write_episodes_to_lerobot_dataset
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

# Step 1: generate synthetic data
episodes = replay_with_randomization(
    source_dataset_path="/data/real",
    n_variants_per_episode=10,
    seed=0,
)
write_episodes_to_lerobot_dataset(episodes, "/data/synthetic_dr")

# Step 2: merge real + synthetic (70% real, 30% sim)
merged = merge_datasets(
    real_path="/data/real",
    sim_paths=["/data/synthetic_dr"],
    output_path="/data/merged_final",
    sim_weight=0.3,
)

print(f"Final merged dataset: {merged}")
# Pass /data/merged_final as --dataset to lerobot-isaac-train
```
