"""
isaac_dr sub-package
====================
Isaac Lab domain-randomization replay pipeline.

Workflow
--------
1. Load a real teleoperated ``LeRobotDataset`` from ``source_dataset_path``.
2. For each episode, replay the recorded action sequence through an Isaac Lab
   ``ManagerBasedRLEnv`` (``lerobot_isaac_env`` package) whose event-manager
   domain-randomization parameters are re-sampled ``n_variants_per_episode`` times.
3. Capture the resulting observations as new ``Episode`` objects.
4. Write the episodes to a new ``LeRobotDataset`` in Parquet format, tagging each
   row with ``source="sim_dr"``.

Public API
----------
``replay_runner.replay_with_randomization``
``parquet_writer.write_episodes_to_lerobot_dataset``
"""
