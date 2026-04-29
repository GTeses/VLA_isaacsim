cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/run_dualarm_idle_scene.py --enable_cameras --start_paused

python demos/run_dualarm_tabletop.py --enable_cameras --motion-scale 1.0

cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python -u scripts/collect_closing_in_data.py \
--enable_cameras \
--num_episodes 5 \
--max_attempts 20 \
--max_steps 80 \
--warmup_steps 5 \
--dataset_file /root/gpufree-data/arcus/zhishu_dualarm_lab/data/hdf5/zhishu_closing_in_gui_debug.hdf5

cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/run_random_dualarm_ik_targets.py \
--enable_cameras \
--num_rounds 5 \
--max_steps_per_round 180 \
--max_task_step 0.02 \
--success_threshold 0.04

cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/collect_random_dualarm_ik_lerobot.py \
--enable_cameras \
--repo_id local/zhishu_dualarm_random_ik \
--num_episodes 20 \
--max_steps_per_episode 120 \
--fps 10 \
--force_overwrite


cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python -u scripts/collect_random_dualarm_ik_lerobot.py \
--enable_cameras \
--repo_id local/zhishu_dualarm_random_ik \
--num_episodes 5 \
--force_overwrite \
--save_failed \
2>&1 | tee /root/gpufree-data/arcus/zhishu_dualarm_lab/tmp/collect_random_dualarm_ik_lerobot.log

cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python -u scripts/collect_random_dualarm_ik_lerobot.py \
--enable_cameras \
--repo_id local/zhishu_dualarm_random_ik \
--num_episodes 5 \
--action_ramp_steps 20 \
--max_action_magnitude 0.15 \
--force_overwrite
2>&1 | tee /root/gpufree-data/arcus/zhishu_dualarm_lab/tmp/collect_random_dualarm_ik_lerobot.log

cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/run_dualarm_link7_space_targets.py \
    --enable_cameras \
    --num_rounds 10 \
    --max_steps_per_round 150 \
    --max_task_step 0.06 \
    --success_threshold 0.04 \
    --hold_steps 3

cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/run_robot_only_scene.py --enable_cameras

python3 scripts/run_robot_only_random_targets.py --headless \
    --target_min_move 0.04 \
    --target_max_move 0.12 \
    --success_threshold 0.02

python -u scripts/run_robot_only_closing_in_v2.py --enable_cameras