# LeIsaac 风格数据链在干净聚拢版中的作用

这份文档专门说明：

- `run_robot_only_closing_in.py` 这条“干净聚拢版”控制脚本，如何重新接回 LeIsaac 风格的数据采集链
- 哪些文件在起作用
- 采集数据的每一个环节分别在哪个文件/函数里完成
- 怎么启动、怎么采集、怎么转成 LeRobot

这份说明只针对当前干净聚拢版，不针对旧的 `collect_closing_in_data.py` 复杂分支。

---

## 1. 先说结论

当前这条链路的设计原则是：

- `robot_only` 线负责“动作本身”
- LeIsaac 风格数据链负责“把动作录下来并转成训练格式”
- 两者通过一个很薄的 recorder 接口连接

也就是说：

- IK、reset、目标采样、共同目标体、圆盘可视化
  都在 `run_robot_only_closing_in.py` 里
- HDF5 schema、episode 写盘、LeRobot 转换
  都在单独的工具模块和 converter 里

这就是“先做稳定动作，再接回数据链”的最小侵入版本。

---

## 2. 这条链路涉及哪些文件

### 2.1 任务与控制主入口

- [run_robot_only_closing_in.py](/home/mark/zhishu_dualarm_lab/scripts/run_robot_only_closing_in.py)

作用：

- 创建 `robot_only` 环境
- 采样共同目标体
- 构造左右臂 structured target
- 运行双臂 Differential IK
- 可视化目标点和圆盘
- 在每个 step 后抓取 `obs/action`
- 可选写成 LeIsaac 风格 HDF5

这是当前“干净聚拢版”的唯一控制入口。

### 2.2 HDF5 写盘工具

- [robot_only_closing_in_dataset.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/robot_only_closing_in_dataset.py)

作用：

- 定义任务名 `TASK_NAME`
- 定义 HDF5 根组 `ROOT_GROUP`
- 定义动作命名 `ACTION_NAMES`
- 创建/追加 HDF5
- 写入 episode
- 解析 70 维 state

这个文件不参与控制，只参与写盘。

### 2.3 LeRobot 转换脚本

- [convert_robot_only_closing_in_to_lerobot.py](/home/mark/zhishu_dualarm_lab/scripts/convert_robot_only_closing_in_to_lerobot.py)

作用：

- 读取干净聚拢版 HDF5
- 逐条 episode 转成 LeRobot dataset
- 可选只保留成功 episode
- 可选 push 到 hub

### 2.4 机器人环境本体

- [env.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/robot_only/env.py)
- [env_cfg.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/robot_only/env_cfg.py)
- [scene_cfg.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/robot_only/scene_cfg.py)
- [constants.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/robot_only/constants.py)

作用：

- `robot_only` 环境 reset
- camera 挂载
- observation/state 结构
- action 语义

这里不是 LeIsaac 数据框架的一部分，但 recorder 完全依赖它提供观测和相机画面。

---

## 3. LeIsaac 风格数据链是怎么接进来的

### 3.1 控制层仍然完全留在 主脚本 里

核心原则：

- 不改 `robot_only` reset 主逻辑
- 不改双臂 IK 主逻辑
- 不把 HDF5 采集逻辑塞进 env

也就是说，LeIsaac 风格框架在这里不是“控制框架”，而是：

- episode writer
- dataset organizer
- converter

### 3.2 `主脚本` 在每一步执行后抓一帧

`run_robot_only_closing_in.py` 里新增了：

- `_capture_policy_frame(...)`
- `_augment_policy_state(...)`

它们的职责是：

1. 调 `env.get_policy_input()`
2. 取出：
   - `observation/external_image`
   - `observation/left_wrist_image`
   - `observation/right_wrist_image`
   - `observation/state`
3. 因为 `robot_only` env 本身不认识共同目标体，所以它吐出来的：
   - `object_pose`
   - `target_pose`
   默认是零
4. recorder 层再把：
   - 圆盘中心写回 `object_pose`
   - 共同目标体中心写回 `target_pose`

这样做的好处是：

- 不污染 `robot_only env`
- 但落盘时 state 已经带上了真实任务语义

### 3.3 episode 结束后统一写 HDF5

在 `main()` 的每一轮里，`主脚本` 会缓存：

- `episode_actions`
- `episode_states`
- `episode_external`
- `episode_left_wrist`
- `episode_right_wrist`

一轮结束后，构造：

- `RobotOnlyClosingInEpisodeSpec`
- `payload`

然后调用：

- `write_episode(...)`

写进 HDF5。

这就是 LeIsaac 风格数据链在当前脚本里真正“起作用”的位置。

---

## 4. 采集数据时，各个环节分别是谁在做

### 4.1 环境创建

文件：

- [run_robot_only_closing_in.py](/home/mark/zhishu_dualarm_lab/scripts/run_robot_only_closing_in.py)

函数：

- `main()`

这里做：

- 解析命令行参数
- 启动 Isaac app
- 构造 `ZhishuDualArmRobotOnlyEnv`

### 4.2 reset 与初始姿态

文件：

- [env.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/robot_only/env.py)

函数：

- `_reset_idx(...)`

这里做：

- root pose reset
- joint pos/vel reset
- arm joint noise
- 左右 `joint4` 进入桌上工作分支

这一步和 LeIsaac 数据框架无关，纯控制逻辑。

### 4.3 共同目标体采样

文件：

- [run_robot_only_closing_in.py](/home/mark/zhishu_dualarm_lab/scripts/run_robot_only_closing_in.py)

函数：

- `_sample_closing_in_targets(...)`

这里做：

- 读取左右肩、腰、头 link 位置
- 确定 shoulder line
- 采样共同目标体中心
- 生成：
  - `left_target`
  - `right_target`
  - `center_target`
  - `disk_center`

### 4.4 IK 求解

文件：

- [run_robot_only_closing_in.py](/home/mark/zhishu_dualarm_lab/scripts/run_robot_only_closing_in.py)

类 / 函数：

- `DualArmPositionIK`
- `_solve_arm(...)`
- `infer(...)`

这里做：

- 左右臂显式按 joint name 建真实 7 轴链
- 左右臂分别求 DLS differential IK
- 再按 env 真实 action 顺序 scatter 回 action buffer
- 最后对 `joint5` 做手心朝向软偏置

### 4.5 相机与观测

文件：

- [env.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/robot_only/env.py)

函数：

- `_get_observations()`
- `get_policy_input()`

这里做：

- 同步 external / left wrist / right wrist 相机
- 渲染图像
- 构造固定 70D `state`
- 返回 recorder 可直接取用的 policy input

### 4.6 录一帧

文件：

- [run_robot_only_closing_in.py](/home/mark/zhishu_dualarm_lab/scripts/run_robot_only_closing_in.py)

函数：

- `_capture_policy_frame(...)`
- `_augment_policy_state(...)`

这里做：

- 从 env 拿最新图像和 state
- 给 state 补入：
  - 圆盘 pose
  - 共同目标体中心 pose

### 4.7 写一条 episode

文件：

- [robot_only_closing_in_dataset.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/robot_only_closing_in_dataset.py)

函数：

- `create_or_open_dataset(...)`
- `write_episode(...)`

这里做：

- 创建 `/data`
- 写 attrs
- 写 observation / actions / task / metrics

### 4.8 转成 LeRobot

文件：

- [convert_robot_only_closing_in_to_lerobot.py](/home/mark/zhishu_dualarm_lab/scripts/convert_robot_only_closing_in_to_lerobot.py)

函数：

- `main()`
- `_create_dataset(...)`

这里做：

- 读 HDF5
- 逐帧调用 `dataset.add_frame(...)`
- 每条 episode 调 `dataset.save_episode(...)`

---

## 5. 当前 HDF5 里写了什么

根组：

- `/data`

每条 episode：

- `/data/demo_0`
- `/data/demo_1`
- ...

每条 episode 里主要有：

- `observation/state`
- `observation/external_image`
- `observation/left_wrist_image`
- `observation/right_wrist_image`
- `actions`
- `task/prompt`
- `task/left_target`
- `task/right_target`
- `task/center_target`
- `task/disk_center`
- `task/target_gap_m`
- `metrics/duration_s`
- `metrics/left_done`
- `metrics/right_done`
- `metrics/final_left_dist`
- `metrics/final_right_dist`

同时 attrs 里还会写：

- `task_name`
- `episode_index`
- `prompt`
- `gap_m`
- `success`
- `num_samples`

---

## 6. 能改哪些参数

当前主要参数都在：

- [run_robot_only_closing_in.py](/home/mark/zhishu_dualarm_lab/scripts/run_robot_only_closing_in.py)

### 6.1 动作相关

- `--max_steps_per_round`
- `--settle_steps`
- `--success_threshold`
- `--hold_steps`
- `--max_task_step`
- `--ik_lambda`

### 6.2 共同目标体相关

- `--target_gap_m`
- `--target_gap_jitter_m`
- `--min_separation_from_start`
- `--target_sample_attempts`

### 6.3 数据采集相关

- `--dataset_file`
- `--resume`
- `--task_prompt`

### 6.4 运行规模相关

- `--num_rounds`
- `--seed`
- `--log_every`

---

## 7. 启动命令

### 7.1 只看 GUI，不采集

```bash
source /home/mark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-5.1
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /home/mark/zhishu_dualarm_lab
python -u scripts/run_robot_only_closing_in.py --enable_cameras
```

### 7.2 少量 GUI 调试

```bash
python -u scripts/run_robot_only_closing_in.py \
  --enable_cameras \
  --num_rounds 5 \
  --max_steps_per_round 180
```

---

## 8. 最终采集命令

### 8.1 新建一个 HDF5

```bash
source /home/mark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-5.1
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /home/mark/zhishu_dualarm_lab
python -u scripts/run_robot_only_closing_in.py \
  --enable_cameras \
  --num_rounds 20 \
  --max_steps_per_round 180 \
  --dataset_file /home/mark/zhishu_dualarm_lab/data/hdf5/robot_only_closing_in_v0.hdf5
```

### 8.2 继续往已有 HDF5 追加

```bash
python -u scripts/run_robot_only_closing_in.py \
  --enable_cameras \
  --num_rounds 20 \
  --max_steps_per_round 180 \
  --dataset_file /home/mark/zhishu_dualarm_lab/data/hdf5/robot_only_closing_in_v0.hdf5 \
  --resume
```

---

## 9. 转成 LeRobot 的命令

```bash
source /home/mark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-5.1
cd /home/mark/zhishu_dualarm_lab
python -u scripts/convert_robot_only_closing_in_to_lerobot.py \
  --dataset_file /home/mark/zhishu_dualarm_lab/data/hdf5/robot_only_closing_in_v0.hdf5 \
  --repo_id mark/zhishu-robot-only-closing-in-v0 \
  --fps 10
```

如果只想转成功 episode：

```bash
python -u scripts/convert_robot_only_closing_in_to_lerobot.py \
  --dataset_file /home/mark/zhishu_dualarm_lab/data/hdf5/robot_only_closing_in_v0.hdf5 \
  --repo_id mark/zhishu-robot-only-closing-in-v0 \
  --fps 10 \
  --successful_only
```

---

## 10. LeIsaac 框架在这里到底“起了什么作用”

更准确地说，这里不是把 `主脚本` 变成“运行在 LeIsaac runtime 里”，而是把它接回了 **LeIsaac 风格的数据组织链**：

- 有统一的 episode 结构
- 有统一的 HDF5 根组 `/data`
- 有统一的 `obs/action/task/metrics` 分层
- 有单独的 converter

所以它起作用的方式是：

- 不改你已经稳定的动作控制
- 只把稳定动作包进标准化数据链

这也是为什么这条线比旧的复杂 collector 更稳：

- 控制问题留在控制脚本里解决
- 数据问题留在 writer/converter 里解决
- 两边通过很小的接口连接

---

## 11. 当前还没有做什么

当前这条链路已经有：

- 干净聚拢动作
- GUI 可视化
- HDF5 写盘
- LeRobot 转换

当前还没有额外做：

- 专门的 HDF5 replay 脚本
- 更复杂的 success filtering / curriculum
- 把圆盘做成真正 scene 资产并参与物理交互

这些都可以后续再加，但不影响当前先把 teacher 数据录下来。
