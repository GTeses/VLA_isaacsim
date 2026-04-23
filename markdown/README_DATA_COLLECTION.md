# Zhishu 数据采集说明

这份文档是给第一次接手 `zhishu_dualarm_lab` 的协作者用的。

目标很简单：

- 知道需要什么环境
- 知道需要哪些本地文件
- 知道怎么启动 Isaac 场景
- 知道怎么运行 `collect_closing_in_data.py`
- 知道当前脚本的实际状态和输出长什么样

这份文档是补充说明，不替代根目录 [README.md](/home/mark/zhishu_dualarm_lab/README.md)。

## 1. 当前数据采集主线

当前正在采的是：

- 任务名：`LeIsaac-Zhishu-ClosingIn-v0`
- 脚本：`scripts/collect_closing_in_data.py`
- 输出格式：小规模 HDF5
- 图像输入：3 路
  - `external_image`
  - `left_wrist_image`
  - `right_wrist_image`
- 当前不采集腰部相机图像

当前 collector 的定位是：

- 在 Isaac 仿真里采一批“无手双臂聚拢”语义数据
- 先保证轨迹和数据格式合理
- 后续再转 LeRobot、做 `compute_norm_stats` 和 fine-tuning

## 2. 必要环境

### 2.1 Isaac 运行环境

必须先使用 `isaaclab-5.1` 环境。

每次运行 Isaac 相关脚本前，先执行这三步：

```bash
source /home/mark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-5.1
which python
python -c "import isaaclab, isaacsim; print('ok')"
```

预期结果：

- `which python` 应该是：
  `/home/mark/miniconda3/envs/isaaclab-5.1/bin/python`
- 最后一条命令应打印：
  `ok`

如果这两步不通过，不要继续跑 Isaac 脚本。

### 2.2 不要默认用 `isaaclab.sh -p`

这台机器当前的主线是：

- 激活 `isaaclab-5.1`
- 直接用环境里的 `python`

不要默认使用：

```bash
/home/mark/IsaacLab/isaaclab.sh -p
```

### 2.3 当前用到的 Python 包

当前数据采集链默认依赖：

- `isaaclab`
- `isaacsim`
- `torch`
- `numpy`
- `h5py`

它们应该都在 `isaaclab-5.1` 环境里。

## 3. 必要仓库与本地文件

### 3.1 仓库目录

默认目录结构按这台机器约定：

```text
/home/mark/
  zhishu_dualarm_lab/
  openpi/
  zhishu_robot_description-URDF/
```

### 3.2 机器人 URDF

当前项目会通过 `source/zhishu_dualarm_lab/utils/local_paths.py` 自动解析机器人 URDF。

默认优先寻找：

```text
/home/mark/zhishu_robot_description-URDF/zhishu_robot_description/urdf/zhishu_robot_description.urdf
```

如果你的机器路径不同，可以二选一：

1. 配环境变量

```bash
export ZHISHU_ROBOT_URDF=/your/path/to/zhishu_robot_description.urdf
```

2. 或在项目里创建：

```text
config/local_paths.json
```

内容示例：

```json
{
  "robot_urdf_path": "/your/path/to/zhishu_robot_description.urdf",
  "openpi_root": "/your/path/to/openpi",
  "openpi_checkpoint_roots": [
    "/your/path/to/openpi/openpi-assets/checkpoints"
  ]
}
```

### 3.3 当前相机约定

当前场景里有 4 路相机：

- head external camera
- waist camera
- left wrist camera
- right wrist camera

但当前数据采集主线只使用 3 路：

- `observation/external_image`
- `observation/left_wrist_image`
- `observation/right_wrist_image`

腰部相机当前不进入数据集。

## 4. 先验证场景能否启动

如果你只是想确认 Isaac 能起来、场景能加载、机器人不会自己乱动，先跑 idle scene：

```bash
source /home/mark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-5.1
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /home/mark/zhishu_dualarm_lab
python -u scripts/run_dualarm_idle_scene.py --enable_cameras
```

这个入口只会：

- 加载当前桌面场景
- 加载 Zhishu 机器人
- step 零动作
- 不自动退出

适合先看：

- 机器人是否成功加载
- 相机是否正常
- 场景资产是否丢失

## 5. 运行数据采集

### 5.1 最小 GUI 版命令

第一次建议用 GUI 看效果：

```bash
source /home/mark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-5.1
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /home/mark/zhishu_dualarm_lab
python -u scripts/collect_closing_in_data.py \
  --enable_cameras \
  --num_episodes 5 \
  --max_attempts 20 \
  --max_steps 80 \
  --warmup_steps 5 \
  --dataset_file /home/mark/zhishu_dualarm_lab/data/hdf5/zhishu_closing_in_gui_debug.hdf5
```

说明：

- `--num_episodes 5`
  目标录 5 条 episode
- `--max_attempts 20`
  最多尝试 20 次；因为当前 safe-start 可能失败，不是每次 reset 都一定录得下来
- `--max_steps 80`
  每条最多记录 80 步
- `--warmup_steps 5`
  正式记录前先热身几步
- `--dataset_file`
  输出 HDF5 路径

### 5.2 Headless 版命令

如果只是批量采，不看 GUI：

```bash
source /home/mark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-5.1
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /home/mark/zhishu_dualarm_lab
python -u scripts/collect_closing_in_data.py \
  --headless \
  --enable_cameras \
  --num_episodes 20 \
  --max_attempts 80 \
  --max_steps 80 \
  --warmup_steps 5 \
  --dataset_file /home/mark/zhishu_dualarm_lab/data/hdf5/zhishu_closing_in_headless.hdf5
```

### 5.3 继续写入已有 HDF5

如果想在已有文件后面继续追加：

```bash
python -u scripts/collect_closing_in_data.py \
  --enable_cameras \
  --resume \
  --dataset_file /home/mark/zhishu_dualarm_lab/data/hdf5/zhishu_closing_in_gui_debug.hdf5
```

## 6. 当前 collector 会打印什么

成功运行时，典型日志流程是：

```text
[INFO] collecting LeIsaac-Zhishu-ClosingIn-v0 to ... episodes=5 max_steps=80
[INFO] attempt=0 episode=0 reset begin
[INFO] attempt=0 episode=0 reset done
[INFO] attempt=0 episode=0 spec mode=symmetric prompt='...'
[INFO] attempt=0 episode=0 safe-start begin
[INFO] safe-start check ...
[INFO] safe-start finished, entering warmup
[INFO] warmup finished, entering episode rollout
[INFO] rollout step=0 fetching policy input
[INFO] rollout step=0 solving teacher action
[INFO] rollout step=0 stepping env with buffered action
[INFO] rollout step=0 fetching post-step policy input
[INFO] demo_0 mode=... prompt='...' steps=80 success=False ...
[INFO] closing-in collection finished recorded=5/5 attempts=10/20
```

你要重点看这几类字段：

- `spec mode=...`
- `prompt='...'`
- `center_target=[...]`
- `left_target=[...]`
- `right_target=[...]`
- `safe-start check ...`
- 最后一行：
  `recorded=X/Y attempts=A/B`

其中：

- `recorded`
  是真正录下来的 episode 数
- `attempts`
  是总尝试次数

如果 `recorded < num_episodes`，说明 safe-start 或 rollout 失败率还偏高。

## 7. 当前数据内容

当前 HDF5 里，每条 episode 会保存：

- `prompt`
- `task_name`
- `target_mode`
- `speed_scale`
- `hold_steps`
- `jitter_scale`
- `success`
- `num_samples`

以及 rollout payload，包括：

- 3 路图像
- `70D state`
- `14D actions`
- reward / done / timestamp 一类辅助字段

当前固定 contract：

- `state`: `70D float32`
- `actions`: `14D float32`
  - `left_joint1~7`
  - `right_joint1~7`

## 8. 当前情况

截至当前版本，`collect_closing_in_data.py` 的状态是：

- 已经能在当前机器上启动 Isaac 场景并采集 `ClosingIn` 数据
- 已经能写出 HDF5
- 目前 collector 仍在持续调试 safe-start 和双臂姿态质量
- 当前最敏感的问题主要还在：
  - 右臂初始化姿态稳定性
  - 右臂 rollout 时肘部参与度
  - 不是每个 attempt 都能成功变成 recorded episode

也就是说：

- 这条链已经能跑
- 但当前还不是“闭着眼大批量生产高质量数据”的状态

协作者如果只是想确认能否跑起来，当前已经足够。
如果是想稳定采大量高质量数据，需要继续收敛 teacher policy。

## 9. 常见问题

### 9.1 为什么 GUI 运行很慢

Isaac Sim 首次启动可能非常慢。

在这台机器上，第一次打开 GUI 甚至可能需要很长时间。不要因为系统提示“无响应”就立刻关掉。

### 9.2 为什么程序可能只录下 3/5、4/5

因为当前 collector 有 `safe-start` 检查。

有些 attempt 会被判定为：

- 初始姿态不够安全
- 某一侧链条仍过低
- 还没进入正式 rollout 就中止

这时不会写成有效 episode。

### 9.3 为什么 scene 里有 4 路相机，但数据只有 3 路

当前训练主线只吃：

- external
- left wrist
- right wrist

腰部相机暂时没有接进数据 schema。

### 9.4 如果 URDF 路径不对怎么办

优先检查：

```bash
echo $ZHISHU_ROBOT_URDF
ls /home/mark/zhishu_robot_description-URDF/zhishu_robot_description/urdf/zhishu_robot_description.urdf
```

如果默认路径不存在，就用环境变量或 `config/local_paths.json` 指向正确位置。

## 10. 相关文件

主要入口：

- [scripts/collect_closing_in_data.py](/home/mark/zhishu_dualarm_lab/scripts/collect_closing_in_data.py)
- [scripts/run_dualarm_idle_scene.py](/home/mark/zhishu_dualarm_lab/scripts/run_dualarm_idle_scene.py)

数据定义：

- [source/zhishu_dualarm_lab/utils/closing_in_dataset.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/closing_in_dataset.py)

路径解析：

- [source/zhishu_dualarm_lab/utils/local_paths.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/local_paths.py)

总说明：

- [README.md](/home/mark/zhishu_dualarm_lab/README.md)
