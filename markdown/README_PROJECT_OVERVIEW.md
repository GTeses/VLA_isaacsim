# ZhishuLab 工程总览

这份文档给第一次接触 `zhishu_dualarm_lab` 的人用。

目标不是讲某一个脚本，而是让你快速回答这些问题：

- 这个项目现在到底做到哪一步了
- 主要功能入口在哪
- 关键文件分别负责什么
- 机器人本体是什么结构
- 如果我要运行场景、联调 policy、采集数据，应该从哪里开始

如果你还没看过数据采集说明，建议同时阅读：

- [README_DATA_COLLECTION.md](/home/mark/zhishu_dualarm_lab/markdown/README_DATA_COLLECTION.md)

根目录主说明仍然是：

- [README.md](/home/mark/zhishu_dualarm_lab/README.md)

## 1. 这个项目是什么

`zhishu_dualarm_lab` 是一个基于：

- Isaac Sim 5.1
- Isaac Lab 2.3.0

的外部项目，用来搭建“智书上半身双臂桌面任务”的实验平台。

它当前不是一个“已经训练好、任务能稳定成功”的完整产品，而是一个已经具备以下能力的工程骨架：

- 能把 Zhishu 机器人加载进 Isaac 场景
- 能在桌面任务里运行双臂 14 维 joint-delta 控制
- 能输出当前长期 observation contract
- 能通过 fake policy / websocket client / 真实 openpi server 做最小闭环联调
- 能运行仿真数据采集脚本，生成 HDF5 数据
- 已开始面向 `pi05_base -> Zhishu 自定义 fine-tuning` 的长期路线收敛

## 2. 当前项目状态

### 2.1 已经完成的部分

- 仿真场景、机器人资产、桌面物体、目标区已经接好
- 双臂长期动作 contract 已固定为 `14D joint delta`
- 当前长期 observation contract 已固定
- 真实 `openpi` server 已经替换过 fake server，最小 websocket 闭环跑通过
- 当前可以采集 `LeIsaac-Zhishu-ClosingIn-v0` 的仿真数据
- `openpi` 侧 Zhishu 自定义 schema / config scaffold 已进代码

### 2.2 当前还在持续收敛的部分

- closing-in teacher policy 的双臂姿态质量
- 右臂初始化和 rollout 中的姿态自然性
- 数据采集成功率和轨迹一致性
- LeRobot 转换后的训练链路批量验证

### 2.3 当前不属于近期目标的部分

- 灵巧手
- 抓取 / 松开闭环
- pick-and-place 作为近期主 benchmark
- 末端位姿 + IK 作为长期执行主线

## 3. 当前长期 contract

### 3.1 动作 contract

当前 env 内部长期动作定义固定为：

- `14D float32`
- `left_joint1 ~ left_joint7`
- `right_joint1 ~ right_joint7`

语义是：

- 双臂 joint delta

这条 contract 是当前工程主线，不是临时 smoke test 层。

### 3.2 observation contract

当前长期 policy input 固定为：

- `prompt`
- `observation/external_image`
- `observation/left_wrist_image`
- `observation/right_wrist_image`
- `observation/state`

其中：

- 图像格式：`HWC uint8 RGB`
- `state`：`70D float32`

### 3.3 state 拼接顺序

`70D state` 当前顺序固定为：

- `joint_pos`
- `joint_vel`
- `last_action`
- `left_tcp_pose`
- `right_tcp_pose`
- `object_pose`
- `target_pose`

这条顺序已经是当前数据采集和训练主线的一部分，不要随意改。

## 4. 机器人本体概况

### 4.1 机器人类型

当前 Zhishu 机器人可以理解为：

- 上半身平台
- 一个可升降、可转动的腰部
- 一个头部关节
- 左右各一条 7 自由度机械臂
- 当前任务主线不接灵巧手，不做抓取

### 4.2 从工程角度当前真正控制的自由度

虽然完整 URDF 里不止 14 个自由度，但当前 env 对外只暴露双臂 14 个关节：

- 左臂 7 自由度
- 右臂 7 自由度

也就是：

- `left_joint1 ~ left_joint7`
- `right_joint1 ~ right_joint7`

腰部、头部等不作为当前 policy 输出的一部分。

### 4.3 原始 URDF 里还能看到的其他关节

在原始 URDF 里还能看到：

- `waist_up_down_join`
  - 腰部升降
  - `prismatic`
- `waist_y_joint`
  - 腰部绕 y 转动
  - `revolute`
- `head_y_joint`
  - 头部转动
  - `revolute`

这些说明这是一个“带躯干和头部的双臂上半身平台”，不是只有两条孤立机械臂。

### 4.4 双臂结构的工程理解

从工程角度，双臂应理解为：

- 每条手臂都是一条 7 自由度串联链
- 靠近身体的一两级关节决定大臂和肩部姿态
- 中段关节决定主要弯折
- 末端几级关节更偏向前臂 / 腕部调整

要注意：

- 左右臂在原始 URDF 里是“总体镜像设计”
- 但不是“完全同参复制、只改一个符号”的简单镜像
- 特别是近端关节的限位和自然分支并不完全对称

这也是为什么当前很多姿态问题不能简单地把左臂逻辑直接复制到右臂。

### 4.5 原始 URDF 位置

当前默认使用的原始 URDF 在：

- [/home/mark/zhishu_robot_description-URDF/zhishu_robot_description/urdf/zhishu_robot_description.urdf](/home/mark/zhishu_robot_description-URDF/zhishu_robot_description/urdf/zhishu_robot_description.urdf)

项目里通过：

- [local_paths.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/local_paths.py)

解析本地路径。

## 5. 场景和传感器

### 5.1 场景组成

当前桌面任务场景主要有：

- ground
- light
- table
- object
- target_zone
- robot
- 4 路 camera body / camera
- tcp frame transformer

### 5.2 当前相机布局

当前场景中有 4 路相机：

- head external camera
- waist camera
- left wrist camera
- right wrist camera

但当前训练 / 数据采集主线只使用 3 路：

- `external_image`
- `left_wrist_image`
- `right_wrist_image`

腰部相机目前只在场景中存在，不进入主数据 contract。

### 5.3 桌面任务对象

当前桌面任务核心对象有两个：

- `object`
  - 当前通常是桌面上的 cube / puck 一类操作对象
- `target_zone`
  - 当前通常是目标区域

`closing-in` 任务里主要是向目标区域聚拢，不是抓取。

## 6. 工程主入口

### 6.1 只打开场景，不做动作

如果你只是想确认：

- Isaac 能否启动
- 场景能否加载
- 机器人是否正常显示

入口是：

- [run_dualarm_idle_scene.py](/home/mark/zhishu_dualarm_lab/scripts/run_dualarm_idle_scene.py)

用途：

- 加载当前 Zhishu 场景
- step 零动作
- 不自动退出

### 6.2 跑 websocket / openpi 联调

如果你想跑：

- fake / real policy server
- websocket client
- env action buffer
- 最小动作闭环

入口是：

- [demos/run_dualarm_with_openpi.py](/home/mark/zhishu_dualarm_lab/demos/run_dualarm_with_openpi.py)
- [source/zhishu_dualarm_lab/demos/run_dualarm_with_openpi.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/demos/run_dualarm_with_openpi.py)

通常实际执行的是根目录 `demos/` 下那个入口。

### 6.3 跑 closing-in 数据采集

如果你想采当前主线仿真数据，入口是：

- [collect_closing_in_data.py](/home/mark/zhishu_dualarm_lab/scripts/collect_closing_in_data.py)

它当前负责：

- 采样 episode spec
- safe-start
- scripted / heuristic teacher rollout
- 写 HDF5

### 6.4 把 HDF5 转成 LeRobot

入口是：

- [convert_zhishu_sim_to_lerobot.py](/home/mark/zhishu_dualarm_lab/scripts/convert_zhishu_sim_to_lerobot.py)

## 7. 关键文件和作用

下面这部分按“第一次读代码最值得知道的东西”来排。

### 7.1 根目录说明

- [README.md](/home/mark/zhishu_dualarm_lab/README.md)
  - 项目主 README
  - 记录路线、长期计划、阶段性状态

### 7.2 环境本体

- [env.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/dualarm_tabletop/env.py)
  - 当前最核心的环境实现
  - 定义 action 应用、observation 输出、camera mount 同步、reward/done、policy buffer

- [env_cfg.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/dualarm_tabletop/env_cfg.py)
  - 环境配置
  - 仿真步长、scene 绑定、episode length 等参数通常在这里收

- [scene_cfg.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/dualarm_tabletop/scene_cfg.py)
  - 场景组装
  - 把 robot、table、object、target、camera、frame transformer 等放进 scene

### 7.3 场景资产和传感器配置

- [objects.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/dualarm_tabletop/objects.py)
  - ground / light / table / object / target zone / camera body 的配置

- [cameras.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/tasks/dualarm_tabletop/cameras.py)
  - head / waist / left wrist / right wrist camera 的配置

### 7.4 observation / action 处理

- [obs_builder.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/obs_builder.py)
  - 把图像、joint、tcp、object、target 拼成当前长期 observation contract

- [action_adapter.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/action_adapter.py)
  - 把 policy action / action chunk 适配成 env 可消费的 joint delta / joint target

- [policy_client.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/policy_client.py)
  - websocket client
  - `zhishu14` / `libero` 输入输出 schema 兼容入口

### 7.5 数据采集和数据结构

- [collect_closing_in_data.py](/home/mark/zhishu_dualarm_lab/scripts/collect_closing_in_data.py)
  - closing-in 数据采集主脚本

- [closing_in_dataset.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/closing_in_dataset.py)
  - HDF5 schema
  - state 解析
  - episode spec 采样
  - success 判定

- [convert_zhishu_sim_to_lerobot.py](/home/mark/zhishu_dualarm_lab/scripts/convert_zhishu_sim_to_lerobot.py)
  - 原始仿真数据向 LeRobot 数据集转换

### 7.6 路径和本地配置

- [local_paths.py](/home/mark/zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/local_paths.py)
  - 本地路径解析器
  - 负责 URDF、openpi 根目录、checkpoint roots 的本地解析

## 8. 当前最重要的运行路径

### 8.1 空场景检查路径

```text
run_dualarm_idle_scene.py
-> ZhishuDualArmTabletopEnv
-> scene_cfg / env_cfg / env
```

### 8.2 websocket 联调路径

```text
run_dualarm_with_openpi.py
-> policy_client.py
-> openpi server
-> action_adapter.py
-> env.apply_policy_output()
-> action_plan_buffer
-> env.step()
```

### 8.3 数据采集路径

```text
collect_closing_in_data.py
-> ClosingInEpisodeSpec 采样
-> safe-start
-> teacher rollout
-> HDF5 写盘
-> convert_zhishu_sim_to_lerobot.py
```

## 9. 新人最推荐的上手顺序

### 第一步：先确认环境

先执行：

```bash
source /home/mark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-5.1
which python
python -c "import isaaclab, isaacsim; print('ok')"
```

如果不通过，先不要继续。

### 第二步：先打开空场景

```bash
cd /home/mark/zhishu_dualarm_lab
python -u scripts/run_dualarm_idle_scene.py --enable_cameras
```

先看：

- 场景是否加载
- 机器人是否正常
- 相机是否正常

### 第三步：再跑 closing-in 数据采集

```bash
python -u scripts/collect_closing_in_data.py \
  --enable_cameras \
  --num_episodes 5 \
  --max_attempts 20 \
  --max_steps 80 \
  --warmup_steps 5 \
  --dataset_file /home/mark/zhishu_dualarm_lab/data/hdf5/zhishu_closing_in_gui_debug.hdf5
```

### 第四步：最后再碰 websocket / openpi

如果只是想熟悉工程，先不要一上来就进 openpi 联调。

先掌握：

- 场景
- env
- observation / action contract
- 数据采集链

再去看 websocket / openpi，会清楚很多。

## 10. 当前项目最需要知道的限制

### 10.1 现在“能用”不等于“已经稳定”

这个仓库当前是：

- 已经能运行
- 已经能采数据
- 已经有长期路线

但还不是：

- 全部任务稳定成功
- 全部姿态自然
- teacher policy 已完全收敛

### 10.2 当前右臂姿态仍在持续收敛

当前 closing-in teacher 的一个重点难点仍然是：

- 右臂初始化和 rollout 过程中的姿态自然性
- 尤其是肘部参与程度和接近目标时的姿态质量

这不是资产加载问题，而是当前 teacher / safe-start 仍在持续调试。

### 10.3 当前主线不是抓取

如果你进仓库就想做：

- pick-and-place
- dexterous hand
- grasp / release

那会和当前主线冲突。

当前主线是：

- reaching
- move-to-target
- closing-in
- pushing / nudging

## 11. 一句话理解这个仓库

`zhishu_dualarm_lab` 当前是一个面向 Zhishu 双臂无手平台的 Isaac 仿真工程底座：

- env、scene、contract、websocket 最小闭环已经建好
- 数据采集链已经开始工作
- 长期目标是把 `pi05_base` 训练成适配 Zhishu 自定义 schema 的 policy

如果你完全不了解这个项目，最先该掌握的不是训练，而是：

- 场景怎么起来
- env 输出什么
- 动作怎么进 env
- 数据是怎么被采出来的
