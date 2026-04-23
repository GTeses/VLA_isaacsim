# zhishu_dualarm_lab

`zhishu_dualarm_lab` 是一个面向 Isaac Sim 5.1 + Isaac Lab 2.3.0 的外部项目骨架，用来跑“智书机器人上半身双臂桌面任务”的第一版最小原型。

## 当前环境约定

这份 README 已按当前机器上的实际目录布局整理，下面这些路径和约定默认成立：

- 仓库根目录：`/root/gpufree-data/arcus/zhishu_dualarm_lab`
- Isaac 运行环境：`conda activate isaaclab`
- openpi 根目录：`/root/gpufree-data/arcus/openpi`
- 当前仓库内所有 demo / script 都从仓库根目录直接运行
- 当前机器优先使用激活环境后的 `python`，不要默认 `isaaclab.sh -p`

本地路径解析的优先级当前是：

- `ZHISHU_ROBOT_URDF`
- `OPENPI_ROOT`
- `OPENPI_CHECKPOINT_ROOTS`
- `OPENPI_LOCAL_CHECKPOINT_ROOT`（兼容旧变量名）
- `config/local_paths.json`

首次使用时，建议先生成一份本地路径配置：

```bash
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
cp config/local_paths.example.json config/local_paths.json
```

当前 `config/local_paths.example.json` 已经对齐这台机器的默认目录，可直接作为起点。

它当前的定位是：

- 已经把“身体”装好：机器人资产、场景、控制、相机、环境接口都已经打通
- 已经完成“最小可行 policy bridge”：fake client、fake websocket server、真实 openpi websocket server 都能接入当前 demo
- 当前版本不再只是“纯仿真底座”，而是“仿真底座 + 最小可行 websocket policy 闭环”

## 当前状态快照

先给最新状态，避免被后面的历史描述误导。

- 真实 `openpi` server 已经替换过 fake server，完成过最小闭环联调
- 当前链路已经实际打通：
  `obs -> get_policy_input() -> websocket client -> openpi server -> action chunk -> apply_policy_output() -> action_plan_buffer -> consume_action_plan_step() -> env.step()`
- env 内部长期 contract 目前仍然是 `14` 维双臂 joint delta
- 当前 `pi05_libero` 接入依赖一个显式兼容层：
  - 输入适配：本地 `prompt + 3 路图像 + 70D state` 中，映射出 LIBERO 风格输入
  - 输出适配：真实 LIBERO `7` 维动作被临时扩成当前 `14` 维双臂动作
- 这个兼容层的目标只是打通真实 websocket 收发和 env 动作落地
- 它不代表当前控制语义已经正确，也不代表 `pi05_libero` 已适配智书双臂任务
- 第二阶段已经开始落到 env 本身：
  - 任务语义已收缩为“无手双臂靠近 / 聚拢 / 推动 cube 到目标区”
  - reward / done 已不再围绕 grasp success 设计
  - 但这仍然是任务定义收缩，不是长期 policy 语义已经完成
- 第三阶段的 `openpi` 侧脚手架也已经进代码：
  - 已新增 `openpi/src/openpi/policies/zhishu_policy.py`
  - 已新增 `pi05_zhishu_dualarm_nohand` config 注册
  - 但这仍然只是自定义 schema / config scaffold，尚未完成 Zhishu 数据转换、norm stats 和 fine-tuned checkpoint 训练
- 第三阶段的本地辅助脚本也已经补上：
  - `scripts/check_openpi_stage3_prereqs.py`
  - `scripts/convert_zhishu_data_to_lerobot.py`
- 当前机器上已确认：
  - `pi05_libero` 已在本地 cache
  - `pi05_base` 已下载到 `/root/gpufree-data/arcus/checkpoints/pi05_base`
  - 当前 `pi05_zhishu_dualarm_nohand` 已改成优先使用本地 `pi05_base`，找不到时才回退到 `gs://openpi-assets/checkpoints/pi05_base`
  - 所以第三阶段目前已经具备“代码脚手架 + 本地 base checkpoint + 数据转换入口”
  - 剩余 blocker 已经收缩为：真实 Zhishu 数据集生成、`compute_norm_stats`、fine-tuning 和新 checkpoint 验证

## 当前最小闭环的真实 contract

### env -> policy client

当前 env 对外稳定提供：

- `prompt`
- `observation/external_image`
- `observation/left_wrist_image`
- `observation/right_wrist_image`
- `observation/state`

其中：

- 三路图像都是 `HWC uint8 RGB`
- `observation/state` 当前是 `70` 维 `float32`
- 其拼接顺序固定为：
  `joint_pos + joint_vel + last_action + left_tcp_pose + right_tcp_pose + object_pose + target_pose`

### `--policy_input_schema zhishu14`

- 不做额外映射
- 直接把上面的本地 contract 发给 server
- 这是长期 Zhishu 主线应使用的 schema 名字
- `native` 目前只是兼容旧命令的别名

### `--policy_input_schema libero`

这是当前真实 `pi05_libero` 联调用的显式兼容层：

- `prompt` 原样透传
- `observation/external_image -> observation/image`
- `observation/left_wrist_image -> observation/wrist_image`
- 图像会从当前分辨率 resize 到 `224 x 224`
- `observation/state` 只截取前 `8` 维，作为 LIBERO policy 期望的低维输入
- `observation/right_wrist_image` 当前不会发给 `pi05_libero`

### policy server -> env

当前 env 最终只接受：

- 单步动作：`[14]`
- 动作块：`[T, 14]`

### `--policy_output_contract zhishu14`

- server 返回的动作必须已经是当前 `14` 维双臂 contract
- client 只做 shape 检查和标准化，不做语义映射
- 这是长期 Zhishu 主线应使用的 contract 名字
- `native14` 目前只是兼容旧命令的别名

### `--policy_output_contract libero7`

这是当前真实 `pi05_libero` 联调用的输出兼容层，也是 smoke-test only 的短期联调层：

- server 返回 LIBERO 单步 `7` 维或 chunk `[T, 7]`
- 当前实现会把同一个 `7` 维动作直接复制到左右两臂
- 最终得到 `[T, 14]`

注意：

- 这只是一个 transport-level placeholder bridge
- 它只证明真实 server 已参与闭环，不能证明动作语义适合智书双臂
- 如果后续要长期使用真实 policy，建议把 schema / adapter 独立成单独模块继续收敛

## 2026-04-21 真实 `pi05_libero` 烟雾测试结果

这一次测试的目标只验证链路，不验证任务成功率，也不把 `pi05_libero` 当长期语义方案。

### 本次测试结论

- 真实 `openpi` server 已替换 fake server
- 本地缓存 checkpoint 已成功加载：
  `/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero`
- websocket 已成功连上真实 server
- 真实 server 已返回 `dict["actions"]`
- server 原始返回的动作 shape 是 `(10, 7)`
- Isaac 侧经过 `--policy_output_contract libero7` 适配后，实际进入 env 的动作 chunk shape 是 `(10, 14)`
- action chunk 已进入 `action_plan_buffer`
- env 已逐步消费 buffer，并完成 `env.step()`
- 当前烟雾测试通过

### 本次 server 启动命令

注意：`tyro` 参数顺序有要求，`--port` 要放在主命令上，不能放在 `policy:checkpoint` 后面。

```bash
cd /root/gpufree-data/arcus/openpi
source .venv/bin/activate
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

本次 server 端关键日志：

```text
INFO:root:Loading model...
INFO:absl:Restoring checkpoint from /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/params.
INFO:absl:Finished restoring checkpoint in 4.52 seconds from /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/params.
INFO:root:Loaded norm stats from /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero
INFO:websockets.server:server listening on 0.0.0.0:8000
```

### 本次 client 启动命令

注意：真实 `pi05_libero` 当前不能直接吃本地 native schema，也不能直接输出本地 `14` 维双臂动作，所以这次烟雾测试必须显式带上 `libero` / `libero7` 兼容参数。

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python -u demos/run_dualarm_with_openpi.py \
  --policy_mode websocket \
  --policy_host 127.0.0.1 \
  --policy_port 8000 \
  --policy_input_schema libero \
  --policy_output_contract libero7 \
  --enable_cameras \
  --headless \
  --max_steps 8 \
  --replan_steps 3 \
  --debug_print_every 1
```

### 本次成功日志摘录

```text
[INFO] policy client mode=websocket host=127.0.0.1 port=8000 replan_steps=3 input_schema=libero output_contract=libero7
[INFO] server metadata keys: []
[INFO] policy input keys: ['observation/external_image', 'observation/left_wrist_image', 'observation/right_wrist_image', 'observation/state', 'prompt']
[INFO] observation/external_image: shape=(256, 256, 3) dtype=uint8
[INFO] observation/left_wrist_image: shape=(256, 256, 3) dtype=uint8
[INFO] observation/right_wrist_image: shape=(256, 256, 3) dtype=uint8
[INFO] observation/state: shape=(70,) dtype=float32
[INFO] received action chunk shape: (10, 14)
[INFO] staged action buffer length: 3
[INFO] first staged action shape: (14,)
[INFO] consumed action shape: (1, 14) buffer_remaining=2
[INFO] step=0 reward_mean=0.6302 terminated=False truncated=False
```

额外做过一次最小 websocket 直连验证，确认真实 server 原始响应 schema：

```text
response_keys ['actions', 'policy_timing', 'server_timing']
raw_actions_shape (10, 7)
raw_actions_dtype float64
```

### 本次没有出现的失败点

- checkpoint load：未失败
- tokenizer/assets download：未观察到新的远端下载；本次直接使用本地缓存 checkpoint 与 norm stats
- websocket：未失败
- response schema：未失败，真实 server 返回了 `actions`
- action shape：未失败，server 原始是 `(10, 7)`，本地适配后进入 env 的是 `(10, 14)`
- env apply：未失败，buffer 写入和消费都成功

### 对这次烟雾测试的解释

- 这次通过，证明的是“真实 `openpi` server 已进入闭环”
- 不是在证明 `pi05_libero` 的控制语义已经适配智书双臂
- 当前 `libero -> 14D dual-arm joint delta` 仍然只是临时兼容桥

## 长期适配路线：`pi05_base` -> Zhishu 双臂无手平台

这部分是当前推荐主线，用来替代把 `pi05_libero` 或 `pi05_droid` 当长期方案的做法。

### 路线判断

- 短期验证继续允许使用真实 `pi05_libero` 做 smoke test
- 长期主线不再围绕 `pi05_libero` / `pi05_droid`
- 长期 checkpoint 起点改为 `pi05_base`
- 长期任务收缩为“无手双臂简化任务”，不把 grasp / release / dexterous-hand manipulation 当近期目标

### 长期任务范围

第一优先级：

- single-arm reaching to target
- bimanual move-to-target
- bimanual closing-in / gather motion
- pushing / nudging object to target region

第二优先级：

- 双臂协同把物体逼近目标区
- 非抓取接触式操作

当前不作为主线：

- 真正 grasp / release
- proxy gripper 闭环
- 灵巧手动作建模
- pick-and-place 作为近期核心 benchmark

### 当前代码里已经落地的第二阶段约束

`zhishu_dualarm_lab` 这边已经先把 env 语义收紧，作为后续 `pi05_base` 自定义 fine-tuning 的执行端。

当前 env 保持不变的部分：

- 3 路图像：
  - `observation/external_image`
  - `observation/left_wrist_image`
  - `observation/right_wrist_image`
- `observation/state` 仍然固定为 `70D float32`
- `14` 维动作 contract 仍然固定为双臂 joint delta

当前 env 已改成无手任务语义的部分：

- prompt 改成“靠近 cube 并把它 nudging 到 target zone，不做 grasp”
- reward 改成显式 no-hand shaping：
  - 双臂 TCP 靠近 cube
  - 双臂中点向 cube 聚拢
  - cube 向 target zone 靠近
  - cube 向 target zone 的逐步 progress
  - 轻微 action penalty
- done 改成：
  - 双臂都已靠近并聚拢到 cube 周围
  - 或 cube 已进入 target zone
  - 或超时

这一步的含义是：

- 现在环境本身更适合做 reaching / move-to-target / pushing / nudging
- 不再继续把 grasp / release 当成当前 reward 设计中心
- 但它仍然不是最终 benchmark 定稿，后面还可以继续按数据和训练结果调阈值与权重

### Zhishu 自定义 `Inputs/Outputs` 已落地的代码脚手架

```text
openpi/src/openpi/policies/zhishu_policy.py
```

当前已经落地了三个入口：

- `ZhishuInputs`
- `ZhishuOutputs`
- `make_zhishu_example()`

建议固定 env / websocket 输入 contract 为：

```python
{
  "prompt": str,
  "observation/external_image": HWC uint8 RGB,
  "observation/left_wrist_image": HWC uint8 RGB,
  "observation/right_wrist_image": HWC uint8 RGB,
  "observation/state": np.ndarray[float32],
}
```

当前 `ZhishuInputs` 映射为模型输入：

```python
{
  "state": <adapted_state>,
  "image": {
    "base_0_rgb": data["observation/external_image"],
    "left_wrist_0_rgb": data["observation/left_wrist_image"],
    "right_wrist_0_rgb": data["observation/right_wrist_image"],
  },
  "image_mask": {
    "base_0_rgb": np.True_,
    "left_wrist_0_rgb": np.True_,
    "right_wrist_0_rgb": np.True_,
  },
  "prompt": data["prompt"],
}
```

`ZhishuOutputs` 当前直接返回：

```python
{"actions": np.asarray(data["actions"][:, :14])}
```

这里有一个现在必须明确写出来的实现约束：

- env / websocket 对外 contract 仍然保留 `70D` 左右的 native state
- 但 `openpi` 当前 `pi0/pi0.5` state 通路本身要求 state 宽度与 `action_dim` 对齐
- 所以 `ZhishuInputs` 里现在显式做了一次 bootstrap 适配：
  - 如果 native state 长于 `14`，就截前 `14` 维
  - 如果短于 `14`，就补零到 `14`
- 这不是最终语义定稿，而是当前 `pi05_base -> 14D dual-arm action` 路线下最小可行的模型侧适配
- 如果后续要完整吃下 `70D state`，那会是 model/schema 联动升级，不只是改 README 或 config

这里的长期原则仍然是：

- 长期直接以 Zhishu 自己的 `14` 维动作作为 policy output contract
- 不再把 `libero7 -> 14` 这种临时桥接当长期方案
- protocol / serialization
- policy schema transform
- env action adapter

这三层必须分开维护。

### `pi05_zhishu_dualarm_nohand` 已落地的 config 脚手架

当前已经在 `openpi/src/openpi/training/config.py` 新增了这条配置，形态上参考了：

- `pi05_libero`
- `pi05_droid_finetune`
- `pi05_aloha_pen_uncap`

长期主线仍然明确继承 `pi05_base` 的权重，而不是继承 expert checkpoint。

当前代码里的核心形态如下：

```python
TrainConfig(
    name="pi05_zhishu_dualarm_nohand",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=14,
        action_horizon=10,
        discrete_state_input=False,
    ),
    data=LeRobotZhishuDataConfig(
        repo_id="your_hf_username/zhishu_dualarm_nohand",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir=_resolve_checkpoint_subpath("pi05_base", "assets"),
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        _resolve_checkpoint_subpath("pi05_base", "params")
    ),
    num_train_steps=20_000,
    batch_size=64,
)
```

当前这条 config 的实现拆分已经是：

- 新增 `LeRobotZhishuDataConfig`
- 默认 `repack_transforms` 会把更标准的 LeRobot 列名重命名成 native websocket contract：
  - `observation.images.external -> observation/external_image`
  - `observation.images.left_wrist -> observation/left_wrist_image`
  - `observation.images.right_wrist -> observation/right_wrist_image`
  - `observation.state -> observation/state`
  - `action -> actions`
- `data_transforms.inputs` 使用 `ZhishuInputs`
- `data_transforms.outputs` 使用 `ZhishuOutputs`
- `weight_loader` 现在会优先使用本机已存在的本地 `pi05_base/params`
- `assets_dir` 现在会优先使用本机已存在的本地 `pi05_base/assets`
- norm stats 不再预设去复用 base checkpoint 的 stats，而是等 Zhishu 数据准备好后由 `compute_norm_stats` 生成自己的 stats

这一点和长期路线是一致的：

- 权重初始化来自 `pi05_base/params`
- norm stats 来自 Zhishu 自己的数据
- 输入输出 schema 来自 Zhishu 自己的 policy transform
- 如需显式指定 checkpoint 根目录，可在运行前设置：
  - `OPENPI_CHECKPOINT_ROOTS=/root/gpufree-data/arcus/checkpoints`
  - `OPENPI_LOCAL_CHECKPOINT_ROOT=/root/gpufree-data/arcus/checkpoints` 仍可用，但只作为兼容旧变量名

首版可先从这些超参数开始试：

- `action_dim = 14`
- `action_horizon = 10` 或 `15`
- `prompt_from_task = True`
- batch size 先保守，不要直接照搬大规模官方设置

### 需要的数据字段清单

如果要把 Zhishu rollout / teleop / scripted demo 转成 LeRobot 格式，当前推荐固定这些 LeRobot 字段：

- `observation.images.external`
- `observation.images.left_wrist`
- `observation.images.right_wrist`
- `observation.state`
- `action`
- `task`
- `episode_index`
- `frame_index`
- `timestamp`
- `done`

其中核心训练字段最少必须稳定包含：

- 3 路图像
- `70D` 左右的稳定 state 向量
- `14D` 动作
- task instruction

### 当前已经补上的数据转换脚手架

当前新增了：

```text
zhishu_dualarm_lab/scripts/convert_zhishu_data_to_lerobot.py
```

这个脚手架当前定义的 canonical raw layout 是：

```text
raw_dir/
  episode_0000/
    metadata.json
    prompt.txt
    observation_state.npy
    action.npy
    reward.npy
    terminated.npy
    truncated.npy
    timestamp.npy
    external_images/000000.png
    left_wrist_images/000000.png
    right_wrist_images/000000.png
```

其中：

- `observation_state.npy` 期望 shape 为 `[T, state_dim]`
- `action.npy` 或 `actions.npy` 期望 shape 为 `[T, 14]`
- `reward.npy` / `terminated.npy` / `truncated.npy` / `timestamp.npy` 当前是推荐记录字段，便于保留 rollout 语义和后续筛选
- 三路图像目录都必须有同样长度的逐帧图片
- `task` 优先从 `metadata.json["task"]` 读取，其次从 `prompt.txt` 读取

### 当前已经补上的仿真数据采集脚本

当前新增了：

```text
zhishu_dualarm_lab/scripts/collect_zhishu_sim_data.py
```

这个脚本默认使用一个轻量 scripted joint-space policy，专门为第一批无手仿真数据服务：

- 先让双臂向 cube 靠近
- 再让双臂向 cube 聚拢
- 再给一个朝 target zone 的推动偏置

它不是高质量控制器，但足够承担“先把 collect -> convert -> norm stats -> fine-tune 路线打通”的第一版数据源。

推荐先采一小批 scripted rollout：

```bash
conda activate isaaclab
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/collect_zhishu_sim_data.py \
  --policy_mode scripted \
  --policy_input_schema zhishu14 \
  --policy_output_contract zhishu14 \
  --enable_cameras \
  --headless \
  --num_episodes 20 \
  --max_steps 80 \
  --replan_steps 1 \
  --output_dir /root/gpufree-data/arcus/zhishu_dualarm_lab/data/raw/zhishu_stage3_reach_push_v1
```

如果你想记录 websocket policy rollout，也可以把 `--policy_mode` 切成 `websocket`，但长期训练主线仍建议优先围绕 `zhishu14` native contract 采数据，不要继续放大 `libero7 -> 14` 临时桥接的比重。

转换脚本会把数据写成：

- `observation.images.external`
- `observation.images.left_wrist`
- `observation.images.right_wrist`
- `observation.state`
- `action`
- `task`

可直接参考：

```bash
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/convert_zhishu_data_to_lerobot.py \
  --raw_dir /root/gpufree-data/arcus/zhishu_dualarm_lab/data/raw/zhishu_stage3_reach_push_v1 \
  --repo_id <your_hf_username/zhishu_dualarm_nohand>
```

### LeIsaac 参考下的 `LeIsaac-Zhishu-ClosingIn-v0` 小规模 HDF5 采集

这条链路是按 LeIsaac 的思路补的：

- 先录一个小规模 HDF5
- 再用 replay 看轨迹语义是否稳定
- 确认“同任务不同动作，但语义一致”之后再转 LeRobot

这不是新的 gym 注册名，而是当前 closing-in 数据集/任务的规范名字：

- `LeIsaac-Zhishu-ClosingIn-v0`

当前已经新增：

```text
zhishu_dualarm_lab/source/zhishu_dualarm_lab/utils/closing_in_dataset.py
zhishu_dualarm_lab/scripts/collect_closing_in_data.py
zhishu_dualarm_lab/scripts/replay_closing_in_hdf5.py
zhishu_dualarm_lab/scripts/convert_zhishu_sim_to_lerobot.py
```

这个 collector 每个 episode 会随机采样：

- `target_mode=center` 或 `target_mode=symmetric`
- 桌面中部大工作区内的 `center_target`
  - `x` 方向大约覆盖 `base_target +/- 0.12`
  - `y` 方向大约覆盖 `base_target +/- 0.22`
  - `z` 方向只做小幅抬升/下调，保持目标仍在桌面上方
- `symmetric` 模式下额外采样左右偏移，范围约 `0.08 ~ 0.18`
- 不同 `speed_scale`
- 不同 `hold_steps`
- 不同 `jitter_scale`
- 桌面上方安全 home pose 周围的对称为主、小到中等幅度双臂起始扰动

当前 closing-in collector 的 teacher 已不再使用手写 `err -> joint delta` 符号映射，
而是改成基于 Isaac Lab `DifferentialIKController` 的 task-space 近场 teacher：

- 目标是避免“猜关节符号”导致双臂往身后拧
- 每一步先在 task space 上生成一个小的 TCP 目标位移
- 再通过 Jacobian / differential IK 求出 14 维 joint-delta teacher action
- 当前 `jitter_scale` 也被收得很小，优先保证语义稳定而不是强行加噪声

当前起始位姿也不再靠“盲采一组 joint 再看 TCP 有没有碰巧在桌上方”：

- collector 会先把双臂 TCP 抬到桌前上方的自由空间
- 再移动到桌面上方的安全 home 区域
- 只有在两个 TCP 都已经高于桌面、并落在桌前工作区时，才开始正式录制
- 这组 home pose 不再来自单一 seed，而是来自一个小型“自然姿态 seed 库”
  - 每个 seed 都是桌上、可达、肩肘自然参与的双臂姿态
  - 每条 episode 会从其中一组 seed 出发，再叠加小到中等扰动
  - safe-start 的 IK 预摆位后，还会把肩/大臂/肘部轻度回拉向该 seed
    这样做是为了避免“末端位置不同，但肩肘总落到同一套别扭姿态”

这样做的目的很直接：避免从桌下坏姿态起步，生成“目标在桌上、双臂却先卡在桌下”的错误 closing-in 轨迹。

并保存：

- `prompt`
- 3 路 `256x256` RGB 图像
- `70D` state
- `14D` joint-delta action
- `success`
- target 点位和初始 scene state

HDF5 结构固定为：

```text
dataset.hdf5
  /data/demo_0
    attrs:
      task_name=LeIsaac-Zhishu-ClosingIn-v0
      prompt=...
      target_mode=center|symmetric
      success=True|False
      speed_scale=...
      hold_steps=...
      jitter_scale=...
    observation/state
    observation/external_image
    observation/left_wrist_image
    observation/right_wrist_image
    actions
    reward
    done
    timestamp
    task/center_target
    task/left_target
    task/right_target
    initial/arm_joint_pos
    initial/object_root_state
    initial/target_root_state
```

推荐先录一个可视化小样本：

```bash
conda activate isaaclab
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/collect_closing_in_data.py \
  --enable_cameras \
  --num_episodes 8 \
  --max_steps 80 \
  --warmup_steps 5 \
  --dataset_file /root/gpufree-data/arcus/zhishu_dualarm_lab/data/hdf5/zhishu_closing_in_smoke.hdf5
```

如果想关掉 GUI 再批量录制，再加 `--headless`。

录完以后先回放，不要直接拿去训练：

```bash
conda activate isaaclab
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/replay_closing_in_hdf5.py \
  --enable_cameras \
  --dataset_file /root/gpufree-data/arcus/zhishu_dualarm_lab/data/hdf5/zhishu_closing_in_smoke.hdf5 \
  --select_episodes 0 1 2
```

如果回放确认轨迹语义稳定，再转 LeRobot：

```bash
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/convert_zhishu_sim_to_lerobot.py \
  --dataset_file /root/gpufree-data/arcus/zhishu_dualarm_lab/data/hdf5/zhishu_closing_in_smoke.hdf5 \
  --repo_id <your_hf_username>/zhishu_closing_in_v0 \
  --successful_only
```

这条 closing-in HDF5 路线的目的很明确：

- 不采随机摇摆
- 不采抓取
- 先把一个清晰的“无手双臂聚拢”语义数据集录干净
- 再把它并入长期 `pi05_base + Zhishu` 数据主线

如果你怀疑某个关节符号或工作空间方向不对，当前还新增了一个单关节 TCP 体检脚本：

```text
zhishu_dualarm_lab/scripts/probe_zhishu_joint_tcp_response.py
```

它会固定初始位姿，逐个关节给正负小动作，打印 TCP 在世界坐标系下的位移：

```bash
conda activate isaaclab
which python
python -c "import isaaclab, isaacsim; print('ok')"
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/probe_zhishu_joint_tcp_response.py \
  --arm left \
  --enable_cameras
```

这个输出适合直接验证：

- 哪个关节主要影响前后
- 哪个关节主要影响左右
- 哪个关节主要影响上下
- 某个“手写 sign 表”到底是不是反了

### 第三阶段开始前的本地预检

当前新增了：

```text
zhishu_dualarm_lab/scripts/check_openpi_stage3_prereqs.py
```

它会检查：

- `openpi` 根目录和 `.venv`
- 本地 `pi05_libero`
- 本地 `pi05_base`
- 当前会按顺序搜索：
  - `/root/gpufree-data/arcus/checkpoints`
  - `/root/.cache/openpi/openpi-assets/checkpoints`

可直接参考：

```bash
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/check_openpi_stage3_prereqs.py
```

如果需要额外 checkpoint 根目录，也可以显式追加：

```bash
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/check_openpi_stage3_prereqs.py \
  --checkpoint_root /root/gpufree-data/arcus/checkpoints
```

当前这台机器上，`pi05_base` 已经在本地，所以“开始 fine-tuning”的前置条件不再是 checkpoint，而是 Zhishu 数据集和 norm stats。

### 当前第三阶段还没完成的部分

当前还没有完成的是：

1. 把真实 Zhishu rollout / teleop / scripted demo 整理成上面的 canonical raw layout
2. 跑 `convert_zhishu_data_to_lerobot.py` 生成稳定的 LeRobot 数据集
3. 运行 `compute_norm_stats`
4. 真正从 `pi05_base` 开始 fine-tuning
5. 用训练出的 checkpoint 起自己的 server

### 建议训练顺序

可直接参考的命令形态：

```bash
cd /root/gpufree-data/arcus/openpi
source .venv/bin/activate
export OPENPI_LOCAL_CHECKPOINT_ROOT=/root/gpufree-data/arcus/checkpoints
export OPENPI_CHECKPOINT_ROOTS=/root/gpufree-data/arcus/checkpoints
uv run scripts/compute_norm_stats.py --config-name pi05_zhishu_dualarm_nohand
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_zhishu_dualarm_nohand --exp-name=<exp_name> --overwrite
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_zhishu_dualarm_nohand \
  --policy.dir=<your_checkpoint_dir>
```

等 Zhishu checkpoint 真训练出来以后，Isaac 侧应切回长期 `zhishu14` contract：

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python -u demos/run_dualarm_with_openpi.py \
  --policy_mode websocket \
  --policy_host 127.0.0.1 \
  --policy_port 8000 \
  --policy_input_schema zhishu14 \
  --policy_output_contract zhishu14 \
  --enable_cameras \
  --headless
```

## 运行前检查

先确认当前 shell 已经进入这台机器上已验证可用的环境：

```bash
conda activate isaaclab
which python
python -c "import isaaclab, isaacsim; print('env ok')"
```

如果 `which python` 不是：

```bash
/opt/conda/envs/isaaclab/bin/python
```

就不要继续执行下面的命令。

当前版本完成的内容：

- 机器人 USD 资产路径和导入脚本
- 双臂 14 维关节动作骨架
- 桌面 + cube + target zone 场景
- 1 路外部相机 + 2 路腕部相机
- `DirectRLEnv` 风格环境接口
- `obs_builder` / `action_adapter` / policy bridge 预留接口

## 历史阶段记录（已过期：当时 README 还在描述“尚未接入 policy”）

这一节保留，是为了记录项目最初阶段的判断和推进顺序。

如果你只关心当前状态，请优先看前面的“当前状态快照”和后面的“π0.5 / Websocket 接入”。

## 历史进展总结（已过期）

如果要给架构师一个一句话版本，可以这样说：

> 我们已经完成了双臂桌面仿真底座，机器人资产、三路相机、14 维双臂控制、标准 env 接口和 policy bridge 占位接口都已经跑通；现在缺的是正式 policy 接入、末端执行器能力和任务层智能。

更细一点的状态如下。

### 已完成

1. 机器人资产接入

- 已从 URDF 成功离线导入 USD
- 环境默认从项目内 USD 加载，不会每次运行时临时从 URDF 在线导入
- 双臂 14 个关节已单独作为动作空间
- 头、腰、轮子不参与动作空间，保持锁定

2. 最小桌面场景

- 已有地面、桌子、操作物体、目标区域
- reset 已可稳定执行
- 已验证环境能创建、能 reset、能 step

3. 相机链路

- 1 路外部相机
- 2 路腕部相机
- 三路都已实际出图
- 当前统一输出 `HWC`、`uint8`、RGB

4. 环境接口

- 已实现 `DirectRLEnv` 风格的 `_setup_scene`
- 已实现 `_pre_physics_step`
- 已实现 `_apply_action`
- 已实现 `_get_observations`
- 已实现 `_get_rewards`
- 已实现 `_get_dones`
- 已实现 `_reset_idx`

5. 动作与观测结构

- 已有 `action_adapter.py`，用于把外部 action 向量映射成 joint target
- 已有 `obs_builder.py`，用于统一构建 policy-facing observation dict
- 已输出 joint state、last action、TCP pose、object pose、target pose、三路图像

6. policy bridge 预留

- 已有 `get_policy_input()`
- 已有 `apply_policy_output()`
- 已有 `replan_steps`
- 已有 `action_plan_buffer`

### 当时未完成（已过期）

1. 还没有接 π0.5

- 当前没有 websocket client
- 当前没有 obs -> policy request -> action chunk -> env action 的真实网络链路
- 当前没有任何 learned policy 或 inference policy

2. 还没有真实末端执行器能力

- 目前是双臂法兰 / 末端 link 可控
- 还没有真实灵巧手
- 也还没有正式 proxy gripper 的抓取闭环

3. 还没有任务智能

- 这一条在第二阶段开始前是成立的
- 现在 reward / done 已经收缩到“无手双臂靠近 / 聚拢 / 推动”语义
- 但它仍然只是环境任务定义，不代表 learned policy 已经具备任务智能

4. 还没有训练和 benchmark

- 没有 RL 训练流程
- 没有多任务 benchmark
- 没有 domain randomization
- 没有 sim2real

### 当时这套系统能做什么（历史阶段）

- 把智书机器人上半身双臂加载到 Isaac Lab
- 在桌面场景里稳定 reset / step
- 接受 14 维 joint action 驱动双臂
- 输出三路图像和低维状态
- 作为后续 policy server 接入的仿真执行端

### 当时这套系统还不能做什么（历史阶段）

- 不能自主完成桌面操作任务
- 不能执行 π0.5 推理
- 不能做真实抓取
- 不能代表最终任务能力

## 当前架构

当前项目建议按下面的理解去看：

1. `assets/robots/zhishu_robot/`

- 这里负责“机器人身体本体”
- `robot_cfg.py` 负责 Isaac Lab articulation 配置
- `usd/` 目录负责持有离线导入好的机器人 USD

2. `tasks/dualarm_tabletop/`

- 这里负责“任务层和环境层”
- `constants.py` 放核心常量
- `scene_cfg.py` 定义场景有哪些实体
- `cameras.py` 定义三路相机
- `objects.py` 定义桌子、物体、目标区域
- `env_cfg.py` 定义 DirectRLEnv 配置
- `env.py` 定义真正的环境生命周期

3. `utils/`

- 这里负责“接口适配层”
- `action_adapter.py` 面向未来 policy 输出格式变化
- `obs_builder.py` 面向未来 policy 输入格式变化
- `tcp_frames.py` 面向未来 TCP 偏移和末端扩展

4. `demos/`

- 这里负责“可视化验收和最小人工驱动”
- 当前 demo 不是 policy
- 当前 demo 只是手写动作，目的是验证双臂能动、环境能跑

## 目录

- `source/zhishu_dualarm_lab/assets/robots/zhishu_robot/`
  机器人 USD 配置和资产目录。
- `source/zhishu_dualarm_lab/tasks/dualarm_tabletop/`
  桌面双臂环境配置与实现。
- `source/zhishu_dualarm_lab/utils/`
  观测构建、动作映射、TCP frame 工具。
- `scripts/import_zhishu_robot_usd.py`
  一次性 URDF -> USD 导入脚本。

## 安装

在已激活的 `isaaclab` 环境里安装本项目：

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python -m pip install -e .
```

这里使用的是 external project 方式。

也就是说：

- 你的业务代码不堆在 `IsaacLab/source` 里
- `IsaacLab` 作为底层框架保留
- `zhishu_dualarm_lab` 作为你自己的独立项目安装进当前 Python 环境

## 导入机器人 USD

机器人原始 URDF：

```bash
/root/gpufree-data/arcus/zhishu_robot_description-URDF/zhishu_robot_description/urdf/zhishu_robot_description.urdf
```

将其一次性导入到项目资产目录：

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/import_zhishu_robot_usd.py --headless
```

导入后的默认 USD 路径：

```bash
/root/gpufree-data/arcus/zhishu_dualarm_lab/source/zhishu_dualarm_lab/assets/robots/zhishu_robot/usd/zhishu_robot.usd
```

说明：

- 当前运行时默认直接用这个 USD
- 这样做的目的，是把“机器人导入问题”和“任务环境问题”分开
- 后续如果机器人资产重新导出，只需要替换这个 USD 或重新跑导入脚本

## 运行 demo

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python demos/run_dualarm_tabletop.py --enable_cameras
```

如果要纯离屏渲染：

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python demos/run_dualarm_tabletop.py --enable_cameras --headless
```

推荐额外参数：

```bash
python demos/run_dualarm_tabletop.py --enable_cameras --motion-scale 1.0
```

如果想让手写动作更明显，可以调大：

```bash
python demos/run_dualarm_tabletop.py --enable_cameras --motion-scale 1.5
```

## 当前 action space

当前默认是 14 维关节增量动作：

- `left_joint1` 到 `left_joint7`
- `right_joint1` 到 `right_joint7`

动作范围按 `[-1, 1]` 归一化后，经 `action_adapter.py` 映射为关节目标增量。

更具体地说：

- action 不是末端位姿控制
- action 也不是 torque
- action 是“每一步给 14 个双臂关节一个归一化增量”
- 再由 `action_adapter.py` 转成下一拍 joint target

这意味着当前版本非常适合：

- 先接 policy server
- 先验证 obs / action 链路
- 先做最小行为闭环

但它还不是最终任务控制方式的终态。

## 当前 observation keys

当前环境返回的观测 dict 主要包含：

- `prompt`
- `observation/external_image`
- `observation/left_wrist_image`
- `observation/right_wrist_image`
- `observation/joint_pos`
- `observation/joint_vel`
- `observation/last_action`
- `observation/left_tcp_pose`
- `observation/right_tcp_pose`
- `observation/object_pose`
- `observation/target_pose`
- `observation/state`

图像格式固定为 `HWC`，`uint8`，RGB 三通道。

当前 `observation/state` 是一个打平后的低维状态向量，包含：

- `joint_pos`
- `joint_vel`
- `last_action`
- `left_tcp_pose`
- `right_tcp_pose`
- `object_pose`
- `target_pose`

它的目标是：

- 先让 websocket policy 或调试脚本有一个稳定低维入口
- 后续如果要换成更正式的 `observation/state` 结构，也只需要集中改 `obs_builder.py`

## websocket policy 预留位置（历史设计说明，现已基本落地）

这部分是接入前写下的设计意图。当前这些位置已经基本补齐，并实际用于 demo 闭环。

当时预期主要补这几个位置：

- `ZhishuDualArmTabletopEnv.get_policy_input()`
- `ZhishuDualArmTabletopEnv.apply_policy_output()`
- `ZhishuDualArmTabletopEnv.action_plan_buffer`
- `source/zhishu_dualarm_lab/utils/action_adapter.py`

当时预期链路：

```text
obs dict -> websocket client -> action chunk / action vec -> apply_policy_output -> env.step
```

## π0.5 / Websocket 接入

本项目当前已经具备“最小可行策略接入”骨架，目标是打通链路，而不是验证任务效果。

当前数据流固定为：

```text
env observation
-> get_policy_input()
-> websocket client
-> openpi policy server
-> action chunk
-> apply_policy_output()
-> action_plan_buffer
-> consume_action_plan_step()
-> env.step()
```

### 第一版 policy input contract

当前真正发给 policy server 的字段固定为：

```python
{
  "prompt": str,
  "observation/external_image": HWC uint8 RGB,
  "observation/left_wrist_image": HWC uint8 RGB,
  "observation/right_wrist_image": HWC uint8 RGB,
  "observation/state": np.ndarray[float32],
}
```

其中：

- 三路图像全部固定为 `HWC uint8 RGB`
- `observation/state` 的顺序固定为：
  `joint_pos + joint_vel + last_action + left_tcp_pose + right_tcp_pose + object_pose + target_pose`

### 第一版 action output contract

当前环境只接受 14 维双臂 joint delta 动作：

- 支持单步向量：`[14]`
- 支持动作块：`[T, 14]`

如果返回维度不是 14，会直接报错，不做静默截断。

### replan / action buffer 机制

当前环境的策略动作消费逻辑是：

1. 如果 `action_plan_buffer` 为空，则发起一次 websocket 推理请求
2. 拿到返回的 action chunk 后，最多保留前 `replan_steps` 步
3. 每个 env step 消费一拍动作
4. buffer 用尽后再次请求 server

### 当前这轮接入实际新增文件

- `source/zhishu_dualarm_lab/utils/policy_client.py`
  websocket client 和 fake client
- `source/zhishu_dualarm_lab/utils/msgpack_numpy.py`
  与 openpi websocket 协议兼容的 numpy msgpack 序列化
- `source/zhishu_dualarm_lab/demos/run_dualarm_with_openpi.py`
  policy bridge demo
- `demos/run_dualarm_with_openpi.py`
  根目录便捷入口

### fake client 运行命令

先只验证 action buffer / replan / env action 链路：

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python demos/run_dualarm_with_openpi.py --policy_mode fake --enable_cameras --replan_steps 8
```

### fake websocket server 运行命令

如果你要连通真实 websocket 收发，但又不想先依赖真实 π0.5 checkpoint，可以先启动一个 openpi-compatible fake server：

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python scripts/serve_fake_policy.py --host 127.0.0.1 --port 8000 --chunk_length 8
```

然后再跑 websocket client demo：

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python demos/run_dualarm_with_openpi.py \
  --policy_mode websocket \
  --policy_host 127.0.0.1 \
  --policy_port 8000 \
  --enable_cameras \
  --replan_steps 8
```

### 真实 websocket client 运行命令

先启动 openpi policy server，再运行 Isaac 侧 client demo。

启动 server 的推荐命令是：

```bash
cd /root/gpufree-data/arcus/openpi
source .venv/bin/activate
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero
```

如果你想使用 openpi 默认环境 policy，也可以使用：

```bash
cd /root/gpufree-data/arcus/openpi
source .venv/bin/activate
uv run scripts/serve_policy.py --env LIBERO --port 8000
```

然后在 Isaac 侧连接：

```bash
conda activate isaaclab
cd /root/gpufree-data/arcus/zhishu_dualarm_lab
python demos/run_dualarm_with_openpi.py \
  --policy_mode websocket \
  --policy_host 127.0.0.1 \
  --policy_port 8000 \
  --policy_input_schema libero \
  --policy_output_contract libero7 \
  --enable_cameras \
  --headless \
  --max_steps 8 \
  --replan_steps 3 \
  --debug_print_every 1
```

说明：

- `pi05_libero` 不是为当前 Zhishu 双臂环境训练的
- 当前这台机器上，external project 一律优先用激活环境后的 `python`，不要默认 `isaaclab.sh -p`
- 它期望的输入 schema 是 LIBERO 风格：
  `observation/image`、`observation/wrist_image`、`observation/state(8)`
- 它原始输出也是 LIBERO 风格 7 维动作
- 因此当前 demo 在 client / adapter 层做了一个“只为打通链路”的兼容映射：
  - 外部相机 -> `observation/image`
  - 左腕相机 -> `observation/wrist_image`
  - `observation/state` 截前 8 维
  - 真实 7 维动作 -> 当前 14 维双臂动作 contract

这个兼容层的目标只是让真实 openpi server 可以替换 fake server 完成端到端收发，不代表控制语义已经正确。

补充说明：

- 当前 `libero` 输入适配逻辑实现在 `source/zhishu_dualarm_lab/utils/policy_client.py`
- 当前 `libero7 -> 14` 输出适配逻辑实现在 `source/zhishu_dualarm_lab/utils/action_adapter.py`
- env 内部的真实消费入口仍然是 `apply_policy_output()` 和 `consume_action_plan_step()`
- 因此目前“网络协议兼容”和“机器人控制语义”仍是两个分开的层次

### 调试日志

当前 demo 会周期性打印：

- policy input keys
- 三路图像的 shape / dtype
- `observation/state` 的 shape / dtype
- 返回 action chunk shape
- 实际消费的 action shape
- 当前 buffer 剩余长度
- step reward / done

如果失败，报错会尽量归到下面这些类别：

- websocket connection failure
- policy timeout
- policy response error
- env action buffer empty

## 已知运行时注意事项

1. 当前机器人 USD 是从原始 URDF 导入来的

- 原始 URDF 的 mesh 引用里有一个文件名问题
- 我们已经在本机上用一个非破坏性 symlink 兜住了导入
- 如果你以后换一台机器重新导入，最好顺手把原始资产命名问题一起修掉

2. 当前 scene 默认关闭了 `clone_in_fabric` 和 `replicate_physics`

- 这是为了兼容当前导入出来的机器人 USD
- 这一版目标是先稳定单环境、稳定可运行
- 不是为了做并行大规模训练

3. 当前 demo 是“验收脚本”，不是“智能策略”

- demo 的动作是手写正弦
- 它的目标只是让你看见双臂可控
- 不代表 policy 控制效果

## 推荐的下一步

建议按下面顺序推进第二阶段剩余工作：

1. 整理 schema / adapter 边界

- 把当前 `libero` 输入适配和 `libero7` 输出适配从 client 中继续拆清楚
- 明确哪些属于“协议兼容”，哪些属于“控制语义映射”
- 避免临时兼容层继续膨胀成隐式约定

2. 明确 action contract

- 继续使用 14 维 joint delta
  或
- 改成更接近 π0.5 期望的 action schema

3. 把长期主线接到 `pi05_base`

- 在 `openpi` 里新增 `zhishu_policy.py`
- 新增 `pi05_zhishu_dualarm_nohand` config
- 让 server 直接吃本地 3 路图像 + 70D state + 14D 动作 contract

4. 准备 Zhishu 自定义数据

- 转成 LeRobot 格式
- 先做 `compute_norm_stats`
- 再跑第一版 fine-tuning
