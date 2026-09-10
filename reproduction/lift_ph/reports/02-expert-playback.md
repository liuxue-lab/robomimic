# Lift-PH 阶段 2：专家轨迹回放报告

## 1. 实验目标

本阶段验证 Lift-PH 专家示范能否在当前 robomimic、robosuite 和 MuJoCo 环境中被正确恢复、渲染和重新执行。

本阶段不训练神经网络。

验收顺序为：

1. 状态回放；
2. 动作回放；
3. 状态回放与动作回放对比；
4. 工程验收和理解验收。

## 2. 实验环境

- 日期：2026-09-08
- 操作系统：Ubuntu 22.04
- Conda 环境：`robot-il`
- Python：3.11.16
- robomimic：0.4.0
- robomimic 基线 commit：`7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9`
- robosuite：1.5.1
- robosuite commit：`a071383d53568ab798eb315c0e95357911be922d`
- MuJoCo：3.2.7
- 当前分支：`lift-ph-reproduction`
- 阶段 2 开始时 HEAD：`8e8e253`

## 3. 数据集

- 任务：Lift
- 数据来源：PH（Proficient Human）
- 机器人：Panda
- 数据类型：low-dim
- 数据集版本：v1.5
- 控制频率：20 Hz
- 轨迹数：200
- transition 数：9666
- 动作维数：7
- State BC 观测维数：19

数据集路径：

```text
datasets/lift/ph/low_dim_v15.hdf5
```

SHA256：

```text
2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540
```

阶段 1 数据审计结论为 `PASS`。

## 4. 状态回放

### 4.1 回放原理

状态回放读取 HDF5 中保存的 `states[t]`，逐帧恢复 MuJoCo 仿真状态，再通过虚拟相机重新渲染画面。

状态回放没有重新执行专家动作。

### 4.2 执行命令

```bash
set -o pipefail

MUJOCO_GL=egl python -m robomimic.scripts.playback_dataset \
  --dataset datasets/lift/ph/low_dim_v15.hdf5 \
  --render_image_names agentview robot0_eye_in_hand \
  --video_path runs/lift_ph/playback/lift_ph_state_first5.mp4 \
  --n 5 \
  2>&1 | tee runs/lift_ph/playback/state-playback-first5.log

state_playback_exit=${PIPESTATUS[0]}
echo "StatePlaybackExit=${state_playback_exit}"
test "${state_playback_exit}" -eq 0
```

### 4.3 程序证据

```text
Created environment with name Lift
Action size is 7
Playing back episode: demo_0
Playing back episode: demo_1
Playing back episode: demo_2
Playing back episode: demo_3
Playing back episode: demo_4
StatePlaybackExit=0
```

### 4.4 视频证据

- 视频路径：`runs/lift_ph/playback/lift_ph_state_first5.mp4`
- 文件格式：ISO Media、MP4 Base Media v1
- 文件大小：278293 bytes
- 视频帧率：20 FPS
- 视频帧数：58
- 视频时长：2.9 秒
- 相机：`agentview`、`robot0_eye_in_hand`
- Git 忽略规则：`.gitignore:129:/runs/`

### 4.5 人工观察

- 视频能够正常打开；
- 两个相机画面均正常；
- `agentview` 能看到 Panda、桌面和方块；
- `robot0_eye_in_hand` 随机器人末端移动；
- Panda、桌面和方块显示正常；
- 五条轨迹均完成接近、抓取和抬升；
- 成功轨迹数为 5/5；
- 未发现黑屏；
- 未发现轨迹内部的严重瞬移；
- 未发现穿模、异常闪烁或画面损坏；
- 不同轨迹之间的场景复位属于正常 episode 边界。

状态回放结论：`PASS`。

## 5. 动作回放

### 5.1 回放原理

动作回放只恢复轨迹初始状态，随后依次执行数据集中的 `actions[t]`，由控制器和 MuJoCo 产生后续状态。

动作回放属于开环重演，不会在每个时间步强制恢复到数据集记录状态。

### 5.2 执行命令

```bash
set -o pipefail

MUJOCO_GL=egl python -m robomimic.scripts.playback_dataset \
  --dataset datasets/lift/ph/low_dim_v15.hdf5 \
  --use-actions \
  --render_image_names agentview robot0_eye_in_hand \
  --video_path runs/lift_ph/playback/lift_ph_actions_first5.mp4 \
  --n 5 \
  2>&1 | tee runs/lift_ph/playback/action-playback-first5.log

action_playback_exit=${PIPESTATUS[0]}
echo "ActionPlaybackExit=${action_playback_exit}"
test "${action_playback_exit}" -eq 0
```

### 5.3 程序证据

```text
Created environment with name Lift
Action size is 7
Playing back episode: demo_0
Playing back episode: demo_1
Playing back episode: demo_2
Playing back episode: demo_3
Playing back episode: demo_4
ActionPlaybackExit=0
```

### 5.4 状态偏差提示

动作回放过程中，脚本报告了重新执行动作产生的状态与记录状态之间的偏差。

各轨迹观察到的最大偏差约为：

- `demo_0`：0.183，step 47；
- `demo_1`：0.459，step 49；
- `demo_2`：1.048，step 46；
- `demo_3`：0.542，step 47；
- `demo_4`：1.826，step 43。

多数前期时间步偏差较小，较大峰值主要出现在轨迹后半段，可能与夹爪接触、抓取和抬升阶段的接触动力学有关。

这些数值来自混合了关节状态、物体状态等分量的仿真状态向量，不能直接解释为米或弧度。

最终是否成功必须结合视频中的任务行为判断，不能只根据单个偏差数值判断。

### 5.5 视频证据

- 视频路径：`runs/lift_ph/playback/lift_ph_actions_first5.mp4`
- 文件格式：ISO Media、MP4 Base Media v1
- 文件大小：278165 bytes
- 视频帧率：20 FPS
- 视频帧数：58
- 视频时长：2.9 秒
- 相机：`agentview`、`robot0_eye_in_hand`
- Git 忽略规则：`.gitignore:129:/runs/`

### 5.6 人工观察

- 视频能够正常打开；
- 两个相机画面均正常；
- 机器人运动方向合理；
- 夹爪在接近方块后闭合；
- 五条轨迹均抓住方块并完成抬升；
- 成功轨迹数为 5/5；
- 未发现方块中途掉落；
- 未发现明显抖动；
- 未发现运动方向异常；
- 与状态回放总体接近；
- 动作回放在主观观察上略快。

动作回放结论：`PASS`。

## 6. 两种回放的对比

两个视频的客观编码参数相同：

- 帧率均为 20 FPS；
- 帧数均为 58；
- 时长均为 2.9 秒；
- 文件大小仅有轻微差异。

因此，“动作回放略快”不是播放器速度或视频总时长不同，而是主观运动观感或重新执行动作后产生的细微轨迹差异。

虽然动作回放出现状态偏差，但五条轨迹仍全部完成抓取和抬升，没有出现任务语义失败。

## 7. WARNING 判断

### 7.1 robosuite_models

日志出现：

```text
Could not import robosuite_models. Some robots may not be available.
```

当前任务使用标准 Panda Lift。环境已成功创建，两个相机正常，状态回放和动作回放均成功。因此该 WARNING 不阻塞，不需要安装 `robosuite_models`。

### 7.2 robomimic private macro

日志出现：

```text
No private macro file found!
```

该提示未影响环境创建、回放、视频生成或退出码，因此当前不阻塞，不需要执行 `setup_macros.py`。

### 7.3 playback diverged

动作回放中的 `playback diverged` 表明重新执行动作后产生的状态与记录状态不完全一致。

动作回放依赖控制器、物理参数、接触动力学和数值积分，细微差异可能逐步积累。本次五条轨迹均保持任务成功，因此记录该差异，但不判定为失败。

## 8. 理解验收

### 8.1 low-dim 数据为什么仍能生成视频

low-dim 数据集没有保存 RGB observation，但保存了环境元数据和 MuJoCo 状态。回放时可以恢复仿真场景，再通过虚拟相机重新渲染 RGB 视频。

### 8.2 状态回放与动作回放的区别

状态回放在每个时间步写入记录的 `states[t]`。

动作回放只恢复初始状态，随后连续执行 `actions[t]`，让控制器和物理引擎产生后续状态。

### 8.3 为什么先进行状态回放

状态回放先验证数据状态、环境重建和相机渲染。只有这些基础链路正常，才有必要进一步检查控制器和动作重演。

### 8.4 为什么偏差提示不等于任务失败

偏差数值表示仿真状态不完全一致，不等同于成功率，也没有单一的物理单位。必须结合运动方向、抓取过程、方块是否抬升以及是否出现发散进行判断。

## 9. 工程验收

- [x] 使用 `robot-il` 环境；
- [x] 仓库为 `/home/lx/robomimic`；
- [x] 分支为 `lift-ph-reproduction`；
- [x] 状态回放退出码为 0；
- [x] 状态回放 MP4 存在且可播放；
- [x] 两个相机画面正常；
- [x] 五条状态轨迹均完成抓取抬升；
- [x] 动作回放退出码为 0；
- [x] 动作回放 MP4 存在且可播放；
- [x] 五条动作轨迹均完成抓取抬升；
- [x] 状态回放与动作回放差异得到解释；
- [x] 视频和日志保存在 `runs/`；
- [x] 原始运行产物被 Git 忽略；
- [x] 回放没有产生意外 Git 变更。

## 10. 最终结论

```text
StatePlaybackStatus = PASS
ActionPlaybackStatus = PASS
StatePlaybackSuccess = 5/5
ActionPlaybackSuccess = 5/5
PlaybackArtifactsIgnoredByGit = True
Stage02Status = PASS
```

Lift-PH 专家示范能够在当前 robomimic、robosuite 和 MuJoCo 环境中正确恢复、渲染和重新执行。

完成报告检查和独立 Git 提交后，允许进入阶段 3：官方 State BC 配置准备与审计。
