# Lift-PH 普通 State BC 训练与评估报告

## 1. 阶段目标

本报告记录 robomimic v0.4.0 中 Lift-PH low-dim 普通
State BC 的训练、故障恢复、checkpoint 选择、独立评估和官方结果对比。

本阶段只训练普通 State BC，不训练 State BC-RNN。

```text
Stage = 04
Task = Lift
DatasetType = PH
ObservationType = low_dim
PolicyFamily = BC
RuntimePolicyClass = BC_GMM
```

## 2. 基础环境

```text
Repository = /home/lx/robomimic
Branch = lift-ph-reproduction
Stage03Commit = f3d9f6d41134980cc410b093cf79eee844239635
Python = 3.11.16
PythonExecutable = /home/lx/miniforge3/envs/robot-il/bin/python
Torch = 2.7.1+cu128
GPU = NVIDIA GeForce RTX 5060 Laptop GPU
SystemMemory = 14 GiB
Swap = 0
```

训练前确认 Git 工作区干净，CUDA 计算测试通过，可用磁盘约 204 GiB。

## 3. 官方配置与数据

官方配置：

```text
reproduction/lift_ph/configs/state_bc_official.json
```

SHA256：

```text
96c81356b0ccb562f0e8f737d1930201bd27c9895831d2419046fc495acc14b9
```

数据集：

```text
/home/lx/robomimic/datasets/lift/ph/low_dim_v15.hdf5
```

数据划分：

```text
TrainTrajectories = 180
ValidationTrajectories = 20
MaskOverlap = 0
```

观测输入：

| 字段 | 维度 |
|---|---:|
| `robot0_eef_pos` | 3 |
| `robot0_eef_quat` | 4 |
| `robot0_gripper_qpos` | 2 |
| `object` | 10 |
| 总输入维度 | 19 |

动作维度为 7。

## 4. 模型与训练参数

```text
algo_name = bc
RuntimePolicyClass = BC_GMM
rnn.enabled = False
gmm.enabled = True
gmm.num_modes = 5
actor_layer_dims = [1024, 1024]
batch_size = 100
num_epochs = 2000
steps_per_epoch = 100
maximum_gradient_updates = 200000
learning_rate = 1e-4
seed = 1
rollout_rate = 50 epochs
rollouts_per_evaluation = 50
rollout_horizon = 400
terminate_on_success = True
```

## 5. 第一次训练及故障

第一次运行目录：

```text
runs/lift_ph/training/core/bc/lift/ph/low_dim/trained_models/
core_bc_lift_ph_low_dim/20260915165026
```

第一次训练完成到 Epoch 400，但在开始生成
`Lift_epoch_400.mp4` 时中断：

```text
OSError: [Errno 12] Cannot allocate memory
```

故障位于 ImageIO 调用 FFmpeg 创建视频编码子进程的阶段，
不是模型前向传播、反向传播、数据集或 CUDA 错误。

内存记录显示：

```text
Epoch 24 Memory Usage = 2448 MB
Epoch 100 Memory Usage = 3495 MB
Epoch 400 Memory Usage = 5440 MB
```

系统没有 swap。重复 rollout 视频渲染期间内存逐渐增长，
最终使 FFmpeg 无法创建子进程。

第一次运行保留了 Epoch 50 至 Epoch 350 的 checkpoint，
其中 Epoch 300 的 rollout 成功率达到 0.98。

## 6. 第一次恢复启动失败

第一份恢复配置仅关闭视频，但保留了原实验名称。
robomimic 检测到实验名称目录已存在，并要求交互确认覆盖。

由于训练通过 `nohup` 后台启动，标准输入不可用，出现：

```text
OSError: [Errno 9] Bad file descriptor
```

该次尝试在训练开始之前退出，没有生成新的训练目录，
也没有覆盖第一次运行。

不得通过输入 `y` 删除旧实验目录。

## 7. 最终恢复方案

从官方配置生成运行时副本：

```text
runs/lift_ph/runtime_configs/state_bc_official_no_video_retry1.json
```

SHA256：

```text
2c4fa3f7eb12ecf79b948fb16c6f860136a3156a7eac47711f63b1745a2fec42
```

相对于官方配置仅有两个运行基础设施差异：

```text
experiment.render_video:
True -> False

experiment.name:
core_bc_lift_ph_low_dim
-> core_bc_lift_ph_low_dim_no_video_retry1
```

模型结构、数据、优化器、训练步数和 rollout 评估参数均未修改。

新的实验名称避免旧目录覆盖；关闭训练期视频避免重复启动
FFmpeg。训练完成后再用独立进程为最佳 checkpoint 生成视频。

## 8. 正式恢复训练

成功运行目录：

```text
runs/lift_ph/training/core/bc/lift/ph/low_dim/trained_models/
core_bc_lift_ph_low_dim_no_video_retry1/20260915172631
```

完成状态：

```text
LatestTrainEpoch = 2000
FinishedRunSuccessfully = True
TracebackCount = 0
CannotAllocateMemoryCount = 0
VideoWriteRecordCount = 0
CheckpointCount = 40
GeneratedTrainingVideoCount = 0
FinalMemoryUsage = 2948 MB
MaximumMemoryUsage = 2970 MB
RunDirectorySize = 181 MB
```

第一次运行与恢复运行在前 7 个 rollout 检查点上的成功率完全一致：

```text
SharedRolloutCount = 7
SharedRolloutMetricsMatch = True
```

这证明关闭视频没有改变对应训练结果。

## 9. 全部训练期 rollout 结果

每次评估包含 50 条 rollout。

| Epoch | Success Rate | 成功次数 |
|---:|---:|---:|
| 50 | 0.28 | 14/50 |
| 100 | 0.76 | 38/50 |
| 150 | 0.94 | 47/50 |
| 200 | 0.92 | 46/50 |
| 250 | 0.96 | 48/50 |
| 300 | 0.98 | 49/50 |
| 350 | 0.94 | 47/50 |
| 400 | 0.88 | 44/50 |
| 450 | 0.96 | 48/50 |
| 500 | 0.98 | 49/50 |
| 550 | 0.96 | 48/50 |
| 600 | 1.00 | 50/50 |
| 650 | 0.94 | 47/50 |
| 700 | 0.96 | 48/50 |
| 750 | 0.92 | 46/50 |
| 800 | 1.00 | 50/50 |
| 850 | 0.92 | 46/50 |
| 900 | 0.88 | 44/50 |
| 950 | 0.96 | 48/50 |
| 1000 | 0.96 | 48/50 |
| 1050 | 0.98 | 49/50 |
| 1100 | 0.90 | 45/50 |
| 1150 | 0.96 | 48/50 |
| 1200 | 0.90 | 45/50 |
| 1250 | 0.94 | 47/50 |
| 1300 | 0.94 | 47/50 |
| 1350 | 0.96 | 48/50 |
| 1400 | 0.92 | 46/50 |
| 1450 | 0.88 | 44/50 |
| 1500 | 0.92 | 46/50 |
| 1550 | 0.86 | 43/50 |
| 1600 | 0.98 | 49/50 |
| 1650 | 0.92 | 46/50 |
| 1700 | 0.96 | 48/50 |
| 1750 | 0.92 | 46/50 |
| 1800 | 0.92 | 46/50 |
| 1850 | 0.96 | 48/50 |
| 1900 | 0.94 | 47/50 |
| 1950 | 0.98 | 49/50 |
| 2000 | 0.96 | 48/50 |

汇总：

```text
BestSuccessRate = 1.00
BestSuccessEpochs = [600, 800]
FinalSuccessRate = 0.96
MeanSuccessRateLast5 = 0.9520
```

## 10. 最佳 checkpoint 选择

选择规则：

1. 首先选择 rollout 成功率最高的 checkpoint；
2. 最高成功率相同时选择最早达到该结果的 checkpoint。

因此选择 Epoch 600：

```text
models/model_epoch_600_Lift_success_1.0.pth
```

SHA256：

```text
eda6c734695de403517d532b971402aa7ff938b36c375aece8d87ce730eb97f7
```

Epoch 800 同样达到 100%，但不是首次达到最高成功率。

最终 Epoch 2000 checkpoint 的成功率为 96%，不能仅因它是最后
一个 checkpoint 就将其选为最佳模型。

Epoch 2000 checkpoint SHA256：

```text
34fb0cd6b7e392961d9a3fdec7bf57c7babacbe117cb468776aef59ec323c145
```

## 11. 独立定量评估

对 Epoch 600 checkpoint 使用新的固定种子进行独立评估：

```text
EvaluationSeed = 20260915
EvaluationRollouts = 100
EvaluationHorizon = 400
Video = False
```

结果：

```text
Return = 0.99
AverageHorizon = 49.1
SuccessRate = 0.99
NumSuccess = 99
EvaluationExitCode = 0
TracebackDetected = False
```

评估日志：

```text
runs/lift_ph/evaluation/
state_bc_epoch600_seed20260915_100rollouts_20260915-190933.log
```

日志 SHA256：

```text
3872d2193d2de3b59e3f8ed0a1345a7d23907c853478eaff7c3ff5b70855265a
```

这次独立评估没有参与 checkpoint 选择，只用于检查策略在新的
固定随机种子下的表现。

## 12. 策略视频

使用 Epoch 600 checkpoint 对相同评估种子的前 5 条 rollout
单独生成 `agentview` 视频：

```text
Rollouts = 5
Successes = 5
SuccessRate = 1.00
AverageHorizon = 44.2
VideoExitCode = 0
TracebackDetected = False
```

视频路径：

```text
runs/lift_ph/evaluation/videos/
state_bc_epoch600_seed20260915_first5.mp4
```

视频大小约 120 KB，SHA256：

```text
cf00e620005d60e4c20d198b9915281e227ab24545b13f42f2a1bd4cf0c02465
```

视频在训练完成后由新的独立进程生成，没有再次出现内存错误。

## 13. 官方结果对比

robomimic 论文对 Lift-PH low-dim 普通 BC 报告：

```text
OfficialSuccessRate = 100.0 ± 0.0 percent
OfficialTrainingSeeds = 3
```

官方统计协议是：

```text
TrainingEpochs = 2000
GradientStepsPerEpoch = 100
EvaluationInterval = 50 epochs
RolloutsPerEvaluation = 50
ReportedMetric = 每个训练运行期间的最高成功率
FinalAggregation = 3个训练种子的平均值和标准差
```

本次复现结果：

| 对比项 | 官方结果 | 本次结果 |
|---|---:|---:|
| 单个训练运行的最高成功率 | 100% | 100% |
| 独立附加评估 | 未报告 | 99/100 |
| 完整训练种子数量 | 3 | 1 |

结论：

```text
SingleSeedOfficialProtocolMatch = PASS
FullThreeSeedReproduction = NOT_PERFORMED
```

本次官方 seed=1 配置的单次训练结果达到官方单次运行目标，
但不能将一次训练表述为已经复现论文的三种子均值和标准差。

官方来源：

- https://arxiv.org/html/2108.03298v2
- https://github.com/ARISE-Initiative/robomimic/blob/v0.4.0/robomimic/scripts/generate_paper_configs.py

## 14. 损失与成功率现象

训练后期的训练 Log Likelihood 为正，Loss 为负。
这不表示“负概率”：连续概率密度可以大于 1，
因此其对数可以为正，负对数似然可以为负。

验证负对数似然达到数百万，但 rollout 成功率可达到 100%。
GMM 的极小标准差会放大验证动作偏差，同时离线动作拟合损失
与闭环任务成功率本来就是不同目标。

因此不能只根据最低验证损失选择机器人策略，必须实际执行
rollout。此次最佳 checkpoint 也确实通过 rollout 成功率选择。

## 15. 非阻断警告

训练和评估中出现：

```text
No private macro file found
Could not import robosuite_models
```

当前 Panda Lift 环境、BC_GMM 策略、CUDA 和 EGL 均成功初始化，
这些警告没有阻断本项目，因此没有进行额外安装。

## 16. 关键产物

```text
OfficialConfigSHA256 =
96c81356b0ccb562f0e8f737d1930201bd27c9895831d2419046fc495acc14b9

RuntimeConfigSHA256 =
2c4fa3f7eb12ecf79b948fb16c6f860136a3156a7eac47711f63b1745a2fec42

BestCheckpointSHA256 =
eda6c734695de403517d532b971402aa7ff938b36c375aece8d87ce730eb97f7

FinalCheckpointSHA256 =
34fb0cd6b7e392961d9a3fdec7bf57c7babacbe117cb468776aef59ec323c145

IndependentEvaluationLogSHA256 =
3872d2193d2de3b59e3f8ed0a1345a7d23907c853478eaff7c3ff5b70855265a

PolicyVideoSHA256 =
cf00e620005d60e4c20d198b9915281e227ab24545b13f42f2a1bd4cf0c02465
```

`runs/` 中的 checkpoint、运行日志和视频不进入 Git。
Git 只记录可复现步骤、配置来源、结果摘要和产物哈希。

## 17. 当前阶段状态

```text
Stage04Preflight = PASS
Stage04FormalTrainingCompletion = PASS
Stage04BestCheckpointSelection = PASS
Stage04IndependentEvaluation = PASS
Stage04PolicyVideo = PASS
Stage04OfficialSingleSeedComparison = PASS
Stage04FullThreeSeedReproduction = NOT_PERFORMED
Stage04EngineeringStatus = PASS
Stage04DocumentationDraft = PASS
Stage04GitAcceptance = NOT_YET_VERIFIED
Stage04UnderstandingStatus = NOT_YET_VERIFIED
Stage04OverallStatus = IN_PROGRESS
```

阶段 4 只有在报告通过 Git 验收，并完成独立理解问答后，
才能标记整体完成。
