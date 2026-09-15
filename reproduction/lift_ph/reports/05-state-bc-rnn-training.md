# Lift-PH State BC-RNN 训练与评估报告

## 1. 阶段目标与结果

本报告记录 robomimic v0.4.0 中 Lift-PH low-dim State BC-RNN 的配置核对、
序列数据审计、前向计算检查、正式训练、checkpoint 选择、独立评估和视频验收。

本次使用一个训练种子 `seed=1`，运行时策略类为 `BC_RNN_GMM`。
2000 轮训练和验证完整结束，共完成 40 次训练期 rollout 评估，保存 40 个非空 checkpoint。
按“训练期成功率最高、并列取最早 epoch”选择 epoch 300。
冻结该 checkpoint 后，独立评估成功 100/100；另行录制的 5 次 rollout 成功 5/5。

| 项目 | 结果 |
| --- | --- |
| 正式训练轮数 | 2000 |
| 训练期最佳成功率 | 1.00，50/50 |
| 选定 checkpoint | epoch 300 |
| 独立评估 | 100/100，成功率 1.00 |
| 独立评估平均步数 | 42.82 |
| 视频 rollout | 5/5，成功率 1.00 |
| 视频 rollout 平均步数 | 40.6 |
| 本地视频人工检查 | 用户确认正常 |

这里的 100% 是有限次评估的样本成功率，不表示真实成功概率必然为 100%。
本阶段没有完成三个训练种子的统计复现；BC 与 BC-RNN 的正式比较留给阶段 6。

## 2. 环境与代码基线

```text
Repository = /home/lx/robomimic
Branch = lift-ph-reproduction
Stage04BaselineHEAD = 323e2bf3901adc53b82bb4d8d6f73a4404948fd1
TrainingLaunchHEAD = 323e2bf3901adc53b82bb4d8d6f73a4404948fd1
Origin = git@github.com:liuxue-lab/robomimic.git
Upstream = https://github.com/ARISE-Initiative/robomimic.git
PythonExecutable = /home/lx/miniforge3/envs/robot-il/bin/python
PythonVersion = 3.11.16
RobomimicVersion = v0.4.0
RobosuiteVersion = 1.5.1
MuJoCoVersion = 3.2.7
TorchVersion = 2.7.1+cu128
TorchCUDAVersion = 12.8
GPU = NVIDIA GeForce RTX 5060 Laptop GPU
SystemMemory = 14 GiB
Swap = 0
MUJOCO_GL = egl
CUDA_VISIBLE_DEVICES = 0
```

阶段开始时确认分支正确、工作区干净、本地 HEAD 与 GitHub 分支 HEAD 一致。
Python 和 robomimic 导入路径正确，CUDA 计算检查通过。
预检时系统可用内存约 9.5 GiB、可用磁盘约 202.06 GiB，无已有训练进程。
启动训练前再次确认工作区干净，记录了上述启动 HEAD。

## 3. 官方配置与对照条件

官方配置：

```text
reproduction/lift_ph/configs/state_bc_rnn_official.json
SHA256 = 06c9380989e5d449038bc0c5a1a5789eb546961c7380ed4ece8217b985bd97c4
```

与阶段 4 的 `state_bc_official.json` 进行完整 JSON 比较，只有以下五处不同：

| 字段 | State BC | State BC-RNN |
| --- | --- | --- |
| `algo.actor_layer_dims` | `[1024, 1024]` | `[]` |
| `algo.rnn.enabled` | `false` | `true` |
| `train.seq_length` | `1` | `10` |
| `experiment.name` | `core_bc_lift_ph_low_dim` | `core_bc_rnn_lift_ph_low_dim` |
| `train.output_dir` | `core/bc/.../trained_models` | `core/bc_rnn/.../trained_models` |

数据、划分、观测键、batch size、优化器、学习率、训练轮数、每轮更新数、
GMM 配置、保存规则和训练期 rollout 协议保持一致。
这属于对官方 BC 与 BC-RNN 设置的比较；序列长度和 actor 结构也是方法差异的一部分。

```text
Algorithm = bc
RuntimeAlgorithmClass = BC_RNN_GMM
PolicyNetworkClass = RNNGMMActorNetwork
ModelParameters = 1986875
ObservationDimension = 19
ActionDimension = 7
GMMEnabled = True
GMMNumModes = 5
GMMLowNoiseEval = True
ActorLayerDims = []
SequenceLength = 10
RNNHorizon = 10
RNNType = LSTM
RNNHiddenDim = 400
RNNNumLayers = 2
RNNBidirectional = False
RNNOpenLoop = False
BatchSize = 100
TrainingSeed = 1
NumEpochs = 2000
TrainingStepsPerEpoch = 100
ValidationStepsPerEpoch = 10
LearningRate = 0.0001
NumDataWorkers = 0
HDF5NormalizeObs = False
RolloutWarmstart = 0
RolloutRate = 50 epochs
RolloutsPerEvaluation = 50
RolloutHorizon = 400
TerminateOnSuccess = True
SaveEveryNEpochs = 50
SaveOnBestRolloutSuccessRate = True
SaveOnBestValidation = False
```

按配置计算，2000 轮、每轮 100 次训练更新，对应 200000 次参数更新。
这里的 epoch 由固定更新次数定义，不等于完整遍历一次全部数据。

## 4. 数据与序列样本审计

```text
Dataset = /home/lx/robomimic/datasets/lift/ph/low_dim_v15.hdf5
DatasetBytes = 21084088
TrajectoryCount = 200
TransitionCount = 9666
TrainTrajectories = 180
ValidTrajectories = 20
TrainTransitions = 8640
ValidTransitions = 1026
TrajectoryLengthMin = 36
TrajectoryLengthMax = 64
MaskDuplicateCount = 0
TrainValidOverlap = 0
MasksCoverAllTrajectories = True
FrameStack = 1
PadSequenceLength = True
TrainSequenceSamples = 8640
ValidSequenceSamples = 1026
```

全部轨迹的观测与动作形状、数值有限性检查通过。
训练和验证按整条轨迹划分；序列采样使用官方 `SequenceDataset`。
一个序列样本来自同一轨迹的连续 10 个时间步，轨迹尾部不足 10 步时按配置重复尾部数据补齐。
不同窗口可能重叠；一个 batch 的 100 个样本不要求来自 100 条不同轨迹。

训练和验证 batch 均通过以下形状检查：

| 字段 | 含义 | Batch 形状 |
| --- | --- | --- |
| `robot0_eef_pos` | 末端位置 | `(100, 10, 3)` |
| `robot0_eef_quat` | 末端四元数姿态 | `(100, 10, 4)` |
| `robot0_gripper_qpos` | 两个夹爪关节的位置，连续量 | `(100, 10, 2)` |
| `object` | 物体位置、四元数、相对末端位置，共 3+4+3 维 | `(100, 10, 10)` |
| 合计状态维度 | 19 | `(100, 10, 19)` |
| 专家动作 | 7 维控制动作 | `(100, 10, 7)` |

原始 batch 位于 CPU，诊断拼接状态为 float64、动作为 float32。
官方预处理后，全部观测和动作均为 CUDA:0 上的 float32。

## 5. GPU 前向计算审计

使用临时模型和真实 batch，仅执行无梯度前向计算，没有执行优化器更新。

```text
RuntimeAlgorithmClass = BC_RNN_GMM
LSTMStructure = input_size=19, hidden_size=400, num_layers=2, bidirectional=False
DistributionBatchShape = (100, 10)
DistributionEventShape = (7,)
FinalHiddenShape = (2, 100, 400)
FinalCellShape = (2, 100, 400)
LogProbabilityShape = (100, 10)
LogProbabilitiesFinite = True
NegativeLogLikelihood = 5.196191787719727
GradientEnabledDuringForward = False
OptimizerStepsExecuted = 0
AuditPeakTorchAllocatedMiB = 61.55
```

61.55 MiB 是这次无梯度审计中的 PyTorch 显存分配峰值，不能用作正式训练显存峰值。
随机初始化模型的 NLL 只用于确认计算链路可运行，不作为训练结果。

## 6. 辅助脚本问题与修复

### 6.1 CUDA 初始化顺序

第一版前向审计在显示 CUDA 可用后报错：

```text
RuntimeError: Invalid device argument
```

问题出现在 CUDA 显式初始化前的显存统计调用阶段。
随后单独检查确认：初始化前 `torch.cuda.is_initialized()` 为 False，
执行 `torch.cuda.init()` 后为 True，设备编号为 0，重置显存统计和 CUDA 运算均通过。
修订版先初始化 CUDA，再重置显存统计，完整前向审计通过。
这次错误不属于正式训练中的 CUDA OOM。

### 6.2 粘贴造成的语法错误

第二版审计命令出现 `try            try:`，触发 `SyntaxError: expected ':'`。
该命令未开始执行。第三版整理代码结构后重新运行并通过。

### 6.3 进度检查的验证轮次匹配错误

早期监控脚本匹配 `Valid Epoch`，而训练日志实际打印 `Validation Epoch`，
因此一度显示尚无验证记录。修正匹配表达式后可正确读取验证轮次。
最终训练和验证记录均完整覆盖 epoch 1 至 2000。

上述三项均为辅助审计或监控脚本问题；本次 BC-RNN 正式训练正常完成，未发生训练重启。
日志中的缺少 private macro 和可选 `robosuite_models` 提示没有阻止本任务执行。

## 7. 运行配置与训练启动

沿用阶段 4 已验证的策略：训练期关闭视频，保留全部 rollout 评估，训练结束后单独录制视频。
官方配置保持不变；运行配置只修改两个字段：

```text
experiment.name = core_bc_rnn_lift_ph_low_dim_no_video_20260916-041540-695444
experiment.render_video = false
```

运行配置路径与哈希：

```text
/home/lx/robomimic/runs/lift_ph/runtime_configs/state_bc_rnn_official_no_video_20260916-041540-695444.json
SHA256 = 3e0449668e9df86843f051b920858652a668728edc844a1691dd148e438a84ea
```

启动前固定 checkpoint 选择规则和独立评估协议：

```text
CheckpointSelectionRule = highest_training_success_then_earliest_epoch
IndependentEvaluationSeed = 20260915
IndependentEvaluationRollouts = 100
IndependentEvaluationHorizon = 400
IndependentVideoRollouts = 5
```

使用新的实验目录，避免覆盖已有实验和后台交互确认问题。
通过独立会话启动训练，标准输入设为 DEVNULL，输出和错误写入启动日志。
本次训练 PID 为 94124；该 PID 仅作为历史启动记录。

```text
LaunchLog = /home/lx/robomimic/runs/lift_ph/launch_logs/state_bc_rnn_no_video_20260916-041540-695444.log
PIDFile = /home/lx/robomimic/runs/lift_ph/launch_logs/state_bc_rnn_no_video_20260916-041540-695444.pid
RunDirectory = /home/lx/robomimic/runs/lift_ph/training/core/bc_rnn/lift/ph/low_dim/trained_models/core_bc_rnn_lift_ph_low_dim_no_video_20260916-041540-695444/20260916041920
InternalLog = RunDirectory/logs/log.txt
```

核心训练调用为 `robomimic/scripts/train.py --config <上述运行配置>`，
使用指定 robot-il Python、`MUJOCO_GL=egl` 和 `CUDA_VISIBLE_DEVICES=0`。

## 8. 正式训练完成与资源观察

最终完成审计：

```text
TrainingProcessExited = True
FinishedRunSuccessfully = True
TrainEpochCoverage = 1..2000, 2000 records
ValidEpochCoverage = 1..2000, 2000 records
CompletedRolloutCount = 40
RolloutEpochs = 50, 100, ..., 2000
CheckpointCount = 40
EmptyCheckpoints = 0
TrainingVideoCount = 0
TracebackCount = 0
CannotAllocateMemoryCount = 0
CUDAOutOfMemoryCount = 0
RunFailedCount = 0
```

训练中采集的部分资源快照：

| 观察时点 | 日志或系统记录 |
| --- | --- |
| epoch 45 至 49 | 每轮 Memory Usage 2282 MB |
| epoch 150 | Memory Usage 2853 MB |
| epoch 195 至 199 | 每轮 Memory Usage 2592 MB |
| epoch 506 至 510 | 每轮 Memory Usage 2904 MB |
| 进度约 epoch 200 时 | 系统可用内存 7.7 GiB |

这些是离散快照，不是完整运行的内存峰值或内存无泄漏证明。
本次没有记录可用于正式比较的端到端训练总耗时。

训练和验证使用 GMM 负对数似然。训练 Loss 可为负数，因为连续分布的概率密度可以大于 1。
本次配置 `gmm.low_noise_eval=True`，评估模式中的低噪声分布会影响验证 NLL 的尺度；
例如 epoch 510 的训练 Loss 为 -21.3352，验证 Loss 为 2223821.8375。
这些数值如实保留；不能直接据此比较训练与验证拟合程度，也不能把它们解释为任务成功率。
checkpoint 始终按预先规定的闭环 rollout 成功率选择。

## 9. 全部训练期 rollout 结果

每次评估执行 50 次任务尝试。

| Epoch | Success Rate | 成功次数 |
| --- | ---: | ---: |
| 50 | 0.14 | 7/50 |
| 100 | 0.70 | 35/50 |
| 150 | 0.82 | 41/50 |
| 200 | 0.90 | 45/50 |
| 250 | 0.98 | 49/50 |
| 300 | 1.00 | 50/50 |
| 350 | 1.00 | 50/50 |
| 400 | 1.00 | 50/50 |
| 450 | 1.00 | 50/50 |
| 500 | 1.00 | 50/50 |
| 550 | 1.00 | 50/50 |
| 600 | 1.00 | 50/50 |
| 650 | 0.98 | 49/50 |
| 700 | 1.00 | 50/50 |
| 750 | 1.00 | 50/50 |
| 800 | 0.98 | 49/50 |
| 850 | 0.98 | 49/50 |
| 900 | 1.00 | 50/50 |
| 950 | 1.00 | 50/50 |
| 1000 | 0.98 | 49/50 |
| 1050 | 1.00 | 50/50 |
| 1100 | 0.94 | 47/50 |
| 1150 | 0.98 | 49/50 |
| 1200 | 1.00 | 50/50 |
| 1250 | 0.98 | 49/50 |
| 1300 | 0.94 | 47/50 |
| 1350 | 0.96 | 48/50 |
| 1400 | 0.96 | 48/50 |
| 1450 | 0.94 | 47/50 |
| 1500 | 0.94 | 47/50 |
| 1550 | 0.92 | 46/50 |
| 1600 | 0.96 | 48/50 |
| 1650 | 0.98 | 49/50 |
| 1700 | 1.00 | 50/50 |
| 1750 | 0.96 | 48/50 |
| 1800 | 0.90 | 45/50 |
| 1850 | 0.94 | 47/50 |
| 1900 | 0.92 | 46/50 |
| 1950 | 0.90 | 45/50 |
| 2000 | 0.96 | 48/50 |

## 10. Checkpoint 选择与冻结

从训练日志读取所有 rollout 成功率，以成功率最高、并列时 epoch 最小为选择标准。
成功率不从文件名推断，也不根据独立评估结果回选模型。

达到 1.00 的 14 个 epoch：

```text
300, 350, 400, 450, 500, 550, 600, 700, 750, 900, 950, 1050, 1200, 1700
```

因此选择 epoch 300。最后一轮 epoch 2000 的成功率为 0.96；最后保存的模型不自动等于最佳模型。

```text
SelectedCheckpoint = /home/lx/robomimic/runs/lift_ph/training/core/bc_rnn/lift/ph/low_dim/trained_models/core_bc_rnn_lift_ph_low_dim_no_video_20260916-041540-695444/20260916041920/models/model_epoch_300_Lift_success_1.0.pth
SelectedEpoch = 300
SelectedCheckpointBytes = 7961174
SelectedCheckpointSHA256 = 7d3ccdb2f6924cdccb2b21fb759ef188ee27f029d35247c895719221f6deac1c
```

独立评估前将 checkpoint、哈希、选择规则、评估种子及命令写入 `protocol.json`。
100 次独立评估和 5 次视频评估完成后均再次校验该 checkpoint，哈希保持一致。

## 11. 100 次独立评估

使用本地官方 `robomimic/scripts/run_trained_agent.py`：

```text
--agent <epoch 300 checkpoint>
--n_rollouts 100
--horizon 400
--seed 20260915
```

本次没有传入 `--video_path` 或 `--render`。加载时打印的 `train.seed=1` 和
`experiment.rollout.n=50` 属于 checkpoint 保存的训练配置；
本次独立评估使用命令行的 `seed=20260915` 和 `n_rollouts=100`。

```text
Return = 1.0
Horizon = 42.82
Success_Rate = 1.0
Num_Success = 100.0
EvaluationExitCode = 0
EvaluationElapsedSeconds = 104.79
TracebackCount = 0
RolloutExceptionCount = 0
MemoryAllocationErrorCount = 0
CUDAOutOfMemoryCount = 0
CheckpointPreserved = True
```

104.79 秒是包装命令记录的整次评估进程耗时，包含模型、环境初始化等开销。
42.82 是每次任务尝试的平均控制步数，不是秒数。

评估目录：

```text
/home/lx/robomimic/runs/lift_ph/evaluation/state_bc_rnn_epoch300_seed20260915_n100_20260916041920
```

该目录保存 `protocol.json`、`evaluation.log`、`execution.json`。
评估日志 SHA256：

```text
e002f97b6790a2e5933d6502252f5ea734aab21f6846b9e77fe457d51fb57c11
```

这 100 次评估没有参与模型选择。有限样本的 100/100 不应推广为所有初始状态均能成功。

## 12. 五次视频评估与人工检查

另启独立进程，使用同一 checkpoint、同一种子和 horizon，执行 5 次 rollout：

```text
--n_rollouts 5
--horizon 400
--seed 20260915
--video_skip 5
--camera_names agentview
--video_path <下述视频路径>
```

`agentview` 与阶段 4 使用的视角一致。视频仅用于观察策略行为，
这些 5 次 rollout 不与独立评估的 100 次合并统计成 105 次测试。

```text
Return = 1.0
Horizon = 40.6
Success_Rate = 1.0
Num_Success = 5.0
VideoEvaluationExitCode = 0
VideoGenerationElapsedSeconds = 15.44
VideoBytes = 114603
CheckpointPreserved = True
TracebackCount = 0
RolloutExceptionCount = 0
MemoryAllocationErrorCount = 0
CUDAOutOfMemoryCount = 0
```

视频路径：

```text
/home/lx/robomimic/runs/lift_ph/evaluation/state_bc_rnn_epoch300_seed20260915_n100_20260916041920/video5/state_bc_rnn_epoch300_seed20260915_first5.mp4
```

视频 SHA256：

```text
b78c4b5e5b2d82a64aa57fe44c5a370965f278ec917c4788d76a5579619add12
```

视频评估日志 `video5/video.log` 的 SHA256：

```text
826bb92082553144d4a66e9dc189d33d93c4e76be866b6a733a4b1456f1fd905
```

`video5/` 同时保存本次视频的 `protocol.json` 和 `execution.json`。
用户在本地播放视频后逐项确认：

1. 视频正常播放，没有黑屏或花屏。
2. 可以看到机械臂接近物体、夹取并抬起。
3. 场景重置时切换到下一次尝试，显示正常。

人工画面验收依据为用户的本地观看反馈；自动日志证明 5 次 rollout 成功及文件生成完成。

## 13. 本阶段理解要点

阶段推进中已讨论并确认以下核心概念：

1. `(100, 10, 19)` 表示 batch 中 100 个序列样本，每个样本 10 个时间步，每步 19 维状态。
2. 隐藏维度 400 是每层 LSTM 内部状态的维度；两个层次的 hidden 和 cell 均为 `(2, 100, 400)`。
3. 训练中的 10 步输出与输入的 10 步专家动作时间对齐，不是预测未来 10 步动作块。
4. 在线执行每次接收当前观测并输出一个 7 维动作；`open_loop=False` 使用新的观测。
   `rnn.horizon=10` 表示每执行 10 步，在第 11、21 等步前重置内部状态。
5. 行为克隆描述专家监督学习方式，LSTM 描述网络结构，GMM 描述动作分布输出。
6. 关闭训练期视频用于降低视频渲染和编码的资源负担；swap 是磁盘支持的系统内存交换空间，
   不等于 GPU 显存扩容，并行训练会竞争计算和内存资源。

观察状态和短期历史可以提供接近、抓取、抬起等任务阶段的信息；
模型并不需要一个额外的人工阶段标签，也不能据此保证短历史足以解决任意部分可观测任务。

## 14. 验收状态与后续工作

```text
Stage05ConfigComparison = PASS
Stage05DatasetSequenceAudit = PASS
Stage05RNNForwardAudit = PASS
Stage05TrainingCompletionAudit = PASS
Stage05CheckpointSelection = PASS
Stage05IndependentEvaluation = PASS
Stage05PolicyVideoGeneration = PASS
Stage05VideoVisualReview = PASS (user-confirmed)
Stage05EngineeringStatus = PASS
Stage05CoreConceptReview = COMPLETED_DURING_EXECUTION
Stage05FullThreeSeedReproduction = NOT_PERFORMED
```

本报告在文档落盘和 Git 验收步骤中编制。报告检查、提交、推送、远程同步，
以及阶段结束的结果理解核对，以后续实际验收记录为准；此处不预先标记整个阶段完成。

`runs/` 中的 checkpoint、运行配置、日志和视频保持 Git 忽略。
本次待纳入版本管理的文件为 `reproduction/lift_ph/reports/05-state-bc-rnn-training.md`。
阶段 6 再结合 [阶段 4 报告](04-state-bc-training.md) 进行同口径比较和项目总验收。

## 15. 代码与证据来源

- 官方配置：[state_bc_rnn_official.json](../configs/state_bc_rnn_official.json)。
- 对照报告：[04-state-bc-training.md](04-state-bc-training.md)。
- 训练与评估源码基线：`323e2bf3901adc53b82bb4d8d6f73a4404948fd1`。
- 评估入口：[run_trained_agent.py](https://github.com/liuxue-lab/robomimic/blob/323e2bf3901adc53b82bb4d8d6f73a4404948fd1/robomimic/scripts/run_trained_agent.py)。
- 数值依据：本阶段配置、数据、前向审计和训练完成检查的终端输出，以及上述运行日志和协议文件。
- 视频画面依据：用户在本地播放器中完成的观看确认。
