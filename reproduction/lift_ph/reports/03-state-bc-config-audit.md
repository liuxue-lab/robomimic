# Lift-PH 官方 State BC 配置审计报告

## 1. 审计目标

本报告记录 robomimic v0.4.0 中 Lift-PH low-dim 行为克隆配置的来源、生成过程、关键参数和运行时兼容性。

本阶段审计两种独立策略：

1. State BC；
2. State BC-RNN。

本阶段仅准备和审计配置，没有启动正式训练。

```text
Stage = 03
Task = Lift
DatasetType = PH
HDF5Type = low_dim
TrainingStarted = False
```

## 2. 软件与源码基线

```text
Repository = /home/lx/robomimic
Branch = lift-ph-reproduction
Stage02Head = 09bbeec
robomimic = v0.4.0
robosuite = v1.5.1
Python = 3.11.16
PythonExecutable = /home/lx/miniforge3/envs/robot-il/bin/python
RobomimicImport = /home/lx/robomimic/robomimic/__init__.py
```

阶段3开始前，Git 工作区干净。

## 3. 官方配置来源

配置不是手工猜测，而是从当前 robomimic v0.4.0 源码中的论文实验生成器生成。

主要来源文件：

```text
robomimic/scripts/generate_paper_configs.py
robomimic/config/bc_config.py
robomimic/__init__.py
robomimic/algo/bc.py
```

官方生成链路：

```text
generate_experiment_config
    -> config_factory
    -> modify_config_for_default_low_dim_exp
    -> modify_config_for_dataset
    -> modify_bc_config_for_dataset
       或 modify_bc_rnn_config_for_dataset
    -> config.dump
```

普通 BC 使用：

```text
algo_name = bc
modifier = modify_bc_config_for_dataset
```

BC-RNN 使用：

```text
algo_name = bc_rnn
modifier = modify_bc_rnn_config_for_dataset
```

BC-RNN 内部仍使用 `BCConfig`，通过 `experiment.name`、`train.seq_length` 和 `algo.rnn.enabled` 与普通 BC 区分。

## 4. 数据集注册表与路径

Lift-PH 的官方注册表条目为：

```text
raw:
  url = v1.5/lift/ph/demo_v15.hdf5
  horizon = 400

low_dim:
  url = v1.5/lift/ph/low_dim_v15.hdf5
  horizon = 400
```

本项目使用的数据集为：

```text
/home/lx/robomimic/datasets/lift/ph/low_dim_v15.hdf5
```

该路径与官方生成器得到的路径一致。

源码中的 `low_dim_v141.hdf5` 只是在注册表 URL 为空时使用的备用名称。Lift-PH low-dim 的 URL 非空，因此不会进入该备用分支。

## 5. 训练集与验证集

Lift-PH 属于 proficient-human 数据，不是 machine-generated 数据，因此官方配置启用验证：

```text
experiment.validate = True
train.hdf5_filter_key = train
train.hdf5_validation_filter_key = valid
```

运行时检查结果：

```text
TrainTrajectories = 180
ValidTrajectories = 20
TrainValidOverlap = 0
```

训练集和验证集按完整 trajectory 划分，不把同一轨迹中的相邻 transition 随机拆分到两侧。

## 6. 状态输入与动作输出

两种策略使用相同的 low-dim 观测：

| 观测字段 | 维度 |
|---|---:|
| `robot0_eef_pos` | 3 |
| `robot0_eef_quat` | 4 |
| `robot0_gripper_qpos` | 2 |
| `object` | 10 |
| 合计 | 19 |

输入维度为：

\[
3+4+2+10=19
\]

动作维度检查结果：

```text
ActionDim = 7
```

动作语义：

```text
[delta_x, delta_y, delta_z,
 delta_rx, delta_ry, delta_rz,
 gripper]
```

两种配置都设置：

```text
train.hdf5_load_next_obs = False
```

因此普通 BC 的监督关系是：

```text
obs[t] -> actions[t]
```

BC-RNN 使用连续状态序列预测对应动作序列，但同样不把 `next_obs` 作为策略输入。

## 7. 公共训练参数

| 参数 | 官方值 |
|---|---:|
| Batch size | 100 |
| Num epochs | 2000 |
| 每个 epoch 的训练步数 | 100 |
| 每个 validation epoch 的步数 | 10 |
| 优化器 | Adam |
| 初始学习率 | \(10^{-4}\) |
| 学习率衰减计划 | 空 |
| 参数 L2 正则化 | 0 |
| 随机种子 | 1 |
| CUDA | True |
| 观测归一化 | False |
| HDF5 cache mode | all |
| Data workers | 0 |
| GMM 分量数 | 5 |
| GMM minimum std | \(10^{-4}\) |
| GMM low-noise evaluation | True |

最大梯度更新次数为：

\[
2000\times100=200000
\]

这里的 epoch 使用固定梯度步数，不一定等于完整遍历一次数据集。

## 8. Rollout 与 checkpoint 设置

```text
experiment.save.enabled = True
experiment.save.every_n_epochs = 50

experiment.rollout.enabled = True
experiment.rollout.n = 50
experiment.rollout.rate = 50
experiment.rollout.horizon = 400
experiment.rollout.warmstart = 0
experiment.rollout.terminate_on_success = True
```

Lift 控制频率为20 Hz，因此400步对应最长20秒：

\[
400/20=20\ \mathrm{s}
\]

任务成功后 rollout 可以提前终止。

## 9. 普通 State BC

正式配置：

```text
reproduction/lift_ph/configs/state_bc_official.json
```

实验名称：

```text
core_bc_lift_ph_low_dim
```

关键结构：

```text
train.seq_length = 1
algo.actor_layer_dims = [1024, 1024]
algo.rnn.enabled = False
algo.gmm.enabled = True
algo.gmm.num_modes = 5
```

策略关系：

\[
o_t
\rightarrow
\mathrm{MLP}
\rightarrow
p_\theta(a_t\mid o_t)
\]

训练输出目录：

```text
/home/lx/robomimic/runs/lift_ph/training/core/bc/lift/ph/low_dim/trained_models
```

SHA256：

```text
96c81356b0ccb562f0e8f737d1930201bd27c9895831d2419046fc495acc14b9
```

## 10. State BC-RNN

正式配置：

```text
reproduction/lift_ph/configs/state_bc_rnn_official.json
```

实验名称：

```text
core_bc_rnn_lift_ph_low_dim
```

关键结构：

```text
train.seq_length = 10
algo.actor_layer_dims = []
algo.rnn.enabled = True
algo.rnn.horizon = 10
algo.rnn.hidden_dim = 400
algo.rnn.rnn_type = LSTM
algo.rnn.num_layers = 2
algo.rnn.bidirectional = False
algo.rnn.open_loop = False
algo.gmm.enabled = True
algo.gmm.num_modes = 5
```

策略关系：

\[
o_{t-9:t}
\rightarrow
\mathrm{LSTM}
\rightarrow
p_\theta(a_t\mid h_t)
\]

`open_loop=False` 表示策略执行时持续接收当前的新观测。

训练输出目录：

```text
/home/lx/robomimic/runs/lift_ph/training/core/bc_rnn/lift/ph/low_dim/trained_models
```

SHA256：

```text
06c9380989e5d449038bc0c5a1a5789eb546961c7380ed4ece8217b985bd97c4
```

## 11. GMM 动作建模与损失

普通 BC 和 BC-RNN 都启用5分量 Gaussian Mixture Model：

\[
p_\theta(a\mid o)
=
\sum_{k=1}^{5}
\pi_k(o)
\mathcal N(a;\mu_k(o),\Sigma_k(o))
\]

GMM 可以描述相似状态下存在的多种合理专家动作，避免简单 MSE 把不同动作模式平均为不合理动作。

训练源码计算：

```python
log_probs = dists.log_prob(batch["actions"])
action_loss = -log_probs.mean()
```

实际损失为动作负对数似然：

\[
\mathcal L
=
-\frac{1}{N}
\sum_i
\log p_\theta(a_i\mid o_i)
\]

共享配置中的确定性 BC L2 动作损失参数不会作为当前 GMM 分支的最终训练损失。

## 12. 配置生成与正式复制

官方候选配置生成在被 Git 忽略的目录：

```text
runs/lift_ph/config_audit/core/lift/ph/low_dim/bc.json
runs/lift_ph/config_audit/core/lift/ph/low_dim/bc_rnn.json
```

生成结果：

```text
ConfigGenerationStatus = PASS
```

候选配置随后逐字节复制到：

```text
reproduction/lift_ph/configs/state_bc_official.json
reproduction/lift_ph/configs/state_bc_rnn_official.json
```

复制验证结果：

```text
StateBCConfigCopy = PASS
StateBCRNNConfigCopy = PASS
```

候选配置受以下 Git 忽略规则保护：

```text
/runs/
```

训练日志、视频和 checkpoint 不进入 Git。

## 13. 运行时解析与数据兼容性

两份正式配置均通过：

1. Python JSON 语法解析；
2. robomimic `config_factory` 解析；
3. HDF5 观测字段检查；
4. 输入与动作维度检查；
5. train/valid mask 检查。

结果：

```text
BC:
  ConfigClass = BCConfig
  SequenceLength = 1
  RNNEnabled = False
  InputDim = 19
  ActionDim = 7
  TrainTrajectories = 180
  ValidTrajectories = 20
  ConfigDatasetCompatibility = PASS

BC-RNN:
  ConfigClass = BCConfig
  SequenceLength = 10
  RNNEnabled = True
  InputDim = 19
  ActionDim = 7
  TrainTrajectories = 180
  ValidTrajectories = 20
  ConfigDatasetCompatibility = PASS

RuntimeConfigAudit = PASS
```

两份配置的 `ConfigClass` 都是 `BCConfig` 属于正常行为。

## 14. 可移植性与使用限制

正式配置包含本机绝对路径：

```text
/home/lx/robomimic/datasets/...
/home/lx/robomimic/runs/...
```

当前机器可以直接使用。

在其他机器复现时，应重新使用官方生成器生成配置，或者只修改：

```text
train.data
train.output_dir
```

不应同时改变网络结构、数据划分、学习率、训练周期和 rollout 参数。

本报告只证明配置来源和数据兼容性，不包含任何训练结果或成功率。

## 15. 后续执行顺序

阶段4按照以下顺序进行：

1. 训练普通 State BC；
2. 选择 checkpoint 并进行闭环 rollout；
3. 训练 State BC-RNN；
4. 在相同数据和评估条件下比较两者。

两种模型分别训练、分别保存、分别评估。

## 16. 审计结论

```text
OfficialConfigSourceLocated = True
DatasetPathMatch = True
TrainMask = train
ValidationMask = valid
ObservationKeysMatch = True
InputDimension = 19
ActionDimension = 7
StateBCConfigParse = PASS
StateBCRNNConfigParse = PASS
StateBCDatasetCompatibility = PASS
StateBCRNNDatasetCompatibility = PASS
CandidateConfigAudit = PASS
RuntimeConfigAudit = PASS
TrainingStarted = False
Stage03ConfigurationAudit = PASS
```

两份官方配置已经准备完成。正式训练必须等到阶段3文档和 Git 验收完成后再开始。
