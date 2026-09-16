# 阶段 7：Vision BC 数据构建、训练、故障恢复与独立评估

本阶段是阶段 0～6 State BC / BC-RNN 核心复现完成后的视觉策略拓展。在同一批 Lift-PH 专家轨迹上，从保存的仿真状态重建双相机 RGB 观测，训练使用图像与机器人本体状态的 BC-GMM，并完成冻结模型后的独立评估。

本报告依据用户终端返回的审计结果和人工视频确认。归档脚本从本机已有 JSON / 日志导出轻量结果，保留原配置字节与可追溯路径。训练种子为 1；这是一轮教学型实验，未执行多训练种子统计复现。

## 1. 验收结论

| 项目 | 已验收结果 |
| --- | --- |
| 视觉数据 | 200 条轨迹，9666 个时间步；两路 84 × 84 RGB |
| 训练 / 验证划分 | 180 / 20 条轨迹，8640 / 1026 个时间步 |
| 实际算法 / 策略类 | `BC_GMM` / `GMMActorNetwork` |
| 策略输入 | `agentview_image`、`robot0_eye_in_hand_image` + 9 维本体状态 |
| 动作维度 / 参数量 | 7 / 23,661,963 |
| 成功运行的 DataLoader workers | 0 |
| 正式训练 | 600 epoch × 500 次更新 = 300,000 次更新 |
| 训练期评估 | 30 组，每组 50 次 rollout |
| 模型选择规则 | 最高训练期成功率，并列取最早 epoch |
| 选定 checkpoint | epoch 80，训练期 50/50，平均 Horizon 44.06 |
| 独立评估 | seed 20260915，100 次，horizon 上限 400 |
| 独立评估结果 | 成功 100/100，平均 Return 1.0，平均 Horizon 46.47 |
| 视频来源 | 同一次独立评估保存的 `demo_0`～`demo_4` 仿真状态回放 |
| 人工视觉验收 | PASS，用户确认这 5 条回放正常 |

选定的 epoch 80 对应 40,000 次更新，但正式训练完整执行了 300,000 次更新。独立评估不参与 checkpoint 选择；回放视频也没有增加新的评估回合。

## 2. 从低维数据重建视觉数据

原始数据为 `datasets/lift/ph/low_dim_v15.hdf5`。其中保留了每条轨迹的 `states`、`model_file` 与相机信息，可以通过官方 `robomimic/scripts/dataset_states_to_obs.py` 重建观测。

执行顺序为：源数据结构检查、单条轨迹渲染、低维对齐与图像预览、全量提取、全量审计。单条烟雾测试仅用于检查渲染；正式训练使用完整的 200 条轨迹数据。

提取使用两台相机 `agentview`、`robot0_eye_in_hand`，分辨率均为 84 × 84；采用 `--done_mode 2 --copy_rewards --copy_dones --compress --exclude-next-obs`。`--copy_rewards` 与 `--copy_dones` 保留源数据监督信号，最终审计确认逐元素一致。完整调用所用的数据集路径、输出位置和脚本哈希见本机提取记录。

输出为 `datasets/lift/ph/image_v15.hdf5`，大小 111,469,613 字节。全量提取用时约 2 分 53 秒。

| 检查 | 结果 |
| --- | --- |
| `states/actions/rewards/dones` | 与源数据逐元素一致 |
| 两路 RGB | 每路 9666 帧，`(T,84,84,3)`，`uint8`，gzip 压缩 |
| 全量像素范围 | 两路均为 6～255 |
| `robot0_eef_pos` 最大绝对误差 | 6.661338147750939e-16 |
| `robot0_eef_quat` 最大绝对误差 | 1.1102230246251565e-15 |
| `robot0_gripper_qpos` 最大绝对误差 | 0 |
| `object` 最大绝对误差 | 6.661338147750939e-16 |
| masks | 全部 8 个保留，包括 train / valid 及 20% / 50% 子集 |
| `next_obs` | 未写入；训练配置不加载下一时刻观测 |

低维对齐误差处于浮点舍入量级，支持重建观测与源轨迹状态对齐。图像外观另外经过抽样预览检查，不能仅凭形状和像素范围判断渲染正确。

## 3. 官方配置与实际运行配置

官方配置通过本仓库的 `generate_paper_configs.py` 生成，选用 `core/lift/ph/image/bc.json`。归档保留两份原始字节：

- [官方 Vision BC 配置](../configs/vision_bc_official.json)
- [实际成功训练配置](../configs/vision_bc_runtime_workers0.json)

成功运行配置相对于官方配置仅有以下四个字段差异：

| 字段 | 官方生成值 | 成功运行值 |
| --- | --- | --- |
| `experiment.name` | `core_bc_lift_ph_image` | `core_bc_lift_ph_image_no_video_20260916-155536_ebb628cb_workers0` |
| `experiment.render_video` | `true` | `false` |
| `train.data` | `/home/lx/robomimic/datasets/lift/ph/image_v141.hdf5` | `/home/lx/robomimic/datasets/lift/ph/image_v15.hdf5` |
| `train.num_data_workers` | 2 | 0 |

关闭的是训练期视频保存；训练期 rollout 评估保持开启，视觉策略仍需获取相机观测。输出目录沿用官方生成配置中的本机目录。

| 训练设置 | 数值 |
| --- | --- |
| batch size / sequence length / frame stack | 16 / 1 / 1 |
| Adam 初始学习率 / L2 | 1e-4 / 0 |
| 学习率衰减 epoch 列表 | 空列表 |
| HDF5 cache / SWMR | `low_dim` / `true` |
| 观测统计归一化 / 加载 next_obs | `false` / `false` |
| 每 epoch 训练 / 验证批次数 | 500 / 50 |
| 保存与 rollout 周期 | 每 20 epoch |
| 每组 rollout / horizon 上限 | 50 / 400 |
| 成功后终止 | `true` |
| 训练随机种子 | 1 |

RGB 转换为浮点数并除以 255，与 `hdf5_normalize_obs=false` 所控制的观测统计归一化是不同处理。

## 4. 网络与前向检查

输入为双路 RGB 与本体状态：末端位置 3 维、末端四元数 4 维、夹爪位置 2 维。环境可以产生 `object` 观测，但本次策略配置没有把它列入输入。

每台相机分别经过 CropRandomizer（84 → 76）、不使用预训练权重的 ResNet18Conv、32 个关键点的 SpatialSoftmax，以及 64 维视觉特征输出。两路特征与本体状态拼接为 `64 + 64 + 9 = 137` 维，再经过 `[1024,1024]` MLP，输出 5 分量 GMM 的 7 维动作分布。RNN 关闭。

前向探针确认以下形状：

| 张量 | 形状 |
| --- | --- |
| 单路原始 RGB batch | `(16,1,84,84,3)`，CPU `uint8` |
| 单路预处理 RGB | `(16,3,84,84)`，CUDA `float32` |
| 单路 VisualCore 输入 / 输出 | `(16,3,76,76)` / `(16,64)` |
| 融合特征 | `(16,137)` |
| 专家动作 / 采样动作 | `(16,7)` |
| GMM 权重 | `(16,5)` |
| GMM 均值 / 标准差 | `(16,5,7)` |

前向检查不记录梯度，优化器更新次数为 0。随后独立烟雾训练执行 2 epoch × 5 次更新，覆盖训练、验证、rollout 与保存。烟雾 checkpoint 重新加载时 514 个状态张量匹配，单步动作有限，评估模式的 GMM 标准差为约 1e-4。

`low_noise_eval=true` 不等于完全确定性地输出单个均值；不能据此假设策略执行没有分布采样。

## 5. 首次正式训练失败及处理

首次正式运行使用 2 个 DataLoader worker。epoch 40 的训练、验证、rollout 和 checkpoint 保存完成后，下一轮读取数据时失败。关键错误为：

```text
malloc_consolidate(): unaligned fastbin chunk detected
ConnectionResetError: [Errno 104] Connection reset by peer
RuntimeError: DataLoader worker (pid 26885) is killed by signal: Aborted.
```

运行耗时 628.9671982730001 秒，执行记录为 FAILED。该错误表明检测到原生内存堆异常，现有证据没有定位到具体底层库。可读取的内核日志没有匹配项，不能据此断言已经排除所有内存问题。关闭终端不是这份记录所证明的失败原因。

处理措施是保留失败目录及 epoch 20 / 40 checkpoint，把 `train.num_data_workers` 改为 0，使用带 `_workers0` 的新实验名，以 seed 1 从头执行完整训练。没有加载失败运行的权重继续训练，也没有把两个运行的更新次数相加。

成功运行消除了本次 DataLoader 子进程崩溃问题，但这不足以证明底层堆损坏的唯一根因。

## 6. 完整训练与模型选择

成功运行通过 600 次训练日志、600 次验证日志及 600 个完整 epoch 记录审计；更新总数为 300,000，训练期 rollout 共 30 组，checkpoint 共 30 个，已记录数值指标均有限。

| 指标 | 数值 |
| --- | ---: |
| 完整运行耗时 | 16018.83595 秒，约 4 小时 26 分 59 秒 |
| 最后训练 NLL | -31.33991999053955 |
| 最后验证 NLL | 3016043.4275 |
| PyTorch 峰值 allocated | 487.263671875 MiB |
| PyTorch 峰值 reserved | 530.0 MiB |
| epoch 600 训练期评估 | 48/50，平均 Horizon 65.48 |

上述内存值只表示 PyTorch CUDA 分配器统计，不是整个进程、渲染器或整块 GPU 的总显存。日志的 `Epoch ... Memory Usage` 也不应当作 GPU 显存。

连续分布的概率密度可以超过 1，因此负对数似然可以为负。评估时观察到的极小 GMM 标准差会显著放大动作偏差对 NLL 的影响；这里不能仅凭验证 NLL 的量级判断策略在环境中失效，也不能把训练与评估模式的 NLL 当作完全同口径的泛化差距。任务表现单独用 rollout 评估。

完整的 30 个训练期评估点见 [训练期 rollout CSV](../results/stage07-training-rollouts.csv)。epoch 80、140、200、380 都达到 50/50；按固定的并列规则选取最早的 epoch 80。

选定 checkpoint：

```text
runs/lift_ph/vision_training/core/bc/lift/ph/image/trained_models/core_bc_lift_ph_image_no_video_20260916-155536_ebb628cb_workers0/20260916222324/models/model_epoch_80_Lift_success_1.0.pth
```

## 7. 冻结 checkpoint 后的独立评估

使用官方 `robomimic/scripts/run_trained_agent.py`，固定 epoch 80 checkpoint，执行 100 次 rollout，seed 为 20260915，horizon 上限为 400。实际脚本的成功后终止逻辑已在本机检查。评估环境由 checkpoint 元数据恢复，使用 Lift / Panda / OSC_POSE，20 Hz，`lite_physics=false`，`reward_shaping=false`。

评估进程正常结束，返回码为 0，用时 146.5691141319985 秒。checkpoint 评估前后哈希一致。

| 独立评估指标 | 结果 |
| --- | ---: |
| 成功数 / 总回合数 | 100 / 100 |
| 平均 Return | 1.0 |
| 平均 Horizon | 46.47 |

同一次评估保存的 `rollouts.hdf5` 含 100 条轨迹，其平均 Horizon 与日志一致。平均 Horizon 的口径仍为全部回合；本次恰好全部成功。

视频通过 `playback_dataset.py` 对该文件的前 5 条轨迹执行仿真状态回放，两路画面为 `agentview` 和 `robot0_eye_in_hand`，`video_skip=1`。生成视频为 713,552 字节，轨迹文件哈希在回放前后保持一致。

用户已观看前 5 条回放并确认正常，随后将原报告的 `visual_inspection` 从 PENDING 更新为 PASS，同时保存修改前备份。腕部视角下方块靠近镜头、夹爪遮挡和略微倾斜的外观经过讨论；此次人工验收是这 5 条视频样本的外观检查，不是接触力、无穿模或全部 100 条轨迹的逐帧证明。

独立评估目录：

```text
runs/lift_ph/evaluation/vision_bc_epoch80_seed20260915_n100_eg6v6_dh/
```

视频与最终评估报告位于该目录下的 `video5_ngahmsp3/`，文件分别为 `vision_bc_epoch80_eval_first5.mp4` 与 `evaluation_and_video_report.json`。

## 8. 与核心 State 结果的关系

| 项目 | State BC | State BC-RNN | Vision BC |
| --- | ---: | ---: | ---: |
| 输入 | 19 维状态 | 19 维状态序列 | 双 RGB + 9 维本体状态 |
| 使用 object 真值作为输入 | 是 | 是 | 否 |
| batch size | 100 | 100 | 16 |
| 训练更新总数 | 200000 | 200000 | 300000 |
| 选定 epoch | 600 | 300 | 80 |
| 独立评估成功数 | 99/100 | 100/100 | 100/100 |
| 独立评估平均 Horizon | 49.10 | 42.82 | 46.47 |

三组评估使用相同的 seed、回合数与 horizon 上限，但相同 seed 不自动保证逐条初始状态严格配对。观测、网络、序列长度、batch size 和训练预算存在差异，因此这里只做实验结果汇总，不能解释为控制其他因素不变的视觉输入消融，也不足以证明方法之间显著或稳定的优劣。

100/100 是本次有限样本结果，不表示所有初始状态下都能成功。阶段 0～6 的核心结论和既有提交保持独立；阶段 7 不涉及 VLA 微调、强化学习或真机迁移。

## 9. 哈希与证据定位

| 对象 | SHA256 |
| --- | --- |
| low_dim_v15.hdf5 | `2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540` |
| image_v15.hdf5 | `1708dbdd6087073a3665444bb4a4dd65e60564d632e2c712e109b2a0ced81632` |
| 提取时的 dataset_states_to_obs.py | `b61bab874ad9cc10bc4a77a20dc70591926cf5449bf5186854d78f76cafef2f8` |
| 官方视觉配置 | `01928b478a329021a631fae6569c659a8fc72c9cb8f514955900babbe4ea86c1` |
| 首次 workers=2 运行配置 | `a677d753269d58fad6dbd15ceae6f5014056aa67350197e6e1a2c9fe6791db91` |
| 成功 workers=0 运行配置 | `0f66dd225bf4d56c58a35b54f692e2db0d67b76ad8915f55abf47bc9fd0b6f31` |
| 选定 epoch 80 checkpoint | `4a2ebaa236f2621d0ac8ecc54e625f82ce2a374e2b908d330db92e0312314cec` |
| 最终 epoch 600 checkpoint | `66317268b0909c02f8214d2be53a7d34e5bbce25d4e9dc116f0e42559ab9127e` |

成功训练的启动与日志目录为：

```text
runs/lift_ph/vision_full_train/core_bc_lift_ph_image_no_video_20260916-155536_ebb628cb_workers0/
```

训练审计在该目录的 `audit_73_crdba/training_audit.json`。首次失败运行位于去除 `_workers0` 后的同名目录。

数据审计为 `runs/lift_ph/vision_data/extract_SklstU/audit_2gy0n38h/dataset_audit.json`。完整训练审计和原始日志保留在本机 `runs/`；归档 JSON 收录训练审计摘要、数据审计、执行记录、评估协议、视频验收及其源文件路径 / 哈希。

本次归档复制配置文件时重新核对配置字节哈希；数据集与 checkpoint 的哈希引用此前已验收的审计记录，不表示本次归档重新执行了训练或评估。

## 10. 归档文件与复用

- [训练期评估 CSV](../results/stage07-training-rollouts.csv)：30 个真实记录点，不插值、不平滑。
- [独立评估 CSV](../results/stage07-independent-evaluation.csv)：一行 Vision BC 独立评估结果及其协议。
- [轻量证据 JSON](../results/stage07-vision-bc-evidence.json)：来源、配置差异和审计记录。
- [导出脚本](../scripts/export_stage07_vision_results.py)：只读取已有记录，不加载模型，不启动仿真。

需要重新导出预览时，可在仓库根目录运行：

```bash
/home/lx/miniforge3/envs/robot-il/bin/python \
  reproduction/lift_ph/scripts/export_stage07_vision_results.py \
  --repo /home/lx/robomimic \
  --output-dir runs/lift_ph/archive_preview/stage07
```

脚本依赖 Python 标准库，预览结果写到独立目录。官方配置保留原始绝对路径及旧的 `image_v141.hdf5` 文件名以追溯来源；实际运行配置使用本机 `image_v15.hdf5`。迁移至其他机器时需要在新的运行副本中调整路径。

数据集、checkpoint、完整日志和视频继续保存在 `datasets/` 与 `runs/`，不纳入本次 Git 归档。归档前仓库基线为 `9eeba89896d48b3c015840df900b6dd8ea1ef70e`；本报告不预先声称阶段 7 已提交或推送，Git 验收以之后的实际终端输出为准。
