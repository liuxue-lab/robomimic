# robomimic Lift-PH 教学型复现

使用 robomimic v0.4.0 与 robosuite v1.5.1，在 Lift-PH 数据集上完成普通 State BC、State BC-RNN 的核心复现，并拓展到双相机 RGB 与机器人本体状态输入的 Vision BC。通过数据审计、仿真回放、配置核对、正式训练和冻结 checkpoint 后的独立评估，形成可追溯的机器人模仿学习实验记录。阶段 8 进一步把冻结的 Vision BC 部署到 Jetson Orin，通过独立以太网完成处理器在环推理、网络闭环和 20 Hz 时序验收。

本项目的 plant 仍是笔记本上的 Lift 仿真环境，不需要实体机械臂。阶段 0～6 完成低维状态策略的核心复现，阶段 7 完成视觉 BC 拓展，阶段 8 验证 Jetson 实机处理器上的推理与网络化软实时闭环；各方法均只有一个训练种子，未完成多训练种子统计复现或实体机械臂迁移验证。

## 1. 核心 State 复现结果

| 项目 | State BC | State BC-RNN |
| --- | ---: | ---: |
| 实际算法类 | `BC_GMM` | `BC_RNN_GMM` |
| 网络结构 | MLP `[1024,1024]` | 两层 LSTM，hidden 400 |
| 状态维度 / 动作维度 | 19 / 7 | 19 / 7 |
| 训练序列长度 | 1 | 10 |
| 训练种子 | 1 | 1 |
| 完整训练 epoch | 2000 | 2000 |
| 参数更新次数 | 200000 | 200000 |
| 选定 checkpoint | epoch 600 | epoch 300 |
| 选定点训练期成功次数 | 50/50 | 50/50 |
| 独立评估成功次数 | 99/100 | 100/100 |
| 独立评估平均执行步数，包含失败 | 49.10 | 42.82 |

独立评估均为 seed `20260915`、100 次 rollout、horizon 上限 400。两种方法按“最高训练期成功率，并列取最早 epoch”的固定规则选模型，独立评估不参与选择。

在本次单训练种子实验中，BC-RNN 的独立评估样本成功率为 100%，BC 为 99%；现有证据不足以证明 BC-RNN 显著或稳定优于 BC。相同评估 seed 也不自动构成逐条初始状态严格配对。

![State BC 与 State BC-RNN 的训练曲线及独立评估](media/stage06-bc-vs-bc-rnn.png)

图中完整显示每种方法的 40 个训练期评估点，每点 50 次 rollout；局部放大明确标注截断纵轴。没有平滑，也没有把相邻 checkpoint 当成独立训练种子绘制误差带。详见[阶段 6 对比报告](reports/06-state-bc-vs-bc-rnn-comparison.md)。

## 2. 阶段与证据导航

| 阶段 | 工作 | 文档 | 当前记录 |
| ---: | --- | --- | --- |
| 0 | 环境与仓库初始化 | [环境审计](reports/00-environment-audit.md) | PASS |
| 1 | 数据集下载与审计 | [数据审计](reports/01-dataset-audit.md) | PASS |
| 2 | 专家状态 / 动作回放 | [回放报告](reports/02-expert-playback.md) | PASS |
| 3 | 官方配置准备与审计 | [配置审计](reports/03-state-bc-config-audit.md) | PASS |
| 4 | 普通 State BC 正式训练与评估 | [BC 报告](reports/04-state-bc-training.md) | PASS |
| 5 | State BC-RNN 正式训练与评估 | [BC-RNN 报告](reports/05-state-bc-rnn-training.md) | PASS |
| 6 | 同口径比较与项目总验收 | [对比报告](reports/06-state-bc-vs-bc-rnn-comparison.md) | PASS，比较材料已提交、推送并完成远程验收 |
| 7（拓展） | 双相机 Vision BC 数据、训练、故障恢复与评估 | [视觉 BC 报告](reports/07-vision-bc-training.md) | PASS，完整训练、独立评估与前 5 条视频人工验收 |
| 8（拓展） | Jetson Orin 处理器在环推理、网络闭环与 20 Hz 评估 | [Jetson HIL 报告](reports/08-jetson-hil-evaluation.md) | `COMPLETED_WITH_RECORDED_DEVIATION`：功能与时序通过，正式评估 99/100 |

阶段 8 的 100 回合执行完整性通过，但严格 100/100 全成功门禁失败。唯一失败的 episode 53 具有连续请求、正常服务器响应、零 deadline miss 和有限动作，分类为有效策略 rollout 达到 horizon；因此保留 99/100 原始结果，不通过重跑或挑选结果改写。

阶段 0～6 的核心复现已完成。阶段 6 比较材料提交为 `5cd56793a2c6cbd1a152429fb1111fe1a141b741`；用户终端已确认本地、远程跟踪分支与 GitHub HEAD 一致，领先/落后为 0/0，工作区干净。该提交是本 README 完成状态的验收依据；后续文档记录同步不改变实验结果。

阶段 6 已修复生成 SVG 的行尾空格检查问题，绘图脚本会自动清理这类空格；恢复后 7 个文件的提交内容哈希与远程归档均已通过核验。具体过程见对比报告第 10 节。

## 3. 核心 State 复现的数据与训练设置

数据文件为 `datasets/lift/ph/low_dim_v15.hdf5`，相对于仓库根目录。共 200 条专家轨迹、9666 个时间步；按整条轨迹划分为 180 条训练轨迹和 20 条验证轨迹，对应 8640 / 1026 个时间步。

| 观测键 | 每步维度 |
| --- | ---: |
| `robot0_eef_pos` | 3 |
| `robot0_eef_quat` | 4 |
| `robot0_gripper_qpos` | 2 |
| `object` | 10 |
| 合计 | 19 |

两种方法都输出 7 维动作，在当前 `OSC_POSE` 设置下对应位置控制 3、姿态控制 3 和夹爪控制 1。训练 batch size 为 100，Adam 初始学习率为 `1e-4`，GMM 为 5 个分量。

- [普通 BC 官方配置](configs/state_bc_official.json)
- [BC-RNN 官方配置](configs/state_bc_rnn_official.json)

官方配置与运行配置的比较已核对：每种方法实际运行时仅修改实验名称、关闭训练期视频，rollout 评估保持开启。两种方法同时改变网络结构和序列长度，因此不把本实验称为“只改变记忆的单因素消融”。

训练序列长度 10 不表示一次输出未来 10 步动作块。当前 BC-RNN 在线执行时逐步读取观测、输出动作，并按既定 RNN horizon 管理循环状态。

## 4. 环境与目录

| 项目 | 已验收版本或路径 |
| --- | --- |
| 用户仓库 | `/home/lx/robomimic` |
| Git 分支 | `lift-ph-reproduction` |
| 环境 | `robot-il`，Python 3.11.16 |
| Python | `/home/lx/miniforge3/envs/robot-il/bin/python` |
| robomimic / robosuite | v0.4.0 / v1.5.1 |
| MuJoCo / PyTorch | 3.2.7 / 2.7.1+cu128 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 渲染 | `MUJOCO_GL=egl` |

更完整的环境记录见 [runtime-summary.txt](environment/runtime-summary.txt)、[conda-list.txt](environment/conda-list.txt) 与 [pip-freeze.txt](environment/pip-freeze.txt)。

| 目录 | 内容 |
| --- | --- |
| `configs/` | State / Vision 官方配置及成功训练的 Vision 运行配置 |
| `environment/` | 环境与源码版本记录 |
| `reports/` | 各阶段报告 |
| `scripts/` | 数据审计、阶段 6 绘图、阶段 7 结果导出及阶段 8 协议、服务器、客户端与分析脚本 |
| `results/` | 已验收日志的轻量 CSV 结果表，以及视觉实验与 Jetson 环境 JSON 证据 |
| `media/` | 对比图 PNG / SVG |
| 仓库根目录下 `runs/lift_ph/` | 运行配置、训练日志、checkpoint、评估日志与视频等本机证据，不入 Git |

## 5. 查看与复用核心 State 结果

- [完整训练期成功率数据](results/stage06-training-rollouts.csv)
- [独立评估结果与参数](results/stage06-independent-evaluation.csv)
- [矢量对比图](media/stage06-bc-vs-bc-rnn.svg)
- [绘图脚本](scripts/plot_stage06_comparison.py)

CSV 是本次已验收终端提取结果的快照，不是新运行结果，也不替代原始日志。绘图脚本只读取两个 CSV，不加载策略或启动训练。

如果需要重绘，在仓库根目录执行以下命令，把预览写到 `runs/`，避免修改正式图件：

```bash
/home/lx/miniforge3/envs/robot-il/bin/python \
  reproduction/lift_ph/scripts/plot_stage06_comparison.py \
  --output-dir runs/lift_ph/figure_preview/stage06
```

脚本使用 matplotlib，默认英文图内文字。中文重绘可通过 `--font` 指定已经存在的中文字体文件。已提供的中文 PNG 可直接查看，SVG 文字转为路径，显示时不要求安装相同字体。用户无需为了本阶段重新安装训练环境。

## 6. 结果解释边界

1. 选中 epoch 300 / 600 不代表只训练到这里；两种方法都完整训练了 2000 epoch。
2. 相同 epoch 数与更新次数不等于相同训练耗时、FLOPs 或监督位置数量。训练总耗时等缺少可靠统一记录时不估造。
3. 一次 rollout 是从环境重置开始，到成功或其他终止条件结束的一回合策略执行；成功、失败都计入尝试次数。
4. 平均 Horizon 包含全部评估回合，单位是控制步数，不能直接称为“成功抓取平均耗时”。
5. 首次达到训练期 50/50 后，按当前选择规则，后面的并列 50/50 不会替换该 checkpoint；本次继续至 2000 epoch 是执行固定预算协议并记录完整曲线。
6. 100/100 是有限样本结果，不是所有初始状态的成功保证。要研究跨训练随机性的稳定性，需要另行设计多个训练种子的实验。

Vision BC 已作为阶段 7 拓展完成实验验收，Jetson 处理器在环推理已作为阶段 8 拓展完成并记录偏差。阶段 0～6 的七阶段核心复现范围保持不变；VLA 微调、强化学习和实体机械臂部署仍属于后续工作。

## 7. Vision BC 拓展结果

在相同 200 条专家轨迹的仿真状态上重建两路 84 × 84 RGB 观测。策略使用双路图像与 9 维机器人本体状态，不使用 `object` 真值作为网络输入。

| 项目 | Vision BC |
| --- | --- |
| 网络 | 两路 ResNet18Conv + SpatialSoftmax，融合 9 维本体状态，MLP + 5 分量 GMM |
| 参数量 / batch size | 23,661,963 / 16 |
| 完整训练预算 | seed 1，600 epoch × 500 更新 = 300000 更新 |
| 成功训练 DataLoader workers | 0 |
| 选定 checkpoint | epoch 80；训练期成功 50/50 |
| 独立评估 | seed 20260915，100 次，horizon 上限 400 |
| 独立评估成功数 / 平均 Horizon | 100/100 / 46.47 |
| 人工视觉检查 | 同一次独立评估的前 5 条状态回放，PASS |

首次 workers=2 运行在 epoch 40 完成后发生 DataLoader 子进程中止；保留失败记录后，以 workers=0 和相同训练种子从头完成整个预算。选定 epoch 80 不表示只训练了 80 epoch；独立评估没有参与模型选择。

State BC、State BC-RNN、Vision BC 的独立评估成功数分别为 99/100、100/100、100/100。三组只有单训练种子，视觉实验的输入、batch size 与更新预算也不同，因此不把这组结果解释为方法优劣的统计结论或单因素消融。相同评估 seed 不自动构成严格的逐回合配对。

- [阶段 7 完整报告](reports/07-vision-bc-training.md)
- [Vision BC 官方配置](configs/vision_bc_official.json) / [成功运行配置](configs/vision_bc_runtime_workers0.json)
- [完整训练期评估数据](results/stage07-training-rollouts.csv) / [独立评估数据](results/stage07-independent-evaluation.csv)
- [轻量审计证据](results/stage07-vision-bc-evidence.json) / [本机记录导出脚本](scripts/export_stage07_vision_results.py)

视觉数据、模型、完整日志和视频保留在本机 `datasets/` 与 `runs/`。视频仅回放原评估的前 5 条轨迹，没有新增评估回合；100/100 仍是有限样本结果。

## 8. Jetson Orin 处理器在环拓展结果

阶段 8 冻结阶段 7 的 epoch 80 checkpoint。RTX 5060 笔记本运行 Lift 仿真，Jetson Orin Nano Super 8GB 通过 `192.168.50.1/24 ↔ 192.168.50.2/24` 独立以太网执行 Vision BC 推理。

| 项目 | 阶段 8 结果 |
| --- | --- |
| checkpoint | epoch 80，SHA256 `4a2ebaa2...2314cec` |
| 固定输入跨设备检查 | 分布参数与 winning-mode action 最大差均小于 `1e-4`，PASS |
| 单回合网络闭环 | 1/1 成功，Horizon 47，视频人工检查 PASS |
| 20 Hz paced benchmark | 1000 次，p99 43.840 ms，最大 44.789 ms，0 timeout，0 deadline miss |
| 正式评估 | seed 20260915，99/100，全部回合平均 Horizon 51.23 |
| 与阶段 7 的配对条件 | 100/100 初始物理状态完全一致，最大差 0 |
| 唯一失败 | episode 53，400 步达到 horizon；网络与协议路径正常 |
| 最终分类 | `COMPLETED_WITH_RECORDED_DEVIATION` |

阶段 7 与阶段 8 的配对成功数分别为 100/100 和 99/100，精确双侧 McNemar p 值为 1.0。该有限样本没有检出显著成功率差异，但不构成等价性证明。阶段 8 严格全成功门禁保持 FAIL，不把完整执行或网络路径 PASS 混写为 100/100 成功。

- [阶段 8 完整报告](reports/08-jetson-hil-evaluation.md)
- [协议配置](configs/stage08-hil-protocol.json)
- [环境审计](results/stage08-environment-audit.json)
- [20 Hz 延迟与资源摘要](results/stage08-latency-summary.csv)
- [正式 100 回合评估摘要](results/stage08-independent-evaluation.csv)

阶段 8 的大型逐步日志、HDF5、checkpoint、视频和 Jetson tegrastats 继续保存在双机 `runs/` 目录，不纳入 Git。当前结果属于网络化处理器在环与软实时评估，不是严格硬实时证明，也不是实体机械臂真机抓取。
