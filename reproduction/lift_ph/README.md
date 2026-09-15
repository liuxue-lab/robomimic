# robomimic Lift-PH 教学型复现

使用 robomimic v0.4.0 与 robosuite v1.5.1，在 Lift-PH low-dim 数据集上复现普通 State BC 和 State BC-RNN。通过专家数据审计、仿真回放、官方配置核对、正式训练、冻结 checkpoint 后的独立评估及比较分析，形成可追溯的机器人模仿学习实验记录。

本项目使用低维状态输入和仿真环境，不需要实体机械臂。它是单训练种子的教学型复现，不代表完成了多种子统计复现、视觉策略或真机迁移验证。

## 1. 当前结果

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

阶段 0～6 的核心复现已完成。阶段 6 比较材料提交为 `5cd56793a2c6cbd1a152429fb1111fe1a141b741`；用户终端已确认本地、远程跟踪分支与 GitHub HEAD 一致，领先/落后为 0/0，工作区干净。该提交是本 README 完成状态的验收依据；后续文档记录同步不改变实验结果。

阶段 6 已修复生成 SVG 的行尾空格检查问题，绘图脚本会自动清理这类空格；恢复后 7 个文件的提交内容哈希与远程归档均已通过核验。具体过程见对比报告第 10 节。

## 3. 数据与训练设置

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
| `configs/` | 已验收的两份官方配置 |
| `environment/` | 环境与源码版本记录 |
| `reports/` | 各阶段报告 |
| `scripts/` | 数据审计与本阶段绘图脚本 |
| `results/` | 从已验收日志提取的轻量 CSV 结果表 |
| `media/` | 对比图 PNG / SVG |
| 仓库根目录下 `runs/lift_ph/` | 运行配置、训练日志、checkpoint、评估日志与视频等本机证据，不入 Git |

## 5. 查看与复用结果

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

Vision BC、VLA 微调、强化学习和真机部署属于后续拓展，不计入当前七阶段核心复现。
