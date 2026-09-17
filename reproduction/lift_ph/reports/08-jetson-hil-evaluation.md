# 阶段 8：Jetson Orin 网络化处理器在环推理与 20 Hz 评估

本阶段把阶段 7 冻结的 Vision BC checkpoint 迁移到 Jetson Orin Nano Super 8GB。RTX 5060 笔记本继续运行 robosuite / MuJoCo Lift 仿真环境，Jetson 通过独立有线网络接收观测并返回 7 维动作。该结构验证的是 Jetson 实机处理器上的策略推理、网络协议和网络化控制时序，不包含实体机械臂执行机构。

本报告依据双机终端验收、已有运行产物、轻量结果表和用户人工视频检查整理。正式 100 回合评估完整执行，但结果为 99/100，因此阶段结论记录为 `COMPLETED_WITH_RECORDED_DEVIATION`，不能写成严格全成功门禁 PASS。

## 1. 验收结论

| 项目 | 结果 |
| --- | --- |
| Jetson 环境、网络、时钟、存储与 CUDA 审计 | PASS |
| checkpoint 传输、哈希与 GPU 加载 | PASS |
| 固定输入跨设备一致性 | PASS |
| 协议自检与单次网络探针 | PASS |
| 单回合网络闭环 | 1/1 成功，PASS |
| 20 Hz 独立时序验收 | 1000 次测量，0 timeout，0 deadline miss，PASS |
| 正式 100 回合执行完整性 | 100/100 回合完成，PASS |
| 正式评估样本成功率 | 99/100 |
| 严格全成功门禁 | FAIL |
| 失败回合诊断 | episode 53 为有效策略 rollout 达到 400 步上限，不是网络或产物损坏 |
| 最终分类 | `COMPLETED_WITH_RECORDED_DEVIATION` |

本阶段可称为处理器在环验证，并完成了网络化 20 Hz 软实时验收。Ubuntu、普通以太网、Python 和 MuJoCo 组合没有提供严格硬实时保证，因此不把结果描述成严格硬实时 HIL。功能闭环中的 plant 仍是笔记本仿真环境，也不把本阶段描述成实体机械臂真机部署。

## 2. 双机环境与网络拓扑

| 项目 | RTX 5060 笔记本 | Jetson Orin |
| --- | --- | --- |
| 用户 / 主机名 | `lx` / `lxlab` | `jetson` / `yahboom` |
| 角色 | Lift 仿真、评估与证据汇总 | Vision BC CUDA 推理服务器 |
| 架构 | x86_64 | aarch64 |
| 直连接口 / 地址 | `enp4s0` / `192.168.50.1/24` | `enP8p1s0` / `192.168.50.2/24` |
| Python | 3.11.16 | 3.10.12 |
| PyTorch / CUDA | 2.7.1+cu128 / 12.8 | 2.5.0a0+872d972e41.nv24.08 / 12.6 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU | Orin |

Jetson 为 NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super，系统为 Ubuntu 22.04.5 LTS、L4T R36.4.3、内核 5.15.148-tegra，功耗模式为 `MAXN_SUPER`。NVIDIA PyTorch wheel 保持不变，没有被通用 PyPI torch 覆盖。

笔记本运行 chrony，并仅允许固定对端 `192.168.50.2/32` 使用 UDP 123。Jetson 的 systemd-timesyncd 使用 `192.168.50.1`，重启后的 `NTP=yes`、`NTPSynchronized=yes` 均通过验收。正式测试前后两台设备的直连地址和路由保持不变。

Jetson 根分区 `/dev/nvme0n1p1` 已从约 90.1 GiB 扩展到 237 GiB，ext4 文件系统显示约 232.242 GiB，本次最终环境审计时可用约 159.349 GiB。分区表与重启挂载均通过检查。扩容前 GPT 备份同时保存在 Jetson 和笔记本，文件哈希一致。

`software-wakeword.service` 保持 `inactive/dead/disabled`，且没有残留语音唤醒进程。最终环境审计还确认 Jetson CUDA 可用、GPU 为 Orin、端口 8765 空闲、必要路径存在以及全部已部署文件哈希一致。

环境审计文件为 [stage08-environment-audit.json](../results/stage08-environment-audit.json)，SHA256 为 `bf15cd78877530227a1d49d125faa99c2e754e9c0533bc44fefd710e2d9975b4`。

## 3. 冻结源码、环境与 checkpoint

Jetson 使用独立目录：

```text
/home/jetson/venvs/robomimic-hil
/home/jetson/robomimic-hil/source/robomimic-src
/home/jetson/robomimic-hil/checkpoints
/home/jetson/robomimic-hil/runtime/stage08
/home/jetson/robomimic-hil/runs
```

源码从笔记本分支 `lift-ph-reproduction` 的提交 `85e16a255d093e423b62388dec2ae8f219049aff` 生成归档并传输。源码归档 SHA256 为 `ed5ca7a324036ec8b36308f204c9d0e700ae335f8e6d4fbff1de5d1cfa5898a5`，Jetson 以 editable 方式安装 robomimic 0.4.0。推理路径缺失的 `huggingface-hub` 0.24.7 通过离线 wheel 安装，没有重新安装 torch 或 torchvision。

冻结 checkpoint：

```text
model_epoch_80_Lift_success_1.0.pth
SHA256: 4a2ebaa236f2621d0ac8ecc54e625f82ce2a374e2b908d330db92e0312314cec
```

CPU 结构审计确认 checkpoint 含 `algo_name`、`config`、`env_metadata`、`model` 和 `shape_metadata`，序列化张量 514 个，数值均有限。策略输入是两路 `(3,84,84)` RGB 和 9 维机器人本体状态，动作维度为 7，不使用深度图。

Jetson GPU 加载得到 `BC_GMM`，参数量 23,661,963，与预期完全一致。参数和缓冲区均位于 `cuda:0`，全部模块处于 eval 模式。模型加载用时约 0.99 秒，加载后的 GPU allocated 约 184.7 MiB。

## 4. 固定输入与跨设备一致性

固定输入使用相同的两路 HWC `uint8` 图像和 `float32` 低维状态。处理后的输入 SHA256 在 Jetson 和笔记本均为：

```text
f98679b6980a07005cbf3625f2592c36ba84240614109ee7ebd430628798d335
```

| 比较对象 | 最大绝对差 | 验收 |
| --- | ---: | --- |
| GMM means | 5.15580177307e-05 | `< 1e-4`，PASS |
| GMM scales | 0 | PASS |
| GMM probabilities | 1.84774398804e-06 | `< 1e-4`，PASS |
| winning-mode mean action | 2.82227993011e-05 | `< 1e-4`，PASS |

两端 winning mode 均为 2，所有输出有限，概率和有效。该检查比较分布参数与 mode action，避免把不同 PyTorch / CUDA / GPU 上的随机采样位流误当成必须逐位一致。

Jetson 固定输入的 30 次核心推理统计为：均值 29.567 ms、中位数 28.591 ms、p95 33.735 ms、最大 34.025 ms，单机核心推理满足 50 ms 预算。

## 5. 网络协议与单次探针

协议配置为 [stage08-hil-protocol.json](../configs/stage08-hil-protocol.json)，版本 1，magic 为 `RMH1`，服务器为 `192.168.50.2:8765`，固定客户端为 `192.168.50.1`。请求包含两路图像和三组低维状态，响应包含 7 维动作。

协议实现包含：

- 固定协议版本与 magic；
- episode ID 和单调 sequence ID；
- dtype、shape、长度和 CRC32 检查；
- 非有限动作和动作范围检查；
- 超时、状态和结构化错误响应；
- 新 episode 必须从 sequence 0 开始；
- 重复和乱序 sequence 拒绝。

笔记本和 Jetson 的协议自检均通过 CRC 损坏、形状错误、dtype 错误、非有限动作、动作越界和序列错误拒绝测试。

seed 对齐后的单次网络探针结果：动作与固定输入参考的最大差为 `4.65661287308e-10`，往返 48.109 ms，服务器推理 41.025 ms，50 ms 网络往返预算 PASS。

## 6. 单回合功能闭环与视觉检查

功能闭环在笔记本创建 Lift 环境，每步把真实仿真观测转换为协议格式，经 Jetson 推理后把动作返回环境。固定 seed `20260915` 的单回合结果为：

| 指标 | 结果 |
| --- | ---: |
| 成功数 | 1/1 |
| Horizon / Return | 47 / 1.0 |
| 动作延迟 p50 / p95 / max | 40.287 / 44.127 / 50.195 ms |
| deadline miss | 1 |
| 控制周期 overrun | 9 |

该回合用于确认观测转换、请求顺序、动作执行、轨迹保存和视频链路正确。时序严格验收使用下一节独立的 paced benchmark，不能用单回合的偶然统计替代。

生成的 HDF5 含 `(47,7)` 动作，视频首帧为 `(336,672,3)`，并包含 agentview 与 eye-in-hand 两路画面。用户人工检查确认两路视角、图像方向和抓取动作均正常，方块被精准夹起。视频观感较模糊是因为策略原始输入只有 84 × 84，回放把两路低分辨率画面放大显示；这不表示传输丢帧或策略输入被再次降采样。

## 7. 20 Hz 独立时序与资源验收

独立 benchmark 先执行 20 次客户端 warm-up，再按 20 Hz 墙钟节拍测量 1000 次请求。Jetson 模型另有 10 次服务器 warm-up。严格稳定条件预先定义为 timeout 数和 deadline miss 数均为 0。

| 指标 | 结果 |
| --- | ---: |
| 测量请求数 | 1000 |
| 控制周期 | 50 ms |
| timeout / deadline miss | 0 / 0 |
| 动作延迟 mean | 37.518 ms |
| 动作延迟 p50 / p95 / p99 / max | 37.790 / 42.616 / 43.840 / 44.789 ms |
| 网络往返 p99 | 43.299 ms |
| 服务器推理 p99 | 40.452 ms |
| release lateness p99 | 0.551 ms |
| 请求间隔 p50 / p99 | 50.000 / 50.412 ms |
| 严格 20 Hz 稳定门禁 | PASS |

与测量窗口对齐的 53 条 tegrastats 样本显示：GPU 利用率 p95 为 98%，GPU 与 junction 最高温度均为 48.781 °C，VDD_IN 平均约 7.027 W、最大约 7.279 W，RAM 使用最大 2860 MiB，swap 使用为 0。功耗模式为 `MAXN_SUPER`。

轻量汇总见 [stage08-latency-summary.csv](../results/stage08-latency-summary.csv)。该 CSV 有 28 行，分别覆盖客户端延迟、服务器阶段延迟和 Jetson 资源。原始逐请求日志与 tegrastats 保留在本机 `runs/lift_ph/hil/`，不纳入 Git。

## 8. 正式 100 回合配对评估

正式评估冻结同一 checkpoint，使用 seed `20260915`、100 回合、horizon 400、成功即终止。服务器在一条持续连接上处理 5123 个请求，没有协议错误。客户端完成全部 100 回合并保存 summary、逐步 JSONL 和 HDF5，执行完整性为 PASS。

| 指标 | 阶段 7 笔记本基线 | 阶段 8 Jetson HIL |
| --- | ---: | ---: |
| 完成回合 | 100 | 100 |
| 成功回合 | 100 | 99 |
| 样本成功率 | 1.00 | 0.99 |
| 成功率 Wilson 95% 区间 | [0.9630, 1.0000] | [0.9455, 0.9982] |
| 全部回合平均 Horizon | 46.47 | 51.23 |
| 成功回合平均 Horizon | 46.47 | 47.7071 |

两份 HDF5 的 100 组初始物理状态逐元素完全一致，最大绝对差为 0；`model_file` 全部一致。阶段 7 未保存 `ep_meta`，阶段 8 保存空字典字符串 `{}`，归一化后语义一致，因此严格语义初始条件配对为 100/100。

配对结果为：两端都成功 99 回合，仅阶段 7 成功 1 回合，仅阶段 8 成功 0 回合，两端都失败 0 回合。精确双侧 McNemar p 值为 1.0。该样本没有检出显著成功率差异，但 p=1.0 不证明两个运行平台等价，也不能把 99/100 改写成 100/100。

### 8.1 episode 53 诊断

唯一失败项为 `stage08-formal100-rollout-053`：执行 400 步、Return 0、以 horizon 结束。该回合具有以下证据：

- 400 个推理请求全部返回 `ok`；
- sequence 0～399 连续，无重复或乱序；
- timeout 和 deadline miss 均为 0；
- 动作全部有限，并满足协议绝对上限 1.001；
- JSONL 与 HDF5 动作逐元素一致；
- 网络往返 p99 41.850 ms、最大 45.991 ms；
- 服务器推理 p99 39.318 ms、最大 40.895 ms。

因此 episode 53 分类为 `VALID_POLICY_ROLLOUT_REACHED_HORIZON`，不能归因于网络丢包、超时、协议错误或证据文件损坏。

在 RTX 5060 上恢复相同初始状态并对齐到相同策略调用位置后，闭环同样运行到 400 步且失败。两端初始观测哈希一致，但长期随机动作序列不逐位相同；BC_GMM 在 `low_noise_eval=true` 下仍执行分布采样，不同 PyTorch、CUDA 和 GPU 后端会形成不同随机采样路径。该反事实结果支持“策略随机轨迹偏差”，但不证明所有平台行为等价，也不改变阶段 8 的 99/100 记录。

正式结果见 [stage08-independent-evaluation.csv](../results/stage08-independent-evaluation.csv)。分析器把最终分类记录为 `COMPLETED_WITH_RECORDED_DEVIATION`，严格全成功门禁保留为 FAIL，没有通过重跑或挑选结果替换失败回合。

## 9. 轻量归档与证据哈希

阶段 8 提交候选包括：

- [协议配置](../configs/stage08-hil-protocol.json)
- [协议模块](../scripts/stage08_protocol.py)
- [Jetson 策略服务器](../scripts/stage08_jetson_policy_server.py)
- [Lift 仿真客户端](../scripts/stage08_lift_sim_client.py)
- [延迟分析器](../scripts/stage08_analyze_latency.py)
- [配对评估分析器](../scripts/stage08_analyze_evaluation.py)
- [环境审计](../results/stage08-environment-audit.json)
- [延迟摘要](../results/stage08-latency-summary.csv)
- [独立评估摘要](../results/stage08-independent-evaluation.csv)

| 对象 | SHA256 |
| --- | --- |
| checkpoint | `4a2ebaa236f2621d0ac8ecc54e625f82ce2a374e2b908d330db92e0312314cec` |
| 协议配置 | `4436233ea0f3e62f57d4c97f12f19eb206bddd24707faab33da9167f00041501` |
| 协议模块 | `14629cd529b6fd5da70f73c4bcc6081b5c6d63e13c0430a3eb55addbbddeba28` |
| Jetson 策略服务器 | `b457a755dc52d38e54a57f33fa9735a6cf122d549341308660bbbcae538b86c3` |
| Lift 仿真客户端 v3 | `b148d8074b581d13a0fcc76f933302d64ee44409e79b7beb11493ab7d756e9f2` |
| 延迟分析器 | `c943d968a19b6e7ee669c4ea332ffa5c369a7e17d20e6572159ca552e542c4cc` |
| 配对评估分析器 | `08c17fa9d186c91640bfb4dd887938b216cc5a3a802c31b5d654e3f58a864d3e` |
| 环境审计 JSON | `bf15cd78877530227a1d49d125faa99c2e754e9c0533bc44fefd710e2d9975b4` |
| 延迟摘要 CSV | `4a7471c69bddcd3241b9223b54fb9d33772a06398e7ab3e7205c8cbeca59d427` |
| 独立评估 CSV | `095144723ec295ae17752cf5f2e18f240fc1917c9e0be366d4c42d790a73f9ab` |
| 正式评估 summary | `61d61627f6b27f4df86489acbc67de2efd8377c00bce48a5ea8427f8106eba50` |
| 配对评估分析 JSON | `3e5768f0421e16d9a5109039cb4bcdb1ad97af1b734b9c409ec1131517e9e183` |

大型 checkpoint、逐步延迟日志、HDF5、视频、服务器 JSONL 和 tegrastats 保留在以下本机目录，不直接提交 Git：

```text
/home/lx/robomimic/runs/lift_ph/hil/
/home/jetson/robomimic-hil/runs/
```

## 10. 解释边界与最终状态

1. 阶段 8 验证的是 Jetson 处理器推理和网络化仿真闭环，不是实体机械臂真机抓取。
2. 20 Hz 结果是当前普通 Linux 和专用直连网络条件下的软实时实测，不是硬实时最坏情况证明。
3. 99/100 和阶段 7 的 100/100 都是单训练种子、有限 100 回合样本，不能外推为所有初始条件的保证。
4. McNemar p=1.0 表示本样本未检出显著差异，不表示统计等价；如需等价或非劣效结论，应预先定义界限并扩大实验设计。
5. GMM 策略仍包含随机采样。固定输入分布参数的一致性与完整闭环轨迹逐步相同是两个不同问题。
6. 功能闭环的少量 overrun 不作为独立时序门禁；专门的 1000 次 paced benchmark 才是 20 Hz 验收依据。
7. 正式评估保留 episode 53，不重跑、不挑选、不用补跑成功回合覆盖原结果。
8. 本报告生成时尚未声明 Git 提交或远程推送完成；最终 Git 状态以后续终端验收为准。

阶段 8 最终记录：

```text
ExecutionStatus=PASS
FunctionalNetworkPath=PASS
StrictStable20HzAcceptance=PASS
FormalEvaluation=99/100
StrictAllSuccessGate=FAIL
Stage08Status=COMPLETED_WITH_RECORDED_DEVIATION
```
