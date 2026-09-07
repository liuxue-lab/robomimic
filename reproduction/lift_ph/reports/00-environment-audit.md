# Lift-PH 复现实验：环境审计报告

## 1. 实验目标

本项目基于官方 robomimic 与 robosuite，复现 Lift-PH 模仿学习基准。

第一阶段实验范围：

- 任务：Lift
- 数据类型：PH（Proficient Human）
- 观测类型：low-dim state
- 首个算法：Behavior Cloning（State BC）
- 评估方式：MuJoCo 闭环 rollout
- 计划数据文件：low_dim_v15.hdf5

当前尚未下载和审计数据集，也尚未开始模型训练。

## 2. 硬件环境

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- 显存：8151 MiB
- NVIDIA 驱动：595.84
- CUDA 可用：True

## 3. 软件环境

- 操作系统：Ubuntu 22.04，Linux 6.8.0-138-generic
- Python：3.11.16
- PyTorch：2.7.1+cu128
- MuJoCo：3.2.7
- robomimic：0.4.0
- robosuite：1.5.1

## 4. 源码版本

### robomimic

- GitHub 仓库：https://github.com/ARISE-Initiative/robomimic
- Git commit：7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9
- 本地源码：/home/lx/robomimic
- 当前实验分支：lift-ph-reproduction

### robosuite

- GitHub 仓库：https://github.com/ARISE-Initiative/robosuite
- Git commit：a071383d53568ab798eb315c0e95357911be922d
- 本地源码：/home/lx/projects/robot-il/vendor/robosuite

## 5. Python 实际导入位置

- robomimic：/home/lx/robomimic/robomimic/__init__.py
- robosuite：/home/lx/projects/robot-il/vendor/robosuite/robosuite/__init__.py

这说明训练时使用的是指定的本地可编辑源码，而不是其他位置残留的安装副本。

## 6. 已完成检查

- robomimic 导入成功
- robosuite 导入成功
- MuJoCo 导入成功
- PyTorch CUDA 可用
- GPU 能够被 PyTorch 识别
- robomimic 与 robosuite 源码提交已固定
- pip 和 Conda 依赖快照已保存
- `/runs/` 已设置为 Git 忽略目录

## 7. 已知非阻塞警告

robosuite 启动时提示未安装 `robosuite_models`。

该扩展主要提供额外机器人模型，当前 Lift 基准不依赖这些额外机器人，因此暂不安装。若后续任务明确需要，再单独处理。

## 8. 审计结论

当前环境满足 Lift-PH low-dim State BC 复现实验的启动条件。

下一步为下载 Lift-PH low-dim v1.5 数据集，并检查文件完整性、HDF5 层级结构、轨迹数量、样本数量和观测维度。
