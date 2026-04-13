# HIL-SERL SurRoL 部署与训练指南

> 适用环境：Ubuntu 22.04 + NVIDIA RTX 3060 + Geomagic Touch 触觉遥操作设备

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统要求](#2-系统要求)
3. [基础环境安装](#3-基础环境安装)
4. [OpenHaptics SDK 安装（Touch 触觉设备）](#4-openhaptics-sdk-安装)
5. [项目部署](#5-项目部署)
6. [数据采集](#6-数据采集)
7. [训练](#7-训练)
8. [监控与调参](#8-监控与调参)
9. [常见问题](#9-常见问题)
10. [配置参数说明](#10-配置参数说明)

---

## 1. 项目概述

本项目基于 **HIL-SERL**（Human-in-the-Loop Sample-Efficient Robotic RL）框架，使用 **SAC**（Soft Actor-Critic）算法在 **SurRoL v2** 仿真环境中训练 Franka Panda 机械臂完成拾取任务。

### 架构

```
┌─────────────────────────────────────────────────────┐
│                    训练流程                            │
│                                                       │
│  ┌──────────┐   gRPC    ┌──────────┐                 │
│  │  Actor   │◄─────────►│ Learner  │                 │
│  │          │  port     │          │                 │
│  │ ·环境交互 │  50051    │ ·SAC更新  │                 │
│  │ ·策略推理 │           │ ·Q函数    │                 │
│  │ ·人工干预 │           │ ·温度调节  │                 │
│  └────┬─────┘           └──────────┘                 │
│       │                                               │
│  ┌────▼─────────────────────────────┐                │
│  │         SurRoL_v2 仿真环境         │                │
│  │  ·PyBullet 物理引擎               │                │
│  │  ·Franka Panda 机械臂             │                │
│  │  ·前置/腕部双相机                  │                │
│  └────┬─────────────────────────────┘                │
│       │                                               │
│  ┌────▼─────┐                                        │
│  │  Touch   │ ← 人类遥操作（HIL）                     │
│  │ 触觉设备  │   Button1: 介入控制                     │
│  │          │   Button2: 夹爪开合                     │
│  └──────────┘                                        │
└─────────────────────────────────────────────────────┘
```

### 人工介入（HIL）机制

- **Button 1（前方按钮）**：按住 = 人工接管控制，松开 = 策略自主控制
- **Button 2（后方按钮）**：切换夹爪开/合
- 人工介入的 transition 同时存入 online buffer 和 offline buffer
- 训练时 batch 采样：50% online（策略+人工混合） + 50% offline（纯人工演示）
- 随着训练进行，逐步减少人工干预频率

---

## 2. 系统要求

### 硬件

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GTX 1080 (8GB) | RTX 3060 (12GB) 或更高 |
| CPU | 4 核 | 8 核以上 |
| 内存 | 16 GB | 32 GB |
| 触觉设备 | Geomagic Touch / Phantom Omni | - |
| USB | 1 个 USB 端口（Touch 设备） | - |

### 软件

| 软件 | 版本 |
|------|------|
| Ubuntu | 22.04 LTS |
| NVIDIA 驱动 | >= 535 |
| CUDA | 12.x |
| Python | 3.12 |
| Conda | Miniconda 或 Anaconda |

---

## 3. 基础环境安装

### 3.1 NVIDIA 驱动

```bash
# 检查当前驱动
nvidia-smi

# 如果未安装
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

### 3.2 CUDA Toolkit

```bash
# 安装 CUDA 12.4（推荐）
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent

# 添加环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3.3 Conda

```bash
# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 3.4 系统依赖

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake git swig \
    libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    libusb-1.0-0-dev libudev-dev \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libncurses5-dev
```

---

## 4. OpenHaptics SDK 安装

Geomagic Touch 需要 OpenHaptics SDK 才能在 Linux 下使用。

### 4.1 下载与安装 OpenHaptics

```bash
# 方法1：从 3D Systems 官网下载
# 访问 https://support.3dsystems.com/s/article/OpenHaptics-for-Linux-Developer-Edition-v34
# 下载 openhaptics_3.4-0-developer-edition-amd64.deb

# 安装
sudo dpkg -i openhaptics_3.4-0-developer-edition-amd64.deb
sudo apt install -f  # 修复依赖

# 方法2：如果有离线包
# sudo dpkg -i openhaptics*.deb TouchDriver*.deb
```

### 4.2 安装 Touch 驱动

```bash
# 安装 Touch/Phantom 设备驱动
# 从 3D Systems 下载 Touch Device Driver for Linux
sudo dpkg -i geomagic_touch_device_driver_*.deb

# 验证安装
ls /opt/OpenHaptics/
# 应该看到 Developer/ Academic/ 等目录
```

### 4.3 设置 USB 权限

```bash
# 添加 udev 规则让普通用户可以访问 Touch 设备
sudo tee /etc/udev/rules.d/99-touch-haptic.rules << 'EOF'
# Geomagic Touch / Phantom Omni
SUBSYSTEM=="usb", ATTR{idVendor}=="0484", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

# 将用户加入 plugdev 组
sudo usermod -aG plugdev $USER
```

### 4.4 验证 Touch 设备

```bash
# 插入 Touch USB，检查是否识别
lsusb | grep -i "haptic\|phantom\|geomagic\|sensable"

# 启动设备配置工具
/opt/OpenHaptics/Developer/3.4-0/bin/TouchSetup
```

---

## 5. 项目部署

### 5.1 克隆项目

```bash
cd ~/projects  # 或你选择的工作目录
git clone --recursive https://github.com/LIJianxuanLeo/hilserl-surrol-improved.git
cd hilserl-surrol-improved
```

### 5.2 一键安装

```bash
bash setup.sh
```

此脚本会自动：
1. 创建 conda 环境 `hilserl`
2. 安装 SurRoL_v2 及其子模块（PyBullet、pybullet_rendering、panda3d-kivy）
3. 编译 Touch 触觉设备 Python 绑定（需要 OpenHaptics SDK）
4. 安装 LeRobot 及依赖
5. 配置所有 JSON 配置文件中的路径
6. 链接演示数据集到 HuggingFace 缓存

### 5.3 手动安装（如果一键脚本失败）

```bash
# 1. 创建环境
conda create -n hilserl python=3.12 -y
conda activate hilserl

# 2. 安装 SurRoL_v2
cd SurRoL_v2
git submodule update --init --recursive
pip install -e .

# 3. 编译 Touch 驱动
bash setup_haptic.sh

# 4. 安装 LeRobot
cd ../lerobot
pip install -e ".[sac]"
pip install grpcio protobuf

# 5. 手动编辑配置文件
# 将 train_config_gym_hil_touch.json 和 env_config_gym_hil_touch_record.json 中的
# "__HAPTIC_MODULE_PATH__" 替换为实际路径，例如：
# /home/你的用户名/projects/hilserl-surrol-improved/SurRoL_v2/haptic_src

# 6. 链接数据集（如果使用预采集数据）
mkdir -p ~/.cache/huggingface/lerobot/local
ln -s $(pwd)/franka_sim_touch_demos ~/.cache/huggingface/lerobot/local/franka_sim_touch_demos
```

### 5.4 验证安装

```bash
conda activate hilserl

# 验证 PyBullet
python -c "import pybullet; print('PyBullet OK')"

# 验证 SurRoL
python -c "import surrol; print('SurRoL OK')"

# 验证 LeRobot
python -c "import lerobot; print('LeRobot OK')"

# 验证 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 验证 Touch 驱动（需要 Touch 设备已连接）
python -c "
import sys
sys.path.insert(0, 'SurRoL_v2/haptic_src')
import touch_haptic
print('Touch haptic driver OK')
"
```

---

## 6. 数据采集

### 6.1 采集演示数据（使用 Touch 设备）

**前提条件**：Touch 设备已连接并通过验证。

```bash
conda activate hilserl
cd lerobot

# 启动数据采集（默认录制 30 个 episode）
python -m lerobot.rl.gym_manipulator \
    --config_path env_config_gym_hil_touch_record.json
```

**操作指南**：
- Touch 手写笔移动 → 控制机械臂末端位置（xyz）
- Touch 手写笔旋转 → 控制机械臂末端姿态（rpy）
- **Button 2（后方按钮）** → 切换夹爪开/合
- **Clutch 模式（采集时默认开启）**：
  - 始终为人工控制
  - 每完成一次成功拾取，自动记录一个 episode

**任务流程**：
1. 环境重置：机械臂回到初始位置，方块随机出现在桌面
2. 操作方块：使用 Touch 控制机械臂移动到方块上方 → 下降 → 合夹爪 → 抬起
3. 成功条件：方块被抬起到一定高度
4. 每个 episode 最长 15 秒

**采集建议**：
- 至少采集 **30 个** 成功的演示
- 保持操作流畅，避免不必要的抖动
- 数据保存在 `franka_sim_touch_demos/` 目录

### 6.2 使用键盘采集（无 Touch 设备时）

```bash
python -m lerobot.rl.gym_manipulator \
    --config_path env_config_gym_hil_il.json
```

---

## 7. 训练

### 7.1 训练概述

训练采用 **Actor-Learner 分布式架构**：

| 组件 | 进程 | 功能 |
|------|------|------|
| Learner | 终端 1 | SAC 策略更新、Q 函数训练、温度调节 |
| Actor | 终端 2 | 环境交互、策略推理、人工干预处理 |

两个进程通过 **gRPC**（端口 50051）通信。

### 7.2 启动训练

**终端 1 - 启动 Learner**：
```bash
conda activate hilserl
cd lerobot

python -m lerobot.rl.learner \
    --config_path train_config_gym_hil_touch.json
```

**终端 2 - 启动 Actor**（等 Learner 显示 "gRPC server started" 后）：
```bash
conda activate hilserl
cd lerobot

python -m lerobot.rl.actor \
    --config_path train_config_gym_hil_touch.json
```

### 7.3 训练中的人工干预（HIL）

训练启动后，你可以通过 Touch 设备 **实时干预**：

- **不按 Button 1**：策略自主控制（观察策略行为）
- **按住 Button 1**：人工接管控制（纠正策略错误）
- **Button 2**：控制夹爪

**干预策略建议**：

| 训练阶段 | 建议干预率 | 说明 |
|----------|-----------|------|
| 0-10k 步 | 70-90% | 策略初期很差，大量人工引导 |
| 10k-30k 步 | 40-60% | 策略开始学习，减少干预 |
| 30k-60k 步 | 20-30% | 仅在关键失败时干预 |
| 60k+ 步 | 5-10% | 几乎不干预，观察策略表现 |

### 7.4 恢复训练

如果训练中断，可以恢复：
```bash
# 修改配置文件中 "resume": true
# 然后按照 7.2 步骤重新启动
```

### 7.5 仅使用离线数据训练（无 Touch 设备）

如果没有 Touch 设备，可以仅使用预采集的演示数据训练。但需要注意：
- Learner 需要 online buffer 中有数据才开始训练
- 需要 Actor 与环境交互（可以使用键盘环境）
- 将 `train_config_gym_hil_touch.json` 中 `teleop` 设为 `null`

---

## 8. 监控与调参

### 8.1 WandB 监控

训练默认 **关闭** WandB（避免未登录时交互提示卡住训练）。启用步骤：

```bash
# 1. 安装并登录
pip install wandb
wandb login  # 输入你的 API key（从 https://wandb.ai/authorize 获取）

# 2. 创建 ~/.netrc（如果登录后仍报错）
touch ~/.netrc

# 3. 修改配置文件 train_config_gym_hil_touch.json
#    将 "wandb": { "enable": false } 改为 "enable": true
```

**关键监控指标**：

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `loss_critic` | Critic TD 损失 | 应逐步下降或稳定 |
| `loss_actor` | Actor 策略损失 | 负值，绝对值增大表示策略改善 |
| `temperature` | SAC 温度 | 从 1.0 开始，逐步下降到 0.01-0.1 |
| `Episodic reward` | 每 episode 累积奖励 | 应逐步增加（max=1.0） |
| `Intervention rate` | 人工干预率 | 应随训练逐步降低 |
| `Episode length` | Episode 步数 | 成功的 episode 通常 30-80 步 |

### 8.2 收敛判断

**正常收敛信号**（使用阶段式奖励）：
- 前 1k 步：episodic reward > 0.05（策略开始学会移动方向）
- 前 5k 步：critic loss 开始下降，reward 达到 0.15+（靠近方块）
- 5k-15k 步：reward 出现 0.35+（开始抓取）
- 15k-30k 步：reward 出现 0.50+（成功抬起）
- temperature 从 1.0 逐步下降到 0.01-0.1
- 终端可见 `reward: 1.0` 的 episode

**不收敛信号**：
- critic loss 持续上升或 NaN
- temperature 卡在 1.0 不变
- 30k 步后 reward 仍低于 0.10（未学会接近）
- actor loss 出现极大值

### 8.3 调参建议

如果训练不收敛，按以下优先级调整：

1. **温度初始化（最重要）**：
   - 当前值：`1.0`（已从原始的 `0.01` 修复）
   - 如果探索不足，可提高到 `2.0`
   - 如果随机性过大，降低到 `0.5`

2. **折扣因子**：
   - 当前值：`0.99`（有效视野 100 步 = 10 秒）
   - pick 任务通常需要 5-10 秒完成
   - 不要低于 `0.97`

3. **学习率**：
   - Actor LR 应 <= Critic LR（当前 0.0001 vs 0.0003）
   - 如果训练不稳定，降低 actor_lr 到 `0.00005`

4. **梯度裁剪**：
   - 当前值：`1.0`（已从原始的 `10.0` 降低）
   - 防止梯度爆炸导致的训练不稳定

---

## 9. 常见问题

### Q1: Touch 设备无法识别

```bash
# 检查 USB 连接
lsusb | grep -i "0484\|2833"

# 检查驱动
ls /opt/OpenHaptics/Developer/3.4-0/

# 重新加载 udev 规则
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Q2: CUDA out of memory / 电脑死机

batch_size 已从 256 降至 128 以适配 RTX 3060 (12GB)。如果仍然 OOM：

```bash
# 进一步减小 batch_size（在 train_config_gym_hil_touch.json 中）
# 128 → 64

# 检查 GPU 内存占用
nvidia-smi

# 如果电脑已经死机，长按电源键重启
# 重启后检查进程是否残留
kill $(lsof -t -i:50051) 2>/dev/null
```

### Q3: gRPC 连接失败

```bash
# 确保 Learner 先启动，Actor 后启动
# 检查端口是否被占用
ss -tlnp | grep 50051

# 如果端口被占用，kill 旧进程
kill $(lsof -t -i:50051)
```

### Q4: SurRoL 仿真窗口不显示

```bash
# 设置显示环境变量
export DISPLAY=:0

# 如果使用远程连接（SSH），需要 X11 转发
ssh -X user@host

# 或者使用虚拟显示（headless）
sudo apt install xvfb
xvfb-run python -m lerobot.rl.actor --config_path train_config_gym_hil_touch.json
```

### Q5: Touch 编译 SWIG 失败

```bash
# 确保安装了 swig
sudo apt install swig

# 确保 OpenHaptics 头文件可找到
export CPATH=/opt/OpenHaptics/Developer/3.4-0/include:$CPATH
export LIBRARY_PATH=/opt/OpenHaptics/Developer/3.4-0/lib/lin-x86_64:$LIBRARY_PATH

# 重新编译
cd SurRoL_v2
bash setup_haptic.sh
```

### Q6: 训练速度慢

RTX 3060 预期训练速度：

| 阶段 | 预期速度 |
|------|---------|
| 纯离线学习 | ~200 步/秒 |
| 在线训练（含环境渲染） | ~5-10 步/秒 |
| 100k 步总训练时间 | ~3-6 小时 |

### Q7: 如何使用已有数据集（不重新采集）

项目已包含 30 个预采集的演示数据。如果 `setup.sh` 正常执行，数据集会自动链接。手动链接：

```bash
mkdir -p ~/.cache/huggingface/lerobot/local
ln -sf $(pwd)/lerobot/franka_sim_touch_demos \
    ~/.cache/huggingface/lerobot/local/franka_sim_touch_demos
```

---

## 10. 配置参数说明

### 训练配置 (`train_config_gym_hil_touch.json`)

#### 原始配置 vs 改进配置对比

| 参数 | 原始值 | 改进值 | 修改原因 |
|------|--------|--------|---------|
| `temperature_init` | 0.01 | **1.0** | 0.01 导致 SAC 无法探索，策略从一开始就接近确定性 |
| `discount` | 0.97 | **0.99** | 0.97 有效视野仅 33 步(3.3s)，pick 需要 5-10s |
| `temperature_lr` | 0.0003 | **0.001** | 加快温度自适应调节速度 |
| `actor_lr` | 0.0003 | **0.0001** | Actor 学习率应低于 Critic，避免策略更新过快 |
| `grad_clip_norm` | 10.0 | **1.0** | 更严格的梯度裁剪，防止训练不稳定 |
| `std_max` | 5.0 | **10.0** | 允许更大的策略标准差，增强初期探索 |
| `eval_freq` | 20000 | **10000** | 更频繁评估，更早发现问题 |
| `save_freq` | 20000 | **10000** | 更频繁保存，减少数据丢失风险 |
| `batch_size` | 256 | **128** | RTX 3060 (12GB) 使用 256 会 OOM 死机 |
| `wandb.enable` | false | **false** | 默认关闭，需要时手动开启（先 `wandb login`） |
| `reward_type` | sparse | **dense + staged** | 阶段式密集奖励加速收敛 |
| `action.min/max` | [-0.4,..,0] / [0.4,..,2] | **[-1.0,..,0] / [1.0,..,2]** | 匹配实际数据中的动作范围 |
| `online_step_before_learning` | 100 | **200** | 积累更多初始样本后再开始学习 |
| `haptic_module_path` | 硬编码绝对路径 | **自动配置** | setup.sh 自动设置为项目内路径 |

#### 关键参数详细说明

**SAC 温度机制**：
- `temperature_init = 1.0`：SAC 的核心是最大熵 RL，温度 alpha 控制探索-利用平衡
- alpha 过低（0.01）→ 策略接近确定性 → 无法发现成功的抓取序列
- alpha = 1.0 → 初始强探索 → 随训练自动下降到最优值（通常 0.01-0.1）
- `target_entropy = -dim(A)/2 = -3`：自动温度调节的目标熵值

**折扣因子**：
- `discount = 0.99`：有效视野 = 1/(1-0.99) = 100 步 = 10 秒（@10fps）
- pick 任务需要：移动(2-3s) → 下降(1-2s) → 抓取(1s) → 抬起(2-3s) = 6-9s
- 因此需要至少 0.99 的折扣因子

**动作空间**：
- 维度 0-2（delta_xyz）：位置增量，范围 [-1, 1]
- 维度 3-5（delta_rpy）：姿态增量，范围 [-0.25, 0.25]
- 维度 6（gripper）：离散夹爪，范围 [0, 2]，经 wrapper 映射为 [-1, 1]
- `num_discrete_actions = 3`：夹爪有 3 种状态（开/中/合）

**阶段式密集奖励（StagedRewardWrapper）**：

替代原始稀疏奖励，提供连续梯度信号引导策略学习：

```
总奖励 = r_reach + r_grasp + r_lift  （范围 [0, 1]）

Stage 1 - 接近 (0~0.25):
  r_reach = 0.25 * exp(-10 * dist_xy) * exp(-10 * dist_z)
  → XY 平面接近方块 + 高度对齐

Stage 2 - 抓取 (0~0.25):
  is_grasped (方块被抬起 > 5mm) → 0.25
  is_near (3D 距离 < 5cm)       → 0.10
  else                           → 0.00

Stage 3 - 抬起 (0~0.50):
  r_lift = 0.50 * min(lift_height / target_height, 1.0)
  → 抬起越高奖励越大

成功完成任务 → reward = 1.0
```

设计目的：
- 随机策略得分 ~0.0（有改进方向）
- 靠近方块得分 ~0.15-0.25（引导学会移动）
- 抓住方块得分 ~0.35-0.50（引导学会合夹爪）
- 成功抬起得分 ~0.50-1.00（引导学会完整序列）
