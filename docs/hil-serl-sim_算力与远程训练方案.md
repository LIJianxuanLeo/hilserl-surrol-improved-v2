# hil-serl-sim 算力配置与远程训练方案

> **目的**：在我们 V1/V2 之外，如需复现/对比参考库 [`ggggfff1/hil-serl-sim`](https://github.com/ggggfff1/hil-serl-sim) 的训练，本文档给出硬件选型、云平台对比、远程训练完整工作流。
>
> **背景**：参考库基于 Berkeley SERL（JAX/Flax）+ REDQ-10 + ResNet-50 + 1280×720 双相机，是为研究级 GPU 服务器设计的。在 3090（24 GB）上会 OOM，详见后文显存解构。

---

## 一、显存需求解构

### 1.1 hil-serl-sim 的"显存吞兽"组件

| 组件 | 设计选择 | 显存代价 |
|------|---------|---------|
| **JAX 框架** | `XLA_PYTHON_CLIENT_PREALLOCATE=true`（默认） | 启动即占 90% 显存 |
| **PyTorch 共存** | `torch 2.3.0` 同时安装 | 与 JAX 抢同一块卡 |
| **REDQ critic ensemble** | `critic_ensemble_size=10` | 10 + 10 target = 20 副本 |
| **图像分辨率** | 双相机 1280×720 RGB | 单帧 2.7 MB |
| **encoder** | `resnet-pretrained`（多为 R50） | 25M 参数 ×N 副本 |
| **DRQ 数据增强** | 随机裁剪/平移 on GPU | 临时张量驻留 |
| **agentlace 异步** | actor/learner/buffer 三进程 | JAX runtime ×3 |
| **replay buffer** | 默认 capacity 1,000,000 | 1M × 多张图 |

### 1.2 不同 GPU 上的显存预算

| GPU | 总显存 | JAX 预分配后 | REDQ-10 | batch (256, 1280×720) | 状态 |
|-----|-------|-------------|---------|---------------------|------|
| RTX 3060 (12 GB) | 12 | 10.8 占用 | OOM | — | ❌ 立刻崩 |
| RTX 3090 (24 GB) | 24 | 21.6 占用 | 1.0 GB | 1.4 GB | ❌ 24.8 GB OOM |
| RTX 4090 (24 GB) | 24 | 21.6 占用 | 1.0 GB | 1.4 GB | ❌ 同样 OOM |
| RTX A6000 (48 GB) | 48 | 43.2 占用 | 1.0 GB | 1.4 GB | ✅ 富余 ~2 GB |
| A100 (40 GB) | 40 | 36 占用 | 1.0 GB | 1.4 GB | ⚠️ 临界（边缘可行） |
| A100 (80 GB) | 80 | 72 占用 | 1.0 GB | 1.4 GB | ✅ 富余 ~6 GB |
| H100 (80 GB) | 80 | 72 占用 | 1.0 GB | 1.4 GB | ✅ 富余 ~6 GB + 速度 2× |

**结论**：跑 hil-serl-sim **原始配置**至少需要 **40 GB** 显存（A100-40G 临界），稳健选 **A6000 或 A100-80G**。

### 1.3 优化后的最低配置

如果做以下显存优化：
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`
- `critic_ensemble_size=2`（放弃 REDQ）
- 相机分辨率降到 256×256
- 选 `resnetv1-10-frozen` encoder

| GPU | 优化后可用 | 状态 |
|-----|----------|------|
| RTX 3060 (12 GB) | 6 GB JAX + ~5 GB PyTorch | ⚠️ 紧张可跑 |
| RTX 3090 (24 GB) | 12 GB JAX + ~10 GB PyTorch | ✅ 舒适 |
| RTX 4090 (24 GB) | 12 GB JAX + ~10 GB PyTorch | ✅ 舒适 + 训练快 30% |

---

## 二、推荐硬件配置（三档）

### 档位 A：原汁原味（保留所有论文设置）

| 项 | 推荐 | 原因 |
|---|------|------|
| GPU | **A100 80G** 或 **H100 80G** | JAX 预分配 + REDQ-10 + 高分辨率 |
| CPU | 32 核以上 | actor 推理 + buffer server + agentlace |
| 内存 | 128 GB | replay buffer 1M 帧的 CPU 缓存 |
| 存储 | 1 TB NVMe SSD | 视频 demos + checkpoints |
| 网络 | 无特别要求 | 单机训练 |

### 档位 B：实用平衡（适度优化，保留 RLPD 核心）

| 项 | 推荐 | 原因 |
|---|------|------|
| GPU | **RTX A6000 (48 GB)** 或 **RTX 4090 (24 GB)** + 优化 | 平衡价格与性能 |
| CPU | 16-24 核 | 够用 |
| 内存 | 64 GB | replay buffer 适度缩小 |
| 存储 | 500 GB NVMe | |

需要的代码修改：
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
# config.py 中改：
critic_ensemble_size=4   # 折中
camera_resolution=(480, 480)
replay_buffer_capacity=300_000
```

### 档位 C：消费级 GPU 极限（仅复现，不追求论文性能）

| 项 | 推荐 | 原因 |
|---|------|------|
| GPU | **RTX 3090 (24 GB)** | 最便宜的可用方案 |
| CPU | 12+ 核 | |
| 内存 | 32 GB | |
| 存储 | 256 GB NVMe | |

需要的代码修改：
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
```
config 改：
```python
critic_ensemble_size=2
camera_resolution=(128, 128)
encoder_type="resnetv1-10-frozen"
batch_size=64
replay_buffer_capacity=100_000
```
**注意**：这套接近我们 V1/V2 的设计，性能会大幅低于论文报告。

---

## 三、云平台对比

### 3.1 单卡云 GPU 价格速查（2026 Q2 行情，按需实例）

| 平台 | A100-80G/h | A6000/h | 4090/h | 3090/h | 优势 |
|------|-----------|---------|--------|--------|------|
| **AWS p4d** | $4.10 | — | — | — | 企业级 SLA |
| **GCP** | $3.67 | — | — | — | 与 BigQuery 集成 |
| **Lambda Labs** | $1.99 | $1.10 | $0.80 | — | 学术友好、价格透明 |
| **RunPod** | $1.89 | $0.79 | $0.74 | $0.34 | 容器化，启动快 |
| **Vast.ai** | $0.90-1.50 | $0.50-0.90 | $0.40-0.70 | $0.20-0.40 | 最便宜（社区算力） |
| **AutoDL** (国内) | ¥9-12 | ¥5-7 | ¥3-5 | ¥1.5-2.5 | 国内访问快 |
| **Hyperstack** | $1.90 | $0.95 | — | — | 欧洲机房 |

### 3.2 单次训练成本估算（以 hil-serl-sim 论文设置为准）

参考库报告：30K steps ≈ 1 小时收敛。但实际复现往往需要 2-4× 论文时间。**按 4 小时/run 算**：

| 配置 | 平台 | 单次成本 | 备注 |
|------|------|---------|------|
| A100-80G × 4h | Lambda | $7.96 | 推荐 |
| A100-80G × 4h | RunPod | $7.56 | 容器更灵活 |
| A100-80G × 4h | Vast.ai | $3.60-6.00 | 取决于具体机器 |
| A6000 × 4h | RunPod | $3.16 | 性价比首选 |
| 4090 × 4h | Vast.ai | $1.60-2.80 | 极致性价比，但显存紧 |

如要做 3 seed × 2 配置（V1 + V2）= 6 次训练：
- A100 路线：$45-60
- A6000 路线：$20-30
- 4090 路线：$10-20（如能塞下）

### 3.3 推荐选择

| 需求 | 平台 | GPU |
|------|------|-----|
| 学术报告（最稳） | **Lambda Labs** | A100-80G |
| 个人项目（性价比） | **RunPod** | A6000 |
| 国内用户 | **AutoDL** | A6000 / 3090 |
| 极致预算 | **Vast.ai** | 4090 (24GB)，需做显存优化 |

---

## 四、远程训练完整工作流

### 4.1 准备阶段（本地）

#### Step 0：前置自检
```bash
# 本地需要的工具
which ssh rsync tmux
gh auth status   # 推送/拉取代码用

# 准备 SSH 公钥（如果还没）
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "you@example.com"
```

#### Step 1：创建云实例
以 **RunPod A6000** 为例：

1. 登录 [runpod.io](https://runpod.io) → Pods → Deploy
2. 选 **GPU**: RTX A6000 (48 GB VRAM)
3. 选 **Template**: `runpod/pytorch:2.3.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
4. 选 **Storage**: 100 GB Container Disk + 50 GB Volume（/workspace 持久化）
5. **SSH key**: 粘贴本地 `~/.ssh/id_ed25519.pub` 内容
6. 点击 **Deploy On-Demand**（约 1 分钟启动）

启动后会拿到 SSH 命令，类似：
```
ssh root@123.456.789.10 -p 22001 -i ~/.ssh/id_ed25519
```

#### Step 2：把 SSH 命令存为别名（提高易用性）
编辑 `~/.ssh/config` 添加：
```
Host hil-train
    HostName 123.456.789.10
    Port 22001
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

后续直接：
```bash
ssh hil-train
```

### 4.2 远程环境一次性配置

#### Step 3：登录后安装系统依赖
```bash
ssh hil-train

# 验证 GPU 可见
nvidia-smi    # 应看到 A6000 / 4090 / A100 等

# 系统包
apt update && apt install -y \
    git tmux htop nvtop ffmpeg \
    libgl1-mesa-glx libxrender1 libsm6 \
    build-essential

# Conda（如果模板没带）
which conda || (
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
)
```

#### Step 4：克隆仓库并装环境
```bash
cd /workspace   # 持久化目录
git clone https://github.com/ggggfff1/hil-serl-sim.git
cd hil-serl-sim

# 创建环境
conda create -n hilserl python=3.10 -y
conda activate hilserl

# JAX (CUDA 12)
pip install --upgrade "jax[cuda12_pip]==0.4.35" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# PyTorch (CUDA 12.1)
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 项目依赖
pip install -e .
pip install -e ./serl_robot_infra   # 如果有此子模块
pip install -e ./serl_launcher
```

#### Step 5：固化显存优化（关键）
在 `~/.bashrc` 末尾追加：
```bash
# JAX 显存优化（避免与 PyTorch 抢显存）
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

# 关闭 TF/XLA 不必要的日志
export TF_CPP_MIN_LOG_LEVEL=3
export JAX_PLATFORMS=cuda
```

激活：`source ~/.bashrc`

### 4.3 训练运行（核心环节）

#### Step 6：在 tmux 中启动长任务
```bash
# 创建会话
tmux new -s hil

# 在 tmux 里启动训练
cd /workspace/hil-serl-sim
conda activate hilserl

# 跑 pick_cube_sim 实验
cd examples
bash run_actor.sh   # 或对应的训练脚本
```

#### Step 7：脱离 tmux 但训练继续
按 `Ctrl+B`，然后按 `D` → 回到 SSH shell，训练在 tmux 后台继续。

```bash
# 查看运行中的会话
tmux ls

# 重新接入
tmux attach -t hil
```

#### Step 8：关键 — 让训练在 SSH 断线后存活

tmux 已经做到这一点。但额外保险：
```bash
# 先用 nohup 启动 tmux server（防止 daemon 被 kill）
nohup tmux new-session -d -s hil 'cd /workspace/hil-serl-sim/examples && conda activate hilserl && bash run_actor.sh > /workspace/train.log 2>&1' &

# SSH 退出后训练继续，重连后看日志：
tail -f /workspace/train.log
```

### 4.4 监控（实时观测）

#### Step 9：开第二个 SSH 会话做监控
```bash
# 本地新开终端
ssh hil-train

# GPU 实时占用
watch -n 1 nvidia-smi

# 或更友好的 nvtop
nvtop

# 系统负载
htop

# 训练日志
tail -f /workspace/train.log

# 训练输出文件（如果有 CSV 日志）
ls -lh /workspace/hil-serl-sim/examples/checkpoints/
```

#### Step 10：远程 GUI 查看仿真画面（可选）

如果想看 MuJoCo 渲染窗口（hil-serl-sim 默认 headless 不需要），用 **X11 forwarding**：
```bash
# 本地（macOS 需先装 XQuartz）
ssh -X hil-train

# 或更快的 VNC（适合 Linux/Windows 客户端）
# 服务端
apt install -y tightvncserver
vncserver :1 -geometry 1920x1080

# 本地（用 VNC Viewer 连接 IP:5901）
```

### 4.5 数据回传

#### Step 11：训练完成后下载结果
```bash
# 本地执行
cd ~/Downloads
mkdir hil_run_$(date +%Y%m%d) && cd "$_"

# 拉回检查点 + 日志（rsync 支持断点续传）
rsync -avzP --partial \
    -e "ssh -p $(awk '/Port/{print $2; exit}' ~/.ssh/config)" \
    hil-train:/workspace/hil-serl-sim/examples/checkpoints/ \
    ./

rsync -avzP \
    hil-train:/workspace/train.log ./

ls -lh
```

如果 checkpoint 巨大（数 GB），可在远程先打包：
```bash
# 远程
cd /workspace/hil-serl-sim/examples
tar -czf /workspace/run_$(date +%Y%m%d).tar.gz checkpoints/
ls -lh /workspace/run_*.tar.gz
```
然后本地 `rsync` 拉这个 tar 包。

### 4.6 关机省钱

#### Step 12：训练完毕立刻关机！
**不要忘记** — 按需实例每分钟都在烧钱。

- **RunPod**：网页端 → Pods → 选实例 → **Stop** （保留 storage，下次恢复）或 **Terminate**（销毁）
- **Lambda Labs**：Dashboard → Terminate
- **Vast.ai**：Console → Destroy

如果用 spot 实例可能被强制回收，确保关键数据已 sync 到本地或 S3。

---

## 五、整合到我们 V1/V2 工作流

如果想用云 GPU 跑**我们的 V1/V2**（而不是 hil-serl-sim 原版），同样适用上述工作流，只是：

1. 仓库改为：
```bash
git clone https://github.com/LIJianxuanLeo/hilserl-surrol-improved.git
git clone https://github.com/LIJianxuanLeo/hilserl-surrol-improved-v2.git
```

2. 显存需求大幅降低（详见前文对比），**RTX 3090 / 4090 / A6000 都富余**，无需极端优化。

3. **限制**：我们的 V1/V2 依赖 Geomagic Touch 触觉设备做人工干预。云 GPU 上无法连接物理 Touch → 必须改用：
   - **键盘 teleop**：用 lerobot 的 keyboard teleoperator
   - **gamepad teleop**：插 Xbox/PS 手柄到本地，通过 SSH X11 forwarding 接入
   - **纯自动训练（无干预）**：在 config 中设 `teleop: null`，使用纯 RL（无 HIL）—— 但样本效率会大幅降低

推荐方案：**云 GPU 跑长训练 + 本地有 Touch 时跑短交互式训练做最终干预微调**。

---

## 六、踩坑清单（务必读）

| 问题 | 现象 | 解决 |
|------|------|------|
| JAX OOM | 启动 5 秒后崩，报 `RESOURCE_EXHAUSTED` | 设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| Disk full | 训练 1 小时后崩，TensorBoard 写不进去 | 容器盘小，把 output 重定向到 `/workspace`（持久卷） |
| SSH 断线丢训练 | 重连发现进程没了 | 用 tmux/nohup，**永远不要直接前台跑训练** |
| 上传带宽慢 | rsync 几十 MB/s 卡顿 | RunPod/Lambda 有专线，Vast.ai 看运气；用 `--bwlimit` 避免占满 |
| 检查点不可恢复 | 重启实例后 checkpoint 找不到 | 必须存到 `/workspace`（Volume），不是 `/root`（容器盘） |
| nvidia-smi 看不到 GPU | 容器没 GPU 权限 | 选模板时确认 GPU passthrough 已启用 |
| MuJoCo 渲染失败 | `RuntimeError: Could not create OpenGL context` | 安装 `libgl1-mesa-glx libegl1-mesa libosmesa6` |
| 多进程通信卡死 | actor/learner agentlace 卡 | 检查 firewall / 用 `127.0.0.1` 而非外网 IP |

---

## 七、快速参考卡

### 启动流程（5 步）
```bash
# 1. 本地：连接
ssh hil-train

# 2. 远程：进入 tmux + 激活环境
tmux new -s hil
cd /workspace/hil-serl-sim/examples
conda activate hilserl

# 3. 远程：跑训练
bash run_actor.sh > /workspace/train_$(date +%s).log 2>&1 &

# 4. 远程：脱离 tmux（Ctrl+B, D）
# 5. 本地：可以关掉 SSH，训练继续
```

### 收尾流程（3 步）
```bash
# 1. 远程：打包结果
ssh hil-train 'cd /workspace/hil-serl-sim/examples && \
    tar -czf /workspace/results.tar.gz checkpoints/ logs/'

# 2. 本地：拉回数据
rsync -avzP hil-train:/workspace/results.tar.gz ./

# 3. 云平台：终止实例（省钱！）
```

### 单次训练预算速算
```
预算 = (训练小时数) × (GPU 单价)
推荐：A6000 × 4 小时 = $3-4 / 次 (RunPod)
推荐：A100-80G × 2 小时 = $4-5 / 次 (Lambda)
```

---

## 八、为什么我们 V1/V2 可以跳过这一切

如对比表显示，我们的 V1/V2 在 **RTX 3060 12GB**（笔记本级 GPU）上即可稳定训练 3 小时 / run。

这归功于以下设计选择（前文已述，此处复盘）：

| 设计 | hil-serl-sim | 我们 V1/V2 |
|------|-------------|-----------|
| 框架 | JAX + PyTorch 双载 | 纯 PyTorch |
| Critic 数 | 10 (REDQ) | 2 (vanilla SAC) |
| Encoder | ResNet-50 (~25M) | ResNet-10 frozen (~5M) |
| 共享 encoder | 否 | 是 (actor + critic) |
| 图像分辨率 | 1280×720 → 256×256 | 128×128 |
| Replay buffer 设备 | GPU (默认) | CPU |
| Batch size | 256 | 128 |
| 显存峰值 | ~25 GB | ~2.5 GB |

**结论**：如果只是想完成 pick-and-lift 这个具体任务（而不是复现完整的 SERL 论文 benchmark），**我们的 V1/V2 是性价比更高的选择** —— 不需要云 GPU，不需要远程训练，本地笔记本就能跑完。

仅当需要做严格的"与原论文对照"实验时，才值得为 hil-serl-sim 投入云 GPU 成本。
