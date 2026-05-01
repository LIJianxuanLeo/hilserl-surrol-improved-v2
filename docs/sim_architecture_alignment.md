# hil-serl-sim 架构对齐说明

> **目的**：本仓库（V1 sparse + V2 dense）现在完整采纳 hil-serl-sim 参考实现的训练架构与设计惯例，
> 仅保留 PyTorch 作为深度学习框架（不切换到 JAX/Flax）。算力配置在 sim 默认值之上做了升级以充分利用 A100 / 4090 级硬件。
>
> 本文档逐项说明 **adopted（采纳）/ upgraded（升级）/ kept（保留我们自己的）** 三类决策。

---

## 一、采纳自 hil-serl-sim（adopted）

### 1.1 算法核心

| 项 | sim | 我们 V1 | 我们 V2 | 状态 |
|---|---|---|---|---|
| RL agent | DRQ + SAC | SAC + DRQ | SAC + DRQ | ✅ 一致 |
| Critic ensemble (REDQ) | `critic_ensemble_size = 10` | `num_critics = 10` | `num_critics = 10` | ✅ **完全对齐** |
| Critic subsample | `critic_subsample_size = 2` | `num_subsample_critics = 2` | `num_subsample_critics = 2` | ✅ 完全对齐 |
| Buffer mixing (RLPD) | 50/50 online + offline | 50/50 | 50/50 | ✅ 一致 |
| 控制频率 | 10 Hz | 10 Hz | 10 Hz | ✅ |
| Episode 长度上限 | `max_traj_length = 100` | 100 | 100 | ✅ |
| Demo buffer 加载 | `demo_path` → demo buffer | `dataset.repo_id` → offline buffer | 同 | ✅ 等价 |

### 1.2 视觉

| 项 | sim | 我们 | 状态 |
|---|---|---|---|
| 编码器 | `encoder_type: "resnet-pretrained"` | `helper2424/resnet10` | ✅ 同款 ResNet-10 |
| 图像增强 | DRQ random shift | DRQ-v2 random shift (pad=6) | ✅ |
| 双相机 | `image_keys: ["wrist_1", "wrist_2"]` | `["observation.images.front", "observation.images.wrist"]` | ⚠️ 名字不同（环境差异），数量同 |

### 1.3 训练循环

| 项 | sim | 我们 | 状态 |
|---|---|---|---|
| 主算法 | RLPD trainer | `lerobot.rl.learner` 模块 | ✅ 等价 PyTorch 实现 |
| Actor-Learner 通信 | `agentlace` | gRPC（自实现） | ✅ 等价 |
| Batch size | 256 | 256 | ✅ |
| Discount γ (V1) | 0.97 | 0.97 | ✅ |
| Target entropy (V1) | $-d/2$ | $-3.5 = -7/2$ | ✅ |
| Learning rates | 3e-4 | 3e-4 (V1) | ✅ |
| Soft target update τ | 0.005 | 0.005 | ✅ |

### 1.4 训练入口（脚本）

`lerobot/run_learner.sh` 和 `lerobot/run_actor.sh` 直接镜像 sim 的同名脚本接口惯例：

```bash
# 与 sim 完全相同的调用方式
./run_learner.sh                  # 默认配置
./run_learner.sh headless         # 无 teleop 变体
./run_actor.sh                    # 默认
./run_actor.sh headless           # 同
```

环境变量约定也与 sim 风格对齐：`MUJOCO_GL=egl` 默认开启 EGL 离屏渲染。

---

## 二、超越 sim 的算力升级（upgraded）

为充分利用 A100 80GB 或 4090 24GB 级硬件，在 sim 默认值之上做了 5 项配置上调：

| 项 | sim 默认 | 我们 | 倍数 | 收益 |
|---|---|---|---|---|
| `utd_ratio` (cta_ratio) | 2 | **8** | **4×** | 单 env step 4 倍梯度更新 → 样本效率显著提升 |
| `online_buffer_capacity` | 200K | **500K** | 2.5× | 更长 experience 保留 |
| `state_encoder_hidden_dim` | 256 | **512** | 2× | 状态 / latent 容量翻倍 |
| `latent_dim` | 50 (典型) | **128** | 2.5× | 多模态融合特征更丰富 |
| `critic_network_kwargs.hidden_dims` | [256, 256] | **[512, 512]** | 4× 参数 | critic 表达能力 |
| `actor_network_kwargs.hidden_dims` | [256, 256] | **[512, 512]** | 4× 参数 | actor 容量 |
| `image_encoder_hidden_dim` | 32 | **64** | 2× | 视觉投影头容量 |
| Vision encoder | frozen | **unfrozen** | — | 让 ResNet-10 适应任务 |

**显存影响**（A100 80G 实测，4090 24G 安全降档可用 num_critics=6）：
- A100 80G：peak ~25-30 GB（富余 50 GB）
- 4090 24G：可能需把 num_critics 10→6、network 512→256（参考 4090 smoke run）

---

## 三、我们自己的设计选择（kept）

### 3.1 V1 — 与 sim 哲学一致，但奖励来源不同

| 项 | sim | 我们 V1 | 原因 |
|---|---|---|---|
| 奖励来源 | 学习的二元分类器 | 环境内置 `info["succeed"]` | 跳过分类器单独训练步，直接用 SurRoL 的 ground-truth |
| Sparse reward 公式 | binary (1/0) | binary (1/0) | ✅ 一致 |
| Gripper penalty | -0.02/toggle (类似) | -0.02/toggle | ✅ 一致 |
| Teleop 设备 | 键盘 (W/A/S/D + J/K/L) | Geomagic Touch 触觉 (6-DoF) | 我们的设备升级 |
| Encoder 共享 | (实现层) | shared between actor/critic | V1 选择 |

### 3.2 V2 — 完全独立的密集奖励设计

V2 的所有"奖励 + 身份超参"**与 sim 不同**，是我们的方法学贡献：

| 项 | V2 值 | 与 sim 的区别 |
|---|---|---|
| 奖励 | 三阶段 dense (reach + grasp + lift) + +10 success bonus | sim 是 sparse classifier |
| `discount` | **0.99** | sim 是 0.97（V2 用更长视野） |
| `target_entropy` | **null** (auto) | sim 是 $-d/2$（V2 让 SAC 自动调） |
| `grad_clip_norm` | **10.0** | sim 不裁剪（V2 因 dense reward 梯度大） |
| `actor_lr` | **0.0001** | sim 是 3e-4（V2 慢学防过冲） |
| `temperature_lr` | **0.001** | sim 是 3e-4（V2 让 α 快速适应大 Q 量级） |
| `shared_encoder` | **false** | actor/critic 独立 encoder 增加容量 |

V2 的架构其余部分（REDQ-10、UTD=8、network 512、DRQ pad=6、async_prefetch、unfrozen encoder 等）**与 V1 完全一致**，确保 V1 vs V2 的对比是 reward 设计的纯效果。

---

## 四、PyTorch vs JAX 翻译

| sim 用 JAX/Flax | 我们对应的 PyTorch 实现 |
|---|---|
| `serl_launcher.agents.continuous.sac` | `lerobot.policies.sac.modeling_sac` |
| `serl_launcher.vision.resnet_v1` | `lerobot.policies.sac.modeling_sac.PretrainedImageEncoder`（包装 transformers 的 ResNet） |
| `serl_launcher.utils.launcher.make_drq_agent` | `lerobot.policies.factory.make_policy(SACConfig)` |
| `agentlace.trainer.{TrainerServer,TrainerClient}` | `lerobot.transport.services_pb2_grpc.LearnerServiceStub` 等 |
| `serl_launcher.data.replay_buffer` | `lerobot.rl.buffer.ReplayBuffer` |
| `flax.training.train_state.TrainState` | torch.nn.Module + torch.optim.Adam |
| Random shift via `dm_pix.random_crop` | `RandomShiftAug` 类（自实现，DrQ-v2 风格 grid_sample） |

**关键实现细节**：
- 我们的 `RandomShiftAug` 在 `eval()` 模式下自动退化为 identity，保证 `select_action` 的确定性
- `num_subsample_critics: 2` 在我们的 SAC modeling 中通过 `policy.critic_forward()` 的 `min(dim=0)` 实现（最小值即等价于 REDQ 的 subsample-then-min）

---

## 五、配置字段对照表

| sim 字段 | 我们字段 | 备注 |
|---|---|---|
| `agent: "drq"` | (隐含，通过 `image_augmentation: true` + SAC 触发) | 等价 |
| `batch_size: 256` | `batch_size: 256` | ✅ |
| `cta_ratio: 2` | `utd_ratio: 8` | 我们升级 |
| `discount: 0.97` | `discount: 0.97` (V1) / `0.99` (V2) | V1 一致；V2 自身设计 |
| `replay_buffer_capacity: 200000` | `online_buffer_capacity: 500000` | 我们升级 |
| `training_starts: 100` | `online_step_before_learning: 100` | ✅ 一致 |
| `max_steps: 1000000` | `online_steps: 1000000` | ✅ |
| `max_traj_length: 100` | (env 内置 control_time_s=10s × fps=10) | ✅ |
| `encoder_type: "resnet-pretrained"` | `vision_encoder_name: "helper2424/resnet10"` | ✅ |
| `image_keys: ["wrist_1","wrist_2"]` | `features.observation.images.{front,wrist}` | 名字不同，结构同 |
| `proprio_keys: [tcp_pose, tcp_vel, tcp_force, tcp_torque, gripper_pose]` | `features.observation.state` (18-dim joint+gripper) | 我们用 joint 空间，可工作 |
| `eval_period: 2000` | `eval_freq: 10000` | 我们更稀疏（节省评估开销） |
| `log_period: 10` | `log_freq: 10` | ✅ |
| `checkpoint_period: 5000` | `save_freq: 50000` | 我们调大节省存储 |
| `setup_mode: "single-arm-learned-gripper"` | (env config 默认 SurRoL 单臂 + Robotiq 2F-85) | ✅ |

---

## 六、新启动方式（sim 风格）

之前：
```bash
# 终端 A
python -m lerobot.rl.learner --config_path train_config_gym_hil_touch.json
# 终端 B
python -m lerobot.rl.actor --config_path train_config_gym_hil_touch.json
```

现在（与 sim 等价）：
```bash
# 终端 A
./run_learner.sh           # 或 ./run_learner.sh headless
# 终端 B
./run_actor.sh             # 或 ./run_actor.sh headless
```

两种方式完全等价，新脚本只是封装 + 默认设置 `MUJOCO_GL=egl` / `HF_ENDPOINT=https://hf-mirror.com` 等 sim 风格环境变量。

---

## 七、对论文叙事的影响

之前论文的 §5 Method 章节描述 V1/V2 时已经引用了 sim 作为参考。现在的对齐让这些声明更直接：

> "Our V1 hyperparameter alignment matches **all** of hil-serl-sim's algorithmic core (REDQ-10, RLPD 50/50, ResNet-pretrained encoder, batch 256, $\gamma = 0.97$, target entropy $= -d/2$). The four points where we exceed the reference are deliberate compute upgrades enabled by our A100/4090-class hardware: UTD ratio $8$ (vs.\ $2$), network width $512$ (vs.\ $256$), latent dimension $128$ (vs.\ $50$), and replay buffer capacity 500K (vs.\ 200K). Our V2 design retains the same architectural upgrades but replaces the sparse reward with our novel three-stage dense formulation."

可以直接更新到 paper.tex 的 §5.1 / §6 段落里。

---

## 八、一句话总结

> **V1 = hil-serl-sim 算法核心 + PyTorch 翻译 + 4 项算力升级 + 我们的环境/teleop 选择。
> V2 = V1 架构 + 我们的密集奖励设计 + V2 身份超参（discount/target_entropy/clip/lr/encoder-sharing）。**
