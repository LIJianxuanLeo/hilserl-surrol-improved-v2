# 改进记录

基于 [GeorgeAuburn/hilserl-surrol](https://github.com/GeorgeAuburn/hilserl-surrol) 的改进。

## 训练不收敛根因分析

原始项目在 3 小时训练后无明显收敛，经分析确认以下 7 个问题：

### 1. temperature_init = 0.01（核心问题）
SAC 算法依赖最大熵框架，温度 alpha 控制策略的随机性。`alpha = 0.01` 意味着 `log(alpha) = -4.6`，策略从训练开始就几乎是确定性的，无法进行有效探索。对于需要发现复杂抓取动作序列的任务，这是致命的。

**修复**: `temperature_init = 1.0`

### 2. discount = 0.97
有效视野 = 1/(1-0.97) = 33 步 = 3.3 秒（@10fps）。完成一次完整的 pick 操作需要 5-10 秒，因此 Q 函数无法"看到"完整的成功序列。

**修复**: `discount = 0.99`（有效视野 100 步 = 10 秒）

### 3. temperature_lr = 0.0003
温度自适应调节速度太慢，SAC 无法及时调整探索-利用平衡。

**修复**: `temperature_lr = 0.001`

### 4. actor_lr = 0.0003（与 critic_lr 相同）
Actor 学习率不应高于 Critic。如果 Actor 更新太快，会追逐不准确的 Q 值估计。

**修复**: `actor_lr = 0.0001`

### 5. grad_clip_norm = 10.0
梯度裁剪太宽松，在稀疏奖励任务中容易出现梯度爆炸。

**修复**: `grad_clip_norm = 1.0`

### 6. action range 不匹配
配置中 `action.min/max = [-0.4, 0.4]`（位置轴），但实际数据中动作范围达到 `[-1.0, 1.0]`。这导致演示数据在归一化时被截断，信息丢失。

**修复**: `action.min/max = [-1.0, 1.0]`（位置轴），与数据实际范围匹配

### 7. haptic_module_path 硬编码
路径硬编码为 `/home/zjj/projects/...`，其他机器无法直接使用。

**修复**: 使用占位符 `__HAPTIC_MODULE_PATH__`，由 `setup.sh` 自动替换

## 收敛加速改进

### 8. 稀疏奖励 → 阶段式密集奖励
原始项目使用稀疏奖励（仅成功时 reward=1.0），策略在初期完全没有梯度信号引导。在 32k 步时仍无法学会接近方块。

**修复**: 
1. 将 `reward_type` 从 `"sparse"` 改为 `"dense"`
2. 新增 `StagedRewardWrapper`（`staged_reward_wrapper.py`），提供三阶段密集奖励：

| 阶段 | 每步原始奖励 | 条件 |
|------|------------|------|
| 接近（Approach） | 0 ~ 0.25 | `0.25 * exp(-10*dist_xy) * exp(-10*dist_z)` |
| 抓取（Grasp） | 0 ~ 0.25 | 方块在手内 = 0.25，靠近 = 0.10 |
| 抬起（Lift） | 0 ~ 0.50 | `0.50 * min(lift/target, 1.0)` |
| 成功 | 1.0 | 完成任务（不缩放） |

**关键**：每步奖励乘以 `1/max_episode_steps`（默认 1/100），使 episode 累积奖励保持在 [0, ~1] 范围，与 SAC temperature=1.0 兼容。

奖励设计特点（episode 累积）：
- 随机策略约得 ~0.02-0.08
- 靠近方块约得 ~0.15-0.25
- 抓住方块约得 ~0.35-0.50
- 成功抬起约得 ~1.0-1.3（shaping + 1.0 成功奖励）
- 连续梯度信号引导策略逐步学会 接近→抓取→抬起

### 9. batch_size = 256 → 128
RTX 3060 (12GB) 使用 batch_size=256 时 GPU 内存溢出导致系统死机。

**修复**: `batch_size = 128`

## 其他改进

- WandB 监控默认关闭（`wandb.enable = false`），避免未登录时出现交互提示
  - 启用方法：先 `wandb login`，再将配置中 `wandb.enable` 改为 `true`
- 提高评估和保存频率 (`eval_freq/save_freq: 20000 → 10000`)
- 增加 `online_step_before_learning: 100 → 200`
- 放宽策略标准差上限 (`std_max: 5.0 → 10.0`)
- 添加 `setup.sh` 一键部署脚本
- 添加完整中文部署文档 `docs/DEPLOY_GUIDE.md`
- 将预采集数据集纳入 git 追踪
- 清理 `.DS_Store` 等临时文件
