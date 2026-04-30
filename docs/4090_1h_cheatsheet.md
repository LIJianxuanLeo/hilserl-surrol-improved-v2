# 4090 一小时实战 cheatsheet

> **目的**：在 4090 服务器上 1 小时内拿到 V1 + V2 真实早期训练数据，论文 Section 8 直接用。
> **核心妥协**：无 Touch 干预（headless 模式），网络宽度从 512 降到 256（4090 24G 显存安全）。
> **保留**：UTD=8 / REDQ-6 / DRQ pad=6 / async_prefetch / unfrozen encoder / F6 完整 32 列日志。

---

## 60 分钟时间表

```
00:00 ─ 05:00   Phase 1  Pre-flight
05:00 ─ 25:00   Phase 2  V1 训练 20 min
25:00 ─ 45:00   Phase 3  V2 训练 20 min
45:00 ─ 55:00   Phase 4  生成对比图
55:00 ─ 60:00   Phase 5  push 到 GitHub
```

---

## Phase 1：Pre-flight（5 min）

```bash
ssh <4090>
cd ~/project   # 或新建

# 拉最新代码（用 ghproxy 加速）
git clone --depth 1 https://ghproxy.com/https://github.com/LIJianxuanLeo/hilserl-surrol-improved.git
git clone --depth 1 https://ghproxy.com/https://github.com/LIJianxuanLeo/hilserl-surrol-improved-v2.git
# 已 clone 过则：cd <repo> && git pull

# 配置已是 4090-safe（REDQ-6, network 256），headless 变体已自带，无需任何改动

# 激活环境，验证 GPU
conda activate hilserl
nvidia-smi   # 期望看到 RTX 4090, 24576 MiB

# 配置 MuJoCo headless 渲染
export MUJOCO_GL=egl
```

---

## Phase 2：V1 训练（20 min）

### 启动
```bash
cd ~/project/hilserl-surrol-improved/lerobot

# 1) 自动归档 + 提示
./start_paper_run.sh   # 但要忽略它的提示，因为咱们用 headless 配置
                       # 它生成的 paper_run_v1 不会用到，会被 paper_run_headless_v1 覆盖

# 2) 终端 A：启动 learner（后台 + tee 日志）
nohup python -m lerobot.rl.learner \
    --config_path train_config_gym_hil_headless.json \
    > learner_v1.log 2>&1 &
echo "Learner PID: $!"

# 等待 30 秒让 learner 起来
sleep 30
tail learner_v1.log
# 应看到 "[LEARNER] Starting learner thread"

# 3) 终端 B（同 SSH 会话也行）：启动 actor
nohup python -m lerobot.rl.actor \
    --config_path train_config_gym_hil_headless.json \
    > actor_v1.log 2>&1 &
echo "Actor PID: $!"
```

### 监控（每 2-3 分钟看一次）
```bash
# 训练数据是否出现
ls outputs/train/paper_run_headless_v1/training_logs/
wc -l outputs/train/paper_run_headless_v1/training_logs/training_metrics.csv
wc -l outputs/train/paper_run_headless_v1/training_logs/episode_metrics.csv

# 最新数据点
tail -3 outputs/train/paper_run_headless_v1/training_logs/training_metrics.csv | cut -d, -f1,2,3,15
# 字段：timestamp, opt_step, loss_critic, q_mean

# GPU 显存（应稳定在 8-15 GB）
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 期望（4090 + REDQ-6 + UTD=8）
| 时间 | training_metrics.csv | episode_metrics.csv |
|------|---------------------|---------------------|
| 5 min | > 200 行 | ≥ 5 episodes |
| 10 min | > 500 行 | ≥ 15 episodes |
| 15 min | > 1000 行 | ≥ 30 episodes |
| 20 min | > 2000 行 | ≥ 50 episodes |

### 20 分钟整 — 优雅停止
```bash
# 计时：20 分钟后执行
ACTOR_PID=<from_above>
LEARNER_PID=<from_above>

kill -INT $ACTOR_PID    # SIGINT，触发 graceful shutdown
sleep 5                  # 等 actor flush transitions
kill -INT $LEARNER_PID   # 然后停 learner
sleep 10                 # 等 learner save_summary

# 验证 summary 已保存
cat outputs/train/paper_run_headless_v1/training_logs/training_summary.json
# 应见 total_optimization_steps > 5000
```

---

## Phase 3：V2 训练（20 min）— 同流程切到 V2 仓库

```bash
cd ~/project/hilserl-surrol-improved-v2/lerobot
./start_paper_run.sh    # 同样忽略提示

nohup python -m lerobot.rl.learner \
    --config_path train_config_gym_hil_headless.json \
    > learner_v2.log 2>&1 &
sleep 30

nohup python -m lerobot.rl.actor \
    --config_path train_config_gym_hil_headless.json \
    > actor_v2.log 2>&1 &

# 监控目录是 outputs/train/paper_run_headless_v2/
# 20 分钟后同样 SIGINT actor 然后 learner
```

---

## Phase 4：生成对比图（10 min）

```bash
cd ~/project/hilserl-surrol-improved/lerobot

# 三个 PDF 一次出
python plot_paper.py \
    --v1 ~/project/hilserl-surrol-improved/lerobot/outputs/train/paper_run_headless_v1 \
    --v2 ~/project/hilserl-surrol-improved-v2/lerobot/outputs/train/paper_run_headless_v2 \
    --out figures_4090_1h

ls figures_4090_1h/
# fig_paper_run_headless_v1_curves.pdf
# fig_v1_curves.pdf
# fig_v2_curves.pdf
# fig_compare_v1v2.pdf  ← 论文最关键
```

### 提取头条数字
```bash
echo "=== V1 results ==="
python3 -c "
import json
s = json.load(open('outputs/train/paper_run_headless_v1/training_logs/training_summary.json'))
for k in ['training_duration_s', 'total_optimization_steps', 'total_episodes',
         'total_successes', 'training_success_rate', 'best_episodic_reward']:
    print(f'  {k}: {s.get(k)}')
"

echo ""
echo "=== V2 results ==="
python3 -c "
import json
s = json.load(open('/root/project/hilserl-surrol-improved-v2/lerobot/outputs/train/paper_run_headless_v2/training_logs/training_summary.json'))
for k in ['training_duration_s', 'total_optimization_steps', 'total_episodes',
         'total_successes', 'training_success_rate', 'best_episodic_reward']:
    print(f'  {k}: {s.get(k)}')
"
```

---

## Phase 5：Push 到 GitHub（5 min）

```bash
# V1
cd ~/project/hilserl-surrol-improved
mkdir -p docs/results/4090_1h_run
cp lerobot/outputs/train/paper_run_headless_v1/training_logs/*.csv  docs/results/4090_1h_run/
cp lerobot/outputs/train/paper_run_headless_v1/training_logs/*.json docs/results/4090_1h_run/
cp -r lerobot/figures_4090_1h docs/results/4090_1h_run/figures
git add docs/results/4090_1h_run
git commit -m "feat: 4090 1-hour real training data — V1 sparse headless"
git push origin main

# V2
cd ~/project/hilserl-surrol-improved-v2
mkdir -p docs/results/4090_1h_run
cp lerobot/outputs/train/paper_run_headless_v2/training_logs/*.csv  docs/results/4090_1h_run/
cp lerobot/outputs/train/paper_run_headless_v2/training_logs/*.json docs/results/4090_1h_run/
git add docs/results/4090_1h_run
git commit -m "feat: 4090 1-hour real training data — V2 dense headless"
git push origin main
```

---

## 故障应急

### 问题 1：Learner 5 分钟内 training_metrics.csv 仍是空表头
```bash
tail -30 learner_v1.log     # 看是否在 warmup
# 应看到：[LEARNER] Warmup: X/50 transitions collected
# 如果一直没动，actor 没在送数据 → 检查 actor.log
```

### 问题 2：CUDA OOM
```bash
# 编辑 lerobot/train_config_gym_hil_headless.json：
#   "batch_size": 256 → 192
#   "num_critics": 6 → 4
# 重启训练
```

### 问题 3：MuJoCo 渲染失败
```bash
export MUJOCO_GL=egl              # 已设置则确认
export MUJOCO_GL=osmesa            # 备选 1
# 或
export PYOPENGL_PLATFORM=egl
```

### 问题 4：进程残留
```bash
# 找出所有 lerobot 相关进程
ps aux | grep lerobot | grep -v grep
# 全杀
pkill -f lerobot.rl.learner
pkill -f lerobot.rl.actor
```

### 问题 5：Git push 慢
```bash
# 数据先打 tar，scp 回本地
tar -czf 4090_results.tar.gz docs/results/4090_1h_run
# 在本地执行
scp 4090:~/project/hilserl-surrol-improved/4090_results.tar.gz ~/Downloads/
# 解压后本地 commit + push
```

---

## 论文中怎么用这批数据

### 进 Section 8.2 / 8.3
- `figures/fig_v1_curves.pdf` → V1 6 子图（reward / loss / Q / temperature / success / intervention）
- `figures/fig_v2_curves.pdf` → V2 同上

### 进 Section 8.4
- `figures/fig_compare_v1v2.pdf` → V1 vs V2 4 子图叠加
- 可在 caption 写："Both versions trained for 20 min on RTX 4090 in headless mode (no human intervention) to enable direct comparison with the hil-serl-sim reference reproduction."

### 进 Section 4.3
- 直接对比：hil-serl-sim 220K 步无干预 = 10% 成功率
- 我们 V1 在 ~10K 步无干预 = ?% 成功率（看 training_summary.json）
- 我们 V2 在 ~10K 步无干预 = ?% 成功率
- 这是 paper 最有冲击力的对比

---

## 一句话

> **照着这文档逐步执行，60 分钟就能完成 V1 + V2 真实训练数据采集 + 生成 3 张论文级 PDF + push 到 GitHub。所有配置已预先降档为 4090-safe，不需要任何调参。**
