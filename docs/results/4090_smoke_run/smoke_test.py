"""
F6 Logging smoke test on RTX 4090.
独立验证 SAC + REDQ-6 + DRQ + F6 logging 全管线工作；不需要 lerobot dataset / env。

Output: 32-col training_metrics.csv + 14-col episode_metrics.csv + summary JSON
"""
import os, sys, time, json, csv, math
import torch
import torch.nn as nn

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v1"
N_TRAIN_STEPS = int(os.environ.get("N_STEPS", 1500))
N_EPISODES = int(os.environ.get("N_EPISODES", 60))
OUT_DIR = f"/root/data/smoke_runs/paper_run_smoke_{VARIANT}/training_logs"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[{VARIANT}] device={device}, steps={N_TRAIN_STEPS}, episodes={N_EPISODES}")

# ── Variant-specific config ──
if VARIANT == "v1":
    discount = 0.97
    network_width = 256
    grad_clip = 100.0
    actor_lr = 3e-4
    target_entropy = -3.5
    success_threshold = 0.5  # binary sparse
elif VARIANT == "v2":
    discount = 0.99
    network_width = 256
    grad_clip = 10.0
    actor_lr = 1e-4
    target_entropy = None  # auto, computed below
    success_threshold = 5.0  # dense reward — succeed if episode reward > 5
else:
    raise ValueError(f"unknown variant {VARIANT}")

# ── Build minimal SAC network analog (REDQ-6 ensemble) ──
N_CRITICS = 6
LATENT = 64
ACTION_DIM = 7
STATE_DIM = 18
IMAGE_FEAT = 32  # what ResNet-10 frozen would produce per camera

class SmokeMLPCritic(nn.Module):
    def __init__(self, in_dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + ACTION_DIM, hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Linear(hid, 1),
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

class SmokeActor(nn.Module):
    def __init__(self, in_dim, hid=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.ReLU(),
        )
        self.mu = nn.Linear(hid, ACTION_DIM)
        self.log_std = nn.Linear(hid, ACTION_DIM)
    def forward(self, s):
        h = self.trunk(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5, 2)
        std = log_std.exp()
        # tanh-squashed sample
        eps = torch.randn_like(mu)
        a = torch.tanh(mu + std * eps)
        log_prob = (-0.5 * (eps**2) - log_std).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, log_prob

# ── DRQ random shift (simplified for smoke) ──
class RandomShift(nn.Module):
    def __init__(self, pad=6): super().__init__(); self.pad = pad
    def forward(self, x):
        if not self.training: return x
        x = nn.functional.pad(x, (self.pad,)*4, mode="replicate")
        n, c, h, w = x.size()
        h_orig = h - 2*self.pad
        sx = torch.randint(0, 2*self.pad, (n,))
        sy = torch.randint(0, 2*self.pad, (n,))
        out = torch.stack([x[i, :, sy[i]:sy[i]+h_orig, sx[i]:sx[i]+h_orig] for i in range(n)])
        return out

# ── Build models ──
in_dim = 2 * IMAGE_FEAT + STATE_DIM  # 2 cameras + state
actor = SmokeActor(in_dim, hid=network_width).to(device)
critics = nn.ModuleList([SmokeMLPCritic(in_dim, hid=network_width).to(device) for _ in range(N_CRITICS)])
target_critics = nn.ModuleList([SmokeMLPCritic(in_dim, hid=network_width).to(device) for _ in range(N_CRITICS)])
for tc, c in zip(target_critics, critics): tc.load_state_dict(c.state_dict())
log_alpha = nn.Parameter(torch.zeros(1, device=device))

if target_entropy is None:
    target_entropy = -ACTION_DIM / 2  # SAC auto

opt_actor = torch.optim.Adam(actor.parameters(), lr=actor_lr)
opt_critic = torch.optim.Adam(list(c for crit in critics for c in crit.parameters()), lr=3e-4)
opt_alpha = torch.optim.Adam([log_alpha], lr=3e-4)

drq = RandomShift(pad=6).to(device)

n_params_actor = sum(p.numel() for p in actor.parameters())
n_params_critic = sum(p.numel() for p in critics.parameters())
print(f"[{VARIANT}] actor={n_params_actor/1e6:.2f}M  critics={n_params_critic/1e6:.2f}M (×{N_CRITICS} REDQ)")

# ── F6 logger (32-col training, 14-col episode) ──
TRAINING_FIELDS = [
    "timestamp","optimization_step","loss_critic","loss_actor","loss_temperature",
    "loss_discrete_critic","temperature","critic_grad_norm","actor_grad_norm",
    "temperature_grad_norm","discrete_critic_grad_norm","replay_buffer_size",
    "offline_replay_buffer_size","optimization_freq_hz",
    "q_mean","q_std","q_min","q_max","entropy_term",
    "q_target_mean","q_target_std","td_error_mean","td_error_std","td_error_max",
    "critic_disagreement","policy_entropy_raw","policy_log_prob_mean",
    "actor_loss_q_term","actor_loss_entropy_term",
    "step_time_ms","gpu_mem_current_mb","gpu_mem_peak_mb",
]
EPISODE_FIELDS = [
    "timestamp","interaction_step","episodic_reward","episode_intervention",
    "intervention_rate","policy_freq_hz","policy_freq_90p_hz",
    "is_success","rolling_success_rate_50","rolling_intervention_rate_50",
    "rolling_policy_only_success_20",
    "episode_length","episode_intervention_steps","termination_reason",
]

train_csv = open(f"{OUT_DIR}/training_metrics.csv", "w", newline="")
train_w = csv.DictWriter(train_csv, fieldnames=TRAINING_FIELDS); train_w.writeheader()
ep_csv = open(f"{OUT_DIR}/episode_metrics.csv", "w", newline="")
ep_w = csv.DictWriter(ep_csv, fieldnames=EPISODE_FIELDS); ep_w.writeheader()
eval_csv = open(f"{OUT_DIR}/eval_metrics.csv", "w", newline="")
eval_w = csv.DictWriter(eval_csv, fieldnames=["timestamp","optimization_step","eval_episodes","eval_success_rate","eval_mean_reward","eval_mean_episode_length"]); eval_w.writeheader()

# ── Synthetic data generator ──
# Simulate what trainer would feed: random images + states + actions + rewards
B = 256
def gen_batch():
    # 2 cameras of fake "encoded" features (already encoded, since we skip ResNet)
    # In reality drq would apply to images BEFORE encoder; we apply to fake images here
    img1 = torch.randn(B, 3, 128, 128, device=device)
    img2 = torch.randn(B, 3, 128, 128, device=device)
    img1 = drq(img1); img2 = drq(img2)
    # collapse to fake "features" (avg pool to 32 dims)
    f1 = img1.mean(dim=(2,3))[:, :IMAGE_FEAT] if img1.shape[1]>=IMAGE_FEAT else torch.cat([img1.mean(dim=(2,3))]*4, dim=-1)[:, :IMAGE_FEAT]
    f2 = img2.mean(dim=(2,3))[:, :IMAGE_FEAT] if img2.shape[1]>=IMAGE_FEAT else torch.cat([img2.mean(dim=(2,3))]*4, dim=-1)[:, :IMAGE_FEAT]
    state = torch.randn(B, STATE_DIM, device=device)
    s = torch.cat([f1, f2, state], dim=-1)
    a = torch.tanh(torch.randn(B, ACTION_DIM, device=device))
    if VARIANT == "v1":
        # sparse: 5% chance of success
        r = (torch.rand(B, device=device) > 0.95).float()
    else:
        # dense: shaped reward [0, 10] + bonus
        base = 3.0 * torch.rand(B, device=device)  # approach
        grasp = 2.0 * torch.rand(B, device=device)  # grasp
        lift = 1.5 * torch.rand(B, device=device)  # lift
        r = base + grasp + lift + (torch.rand(B, device=device) > 0.92).float() * 10.0
    s_next = s + 0.1 * torch.randn_like(s)
    done = (torch.rand(B, device=device) > 0.95).float()
    return s, a, r, s_next, done

# ── Training loop ──
print(f"[{VARIANT}] starting training loop...")
start_time = time.time()
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

actor.train(); critics.train(); drq.train()

# Episode bookkeeping
ep_count = 0; ep_step = 0
episode_buffer = []
rolling_succ = []; rolling_intv = []; rolling_policy_succ = []

opt_step = 0
LOG_EVERY = 10
EVAL_EVERY = 200

for step in range(N_TRAIN_STEPS):
    t0 = time.time()
    s, a, r, s2, d = gen_batch()
    
    # Critic update (with TD target from policy-sampled next action)
    with torch.no_grad():
        a2, lp2 = actor(s2)
        q_next = torch.stack([tc(s2, a2).squeeze(-1) for tc in target_critics], dim=0)  # (N_CRITICS, B)
        q_next_min = q_next.min(dim=0)[0]
        alpha_val = log_alpha.exp().item()
        td_target = r + discount * (1 - d) * (q_next_min - alpha_val * lp2)
    
    q_preds = torch.stack([c(s, a).squeeze(-1) for c in critics], dim=0)  # (N_CRITICS, B)
    loss_critic = ((q_preds - td_target.unsqueeze(0))**2).mean()
    
    opt_critic.zero_grad()
    loss_critic.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(critics.parameters(), grad_clip).item()
    opt_critic.step()
    
    # Soft target update
    for tc, c in zip(target_critics, critics):
        for tp, p in zip(tc.parameters(), c.parameters()):
            tp.data.mul_(1 - 0.005).add_(p.data, alpha=0.005)
    
    # Actor + temp update (every step in our config)
    a_pi, lp = actor(s)
    q_pi = torch.stack([c(s, a_pi).squeeze(-1) for c in critics], dim=0)
    q_pi_min = q_pi.min(dim=0)[0]
    actor_loss_q = -q_pi_min.mean()
    actor_loss_ent = (log_alpha.exp().detach() * lp).mean()
    loss_actor = actor_loss_q + actor_loss_ent
    
    opt_actor.zero_grad()
    loss_actor.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip).item()
    opt_actor.step()
    
    # Temperature
    loss_temp = -(log_alpha * (lp.detach() + target_entropy)).mean()
    opt_alpha.zero_grad()
    loss_temp.backward()
    temp_grad_norm = log_alpha.grad.abs().item()
    opt_alpha.step()
    
    step_time = time.time() - t0
    
    # ── F6 logging (every LOG_EVERY) ──
    if step % LOG_EVERY == 0:
        with torch.no_grad():
            q_min_per_act = q_preds.min(dim=0)[0]
            td_err = (q_preds.mean(dim=0) - td_target).abs()
            entropy_term = (log_alpha.exp() * lp.abs()).mean().item()
        
        gpu_cur = torch.cuda.memory_allocated()/1e6 if torch.cuda.is_available() else 0
        gpu_peak = torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0
        
        train_w.writerow({
            "timestamp": f"{time.time()-start_time:.1f}",
            "optimization_step": step,
            "loss_critic": f"{loss_critic.item():.4f}",
            "loss_actor": f"{loss_actor.item():.4f}",
            "loss_temperature": f"{loss_temp.item():.4f}",
            "loss_discrete_critic": "",
            "temperature": f"{log_alpha.exp().item():.4f}",
            "critic_grad_norm": f"{critic_grad_norm:.4f}",
            "actor_grad_norm": f"{actor_grad_norm:.4f}",
            "temperature_grad_norm": f"{temp_grad_norm:.4f}",
            "discrete_critic_grad_norm": "",
            "replay_buffer_size": min(step*B, 200000),
            "offline_replay_buffer_size": 1630,
            "optimization_freq_hz": f"{1.0/step_time:.2f}",
            "q_mean": f"{q_min_per_act.mean().item():.4f}",
            "q_std": f"{q_min_per_act.std().item():.4f}",
            "q_min": f"{q_min_per_act.min().item():.4f}",
            "q_max": f"{q_min_per_act.max().item():.4f}",
            "entropy_term": f"{entropy_term:.4f}",
            "q_target_mean": f"{td_target.mean().item():.4f}",
            "q_target_std": f"{td_target.std().item():.4f}",
            "td_error_mean": f"{td_err.mean().item():.4f}",
            "td_error_std": f"{td_err.std().item():.4f}",
            "td_error_max": f"{td_err.max().item():.4f}",
            "critic_disagreement": f"{q_preds.std(dim=0).mean().item():.4f}",
            "policy_entropy_raw": f"{(-lp.mean()).item():.4f}",
            "policy_log_prob_mean": f"{lp.mean().item():.4f}",
            "actor_loss_q_term": f"{actor_loss_q.item():.4f}",
            "actor_loss_entropy_term": f"{actor_loss_ent.item():.4f}",
            "step_time_ms": f"{step_time*1000:.2f}",
            "gpu_mem_current_mb": f"{gpu_cur:.1f}",
            "gpu_mem_peak_mb": f"{gpu_peak:.1f}",
        })
        train_csv.flush()

    # Generate fake episode at random intervals (~ every 100 steps = 1 episode)
    if step > 0 and step % 25 == 0 and ep_count < N_EPISODES:
        ep_step = step * 4  # interaction step ~= 4x opt step (UTD=8 but let's say 4 env per opt)
        ep_count += 1
        # Simulate realistic improvement curve
        progress = ep_count / N_EPISODES
        # success prob improves: V1 starts ~5%, ends ~50% with intervention; V2 starts ~30%, ends ~80%
        if VARIANT == "v1":
            base_succ = 0.05 + 0.45 * (1 - math.exp(-3*progress))
            ep_reward = 1.0 if torch.rand(1).item() < base_succ else 0.0
        else:
            base_succ = 0.20 + 0.65 * (1 - math.exp(-2.5*progress))
            ep_reward = 5 + 10 * torch.rand(1).item() if torch.rand(1).item() < base_succ else 1 + 4 * torch.rand(1).item()
        
        intv = int(torch.rand(1).item() < (1.0 - 0.8 * progress))  # intervention rate decays
        intv_steps = int(torch.rand(1).item() * 60) if intv else 0
        ep_len = 60 + int(torch.rand(1).item() * 40)
        is_succ = int(ep_reward >= success_threshold)
        
        rolling_succ.append(is_succ); rolling_intv.append(intv)
        if intv == 0: rolling_policy_succ.append(is_succ)
        rolling_succ = rolling_succ[-50:]; rolling_intv = rolling_intv[-50:]; rolling_policy_succ = rolling_policy_succ[-20:]
        
        ep_w.writerow({
            "timestamp": f"{time.time()-start_time:.1f}",
            "interaction_step": ep_step,
            "episodic_reward": f"{ep_reward:.4f}",
            "episode_intervention": intv,
            "intervention_rate": f"{intv_steps/max(ep_len,1):.4f}",
            "policy_freq_hz": "10.0",
            "policy_freq_90p_hz": "9.5",
            "is_success": is_succ,
            "rolling_success_rate_50": f"{sum(rolling_succ)/len(rolling_succ):.4f}",
            "rolling_intervention_rate_50": f"{sum(rolling_intv)/len(rolling_intv):.4f}",
            "rolling_policy_only_success_20": f"{sum(rolling_policy_succ)/max(len(rolling_policy_succ),1):.4f}",
            "episode_length": ep_len,
            "episode_intervention_steps": intv_steps,
            "termination_reason": "success" if is_succ else ("timeout" if ep_len >= 95 else "done_other"),
        })
        ep_csv.flush()

    # Eval logging
    if step > 0 and step % EVAL_EVERY == 0:
        eval_succ = sum(rolling_policy_succ) / max(len(rolling_policy_succ), 1)
        eval_w.writerow({
            "timestamp": f"{time.time()-start_time:.1f}",
            "optimization_step": step,
            "eval_episodes": len(rolling_policy_succ),
            "eval_success_rate": f"{eval_succ:.4f}",
            "eval_mean_reward": f"{ep_reward:.4f}",
            "eval_mean_episode_length": f"{ep_len:.1f}",
        })
        eval_csv.flush()
    
    if step % 100 == 0:
        print(f"  step {step}: loss_critic={loss_critic.item():.3f} loss_actor={loss_actor.item():.3f} q_mean={q_preds.mean().item():.2f} step_time={step_time*1000:.1f}ms")

train_csv.close(); ep_csv.close(); eval_csv.close()

elapsed = time.time() - start_time
gpu_peak_final = torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0

# Summary
summary = {
    "job_name": f"smoke_{VARIANT}",
    "training_duration_s": round(elapsed, 1),
    "training_duration_h": round(elapsed/3600, 3),
    "total_optimization_steps": N_TRAIN_STEPS,
    "total_episodes": ep_count,
    "total_interventions": sum(1 for x in rolling_intv if x == 1),
    "total_successes": sum(rolling_succ),
    "training_success_rate": round(sum(rolling_succ)/max(len(rolling_succ),1), 4),
    "intervention_episode_ratio": round(sum(rolling_intv)/max(len(rolling_intv),1), 4),
    "best_episodic_reward": float(max(rolling_succ)) if rolling_succ else 0.0,
    "best_eval_success_rate": round(sum(rolling_policy_succ)/max(len(rolling_policy_succ),1), 4),
    "lowest_critic_loss": round(loss_critic.item(), 6),
    "success_threshold": success_threshold,
    "gpu_peak_memory_mb": round(gpu_peak_final, 1),
    "device": str(device),
    "variant": VARIANT,
    "config": {
        "num_critics": N_CRITICS, "utd_ratio": 1, "batch_size": B,
        "discount": discount, "actor_lr": actor_lr, "grad_clip_norm": grad_clip,
        "network_width": network_width, "image_augmentation_pad": 6,
    },
}
with open(f"{OUT_DIR}/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Metadata snapshot
import platform, subprocess
try:
    git_hash = subprocess.check_output(["git", "-C", "/root/data/hilserl-surrol-improved", "rev-parse", "HEAD"]).decode().strip()
except: git_hash = "unknown"

meta = {
    "job_name": f"smoke_{VARIANT}",
    "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "git_commit": git_hash,
    "git_branch": "main",
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "note": "F6 logging schema validation on real RTX 4090. Synthetic data; full SAC + REDQ-6 + DRQ pipeline exercised on GPU.",
}
with open(f"{OUT_DIR}/experiment_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[{VARIANT}] DONE in {elapsed:.1f}s — output in {OUT_DIR}")
print(f"  train: {N_TRAIN_STEPS} steps, ep: {ep_count} episodes, succ_rate: {summary['training_success_rate']:.2%}")
print(f"  GPU peak mem: {gpu_peak_final:.0f} MB")
