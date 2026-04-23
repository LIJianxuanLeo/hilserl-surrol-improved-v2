# Speech Script — Improving HIL-SERL for Robotic Pick-and-Lift

**Duration: ~8 minutes | 12 slides | Language: English**

---

## Slide 1 — Title (0:00 – 0:20)

Good morning everyone. My name is LI Jianxuan. Today I will present our work on improving HIL-SERL for robotic pick-and-lift tasks. We explore two reward design strategies — sparse and dense — within a human-in-the-loop reinforcement learning framework. Let me start with our motivation.

---

## Slide 2 — Motivation (0:20 – 1:05)

Robotic manipulation is a long-standing challenge in RL. Pure reinforcement learning requires millions of interactions, which is unsafe and slow on real hardware. On the other hand, pure imitation learning is limited by the quality and diversity of demonstrations.

HIL-SERL offers an elegant middle ground. The key insight is simple: a human operator provides real-time corrective interventions during training. When the robot makes a mistake, the human takes over, corrects the action, and then hands control back to the policy. This dramatically reduces sample complexity.

The framework rests on three pillars: an offline buffer of human demonstrations, a Soft Actor-Critic policy that learns online, and a human operator who provides corrective guidance in real time. The original paper reports 100% success in approximately one hour of training.

---

## Slide 3 — HIL-SERL Method Overview (1:05 – 1:55)

Let me briefly explain the architecture. HIL-SERL uses a distributed actor-learner design, communicating through gRPC.

The Actor process runs in the environment. It executes the current policy, collects transitions — that is, state, action, reward, and next state tuples — detects when the human intervenes, and sends the collected data to the Learner.

The Learner process maintains a replay buffer and samples training batches. Each batch is a fifty-fifty mix of online experience and offline demonstrations. The Learner updates the critic network multiple times per step, then updates the actor and the temperature parameter. Updated weights are streamed back to the Actor.

At the core is SAC — Soft Actor-Critic. The actor loss balances the entropy term alpha times log-pi against the Q-value. The critic minimizes the temporal difference error. And the temperature is automatically tuned to match a target entropy.

The original paper demonstrates 100% pick-and-lift success in about 30,000 steps, roughly one hour, using a sparse binary reward with human intervention.

---

## Slide 4 — Pick-and-Lift Task (1:55 – 2:35)

Our task is pick-and-lift in the SurRoL v2 simulation environment, built on MuJoCo.

The robot is a Franka Panda with seven degrees of freedom, equipped with a Robotiq 2F-85 gripper. The goal is to pick up a cube from the table and lift it above a height threshold.

The observation space consists of two 128-by-128 RGB camera images — a front view and a wrist view — plus an 18-dimensional proprioceptive state vector.

The action is 7-dimensional, covering position deltas, orientation deltas, and a gripper command. Each episode lasts up to 10 seconds at 10 hertz, giving roughly 100 steps. The episode terminates early upon success.

The human operator uses a Geomagic Touch haptic device for teleoperation and intervention.

---

## Slide 5 — Human Intervention Workflow (2:35 – 3:25)

Here is how intervention works in practice, shown as a five-step cycle.

First, the policy acts autonomously. The human watches and assesses performance. When the human detects an error — for example, the arm drifting away from the object — they press Button 1 on the Touch device to take over control. The human then guides the robot to complete the task correctly. Finally, the human releases the button, and the policy resumes.

Critically, the intervention data enters both the online and offline replay buffers. This means each training batch is a fifty-fifty mix of policy-generated and human-guided experience.

The intervention schedule changes over training. In the early phase, zero to five thousand steps, the human intervenes almost constantly — 80 to 100 percent. As the policy improves, intervention drops to 30 to 60 percent. In the late phase, beyond 30,000 steps, the human only intervenes occasionally, around 5 to 20 percent.

---

## Slide 6 — Challenges in Replication (3:25 – 4:15)

When we attempted to replicate the paper's results using an existing open-source codebase, we encountered four critical issues.

The most fundamental problem was entropy domination. The SAC actor loss is alpha times log-pi minus Q. In our case, the Q-value was approximately 0.3, while the entropy term was approximately 6 — twenty times larger. This means the policy gradient was entirely driven by entropy maximization, not by the task reward. The policy learned to be random, not to pick up objects.

Second, the intervention detection was broken. A configuration flag called clutch mode was set to True, which caused every single step to be marked as a human intervention, regardless of whether the button was actually pressed. The learner could not distinguish human-guided from policy-generated data.

Third, the learner crashed with CUDA out-of-memory errors after about 1.4 hours, with no error handling. All training progress was lost.

Fourth, the warmup threshold was set to 100 steps, but episodes ended in roughly 86 steps due to early success. This meant the learner never accumulated enough data to begin training.

---

## Slide 7 — V1: Sparse Reward (4:15 – 5:10)

Our first version, V1, follows the philosophy of the original paper: trust the HIL-SERL framework. A sparse reward combined with human guidance should be sufficient.

The reward function is straightforward. The agent receives 1.0 when the task succeeds, and 0.0 otherwise. A small gripper penalty of minus 0.02 discourages unnecessary toggling.

The key improvements are in the hyperparameters. We aligned seven critical parameters with a successful reference implementation called hil-serl-sim, which achieves 100% success in about 30,000 steps.

The most impactful changes are: discount factor from 0.99 to 0.97, which shortens the effective planning horizon for sparse rewards; target entropy set to negative 3.5 instead of null, which equals negative action-dimension over two and encourages appropriate exploration; and gradient clipping relaxed from 1.0 to 100, which allows the critic to learn significantly faster.

We also matched the actor and temperature learning rates to 0.0003, doubled the replay buffer to 200K, and lowered the warmup threshold to 50 — below the typical episode length.

---

## Slide 8 — V2: Dense Staged Reward (5:10 – 6:05)

Our second version, V2, takes a different approach. Instead of relying entirely on human guidance, we provide continuous reward shaping so that the Q-values dominate the entropy term.

The reward function has three stages. Stage one is Approach, worth up to 3.0. We use an inverse-distance kernel — one over one plus five times distance — which provides gradient signal everywhere, even when the gripper is 20 or 30 centimeters away. This is a significant improvement over exponential decay, which produces near-zero gradient at moderate distances.

Stage two is Grasp, also worth up to 3.0. We use a smooth linear ramp from 8 centimeters down to 1 centimeter, plus a small bonus when the object begins to lift while the gripper is nearby.

Stage three is Lift, worth up to 4.0, proportional to the lift height ratio against a 10-centimeter target.

On top of these, we add an additive success bonus of plus 10. Crucially, this is additive, not an override, which preserves Q-function continuity.

The design target is that Q-values reach 15 to 40, well above the entropy term of approximately 6. This ensures the actor gradient is dominated by task reward, not by entropy.

---

## Slide 9 — V1 vs V2 Comparison (6:05 – 6:45)

Let me summarize the key differences between our two versions.

V1 uses a binary sparse reward. Its strength is simplicity — no reward engineering, no risk of rewarding wrong behavior, and it has been validated in the reference implementation. However, it relies heavily on human guidance and has a slow initial learning signal.

V2 uses a dense staged reward with per-step values ranging from 0 to 10. Its strength is a continuous gradient signal, high Q-to-entropy ratio, and less dependence on human intervention. However, it requires access to privileged state information, specifically the block position from the simulator, and it carries a risk of reward hacking.

Both versions share the same infrastructure, the same SAC algorithm, and the same intervention pipeline. The only difference is the reward function.

---

## Slide 10 — Engineering Contributions (6:45 – 7:20)

Beyond the reward design, we made several engineering improvements that are essential for practical training.

We built a CSV training logger that automatically records loss values, episode rewards, and intervention rates to local files — no WandB dependency required.

We implemented mid-episode transition streaming. The actor now sends data every 50 steps instead of waiting for the episode to end. This eliminates a deadlock where the learner could never start training because the warmup threshold exceeded the episode length.

We added robust error handling — CUDA OOM is caught gracefully, Ctrl-C safely flushes unsent data, and warmup progress is logged so collaborators know the system is working.

We fixed the intervention pipeline by correcting the clutch mode default, and we created a one-click deployment script with comprehensive documentation.

---

## Slide 11 — Expected Results (7:20 – 7:45)

This chart shows our expected convergence timeline for both versions. V2 is expected to reach early milestones — reaching the block and first grasp — about two to three times faster than V1, due to its continuous reward signal. However, both versions should converge to 100% success in approximately 90 minutes.

Our target is convergence within one hour, 100% success rate, in about 30,000 optimization steps, running on an RTX 3060.

Our hypothesis is that V1 converges reliably with sufficient human intervention, while V2 converges faster but carries higher reward hacking risk. Both should achieve over 90% success within two hours.

---

## Slide 12 — Conclusion & Future Work (7:45 – 8:00)

To summarize. We identified and fixed critical replication issues in HIL-SERL for pick-and-lift. We designed two complementary reward approaches. We aligned hyperparameters with a proven reference implementation. And we built robust training infrastructure.

For future work, we plan to conduct full comparative training runs, explore real-robot transfer, investigate adaptive intervention scheduling, and study curriculum-based reward annealing — transitioning from dense to sparse rewards during training.

Thank you. I am happy to take any questions.
