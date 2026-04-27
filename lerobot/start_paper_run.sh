#!/usr/bin/env bash
# start_paper_run.sh — Pre-flight wrapper for paper-quality training runs.
#
# Why this script exists:
#   - train_config_gym_hil_touch.json now uses a FIXED output_dir
#     (outputs/train/paper_run_v1 for V1, paper_run_v2 for V2)
#   - This guarantees Learner and Actor write to the SAME directory
#     (previously, with output_dir: null, each process auto-generated
#      its own timestamped dir → CSV data scattered across two dirs)
#   - But the training pipeline refuses to start if the fixed dir
#     already exists (FileExistsError)
#
# What it does:
#   1) If $RUN_DIR exists → rename to ${RUN_DIR}_archived_<timestamp>
#   2) Print the two commands you need to run (Learner + Actor)
#
# Usage (from lerobot/ directory):
#   ./start_paper_run.sh
#
# Then open two terminals as instructed and run the commands.

set -e

# ── Read the configured output_dir from JSON ──
CONFIG="train_config_gym_hil_touch.json"
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: $CONFIG not found. Run this script from the lerobot/ directory."
  exit 1
fi

# Extract output_dir using python (no jq dependency)
RUN_DIR=$(python3 -c "
import json, sys
with open('$CONFIG') as f:
    cfg = json.load(f)
od = cfg.get('output_dir')
if not od:
    print('NULL', file=sys.stderr)
    sys.exit(1)
print(od)
")

if [ -z "$RUN_DIR" ] || [ "$RUN_DIR" = "NULL" ]; then
  echo "ERROR: 'output_dir' is null in $CONFIG. Set it to a fixed path first."
  exit 1
fi

# ── Archive existing run dir if present ──
if [ -d "$RUN_DIR" ]; then
  ARCHIVE="${RUN_DIR}_archived_$(date +%Y%m%d_%H%M%S)"
  echo ">>> Existing run dir found: $RUN_DIR"
  echo ">>> Archiving to:           $ARCHIVE"
  mv "$RUN_DIR" "$ARCHIVE"
  echo ">>> Archive complete."
fi

# Note: don't pre-create $RUN_DIR — let the training code create it
# so it owns the lifecycle (avoids permissions weirdness).

# ── Print next-step instructions ──
PWD_ABS=$(pwd)
cat <<EOF

================================================================
 Ready to start paper-quality run
   config:     $CONFIG
   output_dir: $RUN_DIR
================================================================

Open TWO terminals and run these commands:

  ┌─ Terminal A (LEARNER) ─────────────────────────────────────┐
  │ cd $PWD_ABS
  │ conda activate hilserl
  │ python -m lerobot.rl.learner --config_path $CONFIG
  └────────────────────────────────────────────────────────────┘

  Wait for the Learner to print:
      [LEARNER] Starting learner thread

  ┌─ Terminal B (ACTOR) ───────────────────────────────────────┐
  │ cd $PWD_ABS
  │ conda activate hilserl
  │ python -m lerobot.rl.actor --config_path $CONFIG
  └────────────────────────────────────────────────────────────┘

After ~5 minutes, verify all 5 paper-data files have content:

  cd $RUN_DIR/training_logs
  for f in training_metrics.csv episode_metrics.csv eval_metrics.csv \\
           experiment_metadata.json training_summary.json; do
    echo "\$f: \$(wc -l < \$f) lines"
  done

Expected after 5 min:
  training_metrics.csv: > 1 line  (i.e. >0 optimization steps)
  episode_metrics.csv:  >= 1 line (i.e. >=1 episode finished)

If training_metrics.csv stays at 1 line (header only), the Learner
isn't training — see docs/实验合作内容.md → 故障排查.

================================================================

EOF
