#!/usr/bin/env bash
# run_actor.sh — sim-style actor launcher
#
# Mirrors hil-serl-sim's run_actor.sh interface convention,
# but invokes our PyTorch-based lerobot.rl.actor module.
#
# Usage:
#   ./run_actor.sh                      # uses train_config_gym_hil_touch.json
#   ./run_actor.sh headless             # uses train_config_gym_hil_headless.json
#   ./run_actor.sh <config_path>        # uses arbitrary config
#
# IMPORTANT: start ./run_learner.sh first, then ./run_actor.sh in a separate
# terminal once the learner reports "[LEARNER] gRPC server started".

set -euo pipefail
cd "$(dirname "$0")"

case "${1:-touch}" in
  touch)    CONFIG="train_config_gym_hil_touch.json" ;;
  headless) CONFIG="train_config_gym_hil_headless.json" ;;
  *)        CONFIG="$1" ;;
esac

[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ -z "${PYTHON:-}" ]; then
  for cand in \
      /root/data/conda-envs/hilserl/bin/python \
      ~/miniconda3/envs/hilserl/bin/python \
      ~/.conda/envs/hilserl/bin/python \
      python3; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
  done
fi

echo "=== run_actor.sh ==="
echo "  config:        $CONFIG"
echo "  python:        $PYTHON"
echo "  MUJOCO_GL:     $MUJOCO_GL"
echo "==================="

exec "$PYTHON" -m lerobot.rl.actor --config_path "$CONFIG"
