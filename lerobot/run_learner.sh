#!/usr/bin/env bash
# run_learner.sh — sim-style learner launcher
#
# Mirrors hil-serl-sim's run_actor.sh / run_learner.sh interface convention,
# but invokes our PyTorch-based lerobot.rl.learner module.
#
# Usage:
#   ./run_learner.sh                    # uses train_config_gym_hil_touch.json
#   ./run_learner.sh headless           # uses train_config_gym_hil_headless.json
#   ./run_learner.sh <config_path>      # uses arbitrary config
#
# Environment variables (sim-aligned):
#   MUJOCO_GL              egl|osmesa (headless rendering, default: egl)
#   HF_HUB_OFFLINE         1 to skip HuggingFace Hub validation (default: 1)
#   HF_ENDPOINT            mirror URL for HF model downloads (default: hf-mirror.com)
#   PYTHON                 path to python interpreter (auto-detect if unset)

set -euo pipefail
cd "$(dirname "$0")"

# Resolve config
case "${1:-touch}" in
  touch)    CONFIG="train_config_gym_hil_touch.json" ;;
  headless) CONFIG="train_config_gym_hil_headless.json" ;;
  *)        CONFIG="$1" ;;
esac

[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }

# Defaults aligned with sim's headless-cloud expectations
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Auto-detect python (prefer hilserl env, fall back to system)
if [ -z "${PYTHON:-}" ]; then
  for cand in \
      /root/data/conda-envs/hilserl/bin/python \
      ~/miniconda3/envs/hilserl/bin/python \
      ~/.conda/envs/hilserl/bin/python \
      python3; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
  done
fi

echo "=== run_learner.sh ==="
echo "  config:        $CONFIG"
echo "  python:        $PYTHON"
echo "  MUJOCO_GL:     $MUJOCO_GL"
echo "  HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "==================="

exec "$PYTHON" -m lerobot.rl.learner --config_path "$CONFIG"
