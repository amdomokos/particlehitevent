#!/usr/bin/env bash
# Bootstrap a RunPod pod for training runs. Run from anywhere:
#   bash /workspace/particleHitEvent/scripts/runpod_bootstrap.sh [DATA_DIR]
# Exit non-zero on any failure. RunPod's job config should treat non-zero as
# fatal and stop the pod, saving cost.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$REPO_ROOT/preprocessed_data}"   # network volume mount
SMOKE_DIR="${SMOKE_DIR:-/tmp/phe_smoke}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[bootstrap] 1/5 repo:   $REPO_ROOT"
git rev-parse HEAD >/dev/null || { echo "git clone incomplete"; exit 1; }

echo "[bootstrap] 2/5 frozen artifacts intact?"
python -m Data.compute_stats_hashes --verify

echo "[bootstrap] 3/5 data mount: $DATA_DIR"
ls "$DATA_DIR"/chunk_*.pt >/dev/null 2>&1 \
    || { echo "no chunk_*.pt under $DATA_DIR — network volume not mounted?"; exit 1; }

echo "[bootstrap] 4/5 GPU visible?"
python - <<'EOF'
import torch
assert torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available()), \
    "no GPU visible to torch"
print("GPU OK:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "xpu")
EOF

# Step 4 asked torch whether a GPU exists; --require-gpu makes this smoke die
# unless the REAL training path (get_device() inside fit()) actually resolves
# onto one. A pod where torch sees a GPU but the trainer still lands on CPU
# fails here, in seconds — not as "loss converges weirdly slowly" 6h in. The
# check raises before any data loads, so enforcing it costs nothing extra.
echo "[bootstrap] 5/5 pipeline smoke test (mlp), GPU-enforced (--require-gpu)"
python -m Models.train --model mlp --smoke --num-workers 0 --require-gpu \
    --data-dir "$DATA_DIR" --checkpoint-dir "$SMOKE_DIR"

echo "[bootstrap] pod is good — launch: python -m scripts.sweep --output-root <persistent-dir> --data-dir $DATA_DIR"
