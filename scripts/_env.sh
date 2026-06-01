# Shared run environment. Source this from train.sh / eval.sh.
# Prefers LOCAL weights so runs never hit the Hugging Face Hub by accident.
#
#   - If VERIFIER_MODEL_ID is already set, respect it.
#   - Else, if ./qwen exists in the repo root, use it.
#   - Else, warn that a HF download will happen (cfg.model_id is a hub id).

# cd to repo root (this file lives in scripts/)
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_DIR="${VERIFIER_MODEL_ID:-$PWD/qwen}"
if [ -d "$MODEL_DIR" ]; then
  export VERIFIER_MODEL_ID="$MODEL_DIR"
  export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
  echo "[env] using LOCAL weights: $VERIFIER_MODEL_ID (offline mode)"
else
  echo "[env] no local weights at '$MODEL_DIR'."
  echo "[env] Set VERIFIER_MODEL_ID=/path/to/weights, or place them at ./qwen,"
  echo "[env] otherwise from_pretrained will DOWNLOAD cfg.model_id from HF Hub."
fi

# avoid CUDA fragmentation OOM (see SERVER.md)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
