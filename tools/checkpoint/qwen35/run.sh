#!/bin/bash
# Unified checkpoint conversion entry point for Qwen3.5 (dense + MoE).
# Automatically selects the correct script based on direction and model type.
#
# Usage: ./run.sh <direction> <model_type> [python_args...]
#
#   direction:  meg2hf | hf2meg
#   model_type: dense | moe
#
# All remaining arguments are passed directly to the underlying Python script.
#
# Examples:
#   ./run.sh meg2hf dense \
#       --yaml /workspace/FlagScale/examples/qwen35/conf/train/4b.yaml \
#       --meg-ckpt-dir /workspace/FlagScale/train_qwen35_4b/checkpoints/iter_0000001 \
#       --save-dir /workspace/FlagScale/converted_hf_qwen35_4b
#
#   ./run.sh hf2meg moe \
#       --yaml /workspace/FlagScale/examples/qwen35/conf/train/35b_a3b.yaml \
#       --hf-dir /workspace/data/qwen/qwen35_data/qwen35_35ba3b \
#       --save-dir /workspace/FlagScale/checkpoints/qwen35_35ba3b_hf2meg/
#
#   ./run.sh meg2hf dense \
#       --yaml /path/to/4b.yaml \
#       --meg-ckpt-dir /path/to/ckpt \
#       --save-dir /path/to/out \
#       --hf-ref-dir /path/to/hf/ref

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat <<'EOF'
Usage: ./run.sh <direction> <model_type> [python_args...]

  direction:  meg2hf | hf2meg
  model_type: dense | moe

All remaining arguments are passed directly to the underlying Python script.

Common arguments:
  --yaml PATH               Path to training yaml config (required)
  --save-dir PATH           Output directory (required)

  For meg2hf:
    --meg-ckpt-dir PATH     Megatron checkpoint directory (required)
    --hf-ref-dir PATH       Reference HF model for validation (optional)

  For hf2meg:
    --hf-dir PATH           HF checkpoint directory (required)
    --ref-ckpt-dir PATH     Reference Megatron checkpoint for validation (optional)
    --adjust-embedding      Adjust embedding vocab size to match reference (optional)

Examples:
  # Dense: Megatron -> HF
  ./run.sh meg2hf dense \
      --yaml /path/to/4b.yaml \
      --meg-ckpt-dir /path/to/ckpt \
      --save-dir /path/to/out

  # MoE: HF -> Megatron
  ./run.sh hf2meg moe \
      --yaml /path/to/35b_a3b.yaml \
      --hf-dir /path/to/hf \
      --save-dir /path/to/out

  # With validation
  ./run.sh meg2hf dense \
      --yaml /path/to/4b.yaml \
      --meg-ckpt-dir /path/to/ckpt \
      --save-dir /path/to/out \
      --hf-ref-dir /path/to/hf/ref
EOF
}

# Help / insufficient args
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -lt 2 ]; then
    show_help
    exit 0
fi

DIRECTION=$1
MODEL_TYPE=$2
shift 2
# Resolve script path
case "$DIRECTION" in
    meg2hf)
        case "$MODEL_TYPE" in
            dense) SCRIPT="$SCRIPT_DIR/meg2hf_qwen35_dense.py" ;;
            moe)   SCRIPT="$SCRIPT_DIR/meg2hf_qwen35_moe.py"   ;;
            *) echo "Error: unknown model_type '$MODEL_TYPE'. Use: dense | moe" >&2; exit 1 ;;
        esac
        ;;
    hf2meg)
        case "$MODEL_TYPE" in
            dense) SCRIPT="$SCRIPT_DIR/hf2meg_qwen35_dense.py" ;;
            moe)   SCRIPT="$SCRIPT_DIR/hf2meg_qwen35_moe.py"   ;;
            *) echo "Error: unknown model_type '$MODEL_TYPE'. Use: dense | moe" >&2; exit 1 ;;
        esac
        ;;
    *)
        echo "Error: unknown direction '$DIRECTION'. Use: meg2hf | hf2meg" >&2
        exit 1
        ;;
esac

echo "=================================================="
echo "Direction : $DIRECTION"
echo "Model type: $MODEL_TYPE"
echo "Script    : $SCRIPT"
echo "Args      : $*"
echo "=================================================="
echo ""

python "$SCRIPT" "$@"
