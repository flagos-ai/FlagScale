#!/bin/bash
# Unified checkpoint conversion entry point for Qwen3.5
# Dispatches to convert_qwen35.py with --direction.
#
# Usage: ./run.sh <direction> [python_args...]
#
#   direction:  meg2hf | hf2meg
#
# All remaining arguments are passed directly to convert_qwen35.py.
#
# Examples:
#   ./run.sh hf2meg \
#       --yaml /path/to/4b.yaml \
#       --hf-path Qwen/Qwen3.5-4B \
#       --meg-path /path/to/meg/save \
#       [--ref-path /path/to/ref]
#
#   ./run.sh meg2hf \
#       --yaml /path/to/4b.yaml \
#       --meg-path /path/to/meg \
#       --hf-path /path/to/hf/save \
#       [--ref-path /path/to/ref]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat <<'EOF'
Usage: ./run.sh <direction> [python_args...]

  direction:  meg2hf | hf2meg

All remaining arguments are passed directly to convert_qwen35.py.

Required arguments:
  --yaml PATH          Path to training yaml config
  --hf-path PATH|ID    Path to HF checkpoint directory, or a ModelScope model ID (hf2meg only)
  --meg-path PATH      Path to Megatron checkpoint directory (input or output)

Optional arguments:
  --ref-path PATH      Reference checkpoint for validation
  --tp N               Override tensor model parallel size
  --pp N               Override pipeline model parallel size
  --ep N               Override expert model parallel size (MoE only)
  --adjust-ln          Enable legacy layer-norm adjustment
  --adjust-embedding   Adjust embedding vocab size to reference (hf2meg only)

Examples:
  # HF -> Megatron (download Qwen3.5-4B from ModelScope automatically)
  ./run.sh hf2meg \
      --yaml /path/to/4b.yaml \
      --hf-path Qwen/Qwen3.5-4B \
      --meg-path /path/to/meg/save \
      --ref-path /path/to/ref/meg

  # Megatron -> HF
  ./run.sh meg2hf \
      --yaml /path/to/4b.yaml \
      --meg-path /path/to/meg \
      --hf-path /path/to/hf/save \
      --ref-path /path/to/ref/hf
EOF
}

# Help / insufficient args
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -lt 2 ]; then
    show_help
    exit 0
fi

DIRECTION=$1
shift

echo "=================================================="
echo "Direction : $DIRECTION"
echo "Script    : $SCRIPT_DIR/convert_qwen35.py"
echo "Args      : $*"
echo "=================================================="
echo ""

python "$SCRIPT_DIR/convert_qwen35.py" --direction "$DIRECTION" "$@"
