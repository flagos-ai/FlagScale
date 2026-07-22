#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
#
# When --ref-path is provided, the output path for the current direction
# may be omitted; the result is written to a temporary directory, compared
# against the reference, and then cleaned up.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat <<'EOF'
Usage: ./run.sh <direction> [python_args...]

  direction:  meg2hf | hf2meg

All remaining arguments are passed directly to convert_qwen35.py.

Required arguments:
  --yaml PATH          Path to training yaml config

Direction-specific arguments:
  --hf-path PATH|ID    For hf2meg: input HF checkpoint or ModelScope model ID (required).
                       For meg2hf: output HF checkpoint directory (required unless --ref-path is given).
  --meg-path PATH      For hf2meg: output Megatron checkpoint directory (required unless --ref-path is given).
                       For meg2hf: input Megatron checkpoint directory (required).

Optional arguments:
  --ref-path PATH      Reference checkpoint for validation. When provided, the
                       output path for the current direction may be omitted; the
                       converted checkpoint is written to a temporary directory,
                       compared against the reference, and then deleted.
  --ref-skip-value     When validating against --ref-path, skip numerical value
                       comparison and only compare structure, keys, and shapes.
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

  # HF -> Megatron validation only (no permanent output)
  ./run.sh hf2meg \
      --yaml /path/to/4b.yaml \
      --hf-path Qwen/Qwen3.5-4B \
      --ref-path /path/to/ref/meg

  # Megatron -> HF validation only (no permanent output)
  ./run.sh meg2hf \
      --yaml /path/to/4b.yaml \
      --meg-path /path/to/meg \
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
