#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/utils.sh"

PLATFORM="default"

while [[ $# -gt 0 ]]; do
    case $1 in
        --platform) PLATFORM="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

# Set up PYTHONPATH for megatron imports
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/flagscale/train:${PYTHONPATH:-}"

echo "=========================================="
echo "Running Unit Tests"
echo "Platform: $PLATFORM"
echo "PYTHONPATH: $PYTHONPATH"
echo "=========================================="

# Get unit test patterns from platform configuration
PATTERNS=$(python "$SCRIPT_DIR/parse_config.py" --platform "$PLATFORM" --type unit_tests 2>/dev/null || echo '{"include":"*","exclude":[]}')

# Extract include and exclude patterns
INCLUDE=$(echo "$PATTERNS" | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('include','*'))")
EXCLUDE=$(echo "$PATTERNS" | python -c "import sys, json; data=json.load(sys.stdin); exc=data.get('exclude',[]); print(' '.join(['--ignore=' + e for e in exc]) if exc else '')")

# Build pytest command
PYTEST_CMD="torchrun --nproc_per_node=8 -m pytest tests/unit_tests/ -v --tb=short"

# Apply exclude patterns if any
if [ -n "$EXCLUDE" ]; then
    PYTEST_CMD="torchrun --nproc_per_node=8 -m pytest $EXCLUDE tests/unit_tests/ -v --tb=short"
fi

echo "Running: $PYTEST_CMD"

# Run unit tests with patterns from platform config
eval "$PYTEST_CMD"

echo "=========================================="
echo "Unit tests completed successfully"
echo "=========================================="
