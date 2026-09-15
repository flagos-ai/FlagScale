#!/usr/bin/env bash

set -euo pipefail

: "${TE_FL_WHEEL_DIR:?TE_FL_WHEEL_DIR is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_common.sh"
python_bin="${CI_PYTHON_BIN:-$(command -v python3)}"
if [ ! -x "$python_bin" ]; then
  echo "::error::TE-FL Python executable not found: $python_bin" >&2
  exit 1
fi

if [ ! -d "$TE_FL_WHEEL_DIR" ]; then
  echo "::error::TE-FL wheel cache directory is missing: $TE_FL_WHEEL_DIR" >&2
  exit 1
fi

mapfile -t wheels < <(
  find "$TE_FL_WHEEL_DIR" -maxdepth 1 -type f -name 'transformer_engine*.whl' -print | sort
)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "::error::Expected exactly one TE-FL wheel, found ${#wheels[@]}" >&2
  printf '  %s\n' "${wheels[@]}" >&2
  exit 1
fi

install_pip_args=()
install_pip_args_json="${CI_RUNTIME_PIP_INSTALL_ARGS_JSON:-[]}"
parsed_install_pip_args=''
if ! parsed_install_pip_args=$("$python_bin" - "$install_pip_args_json" <<'PY'
import json
import sys

values = json.loads(sys.argv[1])
if not isinstance(values, list) or not all(
    isinstance(value, str)
    and value
    and not any(character in value for character in ("\n", "\r", "\t"))
    for value in values
):
    raise SystemExit("TE-FL install_pip_args must be a JSON string array")
for value in values:
    print(f"__CI_TE_FL_ARG__\t{value}")
PY
); then
  echo "::error::Invalid TE-FL pip install argument configuration" >&2
  exit 1
fi

while IFS=$'\t' read -r record_type arg; do
  case "$record_type" in
    __CI_TE_FL_ARG__)
      install_pip_args+=("$arg")
      ;;
    '')
      ;;
    *)
      ;;
  esac
done <<< "$parsed_install_pip_args"

ci_configure_training_pythonpath
ci_export_env PYTHONNOUSERSITE 1

# Uninstall all existing TransformerEngine variants
"$python_bin" -m pip uninstall -y \
  transformer-engine transformer-engine-torch \
  transformer-engine-cu11 transformer-engine-cu12 transformer-engine-cu13 \
  >/dev/null 2>&1 || true

# Install TE-FL wheel with --no-deps to prevent pip from resolving and downgrading dependencies
"$python_bin" -m pip install \
  --force-reinstall \
  --no-deps \
  --no-cache-dir \
  "${install_pip_args[@]}" \
  "${wheels[0]}"

"$python_bin" - <<'PY'
import os
import sys
import sysconfig
import transformer_engine

print(f"TE-FL Python: {sys.executable}")
print(f"TE-FL wheel import passed: {transformer_engine.__file__}")

actual_path = os.path.realpath(transformer_engine.__file__ or "")
install_roots = {
    os.path.realpath(path)
    for path in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib"))
    if path
}
if not actual_path or not any(
    os.path.commonpath([root, actual_path]) == root for root in install_roots
):
    raise ImportError(
        f"transformer_engine resolved outside the active Python environment: {actual_path}"
    )
PY
