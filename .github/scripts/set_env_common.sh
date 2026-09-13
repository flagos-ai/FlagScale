#!/usr/bin/env bash

# Shared, platform-neutral helpers for CI environment setup scripts.
# Callers own their shell options because this file is sourced by both strict
# CI setup scripts and runners that intentionally do not enable nounset.

CI_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_PROJECT_ROOT="$(cd "$CI_SETUP_DIR/../.." && pwd)"

ci_require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "::error::Required environment variable is empty: $name"
    exit 1
  fi
}

ci_export_env() {
  local name="$1"
  local value="$2"

  export "$name=$value"
  if [ -n "${GITHUB_ENV:-}" ]; then
    printf '%s=%s\n' "$name" "$value" >> "$GITHUB_ENV"
  fi
}

ci_resolve_python_bin() {
  local python_bin="${CI_PYTHON_BIN:-}"

  if [ -n "$python_bin" ]; then
    python_bin=$(command -v "$python_bin" 2>/dev/null || true)
  else
    python_bin=$(command -v python || command -v python3 || true)
  fi
  if [ -z "$python_bin" ] || [ ! -x "$python_bin" ]; then
    echo "::error::Python executable not found" >&2
    return 1
  fi

  export CI_PYTHON_BIN="$python_bin"
}

ci_prepend_pythonpath() {
  local entry="$1"
  local current="${PYTHONPATH:-}"
  local updated

  [ -n "$entry" ] || return 0
  updated=$("${CI_PYTHON_BIN:-python3}" -S - "$entry" "$current" <<'PY'
import os
import sys

entry = sys.argv[1]
paths = [entry]
entry_real_path = os.path.realpath(entry)
paths.extend(
    path
    for path in sys.argv[2].split(os.pathsep)
    if path and os.path.realpath(path) != entry_real_path
)
print(os.pathsep.join(paths))
PY
  )
  ci_export_env PYTHONPATH "$updated"
}

ci_sanitize_training_pythonpath() {
  local current="${PYTHONPATH:-}"
  local sanitized
  local training_overlay="${CI_TRAINING_OVERLAY_DIR:-$CI_PROJECT_ROOT/flagscale/train}"

  sanitized=$("${CI_PYTHON_BIN:-python3}" -S - \
    "$current" \
    "${MEGATRON_INSTALL_DIR:-}" \
    "$training_overlay" \
    "$CI_PROJECT_ROOT" <<'PY'
import os
import sys
import sysconfig

paths = sys.argv[1].split(os.pathsep)
preserved = {
    os.path.realpath(path)
    for path in sys.argv[2:]
    if path
}
preserved.update(
    os.path.realpath(path)
    for path in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib"))
    if path
)
result = []
seen = set()

for path in paths:
    if not path:
        continue
    real_path = os.path.realpath(path)
    if real_path in seen:
        continue
    seen.add(real_path)

    shadows_training_dependency = any(
        os.path.isdir(os.path.join(real_path, package))
        for package in ("megatron", "transformer_engine")
    )
    if real_path in preserved or not shadows_training_dependency:
        result.append(path)
    else:
        print(
            f"Removed conflicting training dependency path: {path}",
            file=sys.stderr,
        )

print(os.pathsep.join(result))
PY
  )
  ci_export_env PYTHONPATH "$sanitized"
}

ci_configure_training_pythonpath() {
  local training_overlay="${CI_TRAINING_OVERLAY_DIR:-$CI_PROJECT_ROOT/flagscale/train}"

  ci_sanitize_training_pythonpath
  ci_prepend_pythonpath "$CI_PROJECT_ROOT"
  ci_prepend_pythonpath "${MEGATRON_INSTALL_DIR:-}"
  ci_prepend_pythonpath "$training_overlay"
}

ci_apply_env_json() {
  local environment_json="$1"
  local python_bin="${CI_PYTHON_BIN:-python3}"
  local entries
  local name
  local value

  if ! entries=$("$python_bin" -S - "$environment_json" <<'PY'
import json
import re
import sys

values = json.loads(sys.argv[1])
if not isinstance(values, dict) or not values:
    raise SystemExit("environment must be a non-empty JSON object")
for key, value in values.items():
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise SystemExit(f"invalid environment variable name: {key!r}")
    if not isinstance(value, (str, int, float, bool)):
        raise SystemExit(f"environment value must be scalar: {key}")
    text = str(value)
    if any(character in text for character in ("\n", "\r", "\t")):
        raise SystemExit(f"environment value contains unsupported control characters: {key}")
    print(f"__CI_ENV__\t{key}\t{text}")
PY
  ); then
    echo "::error::Invalid environment JSON" >&2
    return 1
  fi

  while IFS=$'\t' read -r record_type name value; do
    if [ "$record_type" != "__CI_ENV__" ]; then
      [ -z "$record_type" ] || printf '%s\n' "$record_type" >&2
      continue
    fi
    [ -n "$name" ] || continue
    ci_export_env "$name" "$value"
  done <<< "$entries"
}

ci_activate_python_environment() {
  local pkg_mgr="${CI_RUNTIME_PKG_MGR:-pip}"
  local env_name="${CI_RUNTIME_ENV_NAME:-}"
  local env_path="${CI_RUNTIME_ENV_PATH:-}"

  case "$pkg_mgr" in
    conda)
      [ -n "$env_name" ] || {
        echo "::error::CI_RUNTIME_ENV_NAME is required for conda" >&2
        return 1
      }
      [ -f "$env_path/etc/profile.d/conda.sh" ] || {
        echo "::error::Invalid conda installation: $env_path" >&2
        return 1
      }
      source "$env_path/etc/profile.d/conda.sh"
      conda activate "$env_name"
      ;;
    uv)
      [ -f "$env_path/bin/activate" ] || {
        echo "::error::Invalid uv environment: $env_path" >&2
        return 1
      }
      source "$env_path/bin/activate"
      ;;
    pip)
      ;;
    *)
      echo "::error::Unsupported runtime package manager: $pkg_mgr" >&2
      return 1
      ;;
  esac

  local python_bin
  python_bin=$(command -v python || command -v python3 || true)
  if [ -z "$python_bin" ] || [ ! -x "$python_bin" ]; then
    echo "::error::Python executable not found after environment activation" >&2
    return 1
  fi
  ci_export_env PATH "$PATH"
  ci_export_env CI_PYTHON_BIN "$python_bin"
  ci_sanitize_training_pythonpath
  echo "Python: $python_bin ($($python_bin --version 2>&1))"
}

ci_install_runtime_packages() {
  local python_bin="${CI_PYTHON_BIN:-$(command -v python3)}"
  local packages_json="${CI_RUNTIME_PIP_PACKAGES_JSON:-[]}"
  local install_args_json="${CI_RUNTIME_PIP_INSTALL_ARGS_JSON:-[]}"
  local parsed
  local kind
  local value
  local -a packages=()
  local -a install_args=()

  if ! parsed=$("$python_bin" -S - "$packages_json" "$install_args_json" <<'PY'
import json
import sys

packages = json.loads(sys.argv[1])
install_args = json.loads(sys.argv[2])
def valid_value(item):
    return isinstance(item, str) and item and not any(
        character in item for character in ("\n", "\r", "\t")
    )

if not isinstance(packages, list) or not all(valid_value(item) for item in packages):
    raise SystemExit("runtime pip packages must be a JSON string array")
if not isinstance(install_args, list) or not all(valid_value(item) for item in install_args):
    raise SystemExit("runtime pip install args must be a JSON string array")

for item in packages:
    print(f"__CI_RUNTIME_PIP__\tpackage\t{item}")
for item in install_args:
    print(f"__CI_RUNTIME_PIP__\targ\t{item}")
PY
  ); then
    echo "::error::Invalid runtime pip package configuration" >&2
    return 1
  fi

  local filtered_lines
  filtered_lines=$(grep '^__CI_RUNTIME_PIP__' <<< "$parsed" || true)

  while IFS=$'\t' read -r record_type kind value; do
    case "$record_type" in
      __CI_RUNTIME_PIP__)
        case "$kind" in
          package) packages+=("$value") ;;
          arg) install_args+=("$value") ;;
          *) echo "::error::Invalid runtime pip package record: $kind" >&2; return 1 ;;
        esac
        ;;
      '') ;;
      *)
        echo "::warning::Unexpected line in runtime pip config: $record_type" >&2
        ;;
    esac
  done <<< "$filtered_lines"

  if [ "${#packages[@]}" -eq 0 ]; then
    echo "Configured runtime pip packages: none"
    return 0
  fi

  echo "Installing configured runtime pip packages: ${packages[*]}"
  "$python_bin" -m pip install --no-cache-dir "${install_args[@]}" "${packages[@]}"
}

ci_install_project() {
  cd "$CI_PROJECT_ROOT"
  "${CI_PYTHON_BIN:-$(command -v python3)}" -m pip install -e . --no-deps --no-build-isolation --no-cache-dir "$@"
}
