#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"

cd "$PROJECT_ROOT"
source .github/scripts/set_env_common.sh

workspace_root="${GITHUB_WORKSPACE:-$(dirname "$PROJECT_ROOT")}"
if [ "${MEGATRON_LM_FL_ENABLED:-false}" = "true" ]; then
  export MEGATRON_INSTALL_DIR="$workspace_root/megatron-lm-fl-install"
fi
if [ "${TE_FL_ENABLED:-false}" = "true" ]; then
  export TE_FL_WHEEL_DIR="$workspace_root/te-fl-wheel"
fi

ci_activate_python_environment
export PYTHONNOUSERSITE=1
ci_export_env PYTHONNOUSERSITE "$PYTHONNOUSERSITE"

megatron_environment="${MEGATRON_RUNTIME_ENVIRONMENT-}"
te_fl_environment="${TE_FL_RUNTIME_ENVIRONMENT-}"
[ -n "$megatron_environment" ] || megatron_environment='{}'
[ -n "$te_fl_environment" ] || te_fl_environment='{}'
if [ "$megatron_environment" != "{}" ]; then
  ci_apply_env_json "$megatron_environment"
fi
if [ "$te_fl_environment" != "{}" ]; then
  ci_apply_env_json "$te_fl_environment"
fi

# Runtime configuration may set PYTHONPATH. Re-establish the prepared
# dependency order after applying all configured environment variables.
ci_configure_training_pythonpath

if [ "${TE_FL_ENABLED:-false}" = "true" ]; then
  : "${TE_FL_CACHE_KEY:?TE_FL_CACHE_KEY is required when TE-FL is enabled}"
  if [ -d "$TE_FL_WHEEL_DIR" ]; then
    export CI_RUNTIME_PIP_INSTALL_ARGS_JSON="${TE_FL_INSTALL_PIP_ARGS:-[]}"
    export CI_RUNTIME_PIP_PACKAGES_JSON="${TE_FL_RUNTIME_PIP_PACKAGES:-[]}"
    ci_install_runtime_packages
    bash .github/scripts/install_te_fl_runtime.sh
  else
    echo "::error::TE-FL cache was not restored: $TE_FL_WHEEL_DIR" >&2
    exit 1
  fi
fi

if [ "${MEGATRON_LM_FL_ENABLED:-false}" = "true" ]; then
  : "${MEGATRON_CACHE_KEY:?MEGATRON_CACHE_KEY is required when Megatron-LM-FL is enabled}"
  if [ -d "$MEGATRON_INSTALL_DIR" ]; then
    bash .github/scripts/install_megatron_runtime.sh
  else
    echo "::error::Megatron-LM-FL cache was not restored: $MEGATRON_INSTALL_DIR" >&2
    exit 1
  fi
fi

ci_export_env FLAGSCALE_PREPARED_TRAINING_RUNTIME 1
