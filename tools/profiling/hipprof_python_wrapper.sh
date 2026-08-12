#!/usr/bin/env bash
set -euo pipefail

real_python="${HIPPROF_REAL_PYTHON:-}"
if [[ -z "$real_python" ]]; then
  real_python="$(command -v python3 || command -v python)"
fi

infer_cli_values() {
  local flag="$1"
  shift
  local collecting=0
  local values=()
  for arg in "$@"; do
    if [[ "$arg" == "$flag" ]]; then
      collecting=1
      continue
    fi
    if [[ "$collecting" == "1" ]]; then
      if [[ "$arg" == --* ]]; then
        break
      fi
      values+=("$arg")
    fi
  done
  if [[ "${#values[@]}" -gt 0 ]]; then
    local IFS=,
    echo "${values[*]}"
  fi
}

global_rank="${RANK:-na}"
profile_ranks="${HIPPROF_PROFILE_RANKS:-}"
if [[ -z "$profile_ranks" ]]; then
  profile_ranks="$(infer_cli_values --profile-ranks "$@")"
fi
profile_ranks="${profile_ranks// /}"
profile_ranks="${profile_ranks#[}"
profile_ranks="${profile_ranks%]}"
if [[ -n "$profile_ranks" && "$global_rank" != "na" ]]; then
  case ",${profile_ranks}," in
    *,"${global_rank}",*) ;;
    *) exec "$real_python" "$@" ;;
  esac
fi

hipprof_bin="${HIPPROF_BIN_PATH:-hipprof}"
output_root="${HIPPROF_OUTPUT_DIR:-}"
if [[ -z "$output_root" ]]; then
  echo "HIPPROF_OUTPUT_DIR is required when using hipprof_python_wrapper.sh" >&2
  exit 2
fi

host="${HOSTNAME:-$(hostname)}"
local_rank="${LOCAL_RANK:-na}"
session_prefix="${HIPPROF_SESSION_ID_PREFIX:-flagscale}"
session_id="${session_prefix}_${host}_rank${global_rank}_pid$$"
export HIPPROF_BIN_PATH="$hipprof_bin"
export HIPPROF_SESSION_ID="$session_id"

out_dir="${output_root}/${host}_global${global_rank}_local${local_rank}_pid$$"
mkdir -p "$out_dir"

trace_args=()
trace_list="${HIPPROF_TRACE:-HIP,RCCL,HSA}"
IFS=',' read -r -a traces <<< "$trace_list"
for trace in "${traces[@]}"; do
  trace_key="$(printf '%s' "${trace// /}" | tr '[:lower:]' '[:upper:]')"
  trace_arg="$(printf '%s' "$trace_key" | tr '[:upper:]' '[:lower:]')"
  if [[ "$trace_key" == "" || "$trace_key" == "NONE" ]]; then
    continue
  fi
  if [[ "$trace_key" == "HIP" || "$trace_key" == "RCCL" || "$trace_key" == "HSA" ]]; then
    trace_args+=("--${trace_arg}-trace")
    continue
  fi
  echo "Unsupported HIPPROF_TRACE entry: $trace" >&2
  exit 2
done

group_stream="${HIPPROF_GROUP_STREAM:-1}"
group_args=()
if [[ "$group_stream" == "1" || "$group_stream" == "true" ]]; then
  group_args+=(--group-stream)
fi

segment_size="${HIPPROF_SEGMENT_SIZE:-50000}"
segment_args=()
if [[ -n "$segment_size" ]]; then
  segment_args+=(--segment-size "$segment_size")
fi

exec "$hipprof_bin" \
  --session "$session_id" \
  --trace-off \
  "${trace_args[@]}" \
  "${group_args[@]}" \
  "${segment_args[@]}" \
  -d "$out_dir" \
  -o "$out_dir/result" \
  "$real_python" "$@"
