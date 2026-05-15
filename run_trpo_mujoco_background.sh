#!/usr/bin/env bash
set -euo pipefail

ENVS=(
  "Humanoid-v5"
  "HalfCheetah-v5"
  "Walker2d-v5"
)

LOG_ROOT="${LOG_ROOT:-outputs/background_logs}"
mkdir -p "$LOG_ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"

start_run() {
  local env_name="$1"
  local mode="$2"
  local collect_traj="$3"
  local buffer_name="$4"

  local log_file="${LOG_ROOT}/${timestamp}_${env_name}_${mode}.log"

  if [[ "$mode" == "rollout" ]]; then
    nohup python run_trpo.py \
      algorithm.collect_traj="$collect_traj" \
      algorithm.buffer_name="$buffer_name" \
      env.name="$env_name" \
      experiment_name="${env_name}-TRPO-${mode}" \
      > "$log_file" 2>&1 &
  else
    nohup python run_trpo.py \
      algorithm.collect_traj="$collect_traj" \
      algorithm.buffer_name="$buffer_name" \
      env.name="$env_name" \
      experiment_name="${env_name}-TRPO-${mode}" \
      > "$log_file" 2>&1 &
  fi

  echo "$!  ${env_name}  ${mode}  ${log_file}"
}

echo "Started TRPO background runs:"
echo "PID  ENV  MODE  LOG"

for env_name in "${ENVS[@]}"; do
  start_run "$env_name" "rollout" "false" "ReplayBuffer"
  start_run "$env_name" "trajectory" "true" "TrajectoryRollout"
done
