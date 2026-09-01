#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_exe="$repo_dir/.venv/bin/python"

if [[ ! -x "$python_exe" ]]; then
  echo "Missing virtual environment: $python_exe" >&2
  echo "Follow the Setup and run section in README.md first." >&2
  exit 1
fi

cd "$repo_dir"
AGENT_AUTO_RESUME_QUOTA="${AGENT_AUTO_RESUME_QUOTA:-1}" \
AGENT_MAX_RUN_COST_USD="${AGENT_MAX_RUN_COST_USD:-6.00}" \
exec "$python_exe" -m mle_agent.harness.agent_main \
  --max-iter "${AGENT_MAX_ITER:-12}" \
  --wall-hours "${AGENT_WALL_HOURS:-2}" \
  --bootstrap-turns "${AGENT_BOOTSTRAP_MAX_TURNS:-24}" \
  --agent-turns "${AGENT_MAX_TURNS:-16}" \
  "$@"
