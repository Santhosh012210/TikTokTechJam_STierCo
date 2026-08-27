#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_exe="$repo_dir/.venv/bin/python"

if [[ ! -x "$python_exe" ]]; then
  echo "Missing virtual environment: $python_exe" >&2
  echo "Follow SETUP.md first." >&2
  exit 1
fi

cd "$repo_dir"
exec "$python_exe" -m harness.agent_main \
  --max-iter "${AGENT_MAX_ITER:-3}" \
  --wall-hours "${AGENT_WALL_HOURS:-0.5}" \
  --agent-turns "${AGENT_MAX_TURNS:-8}" \
  "$@"
