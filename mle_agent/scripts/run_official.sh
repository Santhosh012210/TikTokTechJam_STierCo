#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${TASK_DEFINITION_CONFIRMED:-}" != "1" ]]; then
  echo "Refusing official run until the Starter Kit task definition is confirmed." >&2
  echo "After confirmation, run with TASK_DEFINITION_CONFIRMED=1." >&2
  exit 2
fi

"$repo_dir/mle_agent/scripts/test_offline.sh"

AGENT_MAX_ITER=50 \
AGENT_WALL_HOURS=6 \
AGENT_BOOTSTRAP_MAX_TURNS=32 \
AGENT_MAX_TURNS=24 \
exec "$repo_dir/mle_agent/scripts/run_agent.sh" \
  --task-definition-confirmed \
  "$@"
