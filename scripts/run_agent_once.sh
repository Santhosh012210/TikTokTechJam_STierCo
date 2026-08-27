#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_dir"
AGENT_MAX_ITER=1 \
AGENT_MAX_TURNS=10 \
AGENT_WALL_HOURS=0.5 \
exec "$repo_dir/scripts/run_agent.sh" "$@"
