#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$repo_dir"
AGENT_MAX_ITER=1 \
AGENT_BOOTSTRAP_MAX_TURNS=24 \
AGENT_MAX_TURNS=16 \
AGENT_WALL_HOURS=0.5 \
exec "$repo_dir/mle_agent/scripts/run_agent.sh" "$@"
