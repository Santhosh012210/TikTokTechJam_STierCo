#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

AGENT_MAX_ITER="${AGENT_MAX_ITER:-3}" \
AGENT_WALL_HOURS="${AGENT_WALL_HOURS:-0.5}" \
AGENT_BOOTSTRAP_MAX_TURNS="${AGENT_BOOTSTRAP_MAX_TURNS:-24}" \
AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-16}" \
AGENT_AUTO_RESUME_QUOTA="${AGENT_AUTO_RESUME_QUOTA:-0}" \
AGENT_MAX_RUN_COST_USD="${AGENT_MAX_RUN_COST_USD:-1.00}" \
exec "$repo_dir/mle_agent/scripts/run_agent.sh" "$@"
