#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_exe="$repo_dir/.venv/bin/python"

if [[ ! -x "$python_exe" ]]; then
  echo "Missing virtual environment: $python_exe" >&2
  echo "Follow SETUP.md first." >&2
  exit 1
fi

cd "$repo_dir"
"$python_exe" -m mle_agent.tests.test_agent_sdk
"$python_exe" -m mle_agent.tests.test_memory_and_prefetch
"$python_exe" -m mle_agent.tests.test_knowledge
"$repo_dir/mle_agent/scripts/check_starter_kit.sh"
