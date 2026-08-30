#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_exe="$repo_dir/.venv/bin/python"

if [[ ! -x "$python_exe" ]]; then
  echo "Missing virtual environment: $python_exe" >&2
  exit 1
fi

cd "$repo_dir"
exec "$python_exe" -m mle_agent.harness.finalize "$@"
