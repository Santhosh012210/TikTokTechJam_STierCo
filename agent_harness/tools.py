"""Anthropic tool schemas and executor functions for the Builder session.

Three tools:
  write_file  — write text content to a path inside the candidate directory
  read_file   — read a file (candidate dir or starter kit root only)
  run_bash    — run a shell command, with test-set scan before executing model.py
"""
import os
import re
import subprocess
import tempfile
from pathlib import Path

from agent_harness.validator import scan_candidate_source

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic format)
# ---------------------------------------------------------------------------

BUILDER_TOOLS = [
    {
        "name": "write_file",
        "description": (
            "Write text content to a file path inside the candidate directory. "
            "Use this to create or overwrite model.py and any helper files."
        ),
        "parameters": {          # canonical format — provider.py converts to vendor format
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the candidate directory (e.g. 'model.py').",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the text content of a file. "
            "Allowed locations: the candidate directory or the starter kit root."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path to read.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_bash",
        "description": (
            "Run a shell command (CWD = candidate directory). "
            "Use to execute model.py, check syntax, or inspect files. "
            "model.py is scanned for test-set access before execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max runtime in seconds (default 300).",
                    "default": 300,
                },
            },
            "required": ["command"],
        },
    },
]

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9\-_]{10,}")


def redact_secrets(text: str) -> str:
    """Remove API key values from any string before logging or returning to model."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key and api_key in text:
        text = text.replace(api_key, "<REDACTED>")
    text = _SECRET_PATTERN.sub("<REDACTED>", text)
    return text

# ---------------------------------------------------------------------------
# Executor functions
# ---------------------------------------------------------------------------

_MAX_OUTPUT_CHARS = 8000


def exec_write_file(path: str, content: str, candidate_dir: Path) -> str:
    try:
        target = (candidate_dir / path).resolve()
        # Security: must stay inside candidate_dir
        if not str(target).startswith(str(candidate_dir.resolve())):
            return f"ERROR: path '{path}' escapes the candidate directory"
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tmp file + rename
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)
        return f"OK: wrote {len(content)} bytes to {target.name}"
    except Exception as e:
        return f"ERROR: {e}"


def exec_read_file(path: str, candidate_dir: Path, starter_kit_root: Path) -> str:
    try:
        p = Path(path)
        if not p.is_absolute():
            p = (candidate_dir / path).resolve()
        else:
            p = p.resolve()

        # Allow reads from candidate_dir or starter_kit_root only
        allowed = (
            str(p).startswith(str(candidate_dir.resolve()))
            or str(p).startswith(str(starter_kit_root.resolve()))
        )
        if not allowed:
            return (
                f"ERROR: '{path}' is outside allowed directories. "
                "Only candidate dir and starter kit root are readable."
            )
        if not p.exists():
            return f"ERROR: file not found: {p}"

        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_OUTPUT_CHARS:
            content = content[:_MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return content
    except Exception as e:
        return f"ERROR: {e}"


def exec_run_bash(
    command: str,
    timeout_seconds: int,
    candidate_dir: Path,
    starter_kit_root: Path,
) -> str:
    try:
        # If the command runs model.py, scan for test-set access first
        model_py = candidate_dir / "model.py"
        cmd_lower = command.lower()
        if "model.py" in cmd_lower and model_py.exists():
            source = model_py.read_text(encoding="utf-8", errors="replace")
            violations = scan_candidate_source(source)
            if violations:
                return (
                    "REJECTED: test-set access detected in model.py. "
                    f"Violations: {violations}. "
                    "Remove any access to splits['test'] or enc['test'] and retry."
                )

        # Build subprocess environment: full PATH, PYTHONPATH set, no API key
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        pythonpath = str(starter_kit_root)
        if "PYTHONPATH" in env:
            pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = pythonpath

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(candidate_dir),
            env=env,
        )
        output = result.stdout + result.stderr
        if not output:
            output = f"(exit code {result.returncode}, no output)"
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return redact_secrets(output)

    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout_seconds}s"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool_call(
    tool_name: str,
    tool_input: dict,
    candidate_dir: Path,
    starter_kit_root: Path,
) -> str:
    """Route a tool call to the correct executor. Always returns a string."""
    try:
        if tool_name == "write_file":
            return exec_write_file(
                tool_input["path"], tool_input["content"], candidate_dir
            )
        elif tool_name == "read_file":
            return exec_read_file(
                tool_input["path"], candidate_dir, starter_kit_root
            )
        elif tool_name == "run_bash":
            return exec_run_bash(
                tool_input["command"],
                int(tool_input.get("timeout_seconds", 300)),
                candidate_dir,
                starter_kit_root,
            )
        else:
            return f"ERROR: unknown tool '{tool_name}'"
    except Exception as e:
        return f"ERROR in {tool_name}: {e}"
