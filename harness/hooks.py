"""Provider-independent lifecycle hooks for deterministic harness checks."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable

_MAX_HOOK_OUTPUT_CHARS = 4000


@dataclass(frozen=True)
class PostFileSaveHook:
    """A command invoked after a saved file matches ``path_glob``.

    Commands are argument vectors, not shell strings. ``{file}`` is replaced
    with the absolute saved-file path, which keeps hook execution predictable
    even when a generated path contains whitespace or shell metacharacters.
    """

    name: str
    path_glob: str
    command: tuple[str, ...]
    timeout_seconds: float = 10.0

    def matches(self, path: Path) -> bool:
        normalized = path.resolve().as_posix()
        pattern = self.path_glob.lstrip("/")
        return fnmatchcase(normalized, self.path_glob) or fnmatchcase(
            normalized, f"*/{pattern}"
        )


@dataclass(frozen=True)
class HookResult:
    hook_name: str
    success: bool
    output: str


POST_FILE_SAVE_HOOKS: tuple[PostFileSaveHook, ...] = (
    PostFileSaveHook(
        name="model-py-compile",
        path_glob="experiment_workspace/**/model.py",
        command=(sys.executable, "-m", "py_compile", "{file}"),
    ),
)


def run_post_file_save_hooks(
    path: Path,
    hooks: Iterable[PostFileSaveHook] = POST_FILE_SAVE_HOOKS,
) -> list[HookResult]:
    """Run every matching PostFileSave hook and return all outcomes."""
    target = path.resolve()
    results: list[HookResult] = []
    for hook in hooks:
        if not hook.matches(target):
            continue
        command = [part.replace("{file}", str(target)) for part in hook.command]
        try:
            completed = subprocess.run(
                command,
                cwd=str(target.parent),
                capture_output=True,
                text=True,
                timeout=hook.timeout_seconds,
                check=False,
            )
            output = (completed.stdout + completed.stderr).strip()
            if len(output) > _MAX_HOOK_OUTPUT_CHARS:
                output = "... [hook output truncated]\n" + output[-_MAX_HOOK_OUTPUT_CHARS:]
            results.append(HookResult(hook.name, completed.returncode == 0, output))
        except subprocess.TimeoutExpired:
            results.append(
                HookResult(
                    hook.name,
                    False,
                    f"timed out after {hook.timeout_seconds:g}s",
                )
            )
        except OSError as exc:
            results.append(HookResult(hook.name, False, f"could not run hook: {exc}"))
    return results
