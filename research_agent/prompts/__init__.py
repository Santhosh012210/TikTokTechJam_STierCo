"""Load and render versioned research-agent prompt templates."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template


PROMPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    path: str
    template_sha256: str
    content: str

    def evidence(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "template_sha256": self.template_sha256,
        }


def render_prompt(name: str, **values: object) -> RenderedPrompt:
    """Render ``name`` with ``string.Template`` and return audit metadata."""
    path = (PROMPT_DIR / name).resolve()
    if not path.is_relative_to(PROMPT_DIR):
        raise ValueError(f"Prompt path escapes prompt directory: {name!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    source = path.read_text(encoding="utf-8")
    content = Template(source).substitute({key: str(value) for key, value in values.items()})
    return RenderedPrompt(
        name=name,
        path=str(path.relative_to(PROMPT_DIR.parent.parent)),
        template_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        content=content,
    )


__all__ = ["PROMPT_DIR", "RenderedPrompt", "render_prompt"]
