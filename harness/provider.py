"""LLM provider abstraction — the only file that knows about any specific SDK.

Switch providers by setting LLM_PROVIDER in .env:
  LLM_PROVIDER=anthropic   → ANTHROPIC_API_KEY, model defaults to claude-haiku-4-5-20251001
  LLM_PROVIDER=groq        → GROQ_API_KEY,      model defaults to llama-3.3-70b-versatile
  LLM_PROVIDER=gemini      → GEMINI_API_KEY,     model defaults to gemini-2.0-flash
  LLM_PROVIDER=ollama      → no key needed,      model defaults to llama3.2
  LLM_PROVIDER=openai      → OPENAI_API_KEY,     model defaults to gpt-4o-mini

Set LLM_MODEL in .env to override the model for any provider.

builder.py and strategist.py import only from this file — never from a vendor SDK.
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Canonical types (provider-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id:    str
    name:  str
    input: dict


@dataclass
class LLMResponse:
    text:          str | None        # assistant text (may be None if only tool calls)
    tool_calls:    list[ToolCall]    # tool calls the model wants to make
    stop_reason:   str               # "end_turn" | "tool_use" | "length"
    input_tokens:  int
    output_tokens: int


# Abstract client interface
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Provider-agnostic interface. Concrete subclasses own all SDK imports."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send messages to the LLM and return a normalised response.

        `tools` uses the canonical format:
          {"name": str, "description": str,
           "parameters": {"type": "object", "properties": {...}, "required": [...]}}
        """

    @abstractmethod
    def add_response_to_history(
        self, messages: list[dict], response: LLMResponse
    ) -> None:
        """Append the assistant message to the history list (mutates in-place)."""

    @abstractmethod
    def add_tool_results_to_history(
        self,
        messages: list[dict],
        tool_calls: list[ToolCall],
        outputs: list[str],
    ) -> None:
        """Append tool-result messages to the history list (mutates in-place)."""


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    _STOP_MAP = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "length"}

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic as _anthropic  # scoped import — only loaded when provider is used
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._exc = _anthropic.APIError  # save for caller to catch if needed

    def _native_tools(self, tools: list[dict]) -> list[dict]:
        # canonical "parameters" → Anthropic "input_schema"
        return [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]

    def complete(self, messages, tools=None, max_tokens=4096) -> LLMResponse:
        kwargs: dict = {"model": self._model, "max_tokens": max_tokens, "messages": messages}
        if tools:
            kwargs["tools"] = self._native_tools(tools)
        response = self._client.messages.create(**kwargs)

        text: str | None = None
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=self._STOP_MAP.get(response.stop_reason, response.stop_reason),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def add_response_to_history(self, messages, response: LLMResponse) -> None:
        # Reconstruct Anthropic-native content blocks so the history stays valid
        content: list[dict] = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
        messages.append({"role": "assistant", "content": content})

    def add_tool_results_to_history(self, messages, tool_calls, outputs) -> None:
        results = [
            {"type": "tool_result", "tool_use_id": tc.id, "content": out}
            for tc, out in zip(tool_calls, outputs)
        ]
        messages.append({"role": "user", "content": results})


# ---------------------------------------------------------------------------
# OpenAI-compatible implementation (Groq, Gemini, Ollama, OpenAI)
# ---------------------------------------------------------------------------

class OpenAICompatClient(LLMClient):
    _STOP_MAP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "length"}

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI  # scoped import
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    def _native_tools(self, tools: list[dict]) -> list[dict]:
        # canonical format → OpenAI function-calling format
        return [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }}
            for t in tools
        ]

    def complete(self, messages, tools=None, max_tokens=4096) -> LLMResponse:
        kwargs: dict = {"model": self._model, "max_tokens": max_tokens, "messages": messages}
        if tools:
            kwargs["tools"] = self._native_tools(tools)
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))

        finish = response.choices[0].finish_reason or "stop"
        return LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            stop_reason=self._STOP_MAP.get(finish, "end_turn"),
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def add_response_to_history(self, messages, response: LLMResponse) -> None:
        msg: dict = {"role": "assistant", "content": response.text or ""}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                }
                for tc in response.tool_calls
            ]
        messages.append(msg)

    def add_tool_results_to_history(self, messages, tool_calls, outputs) -> None:
        for tc, out in zip(tool_calls, outputs):
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})


# ---------------------------------------------------------------------------
# Factory — reads LLM_PROVIDER and related env vars
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "anthropic": {
        "key_env":  "ANTHROPIC_API_KEY",
        "model":    "claude-haiku-4-5-20251001",
        "base_url": None,
    },
    "groq": {
        "key_env":  "GROQ_API_KEY",
        "model":    "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "gemini": {
        "key_env":  "GEMINI_API_KEY",
        "model":    "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "ollama": {
        "key_env":  None,
        "model":    "llama3.2",
        "base_url": "http://localhost:11434/v1",
    },
    "openai": {
        "key_env":  "OPENAI_API_KEY",
        "model":    "gpt-4o-mini",
        "base_url": None,
    },
}


def make_client() -> LLMClient:
    """Instantiate the correct LLMClient based on LLM_PROVIDER env var."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Choose from: {list(_PROVIDER_DEFAULTS)}"
        )

    defaults = _PROVIDER_DEFAULTS[provider]
    model    = os.environ.get("LLM_MODEL", defaults["model"])
    base_url = os.environ.get("LLM_BASE_URL", defaults["base_url"] or "")  or defaults["base_url"]

    # Resolve API key
    key_env = defaults["key_env"]
    api_key = os.environ.get(key_env, "") if key_env else "ollama"
    if key_env and not api_key:
        raise EnvironmentError(
            f"LLM_PROVIDER={provider} requires {key_env} to be set "
            "(in .env or environment)."
        )

    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    else:
        return OpenAICompatClient(api_key=api_key, model=model, base_url=base_url or None)
