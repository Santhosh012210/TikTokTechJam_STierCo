"""LLM provider abstraction — the only file that knows about any specific SDK.

Switch providers by setting LLM_PROVIDER in .env:
  LLM_PROVIDER=anthropic   → model defaults to claude-haiku-4-5-20251001
  LLM_PROVIDER=groq        → model defaults to openai/gpt-oss-120b
  LLM_PROVIDER=gemini      → model defaults to gemini-3.5-flash-lite
  LLM_PROVIDER=ollama      → model defaults to llama3.2
  LLM_PROVIDER=openai      → model defaults to gpt-4o-mini

Hosted providers use the generic LLM_API_KEY. Set LLM_MODEL or LLM_BASE_URL
to override the selected provider's defaults.

builder.py and strategist.py import only from this file — never from a vendor SDK.
"""
import json
import math
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
    provider_metadata: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text:          str | None        # assistant text (may be None if only tool calls)
    tool_calls:    list[ToolCall]    # tool calls the model wants to make
    stop_reason:   str               # "end_turn" | "tool_use" | "length"
    input_tokens:  int
    output_tokens: int
    reasoning_summary: str | None = None  # concise user-visible decision summary
    provider_metadata: dict = field(default_factory=dict)


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


class CostLimitError(RuntimeError):
    """Raised before a provider request that could cross the configured run cap."""

    non_retryable = True


_MODEL_PRICES_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-2026-03-05": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
    },
}


# ---------------------------------------------------------------------------
# Provider capability registry
# ---------------------------------------------------------------------------
#
# ``init_chat_model`` will happily construct a client for any provider it knows.
# That is not the same as that provider supporting the features this harness
# depends on. Strict tool schemas, native json_schema structured output, and
# cache-read token reporting are all optional in practice, and a provider that
# silently lacks one degrades the run in a way that is hard to attribute later:
# structured reflection quietly falls back to text parsing, or the cost gate
# under-counts because cached input is billed as fresh.
#
# So the registry is an allowlist, not a hint. A provider must be declared here,
# with a price entry for the chosen model, before a run may use it.

@dataclass(frozen=True)
class ProviderCapabilities:
    """What one provider is known to support, and how to construct it."""

    #: ``init_chat_model`` provider id.
    provider: str
    #: Environment variable holding this provider's key.
    api_key_env: str
    #: Provider accepts OpenAI-style strict function schemas.
    strict_tools: bool
    #: Provider supports with_structured_output(method="json_schema").
    json_schema_structured_output: bool
    #: Provider reports cache-read input tokens in usage_metadata.
    reports_cache_read_tokens: bool
    #: Provider accepts a reasoning_effort argument.
    reasoning_effort: bool
    #: Extra kwargs passed straight to init_chat_model.
    extra_kwargs: dict = field(default_factory=dict)

    def structured_output_method(self) -> str:
        return (
            "json_schema" if self.json_schema_structured_output else "function_calling"
        )


PROVIDER_CAPABILITIES: dict[str, ProviderCapabilities] = {
    "openai": ProviderCapabilities(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        strict_tools=True,
        json_schema_structured_output=True,
        reports_cache_read_tokens=True,
        reasoning_effort=True,
        # The Responses API is what surfaces reasoning-token accounting and
        # cache-read details this harness bills against.
        extra_kwargs={"use_responses_api": True},
    ),
    "anthropic": ProviderCapabilities(
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        strict_tools=False,
        # Anthropic tool-use is the structured-output mechanism; there is no
        # separate json_schema response format.
        json_schema_structured_output=False,
        reports_cache_read_tokens=True,
        reasoning_effort=False,
    ),
    "google_genai": ProviderCapabilities(
        provider="google_genai",
        api_key_env="GOOGLE_API_KEY",
        strict_tools=False,
        json_schema_structured_output=True,
        # Gemini does not report cache reads in usage_metadata, so every input
        # token is billed at the uncached rate. That over-estimates rather than
        # under-estimates spend, which is the safe direction for a cost cap.
        reports_cache_read_tokens=False,
        reasoning_effort=False,
    ),
}


class UnsupportedProviderError(ValueError):
    """Raised for a provider that is not in the audited capability registry."""

    non_retryable = True


class LangChainModelClient(LLMClient):
    """Thin LangChain model/message adapter; the harness still owns the loop."""

    _STOP_MAP = {
        "stop": "end_turn",
        "completed": "end_turn",
        "tool_calls": "tool_use",
        "length": "length",
        "max_tokens": "length",
        "incomplete": "length",
    }

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        provider: str = "openai",
        reasoning_effort: str = "low",
        timeout_s: float = 120,
        max_cost_usd: float = 1.0,
        model_instance: object | None = None,
    ) -> None:
        if model not in _MODEL_PRICES_USD_PER_MILLION:
            raise ValueError(
                f"No audited price entry for AGENT_MODEL={model!r}; refusing an unbounded run"
            )
        if max_cost_usd <= 0:
            raise ValueError("AGENT_MAX_RUN_COST_USD must be positive")
        if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"Unsupported AGENT_REASONING_EFFORT={reasoning_effort!r}")
        capabilities = PROVIDER_CAPABILITIES.get(provider)
        if capabilities is None:
            raise UnsupportedProviderError(
                f"Provider {provider!r} is not in the audited capability registry "
                f"({', '.join(sorted(PROVIDER_CAPABILITIES))}). Add a ProviderCapabilities "
                "entry with verified strict-tool, structured-output, and cache-token "
                "behaviour before using it for a scored run."
            )
        self._model_name = model
        self._capabilities = capabilities
        self._prices = _MODEL_PRICES_USD_PER_MILLION[model]
        self._max_cost_usd = float(max_cost_usd)
        self._spent_usd = 0.0
        if model_instance is None:
            from langchain.chat_models import init_chat_model

            kwargs: dict[str, object] = {
                "model_provider": capabilities.provider,
                "api_key": api_key,
                "timeout": timeout_s,
                # The harness owns retry. LangChain's own max_retries would
                # silently multiply billed calls underneath the cost gate and
                # produce retries that never reach the run log, so it is off.
                "max_retries": 0,
                **capabilities.extra_kwargs,
            }
            if capabilities.reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            self._model = init_chat_model(model, **kwargs)
        else:
            self._model = model_instance

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def max_cost_usd(self) -> float:
        return self._max_cost_usd

    @staticmethod
    def _estimate_input_tokens(messages: list[dict]) -> int:
        # Code/JSON often tokenize more densely than prose. Three chars/token plus
        # fixed message overhead is intentionally conservative for the cost gate.
        encoded = json.dumps(messages, ensure_ascii=False, default=str)
        return max(1, math.ceil(len(encoded) / 3) + 256)

    def _projected_call_cost(self, messages: list[dict], max_tokens: int) -> float:
        input_tokens = self._estimate_input_tokens(messages)
        return (
            input_tokens * self._prices["input"]
            + max_tokens * self._prices["output"]
        ) / 1_000_000

    @staticmethod
    def _native_tools(tools: list[dict]) -> list[dict]:
        """Canonical tool dicts -> OpenAI function format.

        ``args_model`` is carried on the canonical spec so dispatch can validate
        against the same Pydantic model that produced the schema; it is harness
        bookkeeping and must not be sent to the provider.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _response_text(response: object) -> str | None:
        value = getattr(response, "text", None)
        if isinstance(value, str) and value:
            return value
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            ]
            joined = "\n".join(part for part in parts if part)
            return joined or None
        return None

    def complete(self, messages, tools=None, max_tokens=4096) -> LLMResponse:
        projected_call = self._projected_call_cost(messages, max_tokens)
        if self._spent_usd + projected_call > self._max_cost_usd:
            raise CostLimitError(
                "Next model call could cross AGENT_MAX_RUN_COST_USD: "
                f"spent=${self._spent_usd:.4f}, projected_call=${projected_call:.4f}, "
                f"cap=${self._max_cost_usd:.2f}"
            )

        runnable = self._model
        if tools:
            bind_kwargs: dict[str, object] = {}
            if self._capabilities.strict_tools:
                bind_kwargs["strict"] = True
            runnable = runnable.bind_tools(self._native_tools(tools), **bind_kwargs)
        response = runnable.invoke(messages, max_completion_tokens=max_tokens)
        return self._normalise(response)

    def _normalise(self, response: object) -> LLMResponse:
        """Provider message -> canonical LLMResponse, billing the call as a side effect.

        Shared by the tool-calling and structured-output paths so both are billed
        against the same cap and reach the run log with the same shape.
        """
        normalised_calls: list[ToolCall] = []
        for call in list(getattr(response, "tool_calls", None) or []):
            arguments = call.get("args", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            normalised_calls.append(ToolCall(
                id=str(call.get("id", "")),
                name=str(call.get("name", "")),
                input=dict(arguments),
            ))

        usage = dict(getattr(response, "usage_metadata", None) or {})
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        input_details = dict(usage.get("input_token_details", {}) or {})
        output_details = dict(usage.get("output_token_details", {}) or {})
        cached_tokens = (
            int(input_details.get("cache_read", 0) or 0)
            if self._capabilities.reports_cache_read_tokens
            else 0
        )
        uncached_tokens = max(0, input_tokens - cached_tokens)
        call_cost = (
            uncached_tokens * self._prices["input"]
            + cached_tokens * self._prices["cached_input"]
            + output_tokens * self._prices["output"]
        ) / 1_000_000
        self._spent_usd += call_cost

        metadata = dict(getattr(response, "response_metadata", None) or {})
        finish = str(
            metadata.get("finish_reason")
            or metadata.get("status")
            or ("tool_calls" if normalised_calls else "stop")
        ).lower()
        # Responses API messages can report status="completed" even when the
        # completed response contains function calls. Tool presence is the
        # authoritative signal: returning end_turn here would append an
        # assistant tool call without its required ToolMessage output.
        stop_reason = (
            "tool_use"
            if normalised_calls
            else self._STOP_MAP.get(finish, "end_turn")
        )
        return LLMResponse(
            text=self._response_text(response),
            tool_calls=normalised_calls,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_metadata={
                "model": self._model_name,
                "response_id": metadata.get("id"),
                "cached_input_tokens": cached_tokens,
                "reasoning_tokens": int(output_details.get("reasoning", 0) or 0),
                "call_cost_usd": call_cost,
                "cumulative_cost_usd": self._spent_usd,
                "max_cost_usd": self._max_cost_usd,
            },
        )

    def complete_structured(
        self,
        messages,
        schema: type,
        max_tokens: int = 4096,
    ) -> tuple[object | None, LLMResponse]:
        """Request one response constrained to ``schema``.

        Returns ``(parsed_or_None, raw_response)``. ``include_raw=True`` keeps the
        underlying AIMessage, so token usage, cost, and the exact text still reach
        the run log even when parsing fails — the harness needs the audit record
        regardless of whether the structured result is usable.

        A parse failure is returned as ``None`` rather than raised: a malformed
        reflection is a recoverable research event, not a provider outage, and
        raising here would trip the retry path and buy a second billed call for
        something a retry cannot fix.
        """
        projected_call = self._projected_call_cost(messages, max_tokens)
        if self._spent_usd + projected_call > self._max_cost_usd:
            raise CostLimitError(
                "Next model call could cross AGENT_MAX_RUN_COST_USD: "
                f"spent=${self._spent_usd:.4f}, projected_call=${projected_call:.4f}, "
                f"cap=${self._max_cost_usd:.2f}"
            )

        runnable = self._model.with_structured_output(
            schema,
            method=self._capabilities.structured_output_method(),
            include_raw=True,
        )
        result = runnable.invoke(messages, max_completion_tokens=max_tokens)
        if isinstance(result, dict):
            raw = result.get("raw")
            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")
        else:  # a provider or fake that ignored include_raw
            raw, parsed, parsing_error = None, result, None

        response = (
            self._normalise(raw)
            if raw is not None
            else LLMResponse(
                text=None, tool_calls=[], stop_reason="end_turn",
                input_tokens=0, output_tokens=0,
                provider_metadata={"model": self._model_name},
            )
        )
        if parsing_error is not None:
            response.provider_metadata["structured_output_error"] = str(parsing_error)
        response.provider_metadata["structured_output_method"] = (
            self._capabilities.structured_output_method()
        )
        return (parsed if parsing_error is None else None), response

    def add_response_to_history(self, messages, response: LLMResponse) -> None:
        from langchain_core.messages import AIMessage

        messages.append(AIMessage(
            content=response.text or "",
            tool_calls=[
                {"id": call.id, "name": call.name, "args": call.input, "type": "tool_call"}
                for call in response.tool_calls
            ],
        ))

    def add_tool_results_to_history(self, messages, tool_calls, outputs) -> None:
        from langchain_core.messages import ToolMessage

        for call, output in zip(tool_calls, outputs):
            messages.append(ToolMessage(
                content=output,
                tool_call_id=call.id,
                name=call.name,
            ))


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

    @staticmethod
    def _extra_content(value: object) -> dict:
        """Copy opaque vendor metadata without inspecting or transforming it."""
        extra = getattr(value, "extra_content", None)
        if extra is None and hasattr(value, "model_dump"):
            extra = value.model_dump(exclude_none=True).get("extra_content")
        if hasattr(extra, "model_dump"):
            extra = extra.model_dump(exclude_none=True)
        return dict(extra) if isinstance(extra, dict) else {}

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
                tool_extra = self._extra_content(tc)
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                    provider_metadata={"extra_content": tool_extra} if tool_extra else {},
                ))

        finish = response.choices[0].finish_reason or "stop"
        message_extra = self._extra_content(msg)
        return LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            stop_reason=self._STOP_MAP.get(finish, "end_turn"),
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            provider_metadata={"extra_content": message_extra} if message_extra else {},
        )

    def add_response_to_history(self, messages, response: LLMResponse) -> None:
        msg: dict = {"role": "assistant", "content": response.text or ""}
        if response.tool_calls:
            native_calls = []
            for tc in response.tool_calls:
                native_call = {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                }
                extra_content = tc.provider_metadata.get("extra_content")
                if extra_content:
                    native_call["extra_content"] = extra_content
                native_calls.append(native_call)
            msg["tool_calls"] = native_calls
        message_extra = response.provider_metadata.get("extra_content")
        if message_extra:
            msg["extra_content"] = message_extra
        messages.append(msg)

    def add_tool_results_to_history(self, messages, tool_calls, outputs) -> None:
        for tc, out in zip(tool_calls, outputs):
            result = {"role": "tool", "tool_call_id": tc.id, "content": out}
            # Gemini 3.x identifies function responses by both call ID and name.
            if tc.provider_metadata.get("extra_content"):
                result["name"] = tc.name
            messages.append(result)


# ---------------------------------------------------------------------------
# Factory — reads LLM_PROVIDER and related env vars
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict] = {
    "anthropic": {
        "requires_api_key": True,
        "model":    "claude-haiku-4-5-20251001",
        "base_url": None,
    },
    "groq": {
        "requires_api_key": True,
        "model":    "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "gemini": {
        "requires_api_key": True,
        "model":    "gemini-3.5-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "ollama": {
        "requires_api_key": False,
        "model":    "llama3.2",
        "base_url": "http://localhost:11434/v1",
    },
    "openai": {
        "requires_api_key": True,
        "model":    "gpt-4o-mini",
        "base_url": None,
    },
}


def _resolve_provider_settings() -> tuple[str, str, str, str | None]:
    """Resolve and validate the selected provider without importing an SDK."""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Choose from: {list(_PROVIDER_DEFAULTS)}"
        )

    defaults = _PROVIDER_DEFAULTS[provider]
    model = os.environ.get("LLM_MODEL") or defaults["model"]
    base_url = os.environ.get("LLM_BASE_URL") or defaults["base_url"]

    api_key = os.environ.get("LLM_API_KEY", "")
    if defaults["requires_api_key"] and not api_key:
        raise EnvironmentError(
            f"LLM_PROVIDER={provider} requires LLM_API_KEY to be set "
            "(in .env or environment)."
        )
    if not api_key:
        api_key = "ollama"  # OpenAI-compatible clients require a non-empty value.

    return provider, api_key, model, base_url or None


def validate_provider_environment() -> dict[str, str | None]:
    """Fail fast on invalid provider configuration, without exposing the key."""
    provider, _api_key, model, base_url = _resolve_provider_settings()
    return {"provider": provider, "model": model, "base_url": base_url}


def make_client() -> LLMClient:
    """Instantiate the correct LLMClient based on LLM_PROVIDER env var."""
    provider, api_key, model, base_url = _resolve_provider_settings()

    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    return OpenAICompatClient(api_key=api_key, model=model, base_url=base_url or None)


def resolve_langchain_settings() -> dict[str, object]:
    """Resolve the LangChain runtime against the capability registry, without exposing keys."""
    model_spec = os.environ.get("AGENT_MODEL", "openai:gpt-5.6-terra").strip()
    provider, separator, model = model_spec.partition(":")
    if not separator or not provider or not model:
        raise ValueError(
            "AGENT_MODEL must be '<provider>:<model-id>', for example "
            "openai:gpt-5.6-terra"
        )
    capabilities = PROVIDER_CAPABILITIES.get(provider)
    if capabilities is None:
        raise UnsupportedProviderError(
            f"AGENT_MODEL provider {provider!r} is not in the audited capability "
            f"registry ({', '.join(sorted(PROVIDER_CAPABILITIES))})"
        )
    api_key = os.environ.get(capabilities.api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"AGENT_MODEL={model_spec} requires {capabilities.api_key_env} to be set"
        )
    # This project keeps all traces local. Do not allow an inherited shell value
    # to silently enable a second external telemetry destination.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return {
        "provider": provider,
        "model": model,
        "model_spec": model_spec,
        "api_key": api_key,
        "capabilities": capabilities,
        "reasoning_effort": os.environ.get(
            "AGENT_REASONING_EFFORT", "low"
        ).strip().lower(),
        "timeout_s": float(os.environ.get("AGENT_MODEL_TIMEOUT_S", "120")),
        "max_cost_usd": float(os.environ.get("AGENT_MAX_RUN_COST_USD", "1.00")),
    }


def make_langchain_client() -> LangChainModelClient:
    settings = resolve_langchain_settings()
    return LangChainModelClient(
        model=str(settings["model"]),
        api_key=str(settings["api_key"]),
        provider=str(settings["provider"]),
        reasoning_effort=str(settings["reasoning_effort"]),
        timeout_s=float(settings["timeout_s"]),
        max_cost_usd=float(settings["max_cost_usd"]),
    )
