"""LLM router: provider fallback chain, budget enforcement, tolerant parsing.

Call sites never pick a model. They declare a *purpose* and a *tier*:

    router.complete(purpose="sql_generation", tier="fast", ...)
    router.complete(purpose="analysis",       tier="reasoning", ...)

The router maps tier -> model per provider and walks the configured chain
(Gemini -> OpenRouter -> Ollama) whenever a provider is down, rate limited or
uncredentialed. That means a Gemini quota exhaustion degrades the agent to a
slower model instead of taking it offline.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import setting
from .obs import metrics
from .resilience.policies import (
    CircuitOpenError,
    PermanentError,
    RetryPolicy,
    TransientError,
    call_with_resilience,
)


def _as_list(value: Any) -> List[str]:
    """Accept either a single model id or an ordered list of them."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v]


class BudgetExceeded(RuntimeError):
    """The turn hit a hard cost/time ceiling and must wind down gracefully."""


@dataclass
class TurnBudget:
    """Per-turn cost ceiling. Shared by the LLM router and the SQL executor."""

    max_llm_calls: int = 14
    max_bytes_billed: int = 6_000_000_000
    wall_clock_s: float = 180.0
    started_at: float = field(default_factory=time.time)
    llm_calls: int = 0
    bytes_billed: int = 0

    def charge_llm(self) -> None:
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded(
                f"turn exceeded {self.max_llm_calls} LLM calls - stopping to control cost"
            )
        if time.time() - self.started_at > self.wall_clock_s:
            raise BudgetExceeded(f"turn exceeded {self.wall_clock_s:.0f}s wall clock")
        self.llm_calls += 1

    def charge_bytes(self, n: int) -> None:
        if self.bytes_billed + n > self.max_bytes_billed:
            raise BudgetExceeded(
                f"turn would scan {(self.bytes_billed + n) / 1e9:.2f} GB, over the "
                f"{self.max_bytes_billed / 1e9:.2f} GB per-turn ceiling"
            )
        self.bytes_billed += n

    def remaining_llm_calls(self) -> int:
        return max(0, self.max_llm_calls - self.llm_calls)

    def reset(self) -> "TurnBudget":
        self.started_at = time.time()
        self.llm_calls = 0
        self.bytes_billed = 0
        return self


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    failovers: int = 0

    def json(self, default: Any = None) -> Any:
        return extract_json(self.text, default)


_FENCE = re.compile(r"```(?:json|sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_fences(text: str) -> str:
    m = _FENCE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def extract_json(text: str, default: Any = None) -> Any:
    """Parse JSON out of a model response that may be wrapped in prose or fences."""
    candidate = strip_fences(text)
    for attempt in (candidate, text or ""):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return default


class _Provider:
    """Thin adapter around one chat backend.

    Each tier maps to an ORDERED LIST of models. A model can be unavailable
    independently of its provider - a 503 on one Gemini model while its siblings
    serve fine is routine - so the router walks models within a provider before
    abandoning the provider entirely.
    """

    def __init__(self, name: str, models: Dict[str, List[str]]) -> None:
        self.name = name
        self.models = {tier: _as_list(v) for tier, v in models.items()}
        self._clients: Dict[str, Any] = {}

    def models_for(self, tier: str) -> List[str]:
        return self.models.get(tier) or self.models.get("fast") or []

    def model_for(self, tier: str) -> str:
        candidates = self.models_for(tier)
        return candidates[0] if candidates else ""

    def available(self) -> bool:
        raise NotImplementedError

    def client(self, model: str):
        raise NotImplementedError

    def invoke(self, model: str, system: str, messages: List[Dict[str, str]]) -> tuple[str, int, int]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc: List[Any] = [SystemMessage(content=system)] if system else []
        for m in messages:
            lc.append(
                AIMessage(content=m["content"])
                if m.get("role") == "assistant"
                else HumanMessage(content=m["content"])
            )
        resp = self.client(model).invoke(lc)
        usage = getattr(resp, "usage_metadata", None) or {}
        content = resp.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return content or "", int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


class GeminiProvider(_Provider):
    def available(self) -> bool:
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

    def client(self, model: str):
        if model not in self._clients:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._clients[model] = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
                temperature=setting("llm.temperature", 0.1),
                max_output_tokens=setting("llm.max_output_tokens", 4096),
                timeout=setting("llm.request_timeout_s", 60),
                max_retries=0,  # retry policy lives in resilience.policies
            )
        return self._clients[model]


class OpenRouterProvider(_Provider):
    def available(self) -> bool:
        return bool(os.getenv("OPENROUTER_API_KEY"))

    def client(self, model: str):
        if model not in self._clients:
            from langchain_openai import ChatOpenAI

            self._clients[model] = ChatOpenAI(
                model=model,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                temperature=setting("llm.temperature", 0.1),
                max_tokens=setting("llm.max_output_tokens", 4096),
                timeout=setting("llm.request_timeout_s", 60),
                max_retries=0,
            )
        return self._clients[model]


class OllamaProvider(_Provider):
    def available(self) -> bool:
        return os.getenv("OLLAMA_ENABLED", "0") == "1"

    def client(self, model: str):
        if model not in self._clients:
            from langchain_ollama import ChatOllama

            self._clients[model] = ChatOllama(
                model=model,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=setting("llm.temperature", 0.1),
            )
        return self._clients[model]


class ScriptedProvider(_Provider):
    """Deterministic test double used by the offline test suite and evals.

    Holds a list of (predicate, responder) rules. Never reached in normal
    operation - it is only registered when RIA_LLM_PROVIDER=scripted.
    """

    def __init__(self) -> None:
        super().__init__("scripted", {"fast": ["scripted"], "reasoning": ["scripted"]})
        self.rules: List[tuple[Callable[[str, str], bool], Callable[[str, str], str]]] = []
        self.calls: List[Dict[str, str]] = []

    def available(self) -> bool:
        return True

    def rule(self, predicate, responder) -> "ScriptedProvider":
        self.rules.append((predicate, responder))
        return self

    def invoke(self, model: str, system: str, messages: List[Dict[str, str]]):
        prompt = messages[-1]["content"] if messages else ""
        self.calls.append({"tier": model, "system": system, "prompt": prompt})
        for predicate, responder in self.rules:
            if predicate(system, prompt):
                return responder(system, prompt), 100, 50
        return "{}", 10, 5


_REGISTRY = {
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
}


class LLMRouter:
    def __init__(self, chain: Optional[List[Dict[str, Any]]] = None, tracer=None) -> None:
        self.tracer = tracer
        self.providers: List[_Provider] = []
        forced = os.getenv("RIA_LLM_PROVIDER")
        # `None` means "use the configured chain"; an explicit [] means "none".
        chain = setting("llm.chain", []) if chain is None else chain
        for entry in chain:
            name = entry.get("provider")
            if forced and forced != name:
                continue
            cls = _REGISTRY.get(name)
            if cls is None:
                continue
            self.providers.append(
                cls(
                    name,
                    {
                        "fast": entry.get("model"),
                        "reasoning": entry.get("reasoning_model") or entry.get("model"),
                    },
                )
            )

    def register(self, provider: _Provider, first: bool = True) -> None:
        self.providers.insert(0, provider) if first else self.providers.append(provider)

    def available_providers(self) -> List[str]:
        return [p.name for p in self.providers if p.available()]

    def complete(
        self,
        *,
        purpose: str,
        system: str,
        messages: List[Dict[str, str]],
        tier: str = "fast",
        budget: Optional[TurnBudget] = None,
    ) -> LLMResult:
        if budget is not None:
            budget.charge_llm()
        metrics.incr("llm.calls", purpose=purpose)

        candidates = [p for p in self.providers if p.available()]
        if not candidates:
            raise PermanentError(
                "No LLM provider is configured. Set GOOGLE_API_KEY (or OPENROUTER_API_KEY, "
                "or OLLAMA_ENABLED=1). See docs/SETUP.md."
            )

        failovers = 0
        attempted = 0
        last_exc: BaseException | None = None
        span_cm = self.tracer.span(f"llm:{purpose}", kind="llm", purpose=purpose, tier=tier) if self.tracer else None
        span = span_cm.__enter__() if span_cm else None
        try:
            for provider in candidates:
                for model in provider.models_for(tier):
                    attempted += 1
                    started = time.perf_counter()
                    try:
                        text, tin, tout = call_with_resilience(
                            lambda p=provider, m=model: p.invoke(m, system, messages),
                            # Breaker is per provider+model: one model under load
                            # must not take its healthy siblings offline.
                            dependency=f"llm:{provider.name}:{model}",
                            policy=RetryPolicy(attempts=3, base_delay_s=0.5, max_delay_s=8.0),
                            on_retry=lambda attempt, exc, s=span: s and s.event(
                                "retry", attempt=attempt + 1, error=str(exc)[:200]
                            ),
                        )
                    except (TransientError, PermanentError, CircuitOpenError) as exc:
                        last_exc = exc
                        failovers += 1
                        metrics.incr("llm.provider_failover", provider=provider.name, model=model)
                        if span:
                            span.event("failover", provider=provider.name, model=model,
                                       error=str(exc)[:200])
                        continue

                    latency = (time.perf_counter() - started) * 1000
                    metrics.observe("llm.latency_ms", latency, provider=provider.name)
                    if span:
                        span.set(
                            provider=provider.name, model=model, tokens_in=tin, tokens_out=tout,
                            latency_ms=round(latency, 1), failovers=failovers,
                            prompt_chars=sum(len(m["content"]) for m in messages),
                            response_preview=(text or "")[:400],
                        )
                    return LLMResult(text, provider.name, model, tin, tout, latency, failovers)

            if span:
                span.status = "error"
                span.error = str(last_exc)[:300]
            raise TransientError(
                f"All {attempted} model(s) across {len(candidates)} provider(s) failed. "
                f"Last error: {last_exc}"
            ) from last_exc
        finally:
            if span_cm:
                span_cm.__exit__(None, None, None)

    def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        messages: List[Dict[str, str]],
        default: Any,
        tier: str = "fast",
        budget: Optional[TurnBudget] = None,
    ) -> tuple[Any, LLMResult]:
        """Ask for JSON, and repair once if the model returns prose.

        Bounded to exactly one repair attempt - an unbounded reformat loop is a
        classic way to turn a bad response into a large bill.
        """
        result = self.complete(
            purpose=purpose, system=system, messages=messages, tier=tier, budget=budget
        )
        parsed = result.json(None)
        if parsed is not None:
            return parsed, result

        repair_msgs = messages + [
            {"role": "assistant", "content": result.text[:1500]},
            {
                "role": "user",
                "content": "That was not valid JSON. Reply with the JSON object only - "
                "no prose, no markdown fences, no explanation.",
            },
        ]
        try:
            retry = self.complete(
                purpose=f"{purpose}.json_repair", system=system, messages=repair_msgs,
                tier=tier, budget=budget,
            )
            return retry.json(default), retry
        except (BudgetExceeded, TransientError, PermanentError):
            return default, result
