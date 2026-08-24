"""Self-correction, degradation and failure containment."""
from __future__ import annotations

import pytest

from agent.obs import metrics
from agent.resilience.policies import (
    CircuitBreaker,
    CircuitOpenError,
    PermanentError,
    RetryPolicy,
    call_with_resilience,
    classify,
)
from fakes import (
    SQL_BROKEN_SYNTAX,
    SQL_CATEGORY,
    SQL_EMPTY,
    SQL_PII_LEAK,
)


class TestErrorClassification:
    @pytest.mark.parametrize("message", [
        "503 Service Unavailable", "deadline exceeded", "429 rate limit exceeded",
        "connection reset by peer", "model is overloaded",
    ])
    def test_transient(self, message):
        assert classify(RuntimeError(message)) == "transient"

    @pytest.mark.parametrize("message", [
        "API key not valid", "403 permission denied", "invalid argument: bad field",
    ])
    def test_permanent(self, message):
        assert classify(RuntimeError(message)) == "permanent"


class TestRetryAndBreaker:
    def test_transient_errors_are_retried_then_succeed(self):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("503 unavailable")
            return "ok"

        got = call_with_resilience(flaky, dependency="test:flaky",
                                   policy=RetryPolicy(attempts=4, base_delay_s=0.001))
        assert got == "ok" and attempts["n"] == 3

    def test_permanent_errors_are_not_retried(self):
        attempts = {"n": 0}

        def bad():
            attempts["n"] += 1
            raise RuntimeError("API key not valid")

        with pytest.raises(PermanentError):
            call_with_resilience(bad, dependency="test:permanent",
                                 policy=RetryPolicy(attempts=5, base_delay_s=0.001))
        assert attempts["n"] == 1, "a permanent error must cost exactly one call"

    def test_breaker_opens_and_fails_fast(self):
        cb = CircuitBreaker("dep", failure_threshold=2, reset_timeout_s=60)
        assert cb.state == "closed"
        cb.record_failure(); cb.record_failure()
        assert cb.state == "open"
        with pytest.raises(CircuitOpenError):
            cb.guard()

    def test_breaker_half_opens_after_timeout(self):
        cb = CircuitBreaker("dep2", failure_threshold=1, reset_timeout_s=0.01)
        cb.record_failure()
        import time
        time.sleep(0.02)
        assert cb.state == "half_open"
        cb.guard()  # does not raise
        cb.record_success()
        assert cb.state == "closed"


class TestSQLSelfCorrection:
    def test_syntax_error_is_repaired_and_the_turn_still_answers(self, session, fake_llm):
        fake_llm.queue_sql(SQL_BROKEN_SYNTAX, SQL_CATEGORY)
        result = session.ask("How did each product category perform?")
        step = result.steps[0]
        assert step["status"] == "ok", step.get("error")
        assert step["repairs"] == 1
        assert step["rows"] > 0
        assert result.ok

    def test_pii_violation_is_caught_before_execution_and_repaired(self, session, fake_llm):
        fake_llm.queue_sql(SQL_PII_LEAK, SQL_CATEGORY)
        result = session.ask("Show me how categories performed")
        step = result.steps[0]
        assert step["repairs"] == 1
        assert step["status"] == "ok"
        # The rejected statement must never have reached the warehouse.
        assert step["rows"] > 0 and "first_name" not in (step.get("validated_sql") or "")

    def test_empty_result_is_widened_once_then_reported_honestly(self, session, fake_llm):
        fake_llm.queue_sql(SQL_EMPTY, SQL_EMPTY, SQL_EMPTY)
        result = session.ask("How many orders had that status?")
        step = result.steps[0]
        assert step["status"] == "empty"
        # Exactly one widening attempt - the loop must not chase an empty set.
        assert step["repairs"] == 1
        assert result.ok

    def test_repair_budget_is_bounded(self, session, fake_llm):
        fake_llm.queue_sql(*([SQL_BROKEN_SYNTAX] * 8))
        result = session.ask("How did each product category perform?")
        step = result.steps[0]
        assert step["status"] in ("failed", "rejected")
        assert step["repairs"] <= 2, "repair loop must respect max_sql_repair_attempts"
        assert result.ok, "a permanently broken query must still return a usable turn"
        assert result.answer

    def test_answer_states_the_failure_rather_than_inventing_numbers(self, session, fake_llm):
        fake_llm.queue_sql(*([SQL_BROKEN_SYNTAX] * 8))
        result = session.ask("How did each product category perform?")
        analyst_prompts = [c["prompt"] for c in fake_llm.calls
                           if "QUERY RESULTS" in c["prompt"]]
        assert analyst_prompts, "the analyst stage must still run"
        assert "FAILED" in analyst_prompts[-1]
        assert "You do NOT have these figures" in analyst_prompts[-1]


class TestProviderDegradation:
    def test_planner_outage_falls_back_to_heuristic_routing(self, session, fake_llm):
        fake_llm.fail("plan")
        result = session.ask("How did revenue trend over the last 12 months?")
        assert result.ok
        assert result.state["degraded"] is True
        assert result.route == "analysis"
        assert result.steps and result.steps[0]["status"] == "ok"

    def test_analyst_outage_returns_the_figures_instead_of_crashing(self, session, fake_llm):
        fake_llm.fail("analysis")
        result = session.ask("How did revenue trend over the last 12 months?")
        assert result.ok
        assert result.state["degraded"] is True
        assert "revenue" in result.answer.lower() or "|" in result.answer

    def test_total_provider_outage_degrades_without_crashing(self, session, fake_llm):
        """A transient provider failure trips the breaker, so every later stage
        fails fast too. The turn must still return an explanatory answer."""
        fake_llm.fail_transient("plan")
        result = session.ask("How did revenue trend over the last 12 months?")
        assert result.ok, "a full provider outage must not raise out of the session"
        assert result.answer and "Traceback" not in result.answer
        from agent.resilience.policies import breaker_states
        # Breakers are keyed provider:model so a single overloaded model cannot
        # take its healthy siblings down with it.
        assert breaker_states().get("llm:scripted:scripted") in ("open", "half_open")

    def test_sql_model_outage_does_not_crash_the_turn(self, session, fake_llm):
        fake_llm.fail("sql_generation")
        result = session.ask("How did revenue trend?")
        assert result.ok and result.answer

    def test_no_provider_configured_produces_a_message_not_a_traceback(self, services):
        from agent.llm import LLMRouter
        services.router = LLMRouter(chain=[])
        services.guardrail.router = services.router
        from agent.session import ChatSession
        s = ChatSession(services=services, user_id="manager_a")
        result = s.ask("How did revenue trend?")
        assert result.answer and "Traceback" not in result.answer


class TestCostContainment:
    def test_llm_call_budget_is_enforced(self, session):
        session.budget.max_llm_calls = 2
        result = session.ask("How did revenue trend over the last 12 months?")
        assert result.ok
        assert session.budget.llm_calls <= 3

    def test_turn_records_metrics(self, session):
        metrics.reset()
        session.ask("How did revenue trend over the last 12 months?")
        snap = metrics.snapshot()["counters"]
        assert snap.get("turns.total", 0) >= 1
        assert snap.get("sql.executed", 0) >= 1


class TestRouterChainSemantics:
    def test_an_explicit_empty_chain_means_no_providers(self):
        """A caller passing [] must not silently inherit the configured chain."""
        from agent.llm import LLMRouter
        assert LLMRouter(chain=[]).providers == []

    def test_omitting_the_chain_uses_configuration(self):
        from agent.llm import LLMRouter
        assert len(LLMRouter().providers) >= 1

    def test_models_fail_over_within_a_provider_before_switching_provider(self, fake_llm):
        """A 503 on one model must try its siblings, not abandon the provider."""
        from agent.llm import LLMRouter, ScriptedProvider

        attempts = []

        class FlakyModels(ScriptedProvider):
            def __init__(self):
                super().__init__()
                self.models = {"fast": ["model-a", "model-b"], "reasoning": ["model-a", "model-b"]}

            def invoke(self, model, system, messages):
                attempts.append(model)
                if model == "model-a":
                    raise RuntimeError("503 This model is currently experiencing high demand")
                return "recovered", 5, 5

        router = LLMRouter(chain=[])
        router.register(FlakyModels())
        result = router.complete(purpose="t", system="s", messages=[{"role": "user", "content": "q"}])
        assert result.text == "recovered"
        assert result.model == "model-b"
        assert attempts[0] == "model-a" and "model-b" in attempts
        assert result.failovers >= 1

    def test_a_failing_model_does_not_open_the_breaker_for_its_siblings(self, fake_llm):
        from agent.llm import LLMRouter, ScriptedProvider
        from agent.resilience.policies import breaker_states

        class FlakyModels(ScriptedProvider):
            def __init__(self):
                super().__init__()
                self.models = {"fast": ["bad-model", "good-model"], "reasoning": ["bad-model", "good-model"]}

            def invoke(self, model, system, messages):
                if model == "bad-model":
                    raise RuntimeError("503 overloaded")
                return "ok", 1, 1

        router = LLMRouter(chain=[])
        router.register(FlakyModels())
        for _ in range(3):
            router.complete(purpose="t", system="s", messages=[{"role": "user", "content": "q"}])
        states = breaker_states()
        assert states.get("llm:scripted:bad-model") in ("open", "half_open")
        assert states.get("llm:scripted:good-model") == "closed"
