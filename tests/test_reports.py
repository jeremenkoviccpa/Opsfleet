"""Saved reports and the destructive-operation confirmation flow."""
from __future__ import annotations

import pytest



def _seed(store, user="manager_a", session_id="sess_1"):
    a = store.create(user_id=user, session_id=session_id, title="Northwind Q1 performance",
                     body_md="Northwind revenue rose 12%.", entities=["Northwind"])
    b = store.create(user_id=user, session_id=session_id, title="Jeans margin review",
                     body_md="Jeans discounting is deep.", entities=["Jeans"])
    c = store.create(user_id=user, session_id="sess_other", title="Weekly ops summary",
                     body_md="Routine weekly operations note.", entities=[])
    return a, b, c


class TestResolution:
    def test_resolve_is_read_only(self, services):
        a, b, c = _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a",
                                                 criteria={"mentions": ["northwind"]})
        assert plan.count == 1 and plan.ids() == [a.report_id]
        # Nothing is mutated until confirmation.
        assert all(not r.deleted for r in services.reports.list("manager_a", include_deleted=True))

    def test_mentions_match_title_body_and_entities(self, services):
        a, b, c = _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a",
                                                 criteria={"mentions": ["jeans"]})
        assert plan.ids() == [b.report_id]

    def test_session_scope_selects_only_this_conversation(self, services):
        a, b, c = _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a",
                                                 criteria={"session_id": "sess_1"})
        assert set(plan.ids()) == {a.report_id, b.report_id}
        assert c.report_id not in plan.ids()

    def test_empty_criteria_selects_nothing(self, services):
        _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={})
        assert plan.count == 0, "an under-specified delete must never mean 'everything'"

    def test_a_user_cannot_resolve_another_users_reports(self, services):
        _seed(services.reports, user="manager_a")
        plan = services.reports.resolve_deletion(user_id="manager_b",
                                                 criteria={"mentions": ["northwind"]})
        assert plan.count == 0


class TestConfirmation:
    def test_nothing_is_deleted_without_confirmation(self, services):
        a, _, _ = _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={"mentions": ["northwind"]})
        services.reports.cancel_plan(plan.token, "manager_a")
        assert services.reports.get(a.report_id).deleted is False

    def test_confirmation_deletes_exactly_the_resolved_set(self, services):
        a, b, c = _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={"mentions": ["northwind"]})
        count, ids = services.reports.confirm_deletion(plan.token, "manager_a")
        assert count == 1 and ids == [a.report_id]
        assert services.reports.get(a.report_id).deleted is True
        assert services.reports.get(b.report_id).deleted is False

    def test_a_token_cannot_be_replayed(self, services):
        _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={"mentions": ["northwind"]})
        services.reports.confirm_deletion(plan.token, "manager_a")
        with pytest.raises(KeyError):
            services.reports.confirm_deletion(plan.token, "manager_a")

    def test_a_token_cannot_be_used_by_another_user(self, services):
        _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={"mentions": ["northwind"]})
        with pytest.raises(PermissionError):
            services.reports.confirm_deletion(plan.token, "manager_b")

    def test_bulk_deletes_require_a_typed_phrase(self, services):
        for i in range(6):
            services.reports.create(user_id="manager_a", session_id="s", title=f"Report {i}",
                                    body_md="body")
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={"all": True})
        assert plan.count == 6
        assert plan.requires_phrase is True
        assert plan.confirm_phrase == "delete 6"

    def test_deletes_are_soft_and_restorable(self, services):
        a, _, _ = _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={"mentions": ["northwind"]})
        services.reports.confirm_deletion(plan.token, "manager_a")
        assert services.reports.restore([a.report_id], "manager_a") == 1
        assert services.reports.get(a.report_id).deleted is False

    def test_every_phase_is_audited(self, services):
        _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={"mentions": ["northwind"]})
        services.reports.confirm_deletion(plan.token, "manager_a")
        actions = [e["action"] for e in services.reports.audit_trail("manager_a")]
        assert "report.delete_requested" in actions
        assert "report.delete_confirmed" in actions
        assert "report.create" in actions


class TestConversationalDeletion:
    """End-to-end through the graph, including the interrupt gate."""

    def test_declining_at_the_gate_deletes_nothing(self, session):
        a, _, _ = _seed(session.services.reports, session_id=session.session_id)
        result = session.ask("Delete all reports mentioning Northwind",
                             on_confirm=lambda payload: {"approved": False, "phrase": ""})
        assert session.services.reports.get(a.report_id).deleted is False
        assert "Nothing was deleted" in result.answer

    def test_approving_at_the_gate_deletes(self, session):
        a, b, _ = _seed(session.services.reports, session_id=session.session_id)
        seen = {}

        def approve(payload):
            seen.update(payload)
            return {"approved": True, "phrase": payload.get("confirm_phrase", "")}

        result = session.ask("Delete all reports mentioning Northwind", on_confirm=approve)
        assert seen["matched"] == 1
        assert seen["rows"][0]["title"] == "Northwind Q1 performance"
        assert "matched because" not in result.answer  # preview is CLI-side, not in the answer
        assert session.services.reports.get(a.report_id).deleted is True
        assert session.services.reports.get(b.report_id).deleted is False

    def test_the_gate_is_reached_before_any_mutation(self, session):
        a, _, _ = _seed(session.services.reports, session_id=session.session_id)

        def inspect(payload):
            # At this point the graph is suspended - the report must be intact.
            assert session.services.reports.get(a.report_id).deleted is False
            return {"approved": False, "phrase": ""}

        session.ask("Delete all reports mentioning Northwind", on_confirm=inspect)

    def test_no_match_answers_without_asking_for_confirmation(self, session):
        _seed(session.services.reports, session_id=session.session_id)
        called = {"n": 0}

        def should_not_run(_payload):
            called["n"] += 1
            return {"approved": True, "phrase": ""}

        result = session.ask("Delete all reports mentioning Contoso", on_confirm=should_not_run)
        assert called["n"] == 0
        assert "couldn't find" in result.answer.lower()

    def test_default_handler_denies(self, session):
        a, _, _ = _seed(session.services.reports, session_id=session.session_id)
        result = session.ask("Delete all reports mentioning Northwind")  # no on_confirm
        assert session.services.reports.get(a.report_id).deleted is False
        assert "Nothing was deleted" in result.answer

    def test_wrong_bulk_phrase_aborts(self, session):
        for i in range(5):
            session.services.reports.create(user_id="manager_a", session_id=session.session_id,
                                            title=f"Report {i}", body_md="body")
        result = session.ask("Delete all my reports",
                             on_confirm=lambda p: {"approved": True, "phrase": "yes"})
        assert "didn't match" in result.answer
        assert all(not r.deleted for r in session.services.reports.list("manager_a", include_deleted=True))

    def test_a_report_is_saved_when_one_is_requested(self, session):
        result = session.ask("Create a Q1 report with insights and action items for Q2")
        saved = session.services.reports.list("manager_a")
        assert saved, "the report should be persisted to the library"
        assert saved[0].report_id in result.answer
        assert "Recommended actions" in saved[0].body_md


class TestCriteriaPrecedence:
    """A broad criterion must never widen a narrow one."""

    def test_narrower_criteria_win_over_all(self, services):
        a, b, c = _seed(services.reports)
        plan = services.reports.resolve_deletion(
            user_id="manager_a", criteria={"mentions": ["northwind"], "all": True})
        assert plan.ids() == [a.report_id]

    def test_session_scope_wins_over_all(self, services):
        a, b, c = _seed(services.reports)
        plan = services.reports.resolve_deletion(
            user_id="manager_a", criteria={"session_id": "sess_1", "all": True})
        assert set(plan.ids()) == {a.report_id, b.report_id}

    def test_all_alone_still_selects_everything(self, services):
        a, b, c = _seed(services.reports)
        plan = services.reports.resolve_deletion(user_id="manager_a", criteria={"all": True})
        assert set(plan.ids()) == {a.report_id, b.report_id, c.report_id}


class TestReportDetection:
    """A requested report must be formatted as one AND saved.

    Regression: the planner routed a report request to "analysis" (correct - it
    needed data) but left report_title empty, so the turn was written as a chat
    answer and never reached the report library.
    """

    @pytest.mark.parametrize("question", [
        "Create a Q1 report with insights and action items for Q2",
        "Write me a report on category performance",
        "Put together a briefing on last quarter",
        "Prepare a write-up of the Texas numbers",
        "produce a report including recommendations",
    ])
    def test_report_requests_are_detected(self, question):
        from agent.nodes.compose import _asked_for_a_report
        assert _asked_for_a_report(question) is True

    @pytest.mark.parametrize("question", [
        "How did revenue trend over the last 12 months?",
        "Why are customers in Texas underspending?",
        "What data do we have available?",
        "Delete all reports mentioning Northwind",
    ])
    def test_ordinary_questions_are_not_reports(self, question):
        from agent.nodes.compose import _asked_for_a_report
        assert _asked_for_a_report(question) is False

    def test_a_report_is_saved_even_when_the_planner_omits_a_title(self, session, fake_llm):
        # Planner routes to analysis and supplies NO report_title - the exact
        # shape that silently dropped the report before.
        fake_llm.set_plan({
            "route": "analysis", "reason": "needs data", "time_window": "Q1 2026",
            "notes": "", "steps": [{"step_id": "s1", "goal": "revenue by category"}],
            "deletion_criteria": {}, "report_title": "",
        })
        result = session.ask("Create a Q1 report with insights and action items for Q2")
        assert result.state["answer_is_report"] is True
        saved = session.services.reports.list("manager_a")
        assert saved, "a requested report must reach the library"
        assert saved[0].report_id in result.answer

    def test_the_report_structure_is_used_not_the_chat_structure(self, session, fake_llm):
        fake_llm.set_plan({
            "route": "analysis", "reason": "needs data", "time_window": "",
            "notes": "", "steps": [{"step_id": "s1", "goal": "revenue"}],
            "deletion_criteria": {}, "report_title": "",
        })
        session.ask("Create a Q1 report with insights and action items for Q2")
        analyst = [c["system"] for c in fake_llm.calls if "senior retail data analyst" in c["system"]][-1]
        assert "SAVED REPORT" in analyst, "the report directive must be in the prompt"
        assert "Executive summary" in analyst, "the persona's report_format must be used"

    def test_an_ordinary_question_is_not_saved_as_a_report(self, session):
        result = session.ask("How did revenue trend over the last 12 months?")
        assert result.state.get("answer_is_report") is not True
        assert session.services.reports.list("manager_a") == []
