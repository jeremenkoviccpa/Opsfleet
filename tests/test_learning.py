"""User-level and system-level learning loops."""
from __future__ import annotations


from agent.memory import user_profile


class TestPreferenceMemory:
    def test_an_explicit_preference_applies_immediately(self, isolated_db):
        pref = user_profile.record("m1", "output_format", "table", source="explicit",
                                   evidence="always give me tables")
        assert pref.confidence == 1.0 and pref.active

    def test_an_inferred_preference_needs_corroboration(self, isolated_db):
        first = user_profile.record("m1", "output_format", "bullets", source="inferred")
        assert first.active is False, "one weak signal must not change behaviour"
        second = user_profile.record("m1", "output_format", "bullets", source="inferred")
        assert second.evidence_count == 2 and second.active is True

    def test_confidence_ramps_and_saturates(self, isolated_db):
        confidences = [
            user_profile.record("m1", "analysis_depth", "deep", source="inferred").confidence
            for _ in range(8)
        ]
        assert confidences == sorted(confidences), "confidence must be monotonic"
        assert confidences[-1] <= 0.95

    def test_an_explicit_statement_outranks_a_contradicting_inference(self, isolated_db):
        user_profile.record("m1", "output_format", "table", source="explicit")
        after = user_profile.record("m1", "output_format", "prose", source="inferred")
        assert after.value == "table" and after.source == "explicit"

    def test_a_contradicting_inference_decays_rather_than_flipping(self, isolated_db):
        for _ in range(4):
            user_profile.record("m1", "output_format", "bullets", source="inferred")
        flipped = user_profile.record("m1", "output_format", "table", source="inferred")
        assert flipped.value == "table"
        assert flipped.active is False, "a single contrary signal must not take effect at once"

    def test_unknown_keys_and_values_are_rejected(self, isolated_db):
        assert user_profile.record("m1", "favourite_colour", "blue") is None
        assert user_profile.record("m1", "output_format", "interpretive_dance") is None

    def test_only_active_preferences_reach_the_prompt(self, isolated_db):
        user_profile.record("m1", "output_format", "bullets", source="inferred")
        user_profile.record("m1", "analysis_depth", "deep", source="explicit")
        block = user_profile.render_for_prompt("m1")
        assert "analysis_depth: deep" in block
        assert "output_format" not in block

    def test_preferences_are_per_user(self, isolated_db):
        user_profile.record("manager_a", "output_format", "table", source="explicit")
        user_profile.record("manager_b", "output_format", "bullets", source="explicit")
        assert "table" in user_profile.render_for_prompt("manager_a")
        assert "bullets" in user_profile.render_for_prompt("manager_b")

    def test_forget_clears_memory(self, isolated_db):
        user_profile.record("m1", "output_format", "table", source="explicit")
        assert user_profile.forget("m1") == 1
        assert user_profile.get_preferences("m1") == []


class TestLearningThroughConversation:
    def test_a_stated_preference_is_captured_from_a_real_turn(self, session):
        session.ask("Always give me tables, not prose. How did revenue trend?")
        prefs = {p.key: p for p in user_profile.get_preferences("manager_a")}
        assert "output_format" in prefs
        assert prefs["output_format"].value == "table"
        assert prefs["output_format"].source == "explicit"

    def test_learned_preferences_are_injected_into_later_turns(self, session, fake_llm):
        user_profile.record("manager_a", "output_format", "bullets", source="explicit")
        session.ask("How did revenue trend?")
        analyst_prompts = [c["system"] for c in fake_llm.calls if "senior retail data analyst" in c["system"]]
        assert analyst_prompts
        assert "output_format: bullets" in analyst_prompts[-1]

    def test_a_plain_question_teaches_nothing(self, session):
        session.ask("How did revenue trend over the last 12 months?")
        assert user_profile.get_preferences("manager_a") == []


class TestGoldenBucketLearning:
    def test_a_successful_analysis_proposes_a_candidate_trio(self, session):
        before = len(session.services.golden.list_candidates())
        session.ask("How did revenue trend over the last 12 months?")
        candidates = session.services.golden.list_candidates()
        assert len(candidates) == before + 1
        cand = candidates[-1]
        assert cand.status == "candidate"
        assert cand.sql and cand.intent_tags and cand.analyst_method_notes

    def test_candidates_are_not_retrievable_until_promoted(self, session):
        session.ask("How did revenue trend over the last 12 months?")
        cand = session.services.golden.list_candidates()[-1]
        promoted_ids = {t.trio_id for t in session.services.golden.trios}
        assert cand.trio_id not in promoted_ids, "the agent must not learn from unreviewed output"

    def test_promotion_makes_a_candidate_retrievable(self, session, tmp_path):
        session.ask("How did revenue trend over the last 12 months?")
        cand = session.services.golden.list_candidates()[-1]
        # Promote into an isolated root so the repo's seeded bucket is untouched.
        gb = session.services.golden
        gb.root = tmp_path / "bucket"
        gb.root.mkdir(parents=True, exist_ok=True)
        (gb.root / "candidates").mkdir(exist_ok=True)
        import shutil
        shutil.copy(cand.source_path, gb.root / "candidates" / f"{cand.trio_id}.json")
        gb.candidates_dir = gb.root / "candidates"
        gb.cache_path = gb.root / ".embedding_cache.json"
        assert gb.promote(cand.trio_id, reviewer="tester") is not None
        assert cand.trio_id in {t.trio_id for t in gb.trios}

    def test_a_failed_turn_proposes_nothing(self, session, fake_llm):
        from fakes import SQL_BROKEN_SYNTAX
        fake_llm.queue_sql(*([SQL_BROKEN_SYNTAX] * 8))
        before = len(session.services.golden.list_candidates())
        session.ask("How did each category perform?")
        assert len(session.services.golden.list_candidates()) == before
