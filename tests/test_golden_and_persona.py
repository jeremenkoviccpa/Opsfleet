"""Hybrid retrieval over the Golden Bucket, and runtime persona switching."""
from __future__ import annotations

import time

import pytest

from agent.config import list_personas, load_persona
from agent.golden.store import BM25, GoldenBucket, tokenize


class TestRetrieval:
    @pytest.mark.parametrize("question,expected", [
        ("why are people in texas spending less than in california?", "trio_001"),
        ("our churn went up last month, what happened", "trio_005"),
        ("monthly sales trend for the year", "trio_002"),
        ("compare margin between two clothing categories", "trio_004"),
        ("build me a quarterly summary with next steps", "trio_006"),
        ("what tables can I query", "trio_007"),
        ("who spends the most with us", "trio_003"),
    ])
    def test_paraphrased_questions_retrieve_the_right_precedent(self, question, expected):
        hits = GoldenBucket().search(question, k=3)
        assert hits, "retrieval returned nothing"
        assert hits[0].trio.trio_id == expected

    def test_hybrid_beats_either_leg_alone_on_vocabulary_match(self):
        gb = GoldenBucket()
        hits = gb.search("aov and discount depth by category", k=3)
        assert hits[0].bm25 > 0, "lexical leg must contribute"
        assert hits[0].dense != 0.0, "dense leg must contribute"

    def test_scores_are_normalised_and_bounded(self):
        hits = GoldenBucket().search("revenue trend", k=5)
        assert all(0.0 <= h.score <= 1.05 for h in hits)

    def test_precedent_block_carries_sql_and_method_notes(self):
        hits = GoldenBucket().search("why are texans underspending", k=1)
        block = hits[0].as_prompt_block()
        assert "Analyst's SQL" in block
        assert "Method rules the analyst applied" in block
        assert "frequency" in block.lower()

    def test_bm25_ranks_by_term_specificity(self):
        corpus = [tokenize("monthly revenue trend"), tokenize("customer churn cohort analysis"),
                  tokenize("product margin comparison")]
        scores = BM25(corpus).scores(tokenize("churn cohort"))
        assert scores.argmax() == 1

    def test_retrieval_reaches_the_agent_prompt(self, session, fake_llm):
        session.ask("Why are customers in Texas underspending compared to California?")
        analyst_prompts = [c["prompt"] for c in fake_llm.calls if "ANALYST PRECEDENTS" in c["prompt"]]
        assert analyst_prompts
        assert "trio_001" in analyst_prompts[-1]
        assert "frequency" in analyst_prompts[-1].lower()


class TestPersonaAgility:
    def test_both_shipped_personas_load(self):
        assert {"exec_default", "ceo_q3_terse"} <= set(list_personas())

    def test_personas_differ_in_voice_and_length(self):
        default = load_persona("exec_default")
        terse = load_persona("ceo_q3_terse")
        assert terse["max_answer_words"] < default["max_answer_words"]
        assert terse["tone"] != default["tone"]

    def test_the_active_persona_reaches_the_prompt(self, session, fake_llm):
        session.set_persona("ceo_q3_terse")
        session.ask("How did revenue trend?")
        analyst = [c["system"] for c in fake_llm.calls if "senior retail data analyst" in c["system"]][-1]
        assert "Blunt, numbers-first" in analyst
        assert "Maximum 120 words" in analyst

    def test_switching_persona_changes_the_next_turn(self, session, fake_llm):
        session.ask("How did revenue trend?")
        first = [c["system"] for c in fake_llm.calls if "senior retail data analyst" in c["system"]][-1]
        session.set_persona("ceo_q3_terse")
        session.ask("How did revenue trend?")
        second = [c["system"] for c in fake_llm.calls if "senior retail data analyst" in c["system"]][-1]
        assert first != second

    def test_an_edited_persona_file_is_picked_up_without_a_restart(self, tmp_path, monkeypatch):
        """The CEO's office edits the YAML; the next turn uses it."""
        import agent.config as config

        personas = tmp_path / "personas"
        personas.mkdir()
        target = personas / "weekly.yaml"
        target.write_text("id: weekly\ntone: Formal and measured.\nmax_answer_words: 300\n")
        monkeypatch.setattr(config, "persona_dir", lambda: personas)
        config._persona_cache.clear()

        assert load_persona("weekly")["tone"].startswith("Formal")

        time.sleep(0.01)
        target.write_text("id: weekly\ntone: Punchy and urgent.\nmax_answer_words: 100\n")
        reloaded = load_persona("weekly")
        assert reloaded["tone"].startswith("Punchy")
        assert reloaded["max_answer_words"] == 100

    def test_an_unknown_persona_is_reported_clearly(self):
        with pytest.raises(FileNotFoundError) as exc:
            load_persona("does_not_exist")
        assert "Available:" in str(exc.value)
