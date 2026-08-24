from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

# Keep tests hermetic: no network, no credentials, deterministic retrieval.
os.environ["RIA_EMBEDDINGS"] = "hashing"
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("OPENROUTER_API_KEY", None)
os.environ["OLLAMA_ENABLED"] = "0"

# agent.config runs load_dotenv() at import time, which repopulates any key the
# lines above cleared. Import first, then clear again, so the suite can never reach
# a live provider or spend real quota.
from agent import config as _agent_config  # noqa: E402,F401

for _var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
    os.environ.pop(_var, None)
os.environ["OLLAMA_ENABLED"] = "0"

from agent.memory import store as memory_store  # noqa: E402
from agent.services import build_services  # noqa: E402
from agent.session import ChatSession  # noqa: E402
from fakes import FakeLLM  # noqa: E402


from agent.obs import metrics  # noqa: E402
from agent.resilience import policies  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_process_state():
    """Breakers and metric counters are process-global by design; reset per test."""
    policies.reset_breakers()
    metrics.reset()
    yield
    policies.reset_breakers()


@pytest.fixture()
def isolated_db(tmp_path):
    memory_store.configure(tmp_path / "state.db")
    return tmp_path


@pytest.fixture()
def fake_llm():
    return FakeLLM()


@pytest.fixture()
def services(isolated_db, fake_llm):
    svc = build_services("offline")
    memory_store.configure(isolated_db / "state.db")
    svc.router = fake_llm.router(tracer=svc.tracer)
    svc.guardrail.router = svc.router
    svc.golden.candidates_dir = isolated_db / "candidates"
    svc.golden.candidates_dir.mkdir(parents=True, exist_ok=True)
    return svc


@pytest.fixture()
def session(services):
    return ChatSession(services=services, user_id="manager_a", user_display_name="Dana")
