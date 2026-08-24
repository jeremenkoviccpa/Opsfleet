"""Configuration loading with hot-reload.

Every YAML file under config/ is watched by mtime and re-read when it changes.
This is what makes persona / PII-policy / budget changes take effect without a
restart or a redeploy.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
RUNTIME_DIR = REPO_ROOT / ".runtime"

load_dotenv(REPO_ROOT / ".env")


class _HotFile:
    """A YAML file that reloads itself when its mtime moves."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._data: Dict[str, Any] = {}

    def get(self) -> Dict[str, Any]:
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
            except FileNotFoundError:
                return self._data
            if mtime != self._mtime:
                with self.path.open("r", encoding="utf-8") as fh:
                    self._data = yaml.safe_load(fh) or {}
                self._mtime = mtime
            return self._data

    @property
    def reloaded_at(self) -> float | None:
        return self._mtime


_settings_file = _HotFile(CONFIG_DIR / "settings.yaml")
_pii_file = _HotFile(CONFIG_DIR / "pii_policy.yaml")
_persona_cache: Dict[str, _HotFile] = {}


def settings() -> Dict[str, Any]:
    return _settings_file.get()


def setting(dotted: str, default: Any = None) -> Any:
    node: Any = settings()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def pii_policy() -> Dict[str, Any]:
    return _pii_file.get()


def persona_dir() -> Path:
    return CONFIG_DIR / "personas"


def list_personas() -> list[str]:
    return sorted(p.stem for p in persona_dir().glob("*.yaml"))


def load_persona(persona_id: str) -> Dict[str, Any]:
    path = persona_dir() / f"{persona_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Persona '{persona_id}' not found. Available: {', '.join(list_personas())}"
        )
    if persona_id not in _persona_cache:
        _persona_cache[persona_id] = _HotFile(path)
    return _persona_cache[persona_id].get()


def persona_mtime(persona_id: str) -> float | None:
    path = persona_dir() / f"{persona_id}.yaml"
    return path.stat().st_mtime if path.exists() else None


def runtime_dir() -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class ProviderCredential:
    provider: str
    available: bool
    detail: str = ""


def credential_status() -> list[ProviderCredential]:
    """What the LLM router can actually reach right now."""
    out = []
    gem = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    out.append(ProviderCredential("gemini", bool(gem), "GOOGLE_API_KEY set" if gem else "no GOOGLE_API_KEY"))
    orouter = os.getenv("OPENROUTER_API_KEY")
    out.append(ProviderCredential("openrouter", bool(orouter),
                                  "OPENROUTER_API_KEY set" if orouter else "no OPENROUTER_API_KEY"))
    enabled = os.getenv("OLLAMA_ENABLED", "0") == "1"
    out.append(ProviderCredential("ollama", enabled,
                                  os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                                  if enabled else "OLLAMA_ENABLED != 1"))
    return out
