"""The Golden Knowledge Bucket.

A Trio is (question -> SQL -> analyst report) plus the method notes that explain
*why* the analyst wrote that SQL. Retrieval is hybrid:

  * BM25 over the question text and intent tags catches exact vocabulary
    ("churn", "AOV", a category name) that embeddings routinely blur;
  * a dense vector leg catches paraphrase ("why are people buying less in TX"
    -> the Texas/California spend-gap trio).

Scores are min-max normalised per leg before the weighted sum, because BM25 and
cosine are not on comparable scales and a raw sum silently lets BM25 dominate.

Embeddings come from a pluggable backend. In production that is Vertex AI
`text-embedding-005` with the vectors in Vertex AI Vector Search. Offline, a
deterministic character-n-gram TF-IDF backend stands in, so retrieval behaviour
is testable without a cloud account. It is a weaker signal than a real
embedding model - it is a test stand-in, not a claim of equivalence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import DATA_DIR
from ..obs import metrics

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "is", "are",
    "was", "were", "we", "our", "us", "my", "me", "i", "what", "how", "why",
    "do", "does", "did", "with", "by", "at", "it", "this", "that", "be", "as",
}


def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Trio:
    trio_id: str
    question: str
    sql: str
    analyst_report: str
    intent_tags: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    analyst_method_notes: List[str] = field(default_factory=list)
    author: str = ""
    quality_score: float = 0.0
    status: str = "candidate"
    created_at: str = ""
    used_count: int = 0
    source_path: Optional[Path] = None

    @property
    def index_text(self) -> str:
        return " ".join(
            [self.question, self.question, " ".join(self.intent_tags),
             " ".join(self.tables), self.analyst_report,
             " ".join(self.analyst_method_notes)]
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "source_path"}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Optional[Path] = None) -> "Trio":
        return cls(
            trio_id=data.get("trio_id") or (path.stem if path else "unknown"),
            question=data.get("question", ""),
            sql=data.get("sql", ""),
            analyst_report=data.get("analyst_report", ""),
            intent_tags=list(data.get("intent_tags", [])),
            tables=list(data.get("tables", [])),
            analyst_method_notes=list(data.get("analyst_method_notes", [])),
            author=data.get("author", ""),
            quality_score=float(data.get("quality_score", 0.0)),
            status=data.get("status", "candidate"),
            created_at=data.get("created_at", ""),
            used_count=int(data.get("used_count", 0)),
            source_path=path,
        )


@dataclass
class RetrievedTrio:
    trio: Trio
    score: float
    bm25: float
    dense: float

    def as_prompt_block(self, include_sql: bool = True) -> str:
        parts = [f"### Precedent {self.trio.trio_id} (relevance {self.score:.2f})",
                 f"Analyst was asked: {self.trio.question}"]
        if include_sql and self.trio.sql:
            parts.append(f"Analyst's SQL:\n```sql\n{self.trio.sql}\n```")
        if self.trio.analyst_report:
            parts.append(f"Analyst's interpretation: {self.trio.analyst_report}")
        if self.trio.analyst_method_notes:
            notes = "\n".join(f"- {n}" for n in self.trio.analyst_method_notes)
            parts.append(f"Method rules the analyst applied:\n{notes}")
        return "\n".join(parts)


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------
class EmbeddingBackend:
    name = "base"
    dim = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashingTfidfEmbeddings(EmbeddingBackend):
    """Deterministic offline stand-in: hashed word + character-4-gram TF-IDF.

    Not a semantic model. It exists so the hybrid retrieval path, its scoring
    and its tests run identically with no network and no credentials.
    """

    name = "hashing-tfidf"

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def _features(self, text: str) -> Counter:
        text = (text or "").lower()
        feats = Counter(tokenize(text))
        compact = re.sub(r"\s+", " ", text)
        for i in range(len(compact) - 3):
            feats[f"#{compact[i:i + 4]}"] += 0.35
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat, weight in self._features(text).items():
                h = int(hashlib.md5(feat.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += weight * (1.0 if h & 1 else -1.0)
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class GeminiEmbeddings(EmbeddingBackend):
    name = "gemini-embedding-001"
    dim = 3072

    def __init__(self, model: str = "models/gemini-embedding-001") -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._client = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._client.embed_documents(list(texts))
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


def default_embedding_backend() -> EmbeddingBackend:
    if os.getenv("RIA_EMBEDDINGS", "").lower() == "hashing":
        return HashingTfidfEmbeddings()
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            return GeminiEmbeddings()
        except Exception:
            pass
    return HashingTfidfEmbeddings()


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.n = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.freqs = [Counter(d) for d in corpus]
        df = Counter()
        for doc in corpus:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def scores(self, query: List[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        for i, freq in enumerate(self.freqs):
            denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
            total = 0.0
            for term in query:
                f = freq.get(term, 0)
                if not f:
                    continue
                total += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + denom_norm)
            out[i] = total
        return out


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
class GoldenBucket:
    def __init__(
        self,
        root: Optional[Path] = None,
        embedder: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.root = root or (DATA_DIR / "golden_bucket")
        self.candidates_dir = self.root / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.root / ".embedding_cache.json"
        self.embedder = embedder or default_embedding_backend()
        self.trios: List[Trio] = []
        self._bm25: Optional[BM25] = None
        self._matrix: Optional[np.ndarray] = None
        self._loaded_signature: Optional[str] = None
        self.reload()

    # ---- indexing --------------------------------------------------------
    def _signature(self) -> str:
        files = sorted(self.root.glob("*.json")) + sorted(self.candidates_dir.glob("*.json"))
        return hashlib.md5(
            "".join(f"{p.name}:{p.stat().st_mtime}" for p in files).encode()
        ).hexdigest()

    def reload(self, force: bool = False) -> None:
        signature = self._signature()
        if not force and signature == self._loaded_signature:
            return
        trios: List[Trio] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                trios.append(Trio.from_dict(json.loads(path.read_text("utf-8")), path))
            except (json.JSONDecodeError, OSError):
                continue
        # Only promoted trios are retrievable; candidates await review.
        self.trios = [t for t in trios if t.status == "promoted"]
        self._build_index()
        self._loaded_signature = signature

    def _load_cache(self) -> Dict[str, List[float]]:
        try:
            return json.loads(self.cache_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _build_index(self) -> None:
        if not self.trios:
            self._bm25, self._matrix = None, None
            return
        self._bm25 = BM25([tokenize(t.index_text) for t in self.trios])

        cache = self._load_cache()
        vectors: List[Optional[np.ndarray]] = []
        missing: List[int] = []
        for i, trio in enumerate(self.trios):
            key = f"{self.embedder.name}:{hashlib.md5(trio.index_text.encode()).hexdigest()}"
            if key in cache:
                vectors.append(np.asarray(cache[key], dtype=np.float32))
            else:
                vectors.append(None)
                missing.append(i)

        if missing:
            try:
                fresh = self.embedder.embed([self.trios[i].index_text for i in missing])
                for slot, i in enumerate(missing):
                    vectors[i] = fresh[slot]
                    key = f"{self.embedder.name}:{hashlib.md5(self.trios[i].index_text.encode()).hexdigest()}"
                    cache[key] = fresh[slot].tolist()
                self.cache_path.write_text(json.dumps(cache), "utf-8")
            except Exception:
                # Embedding provider down -> degrade to lexical-only retrieval
                # rather than failing the turn.
                fallback = HashingTfidfEmbeddings()
                fresh = fallback.embed([self.trios[i].index_text for i in missing])
                for slot, i in enumerate(missing):
                    vectors[i] = fresh[slot]
        self._matrix = np.vstack([v for v in vectors if v is not None])

    # ---- retrieval -------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 4,
        bm25_weight: float = 0.45,
        dense_weight: float = 0.55,
        min_score: float = 0.0,
    ) -> List[RetrievedTrio]:
        self.reload()
        if not self.trios or self._bm25 is None or self._matrix is None:
            metrics.incr("golden.miss", reason="empty_bucket")
            return []

        lexical = self._bm25.scores(tokenize(query))
        try:
            qvec = self.embedder.embed([query])[0]
        except Exception:
            qvec = HashingTfidfEmbeddings().embed([query])[0]
        dense = (self._matrix @ qvec) if self._matrix.shape[1] == qvec.shape[0] else np.zeros(len(self.trios))

        combined = bm25_weight * _minmax(lexical) + dense_weight * _minmax(np.asarray(dense))
        # A trio the analysts rated highly is a better precedent at equal relevance.
        combined = combined * (0.85 + 0.15 * np.array([t.quality_score for t in self.trios]))

        order = np.argsort(-combined)[:k]
        results = [
            RetrievedTrio(self.trios[i], float(combined[i]), float(lexical[i]), float(dense[i]))
            for i in order
            if combined[i] >= min_score
        ]
        if results:
            metrics.incr("golden.hits")
            metrics.observe("golden.top_score", results[0].score)
        else:
            metrics.incr("golden.miss", reason="below_threshold")
        return results

    # ---- write path ------------------------------------------------------
    def add_candidate(self, trio: Trio) -> Path:
        trio.status = "candidate"
        path = self.candidates_dir / f"{trio.trio_id}.json"
        path.write_text(json.dumps(trio.to_dict(), indent=2, default=str), "utf-8")
        return path

    def list_candidates(self) -> List[Trio]:
        out = []
        for path in sorted(self.candidates_dir.glob("*.json")):
            try:
                out.append(Trio.from_dict(json.loads(path.read_text("utf-8")), path))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def promote(self, trio_id: str, reviewer: str = "cli") -> Optional[Path]:
        for trio in self.list_candidates():
            if trio.trio_id == trio_id:
                trio.status = "promoted"
                data = trio.to_dict()
                data["promoted_by"] = reviewer
                data["promoted_at"] = time.strftime("%Y-%m-%d")
                target = self.root / f"{trio_id}.json"
                target.write_text(json.dumps(data, indent=2, default=str), "utf-8")
                if trio.source_path and trio.source_path.exists():
                    trio.source_path.unlink()
                self.reload(force=True)
                return target
        return None

    def reject(self, trio_id: str) -> bool:
        for trio in self.list_candidates():
            if trio.trio_id == trio_id and trio.source_path:
                trio.source_path.unlink()
                return True
        return False

    def stats(self) -> Dict[str, Any]:
        return {
            "promoted": len(self.trios),
            "candidates": len(self.list_candidates()),
            "embedder": self.embedder.name,
            "tags": sorted({tag for t in self.trios for tag in t.intent_tags}),
        }
