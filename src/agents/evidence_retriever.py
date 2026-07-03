"""
src/agents/evidence_retriever.py
=================================
Evidence retriever for the Turkish RAG Hallucination Verifier pipeline.

Retrieves supporting evidence for a claim from one or more sources:
  - Local document store (BM25 / TF-IDF)
  - Wikipedia TR (via wikipedia-api)
  - Web search (stub — requires API key)

Architecture position:
    claim → EvidenceRetriever → context chunks → Qwen verifier

The retriever is intentionally decoupled from the verifier:
  - Verifier only checks "does this context support this claim?"
  - Retriever decides "where to look for context"

Usage:
    from src.agents.evidence_retriever import EvidenceRetriever
    retriever = EvidenceRetriever(mode="local", corpus=my_docs)
    result = retriever.retrieve("Lozan Antlaşması 1923'te imzalandı.")
    print(result.context)   # Retrieved passage
    print(result.source)    # "local" | "wikipedia" | "web"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Result of a single retrieval operation."""
    context: str                        # Retrieved passage(s) concatenated
    source: str                         # "local" | "wikipedia" | "web" | "none"
    passages: List[str] = field(default_factory=list)   # Individual passages
    scores: List[float] = field(default_factory=list)   # Relevance scores
    query: str = ""                     # The query used for retrieval
    retrieval_error: Optional[str] = None


# ---------------------------------------------------------------------------
# EvidenceRetriever
# ---------------------------------------------------------------------------

class EvidenceRetriever:
    """
    Multi-source evidence retriever.

    Modes:
        "local"     — BM25-style keyword search over a provided corpus
        "wikipedia" — Wikipedia TR article retrieval (requires wikipedia-api)
        "web"       — Web search stub (requires external API)
        "passthrough" — Returns the provided context as-is (for pipeline testing)

    Args:
        mode:       Retrieval mode. Default: "passthrough".
        corpus:     List of document strings for "local" mode.
        top_k:      Number of passages to retrieve. Default: 3.
        max_chars:  Maximum characters in returned context. Default: 2000.
    """

    def __init__(
        self,
        mode: str = "passthrough",
        corpus: Optional[List[str]] = None,
        top_k: int = 3,
        max_chars: int = 2000,
    ) -> None:
        self.mode = mode
        self.corpus = corpus or []
        self.top_k = top_k
        self.max_chars = max_chars

        if mode == "local" and not corpus:
            raise ValueError(
                "EvidenceRetriever mode='local' requires a non-empty corpus. "
                "Pass corpus=[...] with your document strings."
            )

    def retrieve(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve evidence for a query/claim.

        Args:
            query:   The claim or question to retrieve evidence for.
            context: Pre-existing context (used in passthrough mode).

        Returns:
            RetrievalResult with context and metadata.
        """
        if self.mode == "passthrough":
            return self._passthrough(query, context)
        elif self.mode == "local":
            return self._local_bm25(query)
        elif self.mode == "wikipedia":
            return self._wikipedia(query)
        elif self.mode == "web":
            return self._web_stub(query)
        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode!r}. "
                             f"Choose from: passthrough, local, wikipedia, web")

    # ------------------------------------------------------------------
    # Passthrough mode — use existing context as-is
    # ------------------------------------------------------------------

    def _passthrough(self, query: str, context: Optional[str]) -> RetrievalResult:
        """Return the provided context unchanged."""
        if not context:
            return RetrievalResult(
                context="",
                source="none",
                query=query,
                retrieval_error="No context provided in passthrough mode.",
            )
        truncated = context[:self.max_chars]
        return RetrievalResult(
            context=truncated,
            source="passthrough",
            passages=[truncated],
            scores=[1.0],
            query=query,
        )

    # ------------------------------------------------------------------
    # Local BM25-style retrieval
    # ------------------------------------------------------------------

    def _local_bm25(self, query: str) -> RetrievalResult:
        """
        Simple TF-IDF-style keyword retrieval over the local corpus.
        No external dependencies — uses token overlap scoring.

        For production use, replace with rank_bm25 or faiss.
        """
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return RetrievalResult(
                context="",
                source="local",
                query=query,
                retrieval_error="Empty query after tokenization.",
            )

        scored = []
        for doc in self.corpus:
            doc_tokens = set(_tokenize(doc))
            if not doc_tokens:
                continue
            # Jaccard similarity
            overlap = len(query_tokens & doc_tokens)
            score = overlap / len(query_tokens | doc_tokens)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.top_k]

        if not top or top[0][0] == 0.0:
            return RetrievalResult(
                context="",
                source="local",
                query=query,
                retrieval_error="No relevant passages found in local corpus.",
            )

        passages = [doc for _, doc in top]
        scores = [score for score, _ in top]
        context = "\n\n".join(passages)[:self.max_chars]

        return RetrievalResult(
            context=context,
            source="local",
            passages=passages,
            scores=scores,
            query=query,
        )

    # ------------------------------------------------------------------
    # Wikipedia TR retrieval
    # ------------------------------------------------------------------

    def _wikipedia(self, query: str) -> RetrievalResult:
        """
        Retrieve a Wikipedia TR article summary for the query.
        Requires: pip install wikipedia-api
        """
        try:
            import wikipediaapi
        except ImportError:
            return RetrievalResult(
                context="",
                source="wikipedia",
                query=query,
                retrieval_error=(
                    "wikipedia-api not installed. "
                    "Run: pip install wikipedia-api"
                ),
            )

        try:
            wiki = wikipediaapi.Wikipedia(
                language="tr",
                user_agent="TurkishRAGVerifier/1.0 (research project)",
            )
            # Extract key entity from query for article lookup
            search_term = _extract_search_term(query)
            page = wiki.page(search_term)

            if not page.exists():
                return RetrievalResult(
                    context="",
                    source="wikipedia",
                    query=query,
                    retrieval_error=f"Wikipedia page not found for: {search_term!r}",
                )

            # Use summary (first ~2000 chars)
            context = page.summary[:self.max_chars]
            return RetrievalResult(
                context=context,
                source="wikipedia",
                passages=[context],
                scores=[1.0],
                query=query,
            )

        except Exception as e:
            return RetrievalResult(
                context="",
                source="wikipedia",
                query=query,
                retrieval_error=f"Wikipedia retrieval error: {e}",
            )

    # ------------------------------------------------------------------
    # Web search stub
    # ------------------------------------------------------------------

    def _web_stub(self, query: str) -> RetrievalResult:
        """
        Web search retrieval stub.
        Implement with SerpAPI, Bing Search API, or DuckDuckGo.
        """
        return RetrievalResult(
            context="",
            source="web",
            query=query,
            retrieval_error=(
                "Web search not implemented. "
                "Integrate SerpAPI or Bing Search API and implement _web_search()."
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple Turkish-aware tokenizer: lowercase, remove punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    return [t for t in tokens if len(t) > 2]


def _extract_search_term(query: str) -> str:
    """
    Extract the most likely Wikipedia search term from a claim.
    Heuristic: take the first noun phrase (capitalized words).
    """
    # Find capitalized word sequences (likely named entities)
    matches = re.findall(r"[A-ZÇĞİÖŞÜ][a-zçğışöüA-ZÇĞİÖŞÜ]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğışöüA-ZÇĞİÖŞÜ]+)*", query)
    if matches:
        # Return the longest match (most specific entity)
        return max(matches, key=len)
    # Fallback: use first 3 words
    words = query.split()
    return " ".join(words[:3])