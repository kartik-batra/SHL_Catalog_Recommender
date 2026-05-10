"""
Retriever: wraps the FAISS index and catalog metadata.

Loaded once at service startup; all methods are synchronous and fast
(FAISS search on 400 vectors is sub-millisecond).
"""

from __future__ import annotations

import logging
import pickle
from difflib import get_close_matches
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

TEST_TYPE_LABELS: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


class Retriever:
    """
    Wraps the FAISS index for semantic search over the SHL catalog.

    The retriever is intentionally synchronous.  FastAPI runs agent.run()
    in a thread pool (via run_in_executor) so blocking here is fine.
    """

    def __init__(
        self,
        index_path: str = "data/catalog.index",
        meta_path: str = "data/catalog_meta.pkl",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        for path in (index_path, meta_path):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found. Run 'python scripts/build_catalog.py' first."
                )

        logger.info("Loading FAISS index from %s …", index_path)
        self.index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            self.catalog: list[dict] = pickle.load(f)

        logger.info("Loading FastEmbed model %s …", model_name)
        self.model = TextEmbedding(model_name=model_name)

        # Pre-build lookup structures for O(1) / O(log n) access
        self.url_set: set[str] = {item["url"] for item in self.catalog}
        self.name_lower_map: dict[str, dict] = {
            item["name"].lower(): item for item in self.catalog
        }

        logger.info(
            "Retriever ready — %d assessments indexed.", len(self.catalog)
        )

    # ─────────────────────────── public API ──────────────────────────────────

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """
        Semantic search over the catalog.

        Returns up to top_k items ordered by cosine similarity (descending).
        Each item is the original catalog dict with an extra '_score' key.
        """
        if not query.strip():
            return []

        # Embed and normalise query
        embedding = list(self.model.embed([query]))[0]
        embedding = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding)

        k = min(top_k, len(self.catalog))
        scores, indices = self.index.search(embedding, k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.catalog[idx])
            item["_score"] = float(score)
            results.append(item)

        return results

    def get_by_name(self, name: str) -> dict | None:
        """
        Look up an assessment by name.

        Tries exact (case-insensitive) match first, then fuzzy match
        (difflib with cutoff=0.6) to tolerate minor user typos.
        """
        key = name.lower().strip()
        if key in self.name_lower_map:
            return self.name_lower_map[key]

        close = get_close_matches(key, self.name_lower_map.keys(), n=1, cutoff=0.6)
        if close:
            return self.name_lower_map[close[0]]

        return None

    def validate_url(self, url: str) -> bool:
        """Return True only if the URL is in the scraped catalog."""
        return url in self.url_set

    def format_for_prompt(self, items: list[dict], max_desc_chars: int = 220) -> str:
        """
        Format a list of retrieved items as a compact context block
        suitable for injection into the LLM system prompt.
        """
        if not items:
            return "(No relevant assessments found — ask the user for more details.)"

        lines: list[str] = []
        for item in items:
            types_full = " | ".join(
                TEST_TYPE_LABELS.get(t, t) for t in item.get("test_types", [])
            ) or "General"
            desc = item.get("description", "")[:max_desc_chars].strip()
            levels = ", ".join(item.get("job_levels", [])[:5]) or "—"
            remote_tag = " [Remote✓]" if item.get("remote_testing") else ""
            adaptive_tag = " [Adaptive]" if item.get("adaptive_irt") else ""

            lines.append(
                f'• "{item["name"]}"{remote_tag}{adaptive_tag}\n'
                f'  URL: {item["url"]}\n'
                f"  Types: {types_full}\n"
                f"  Levels: {levels}\n"
                f"  {desc}"
            )

        return "\n\n".join(lines)
