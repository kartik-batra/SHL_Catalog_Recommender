"""
Offline embedder: reads catalog.json, generates FastEmbed embeddings,
builds a FAISS IndexFlatIP (cosine similarity via L2-normalised vectors),
and persists both the index and catalog metadata.

Run standalone:
    python -m src.embedder
"""

from __future__ import annotations

import json
import logging
import pickle
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


def _build_text(item: dict) -> str:
    """
    Create a semantically rich text blob for one catalog item.

    Field ordering mirrors what a recruiter would phrase in a query:
    name → what it measures → description → who it's for → languages → features.
    """
    types_full = ", ".join(
        TEST_TYPE_LABELS.get(t, t) for t in item.get("test_types", [])
    ) or "General assessment"

    description = item.get("description", "").strip()
    levels = ", ".join(item.get("job_levels", [])) or "all job levels"
    langs_raw = item.get("languages", [])
    langs = ", ".join(langs_raw[:6]) + ("..." if len(langs_raw) > 6 else "")

    remote = "Remote testing supported." if item.get("remote_testing") else ""
    adaptive = "Adaptive / IRT." if item.get("adaptive_irt") else ""

    parts = [
        f"{item['name']}.",
        f"Measures: {types_full}.",
        description,
        f"Suitable for: {levels}.",
        f"Available in: {langs}." if langs else "",
        remote,
        adaptive,
    ]
    return " ".join(p for p in parts if p).strip()


def build_index(
    catalog_path: str = "data/catalog.json",
    index_path: str = "data/catalog.index",
    meta_path: str = "data/catalog_meta.pkl",
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
) -> None:
    """
    Build and persist the FAISS index.

    Index type: IndexFlatIP (exact search, inner product).
    Vectors are L2-normalised before insertion so inner product == cosine similarity.
    At ~400 vectors this is faster than HNSW and requires no training.
    """
    for path in (catalog_path,):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python -m src.scraper' first."
            )

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Load catalog ─────────────────────────────────────────────────────────
    with open(catalog_path, encoding="utf-8") as f:
        catalog: list[dict] = json.load(f)
    logger.info("Loaded %d items from %s", len(catalog), catalog_path)

    # ── Build texts ──────────────────────────────────────────────────────────
    texts = [_build_text(item) for item in catalog]
    logger.info("Sample text[0]: %s", texts[0][:200])

    # ── Embed ────────────────────────────────────────────────────────────────
    logger.info("Loading FastEmbed model: %s", model_name)
    model = TextEmbedding(model_name=model_name)

    logger.info("Generating embeddings (batch_size=%d)...", batch_size)
    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = list(model.embed(batch))
        all_embeddings.extend(batch_embs)
        logger.info("  Embedded %d / %d", min(i + batch_size, len(texts)), len(texts))

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info("Embedding matrix shape: %s", embeddings.shape)

    # ── Normalise + index ────────────────────────────────────────────────────
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

    # ── Persist ──────────────────────────────────────────────────────────────
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(catalog, f)

    logger.info("Saved index → %s", index_path)
    logger.info("Saved metadata → %s", meta_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_index()
