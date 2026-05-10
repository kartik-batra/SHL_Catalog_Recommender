"""
Offline pipeline: scrape the SHL catalog → build FAISS index.

Usage:
    python scripts/build_catalog.py               # full run (scrape + index)
    python scripts/build_catalog.py --no-details  # skip detail pages (faster)
    python scripts/build_catalog.py --index-only  # rebuild index from existing catalog.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.embedder import build_index
from src.scraper import scrape_catalog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SHL catalog and FAISS index.")
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Skip fetching individual product detail pages (faster, less info).",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip scraping; just rebuild the FAISS index from an existing catalog.json.",
    )
    args = parser.parse_args()

    # ── Step 1: Scrape ────────────────────────────────────────────────────────
    if not args.index_only:
        logger.info("═" * 60)
        logger.info("STEP 1 — Scraping SHL catalog …")
        logger.info("═" * 60)
        asyncio.run(
            scrape_catalog(
                output_path=settings.catalog_path,
                fetch_details=not args.no_details,
                listing_delay=settings.scrape_delay_listing,
                detail_delay=settings.scrape_delay_detail,
            )
        )
    else:
        logger.info("Skipping scrape (--index-only).")
        if not Path(settings.catalog_path).exists():
            logger.error(
                "catalog.json not found at %s. Remove --index-only to scrape first.",
                settings.catalog_path,
            )
            sys.exit(1)

    # ── Step 2: Build index ───────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 2 — Building FAISS index …")
    logger.info("═" * 60)
    build_index(
        catalog_path=settings.catalog_path,
        index_path=settings.index_path,
        meta_path=settings.meta_path,
        model_name=settings.embed_model,
    )

    logger.info("═" * 60)
    logger.info("Done!  Start the server with:")
    logger.info("  uvicorn main:app --host 0.0.0.0 --port 8000")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
