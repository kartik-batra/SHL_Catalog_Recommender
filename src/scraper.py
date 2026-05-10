"""
SHL catalog scraper.

Fetches all Individual Test Solutions (type=1) from:
  https://www.shl.com/products/product-catalog/?start={n}&type=1

Each listing page gives: name, url, test_type codes, remote_testing, adaptive_irt.
Each detail page gives: description, job_levels, languages.

Run standalone:
    python -m src.scraper
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_BASE = f"{BASE_URL}/products/product-catalog/"

# Maps single-letter codes to human-readable names
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

KNOWN_JOB_LEVELS: set[str] = {
    "Director",
    "Entry-Level",
    "Executive",
    "Front Line Manager",
    "General Population",
    "Graduate",
    "Manager",
    "Mid-Professional",
    "Professional Individual Contributor",
    "Supervisor",
}

# A broad but tractable list of languages that appear in SHL pages
KNOWN_LANGUAGES: set[str] = {
    "Arabic", "Bulgarian", "Chinese Simplified", "Chinese Traditional",
    "Croatian", "Czech", "Danish", "Dutch", "English (Australia)",
    "English (Canada)", "English (Malaysia)", "English (Singapore)",
    "English (South Africa)", "English (USA)", "English International",
    "Estonian", "Finnish", "Flemish", "French", "French (Belgium)",
    "French (Canada)", "German", "Greek", "Hungarian", "Icelandic",
    "Indonesian", "Italian", "Japanese", "Korean", "Latin American Spanish",
    "Latvian", "Lithuanian", "Malay", "Norwegian", "Polish", "Portuguese",
    "Portuguese (Brazil)", "Romanian", "Russian", "Serbian", "Slovak",
    "Spanish", "Swedish", "Thai", "Turkish", "Vietnamese",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ─────────────────────────── listing-page parsing ────────────────────────────

def _parse_listing_table(html: str) -> list[dict]:
    """Parse the Individual Test Solutions table from a catalog listing page."""
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict] = []

    for table in soup.find_all("table"):
        # Identify the correct table by its first header cell
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if not headers or "Individual Test Solutions" not in headers[0]:
            continue

        for row in table.find_all("tr")[1:]:  # skip the header row
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            link = cols[0].find("a")
            if not link or not link.get("href"):
                continue

            href = link["href"]
            url = href if href.startswith("http") else BASE_URL + href
            name = link.get_text(strip=True)
            if not name:
                continue

            # Remote Testing / Adaptive: SHL uses a tick image; presence = True
            remote = bool(cols[1].find("img") or cols[1].get_text(strip=True))
            adaptive = bool(cols[2].find("img") or cols[2].get_text(strip=True))

            # Test type codes: single uppercase letters separated by spaces
            raw_types = cols[3].get_text(" ", strip=True)
            test_types = [t for t in raw_types.split() if t in TEST_TYPE_LABELS]

            items.append({
                "name": name,
                "url": url,
                "test_types": test_types,
                "remote_testing": remote,
                "adaptive_irt": adaptive,
            })

        break  # Only one Individual Test Solutions table per page

    return items


# ─────────────────────────── detail-page parsing ─────────────────────────────

def _parse_detail_page(html: str, item: dict) -> dict:
    """Extract description, job_levels, and languages from a detail page."""
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    # ── Description ──────────────────────────────────────────────────────────
    # SHL detail pages normally have a meaningful paragraph early in the body.
    description = ""
    skip_fragments = {
        "Practice Tests", "Candidate Support", "Client Support",
        "Browser Check", "Cookie", "Privacy", "Login", "Buy Online",
    }
    for tag in soup.find_all(["p", "div"]):
        text = tag.get_text(" ", strip=True)
        if len(text) < 80:
            continue
        if any(skip in text for skip in skip_fragments):
            continue
        description = re.sub(r"\s+", " ", text)[:600]
        break

    # ── Job levels ───────────────────────────────────────────────────────────
    job_levels = [lvl for lvl in KNOWN_JOB_LEVELS if lvl in page_text]

    # ── Languages ────────────────────────────────────────────────────────────
    languages = [lang for lang in KNOWN_LANGUAGES if lang in page_text]

    return {
        **item,
        "description": description,
        "job_levels": sorted(job_levels),
        "languages": sorted(languages),
    }


# ─────────────────────────── async fetch helpers ─────────────────────────────

async def _fetch(client: httpx.AsyncClient, url: str) -> str | None:
    """GET a URL and return its text, or None on error."""
    try:
        resp = await client.get(url, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("Fetch failed for %s: %s", url, exc)
        return None


# ─────────────────────────── main entry point ────────────────────────────────

async def scrape_catalog(
    output_path: str = "data/catalog.json",
    fetch_details: bool = True,
    listing_delay: float = 0.5,
    detail_delay: float = 0.3,
) -> list[dict]:
    """
    Scrape the full Individual Test Solutions catalog.

    Args:
        output_path: where to write catalog.json
        fetch_details: if False, skip detail-page enrichment (faster, less info)
        listing_delay: seconds to wait between listing page requests
        detail_delay: seconds to wait between detail page requests
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_items: list[dict] = []

    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS, follow_redirects=True
    ) as client:
        # ── Step 1: scrape all listing pages ─────────────────────────────────
        page_num = 0
        while True:
            start = page_num * 12
            url = f"{CATALOG_BASE}?start={start}&type=1"
            logger.info("Listing page %d (start=%d)", page_num + 1, start)

            html = await _fetch(client, url)
            if html is None:
                logger.error("Could not fetch listing page %d — stopping.", page_num)
                break

            page_items = _parse_listing_table(html)
            if not page_items:
                logger.info("No items on page %d — done with listing.", page_num + 1)
                break

            all_items.extend(page_items)
            logger.info("  Found %d items (total so far: %d)", len(page_items), len(all_items))
            page_num += 1
            await asyncio.sleep(listing_delay)

        logger.info("Listing phase complete: %d raw items found.", len(all_items))

        # ── Step 2: deduplicate by URL ────────────────────────────────────────
        seen: set[str] = set()
        unique: list[dict] = []
        for item in all_items:
            if item["url"] not in seen:
                seen.add(item["url"])
                unique.append(item)
        logger.info("After dedup: %d unique items.", len(unique))

        # ── Step 3: enrich with detail pages ─────────────────────────────────
        if fetch_details:
            enriched: list[dict] = []
            for i, item in enumerate(unique):
                logger.info(
                    "Detail %d/%d: %s", i + 1, len(unique), item["name"][:60]
                )
                html = await _fetch(client, item["url"])
                if html:
                    enriched.append(_parse_detail_page(html, item))
                else:
                    enriched.append({
                        **item,
                        "description": "",
                        "job_levels": [],
                        "languages": [],
                    })
                await asyncio.sleep(detail_delay)
            unique = enriched

    # ── Persist ───────────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d items to %s", len(unique), output_path)
    return unique


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(scrape_catalog())
