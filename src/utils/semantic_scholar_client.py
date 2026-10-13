"""Lightweight async Semantic Scholar API client for gap resolution.

Rate-limited to 1 req/sec. Gracefully returns None on any failure.
"""

import asyncio
import logging
import os
from typing import Any

import aiohttp

logger = logging.getLogger("s2_client")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
FIELDS = "title,abstract,tldr,externalIds,openAccessPdf,year,citationCount"
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_last_request_time: float = 0


async def _rate_limit():
    """Ensure at least 1 second between requests."""
    global _last_request_time
    import time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < 1.0:
        await asyncio.sleep(1.0 - elapsed)
    _last_request_time = time.monotonic()


def _headers() -> dict[str, str]:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if key:
        return {"x-api-key": key}
    return {}


def _parse_response(data: dict) -> dict[str, Any] | None:
    """Normalize S2 API response to a standard format."""
    if not data or not data.get("title"):
        return None
    tldr = data.get("tldr")
    if isinstance(tldr, dict):
        tldr = tldr.get("text", "")
    return {
        "title": data.get("title", ""),
        "abstract": data.get("abstract", ""),
        "tldr": tldr or "",
        "year": data.get("year"),
        "citation_count": data.get("citationCount"),
        "arxiv_id": (data.get("externalIds") or {}).get("ArXiv"),
        "doi": (data.get("externalIds") or {}).get("DOI"),
        "open_access_url": (data.get("openAccessPdf") or {}).get("url"),
    }


async def fetch_paper(identifier: str) -> dict[str, Any] | None:
    """Fetch a paper by S2 identifier.

    identifier examples:
        "ARXIV:2106.09685"
        "DOI:10.48550/arXiv.2106.09685"
        "CorpusID:235422040"
    """
    await _rate_limit()
    url = f"{BASE_URL}/{identifier}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"fields": FIELDS},
                headers=_headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 404:
                    return None
                if resp.status == 429:
                    logger.warning("S2 API rate limited (429)")
                    return None
                resp.raise_for_status()
                return _parse_response(await resp.json())
    except Exception as e:
        logger.debug(f"S2 fetch_paper failed for {identifier}: {e}")
        return None


async def search_by_title(title: str) -> dict[str, Any] | None:
    """Search for a paper by title, return the best match."""
    await _rate_limit()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                SEARCH_URL,
                params={"query": title, "limit": 3, "fields": FIELDS},
                headers=_headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"S2 search failed with status {resp.status}")
                    return None
                data = await resp.json()
                papers = data.get("data", [])
                if not papers:
                    return None
                # Return the first result (S2 ranks by relevance)
                return _parse_response(papers[0])
    except Exception as e:
        logger.debug(f"S2 search_by_title failed for '{title}': {e}")
        return None


async def resolve_reference(ref: dict) -> dict[str, Any] | None:
    """Try to resolve a paper reference using all available identifiers.

    ref should have optional keys: arxiv_id, doi, title, authors, year
    Returns normalized paper dict or None.
    """
    # Try ArXiv ID first (most precise)
    arxiv_id = ref.get("arxiv_id", "")
    if arxiv_id:
        # Strip version suffix and common prefixes
        clean = arxiv_id.lower().replace("arxiv:", "").strip()
        result = await fetch_paper(f"ARXIV:{clean}")
        if result:
            return result

    # Try DOI
    doi = ref.get("doi", "")
    if doi:
        result = await fetch_paper(f"DOI:{doi}")
        if result:
            return result

    # Fall back to title search
    title = ref.get("title", "")
    if title and len(title) > 5:
        result = await search_by_title(title)
        if result:
            return result

    return None
