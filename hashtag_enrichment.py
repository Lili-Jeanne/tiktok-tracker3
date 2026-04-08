import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

OUTPUT_FILE = Path("data/trends.json")
TIKTOK_HASHTAGS_URL = "https://tiktokhashtags.com/hashtag/{slug}/"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
}


def parse_human_number(value: str) -> Optional[float]:
    cleaned = value.strip().lower().replace(",", "")
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion|k|m|b|t)?$", cleaned)
    if not match:
        return None
    base = float(match.group(1))
    unit = (match.group(2) or "").lower()
    multiplier = {
        "": 1.0,
        "k": 1_000.0,
        "thousand": 1_000.0,
        "m": 1_000_000.0,
        "million": 1_000_000.0,
        "b": 1_000_000_000.0,
        "billion": 1_000_000_000.0,
        "t": 1_000_000_000_000.0,
        "trillion": 1_000_000_000_000.0,
    }
    if unit not in multiplier:
        return None
    return base * multiplier[unit]


def format_int_with_commas(value: float) -> str:
    return f"{int(round(value)):,}"


def extract_hashtag_candidates(trend_keyword: str) -> List[str]:
    hashtags = [h.lower() for h in re.findall(r"#([a-zA-Z0-9_]+)", trend_keyword)]
    if hashtags:
        candidates = list(dict.fromkeys(hashtags))
    else:
        fallback_words = re.findall(r"[a-zA-Z0-9]+", trend_keyword.lower())
        candidates = list(dict.fromkeys(fallback_words[:3]))

    expanded: List[str] = []
    for c in candidates:
        expanded.append(c)
        # Variant utile: "#caughtin4kfr" -> "caughtin4k"
        if c.endswith("fr") and len(c) > 4:
            expanded.append(c[:-2])
    return list(dict.fromkeys(expanded))


def extract_slug_from_url(value: str) -> str:
    if not isinstance(value, str):
        return ""
    match = re.search(r"/tag/([a-zA-Z0-9_]+)", value)
    if match:
        return match.group(1).lower()
    return ""


def parse_hashtag_stats_from_text(page_text: str, hashtag_slug: str) -> Optional[Dict[str, Any]]:
    compact = " ".join(page_text.split())
    tag = re.escape(hashtag_slug.lower())
    row_pattern = re.compile(
        rf"#\s*{tag}\s+(\d+(?:\.\d+)?\s*(?:Thousand|Million|Billion|Trillion))\s+"
        rf"(\d+(?:\.\d+)?\s*(?:Thousand|Million|Billion|Trillion))\s+([\d,]+)",
        flags=re.IGNORECASE,
    )
    row_match = row_pattern.search(compact)

    overall_posts = row_match.group(1).strip() if row_match else None
    overall_views = row_match.group(2).strip() if row_match else None
    views_per_post = row_match.group(3).strip() if row_match else None

    if not overall_posts or not overall_views:
        sentence_match = re.search(
            r"over\s+(\d+(?:\.\d+)?\s*(?:Thousand|Million|Billion|Trillion))\s+overall posts"
            r"\s+and\s+(\d+(?:\.\d+)?\s*(?:Thousand|Million|Billion|Trillion))\s+overall views",
            compact,
            flags=re.IGNORECASE,
        )
        if sentence_match:
            overall_posts = overall_posts or sentence_match.group(1).strip()
            overall_views = overall_views or sentence_match.group(2).strip()

    if not overall_posts or not overall_views:
        label_row_match = re.search(
            r"Hashtag\s+Posts\s+Views\s+Post Views\s+[#\s]*" + tag
            + r"\s+(\d+(?:\.\d+)?\s*(?:Thousand|Million|Billion|Trillion))\s+"
            + r"(\d+(?:\.\d+)?\s*(?:Thousand|Million|Billion|Trillion))\s+([\d,]+)",
            compact,
            flags=re.IGNORECASE,
        )
        if label_row_match:
            overall_posts = overall_posts or label_row_match.group(1).strip()
            overall_views = overall_views or label_row_match.group(2).strip()
            views_per_post = views_per_post or label_row_match.group(3).strip()

    if not overall_posts or not overall_views:
        return None

    if not views_per_post:
        posts_num = parse_human_number(overall_posts)
        views_num = parse_human_number(overall_views)
        if posts_num and posts_num > 0 and views_num is not None:
            views_per_post = format_int_with_commas(views_num / posts_num)

    numeric_views = parse_human_number(overall_views)
    if numeric_views is None:
        return None

    return {
        "Overall Posts": overall_posts,
        "Overall Views": overall_views,
        "Views / Post": views_per_post or "",
        "_overall_views_numeric": numeric_views,
    }


def fetch_hashtag_stats(hashtag_slug: str) -> Optional[Dict[str, Any]]:
    url = TIKTOK_HASHTAGS_URL.format(slug=quote(hashtag_slug))
    response = requests.get(url, headers=HTTP_HEADERS, timeout=45)
    if response.status_code in (403, 429):
        retry_url = url.rstrip("/") + "/"
        response = requests.get(retry_url, headers=HTTP_HEADERS, timeout=45)
    if response.status_code != 200:
        return None

    return parse_hashtag_stats_from_text(response.text, hashtag_slug)


def enrich_trends_with_hashtag_stats(trends: List[Dict[str, Any]]) -> None:
    for trend in trends:
        hashtag_hint = str(trend.get("m", "")).strip()
        title_hint = str(trend.get("ti", "")).strip()
        legacy_keyword = str(trend.get("k", "")).strip()
        video_slug = extract_slug_from_url(str(trend.get("v", "")).strip())
        keyword = hashtag_hint or legacy_keyword or video_slug or title_hint

        stats: Optional[Dict[str, Any]] = None
        used_hashtag = ""
        all_candidates: List[str] = []
        all_candidates.extend(extract_hashtag_candidates(keyword))
        if video_slug:
            all_candidates.extend(extract_hashtag_candidates(video_slug))
        if title_hint:
            all_candidates.extend(extract_hashtag_candidates(title_hint))

        for candidate in list(dict.fromkeys(all_candidates)):
            candidate_stats = fetch_hashtag_stats(candidate)
            if candidate_stats:
                stats = candidate_stats
                used_hashtag = candidate
                break

        trend["hashtag"] = f"#{used_hashtag}" if used_hashtag else ""
        trend["Overall Posts"] = stats["Overall Posts"] if stats else ""
        trend["Overall Views"] = stats["Overall Views"] if stats else ""
        trend["Views / Post"] = stats["Views / Post"] if stats else ""


def main() -> None:
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable: {OUTPUT_FILE}")

    payload = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON invalide: objet racine attendu.")

    trends_value = payload.get("trends")
    if not isinstance(trends_value, list):
        trends_value = payload.get("tendances")
    if not isinstance(trends_value, list):
        OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[OK] Aucun tableau de trends/tendances à enrichir.")
        return

    trends: List[Dict[str, Any]] = [item for item in trends_value if isinstance(item, dict)]
    enrich_trends_with_hashtag_stats(trends[:10])
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Hashtags enrichis dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
