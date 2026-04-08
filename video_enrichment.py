import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests

OUTPUT_FILE = Path("data/trends.json")
APIFY_URL = "https://api.apify.com/v2/acts/clockworks~tiktok-scraper/run-sync-get-dataset-items"


def get_views(video: Dict[str, Any]) -> int:
    if isinstance(video.get("playCount"), int):
        return int(video["playCount"])
    stats = video.get("stats")
    if isinstance(stats, dict) and isinstance(stats.get("playCount"), int):
        return int(stats["playCount"])
    return 0


def normalize_hashtag(raw_value: str) -> str:
    if not isinstance(raw_value, str):
        return ""
    return raw_value.strip().lstrip("#").lower()


def fetch_top_video_for_hashtag(hashtag: str, api_token: str) -> Dict[str, Any]:
    payload = {
        "hashtags": [hashtag],
        "resultsPerPage": 30,
        "maxItems": 30,
    }
    response = requests.post(f"{APIFY_URL}?token={api_token}", json=payload, timeout=90)
    response.raise_for_status()
    videos = response.json()
    if not isinstance(videos, list) or not videos:
        return {}
    return max(videos, key=get_views)


def main() -> None:
    token = os.getenv("APIFY_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("APIFY_API_TOKEN manquant.")
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable: {OUTPUT_FILE}")

    payload = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON invalide: objet racine attendu.")

    trends = payload.get("tendances")
    if not isinstance(trends, list):
        trends = payload.get("trends")
    if not isinstance(trends, list):
        print("[VIDEO] Aucun tableau tendances/trends trouvé.")
        return

    for idx, trend in enumerate(trends[:10], start=1):
        if not isinstance(trend, dict):
            continue
        hashtag = normalize_hashtag(str(trend.get("hashtag", "") or trend.get("m", "")))
        if not hashtag:
            print(f"[VIDEO] Trend {idx}: hashtag absent, skip.")
            continue

        print(f"[VIDEO] Trend {idx}: recherche top vidéo pour #{hashtag}")
        top_video = fetch_top_video_for_hashtag(hashtag, token)
        if not top_video:
            print(f"[VIDEO] Trend {idx}: aucune vidéo trouvée pour #{hashtag}")
            continue

        video_url = top_video.get("webVideoUrl") or top_video.get("url") or ""
        if video_url:
            trend["v"] = video_url
            trend["video_views"] = get_views(top_video)
            print(f"[VIDEO] Trend {idx}: vidéo mise à jour.")
        else:
            print(f"[VIDEO] Trend {idx}: URL vidéo absente, skip.")

    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[VIDEO] Enrichissement terminé: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()