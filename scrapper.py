import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

OUTPUT_FILE = Path("data/trends.json")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
TIKTOK_HASHTAGS_URL = "https://tiktokhashtags.com/hashtag/{slug}/"

PROMPT_TEMPLATE = """Tu es un expert en culture internet, micro-tendances TikTok et comportements numériques des collégiens français (11–15 ans). Nous sommes [mettre la date du jour].

⚙️ PROTOCOLE DE VALIDATION
- Inclure une tendance si observée sur ≥3 comptes distincts ou ≥2 plateformes différentes (TikTok FR, Reels Instagram FR, YouTube Shorts, Snapchat Spotlight)
- Sources : créateurs, reposts, compilations
- Signal faible mais répété → inclure ; isolé ou douteux → exclure

🎯 MISSION
- Génère 10 micro-tendances actives (≤6 semaines)
- Focus pour parents : comprendre leur enfant et identifier risques (pression sociale, imitation risquée, harcèlement, exposition)
- Inclure : slang/expressions IRL, memes/brainrot, formats vidéos, comportements sociaux liés aux trends

🧠 FILTRE PARENTAL
- Pour chaque tendance, évaluer : compréhensible pour un parent ? impact possible ? risque ?

📋 FORMAT DE SORTIE STRICT (JSON COMPACT)
- Clés pour chaque tendance :
  - ti : nom de la trend
  - m : hashtag associé le plus populaire / cohérent
  - d : explication simple pour parent
  - ty : [slang/meme/son/format/comportement]
  - v : lien exemple de la trend
  - da : date approximative de démarrage
  - du : durée de vie estimée (en semaines)
  - c : [physique (danse, challenge corporel) / morale (auras, validations sociales) / virtuelle (brainrot, memes)]

⚠️ CONTRAINTES
- Pas de tendances adultes (18+)
- Pas de panique inutile
- Pas de jargon incompréhensible
- Pas de tendances mortes
- Autorisé : humour absurde, langage débile/brainrot, tendances sociales (exclusion, imitation)

📊 SYNTHÈSE FINALE
- top3_comprendre : tendances clés pour décoder son enfant
- top3_surveiller : comportements ou dynamiques sociales à risque
- normal : tendances sans impact
- conseil : 3 lignes max pour parents

🔹 **Sortie : JSON compact uniquement**, aucune lisibilité nécessaire, rien d'autre que le JSON."""


def parse_model_json(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Réponse Claude non JSON.")
    payload = json.loads(raw_text[start : end + 1])
    if "trends" not in payload or not isinstance(payload["trends"], list) or not payload["trends"]:
        raise ValueError("JSON invalide: clé trends manquante ou vide.")
    return payload


def build_prompt_with_today() -> str:
    today = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    return PROMPT_TEMPLATE.replace("[mettre la date du jour]", today)


def call_claude_api(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY manquant.")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 3000,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(CLAUDE_API_URL, headers=headers, json=body, timeout=120)
    response.raise_for_status()
    payload = response.json()
    chunks: List[Dict[str, Any]] = payload.get("content", [])
    text_parts = [str(chunk.get("text", "")) for chunk in chunks if chunk.get("type") == "text"]
    output_text = "".join(text_parts).strip()
    if not output_text:
        raise RuntimeError("Réponse Claude vide.")
    return output_text


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


def extract_hashtag_candidates(trend_keyword: str) -> List[str]:
    hashtags = [h.lower() for h in re.findall(r"#([a-zA-Z0-9_]+)", trend_keyword)]
    if hashtags:
        return list(dict.fromkeys(hashtags))

    fallback_words = re.findall(r"[a-zA-Z0-9]+", trend_keyword.lower())
    return list(dict.fromkeys(fallback_words[:3]))


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
        return None

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
    response = requests.get(url, timeout=45)
    if response.status_code != 200:
        return None

    return parse_hashtag_stats_from_text(response.text, hashtag_slug)


def enrich_and_filter_trends_with_hashtag_stats(trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for trend in trends:
        keyword = str(trend.get("k", "")).strip()
        if not keyword:
            continue

        stats: Optional[Dict[str, Any]] = None
        used_hashtag = ""
        for candidate in extract_hashtag_candidates(keyword):
            candidate_stats = fetch_hashtag_stats(candidate)
            if candidate_stats:
                stats = candidate_stats
                used_hashtag = candidate
                break

        trend["hashtag"] = f"#{used_hashtag}" if used_hashtag else ""
        trend["Overall Posts"] = stats["Overall Posts"] if stats else ""
        trend["Overall Views"] = stats["Overall Views"] if stats else ""
        trend["Views / Post"] = stats["Views / Post"] if stats else ""
        output.append(trend)

    return output


def normalize_compact_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    trends: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload.get("trends", []), start=1):
        if not isinstance(item, dict):
            continue
        parent_info = item.get("p", {}) if isinstance(item.get("p"), dict) else {}
        impact_info = item.get("i", {}) if isinstance(item.get("i"), dict) else {}
        normalized_item = {
            "id": str(item.get("id", idx)),
            "k": str(item.get("k", "")).strip(),
            "d": str(item.get("d", "")).strip(),
            "t": str(item.get("t", "")).strip(),
            "s": str(item.get("s", "")).strip(),
            "v": str(item.get("v", "")).strip(),
            "p": {
                "traduction": str(parent_info.get("traduction", "")).strip(),
                "raison d’usage": str(parent_info.get("raison d’usage", "")).strip(),
            },
            "i": {
                "niveau": str(impact_info.get("niveau", "")).strip(),
                "risque": str(impact_info.get("risque", "")).strip(),
            },
            "a": str(item.get("a", "")).strip(),
            "r": str(item.get("r", "")).strip(),
            "du": str(item.get("du", "")).strip(),
            "f": str(item.get("f", "")).strip(),
        }
        if normalized_item["k"]:
            trends.append(normalized_item)

    filtered_trends = enrich_and_filter_trends_with_hashtag_stats(trends[:10])

    return {
        "last_update": datetime.now(timezone.utc).isoformat(),
        "trends": filtered_trends,
        "top3_comprendre": payload.get("top3_comprendre", []),
        "top3_surveille": payload.get("top3_surveille", []),
        "normal": payload.get("normal", []),
        "conseil": payload.get("conseil", ""),
    }


def save_output(payload: Dict[str, Any], raw_response: str) -> None:
    output = {
        **payload,
        "generated_by": CLAUDE_MODEL,
        "raw_response": raw_response[:12000],
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Résultat écrit dans {OUTPUT_FILE}")


def main() -> None:
    prompt = build_prompt_with_today()
    raw_response = call_claude_api(prompt)
    parsed = parse_model_json(raw_response)
    compact_output = normalize_compact_output(parsed)
    save_output(compact_output, raw_response)


if __name__ == "__main__":
    main()
