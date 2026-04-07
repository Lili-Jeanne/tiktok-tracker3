import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

OUTPUT_FILE = Path("data/trends.json")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-latest")

PROMPT_TEMPLATE = """Tu es un expert en culture internet, micro-tendances TikTok et comportements numériques des collégiens français (11–15 ans). Nous sommes [mettre la date du jour].
⚙️ PROTOCOLE DE VALIDATION
- Inclure une tendance si observée sur ≥3 comptes distincts ou ≥2 plateformes différentes (TikTok FR, Reels Instagram FR, YouTube Shorts, Snapchat Spotlight)
- Sources : créateurs, reposts, compilations
- Signal faible mais répété → inclure ; isolé ou douteux → exclure
🎯 MISSION
- Génère 15 micro-tendances actives (≤6 semaines)
- Focus pour parents : comprendre leur enfant et identifier risques (pression sociale, imitation risquée, harcèlement, exposition)
- Inclure : slang/expressions IRL, memes/brainrot, formats vidéos, comportements sociaux liés aux trends
🧠 FILTRE PARENTAL
- Pour chaque tendance, évaluer : compréhensible pour un parent ? impact possible ? risque ?
📋 FORMAT DE SORTIE STRICT (JSON COMPACT, clé courte)
- Clés :
  - id : identifiant
  - k : mot-clé / phrase virale
  - d : explication simple (parent, 1 ligne)
  - t : type [slang/meme/son/format/comportement]
  - s : origine [plateforme — type contenu]
  - v : vues estimées
  - p : parent {traduction, raison d’usage}
  - i : impact {niveau, risque}
  - a : à surveiller si
  - r : comment réagir (parent)
  - du : durée vie estimée (semaines)
  - f : score fiabilité (note/justification)
⚠️ CONTRAINTES
- Pas de tendances adultes (18+)
- Pas de panique inutile
- Pas de jargon incompréhensible
- Pas de tendances mortes
- Autorisé : humour absurde, langage débile/brainrot, tendances sociales (exclusion, imitation)
📊 SYNTHÈSE FINALE
- top3_comprendre : tendances clés pour décoder son enfant
- top3_surveille : comportements ou dynamiques sociales à risque
- normal : tendances sans impact
- conseil : 3 lignes max pour parents
🔹 **Sortie : JSON compact uniquement**, aucune lisibilité nécessaire, rien d’autre que le JSON."""


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

    return {
        "last_update": datetime.now(timezone.utc).isoformat(),
        "trends": trends[:15],
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
