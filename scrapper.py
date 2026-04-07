import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"}

SOURCE_URLS = {
    "slang": [
        "https://www.urbandictionary.com/",
        "https://www.urbandictionary.com/define.php?term=skibidi",
    ],
    "culture_news": [
        "https://www.dexerto.com/fr/entertainment/",
        "https://www.melty.fr/",
    ],
}

OUTPUT_FILE = Path("data/trends.json")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def fetch_html(url: str, timeout: int = 20) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_text_candidates_from_html(html: str, max_items: int = 40) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    results: List[str] = []

    for selector in ["h1", "h2", "h3", "a", ".word", ".def-panel", "article"]:
        for element in soup.select(selector):
            text = " ".join(element.get_text(separator=" ", strip=True).split())
            if not text:
                continue
            if len(text) < 4 or len(text) > 180:
                continue
            if text.lower().startswith(("cookie", "privacy", "subscribe", "newsletter")):
                continue
            results.append(text)
            if len(results) >= max_items:
                return dedupe_keep_order(results)

    return dedupe_keep_order(results)


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def collect_sources() -> List[str]:
    collected: List[str] = []
    for source_name, urls in SOURCE_URLS.items():
        for url in urls:
            try:
                html = fetch_html(url)
                snippets = extract_text_candidates_from_html(html)
                collected.extend(snippets)
                print(f"[OK] {source_name}: {url} -> {len(snippets)} éléments")
            except Exception as exc:
                print(f"[WARN] Source inaccessible ({source_name}): {url} ({exc})")

    return dedupe_keep_order(collected)


def build_prompt(raw_items: List[str]) -> str:
    payload = json.dumps(raw_items[:100], ensure_ascii=False)
    return (
        "Voici une liste de termes et titres web : "
        f"{payload}\n\n"
        "Filtre et garde uniquement les micro-trends, memes absurdes ou expressions "
        "typiques des collégiens français (11-15 ans). Ignore les news sérieuses.\n"
        "Format de sortie : JSON pur.\n"
        'Retourne exactement: {"trends":[{"title":"...","context":"..."}]}\n'
        "Pas de markdown. Pas de texte avant/après le JSON."
    )


def parse_json_from_model(text: str) -> Dict:
    text = text.strip()
    json_block_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_block_match:
        text = json_block_match.group(0)
    data = json.loads(text)
    if "trends" not in data or not isinstance(data["trends"], list):
        raise ValueError("JSON invalide: clé 'trends' absente ou invalide")
    return data


def local_fallback_filter(raw_items: List[str]) -> Dict:
    keywords = [
        "skibidi",
        "sigma",
        "rizz",
        "gyatt",
        "npc",
        "brainrot",
        "meme",
        "tiktok",
        "trend",
        "core",
        "forsure",
    ]
    trends = []
    for item in raw_items:
        low = item.lower()
        if any(k in low for k in keywords):
            trends.append(
                {
                    "title": item[:80],
                    "context": "Signal détecté automatiquement depuis des sources web pop-culture.",
                }
            )
        if len(trends) >= 20:
            break
    return {"trends": trends}


def filter_with_huggingface(raw_items: List[str]) -> Dict:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        print("[WARN] Aucun token Hugging Face détecté, fallback local utilisé.")
        return local_fallback_filter(raw_items)

    client = InferenceClient(model=MODEL_ID, token=token)
    prompt = build_prompt(raw_items)

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un classifieur de micro-trends web. "
                    "Tu réponds uniquement en JSON valide."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.2,
    )
    text = completion.choices[0].message.content or ""
    try:
        return parse_json_from_model(text)
    except Exception as exc:
        print(f"[WARN] Réponse IA non parsable ({exc}), fallback local utilisé.")
        return local_fallback_filter(raw_items)


def normalize_output(data: Dict) -> Dict:
    normalized = {
        "last_update": datetime.now(timezone.utc).isoformat(),
        "trends": [],
    }
    for item in data.get("trends", []):
        title = str(item.get("title", "")).strip()
        context = str(item.get("context", "")).strip()
        if not title:
            continue
        normalized["trends"].append({"title": title[:120], "context": context[:240]})
    return normalized


def save_output(payload: Dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Résultat écrit dans {OUTPUT_FILE}")


def main() -> None:
    raw_items = collect_sources()
    if not raw_items:
        print("[WARN] Aucun contenu récupéré, génération d'un fichier vide.")
        payload = normalize_output({"trends": []})
    else:
        ai_output = filter_with_huggingface(raw_items)
        payload = normalize_output(ai_output)
    save_output(payload)


if __name__ == "__main__":
    main()
