import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import anthropic

OUTPUT_FILE = Path("data/trends.json")
CLAUDE_MODEL = "claude-sonnet-4-6" 

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
    if not isinstance(payload, dict):
        raise ValueError("JSON invalide: objet racine attendu.")
    return payload


def build_prompt_with_today() -> str:
    today = datetime.now(timezone.utc).strftime("%d/%m/%Y")
    return PROMPT_TEMPLATE.replace("[mettre la date du jour]", today)


def call_claude_api(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY manquant.")

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    output_text = "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()

    if not output_text:
        raise RuntimeError("Réponse Claude vide.")

    return output_text


def save_output(payload: Dict[str, Any]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Résultat écrit dans {OUTPUT_FILE}")


def main() -> None:
    prompt = build_prompt_with_today()
    raw_response = call_claude_api(prompt)
    parsed = parse_model_json(raw_response)
    save_output(parsed)


if __name__ == "__main__":
    main()
