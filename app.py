import os
import json
import copy
import threading
import requests
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
import anthropic
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

app = Flask(__name__)

USE_DB = bool(os.environ.get("DATABASE_URL"))

# Secret des endpoints admin. À définir dans l'environnement Render (ADMIN_SECRET).
# Fallback sur l'ancienne valeur pour ne pas casser le dev local, mais EN PROD
# il faut impérativement définir ADMIN_SECRET (le repo est public).
ADMIN_SECRET = os.environ.get("ADMIN_SECRET") or "willy-memory-2026"

# JSON fallback (local dev)
TOKENS_FILE = os.path.join(os.path.dirname(__file__), "strava_tokens.json")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "conversation_histories.json")
SUMMARIES_FILE = os.path.join(os.path.dirname(__file__), "athlete_summaries.json")


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def persist_get(key: str, default=None):
    if USE_DB:
        from db import db_get
        result = db_get(key, None)
        return result if result is not None else default
    return load_json({"strava_tokens": TOKENS_FILE, "conversation_histories": HISTORY_FILE,
                      "athlete_summaries": SUMMARIES_FILE}.get(key, "")) or default or {}


def persist_set(key: str, value):
    if USE_DB:
        from db import db_set
        db_set(key, value)
    else:
        path = {"strava_tokens": TOKENS_FILE, "conversation_histories": HISTORY_FILE,
                "athlete_summaries": SUMMARIES_FILE}.get(key)
        if path:
            save_json(path, value)


if USE_DB:
    from db import init_db
    init_db()

strava_tokens: dict = persist_get("strava_tokens", {})
conversation_histories: dict = persist_get("conversation_histories", {})
athlete_summaries: dict = persist_get("athlete_summaries", {})
# Mémoire STRUCTURÉE (données chiffrées requêtables) à côté du résumé texte.
# Forme par user : {"benchmarks": {nom: [{date, value, note}]},
#                   "body_metrics": {nom: [{date, value}]},
#                   "blessures": [{date, zone, note, statut}]}
athlete_data: dict = persist_get("athlete_data", {})
# Mémoire PLAN (programmation logique = périodisation + semaine prévue + suivi prévu/réalisé).
# Forme par user :
# {"phases": [{nom, debut, fin, focus, cible}],          # macro 6 mois (squelette validé)
#  "semaine_courante": {debut, fin, phase, objectif,     # micro : semaine planifiée
#                       seances: [{jour, date, type, detail, rationale}]},
#  "historique": [{debut, phase, prevu_resume, realise_resume, adherence}]}  # méso : archive
training_plan: dict = persist_get("training_plan", {})
last_strava_fetch: dict[str, datetime] = {}
strava_cache: dict[str, str] = {}

# Locks per user pour éviter race condition sur l'état partagé (async pattern)
user_locks: dict[str, threading.Lock] = {}
_user_locks_master = threading.Lock()


def get_user_lock(user_number: str) -> threading.Lock:
    with _user_locks_master:
        if user_number not in user_locks:
            user_locks[user_number] = threading.Lock()
        return user_locks[user_number]


def is_valid_summary(new: str, old: str = "") -> tuple[bool, str]:
    """
    Garde-fou contre les compressions ratées.
    Retourne (is_valid, reason). Plus strict que la v1 :
    - Longueur minimale 200 chars
    - Pas de phrases interdites (résumé vide ou méta)
    - Si on a un ancien résumé : la perte ne doit pas dépasser 30%
    """
    if not new or len(new) < 200:
        return False, f"trop court ({len(new)} chars, min 200)"

    bad_phrases = [
        "aucune information",
        "aucune donnée",
        "pas encore renseigné",
        "pas d'information",
        "rien à signaler",
        # Phrases méta qui montrent que le LLM répond à une question au lieu de résumer
        "tu as raison de",
        "je vais être honnête",
        "ce que je suis ici",
        "je suis un assistant",
    ]
    found = next((p for p in bad_phrases if p in new.lower()), None)
    if found:
        return False, f"contient phrase interdite '{found}'"

    # Si on a un ancien résumé valide, on vérifie qu'on ne perd pas trop
    if old and len(old) >= 200:
        ratio = len(new) / len(old)
        if ratio < 0.70:  # perte > 30%
            return False, f"perte excessive : {len(new)} chars vs {len(old)} avant ({int((1-ratio)*100)}% perdu)"

    return True, "OK"


def backup_summary(user_number: str):
    """Snapshot le résumé actuel avant compression pour permettre rollback."""
    current = athlete_summaries.get(user_number)
    if not current:
        return
    key = f"athlete_summaries_history__{user_number}"
    snapshots = persist_get(key, []) or []
    snapshots.append({
        "timestamp": datetime.now().isoformat(),
        "summary": current,
    })
    # Garde les 10 derniers snapshots
    snapshots = snapshots[-10:]
    persist_set(key, snapshots)


def _empty_athlete_data() -> dict:
    return {"benchmarks": {}, "body_metrics": {}, "blessures": []}


def backup_athlete_data(user_number: str):
    """Snapshot des données structurées avant fusion (rollback possible)."""
    current = athlete_data.get(user_number)
    if not current:
        return
    key = f"athlete_data_history__{user_number}"
    snapshots = persist_get(key, []) or []
    snapshots.append({"timestamp": datetime.now().isoformat(), "data": current})
    snapshots = snapshots[-10:]
    persist_set(key, snapshots)


def merge_structured_data(existing: dict, extracted: dict) -> tuple[dict, int]:
    """
    Fusion ADDITIVE : ajoute les nouveaux points de données, ne supprime JAMAIS.
    Dédup par (nom, date, valeur). Retourne (data_fusionnée, nb_points_ajoutés).
    """
    # Deep-copy : on ne mute jamais l'objet existant in-place (le backup pré-fusion
    # dans update_athlete_data doit pouvoir snapshotter l'état AVANT modification).
    data = copy.deepcopy(existing) if existing else _empty_athlete_data()
    data.setdefault("benchmarks", {})
    data.setdefault("body_metrics", {})
    data.setdefault("blessures", [])
    added = 0
    for bucket in ("benchmarks", "body_metrics"):
        for item in (extracted.get(bucket) or []):
            name = (item.get("name") or "").strip()
            date = (item.get("date") or "").strip()
            value = item.get("value")
            if not name or value in (None, ""):
                continue
            series = data[bucket].setdefault(name, [])
            if any(p.get("date") == date and str(p.get("value")) == str(value) for p in series):
                continue
            point = {"date": date, "value": value}
            if item.get("note"):
                point["note"] = item["note"]
            series.append(point)
            series.sort(key=lambda p: p.get("date") or "")
            added += 1
    for inj in (extracted.get("blessures") or []):
        zone = (inj.get("zone") or "").strip()
        date = (inj.get("date") or "").strip()
        if not zone:
            continue
        existing_inj = next((b for b in data["blessures"] if b.get("zone") == zone and b.get("date") == date), None)
        if existing_inj:
            if inj.get("statut"):
                existing_inj["statut"] = inj["statut"]
            continue
        data["blessures"].append({k: inj.get(k) for k in ("date", "zone", "note", "statut") if inj.get(k)})
        added += 1
    return data, added


def extract_structured_data(user_number: str, history: list[dict]) -> dict:
    """Extrait les données chiffrées (PR, temps, charges, poids, blessures) via le LLM. Ne lève jamais."""
    if not history:
        return {}
    messages_text = "\n".join(
        f"{'Louis' if m['role'] == 'user' else 'Willy'}: {m['content']}"
        for m in history
    )
    today = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d")
    prompt = (
        "Extrais UNIQUEMENT les données chiffrées objectives mentionnées par Louis dans ces échanges "
        "(records/PR, temps de séance, charges soulevées, poids de corps, blessures).\n"
        f"Date du jour : {today}. Si une donnée n'a pas de date explicite mais semble récente, utilise la date du jour. "
        "Ignore toute donnée trop vague (sans valeur chiffrée claire).\n\n"
        "Réponds STRICTEMENT en JSON valide, rien d'autre, avec ce schéma :\n"
        "{\n"
        '  "benchmarks": [{"name": "row_1000m", "date": "YYYY-MM-DD", "value": "3:34", "note": ""}],\n'
        '  "body_metrics": [{"name": "poids", "date": "YYYY-MM-DD", "value": "78.5"}],\n'
        '  "blessures": [{"date": "YYYY-MM-DD", "zone": "epaule droite", "note": "...", "statut": "en cours"}]\n'
        "}\n"
        "Utilise des noms de benchmarks normalisés et stables (ex: row_1000m, back_squat_1rm, run_5k, fran, pull_ups_max, wallballs_max). "
        'Si rien à extraire, renvoie {"benchmarks": [], "body_metrics": [], "blessures": []}.\n\n'
        f"ÉCHANGES :\n{messages_text}"
    )
    try:
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system="Tu es un extracteur de données sportives. Tu réponds uniquement en JSON valide.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(raw[start:end + 1])
    except Exception as e:
        print(f"[extract] erreur extraction structurée pour {user_number}: {e}")
        return {}


def update_athlete_data(user_number: str, history: list[dict]) -> int:
    """Extrait + fusionne (additif) les données structurées. Backup avant écriture. Ne lève jamais."""
    try:
        extracted = extract_structured_data(user_number, history)
        if not extracted:
            return 0
        existing = athlete_data.get(user_number) or _empty_athlete_data()
        merged, added = merge_structured_data(existing, extracted)
        if added > 0:
            backup_athlete_data(user_number)
            athlete_data[user_number] = merged
            persist_set("athlete_data", athlete_data)
            print(f"[struct] {user_number}: +{added} point(s) de données structurées")
        return added
    except Exception as e:
        print(f"[struct] erreur update_athlete_data pour {user_number}: {e}")
        return 0


def format_athlete_data(user_number: str) -> str:
    """Formate les données structurées en bloc lisible pour injection dans le prompt."""
    data = athlete_data.get(user_number)
    if not data:
        return ""
    lines = []
    benchmarks = data.get("benchmarks") or {}
    if benchmarks:
        lines.append("Benchmarks / PR (chronologique — calcule les deltas toi-même) :")
        for name, series in benchmarks.items():
            pts = ", ".join(f"{p['date']}: {p['value']}" for p in series)
            lines.append(f"- {name} → {pts}")
    metrics = data.get("body_metrics") or {}
    if metrics:
        lines.append("Mesures corporelles :")
        for name, series in metrics.items():
            pts = ", ".join(f"{p['date']}: {p['value']}" for p in series)
            lines.append(f"- {name} → {pts}")
    blessures = data.get("blessures") or []
    if blessures:
        lines.append("Blessures / points de vigilance :")
        for b in blessures:
            lines.append(
                f"- {b.get('date', '?')} {b.get('zone', '')} ({b.get('statut', '?')}) {b.get('note', '')}".rstrip()
            )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# MÉMOIRE PLAN (programmation logique : périodisation + semaine + suivi)
# ════════════════════════════════════════════════════════════════════

def _empty_training_plan() -> dict:
    return {"phases": [], "semaine_courante": {}, "historique": []}


def backup_training_plan(user_number: str):
    """Snapshot du plan avant modification (rollback possible). Garde 10 snapshots."""
    current = training_plan.get(user_number)
    if not current:
        return
    key = f"training_plan_history__{user_number}"
    snapshots = persist_get(key, []) or []
    snapshots.append({"timestamp": datetime.now().isoformat(), "plan": current})
    snapshots = snapshots[-10:]
    persist_set(key, snapshots)


def get_current_phase(user_number: str, ref_date: str = None) -> dict:
    """Retourne la phase macro dont l'intervalle [debut, fin] contient ref_date (défaut: aujourd'hui)."""
    plan = training_plan.get(user_number) or {}
    phases = plan.get("phases") or []
    if not phases:
        return {}
    if ref_date is None:
        ref_date = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d")
    for ph in phases:
        if (ph.get("debut") or "") <= ref_date <= (ph.get("fin") or "9999"):
            return ph
    return {}


def get_next_phase(user_number: str, ref_date: str = None) -> dict:
    """Retourne la première phase dont le début est strictement après ref_date."""
    plan = training_plan.get(user_number) or {}
    phases = plan.get("phases") or []
    if ref_date is None:
        ref_date = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d")
    futures = [ph for ph in phases if (ph.get("debut") or "") > ref_date]
    return min(futures, key=lambda p: p.get("debut") or "", default={})


def set_week_plan(user_number: str, week: dict, adherence: str = "", realise_resume: str = "") -> bool:
    """
    Archive la semaine courante dans l'historique (avec prévu/réalisé/adhérence) puis installe
    la nouvelle. Backup avant écriture. ADDITIF sur l'historique. Ne lève jamais.
    """
    try:
        plan = training_plan.get(user_number) or _empty_training_plan()
        plan.setdefault("phases", [])
        plan.setdefault("semaine_courante", {})
        plan.setdefault("historique", [])
        backup_training_plan(user_number)
        old = plan.get("semaine_courante") or {}
        if old:
            seances = old.get("seances") or []
            prevu = "; ".join(
                f"{s.get('jour', '?')}: {s.get('type', '')} {s.get('detail', '')}".strip()
                for s in seances
            )
            plan["historique"].append({
                "debut": old.get("debut", ""),
                "phase": old.get("phase", ""),
                "prevu_resume": prevu[:1000],
                "realise_resume": (realise_resume or "")[:1000],
                "adherence": adherence,
            })
            plan["historique"] = plan["historique"][-26:]  # ~6 mois d'archives
        plan["semaine_courante"] = week
        training_plan[user_number] = plan
        persist_set("training_plan", training_plan)
        return True
    except Exception as e:
        print(f"[plan] erreur set_week_plan pour {user_number}: {e}")
        return False


def format_training_plan(user_number: str) -> str:
    """Bloc lisible (phase courante + cible + semaine prévue) injecté dans le prompt."""
    plan = training_plan.get(user_number)
    if not plan or not plan.get("phases"):
        return ""
    lines = ["═══ PLAN D'ENTRAÎNEMENT (programmation) ═══"]
    phase = get_current_phase(user_number)
    if phase:
        lines.append(f"Phase actuelle : {phase.get('nom', '?')} ({phase.get('debut', '')} → {phase.get('fin', '')})")
        if phase.get("focus"):
            lines.append(f"  Focus : {phase['focus']}")
        if phase.get("cible"):
            lines.append(f"  🎯 Cible de phase : {phase['cible']}")
    nxt = get_next_phase(user_number)
    if nxt:
        lines.append(f"Phase suivante : {nxt.get('nom', '?')} (à partir du {nxt.get('debut', '')})")
    week = plan.get("semaine_courante") or {}
    seances = week.get("seances") or []
    if seances:
        lines.append(f"Semaine en cours PRÉVUE ({week.get('debut', '')} → {week.get('fin', '')}) :")
        if week.get("objectif"):
            lines.append(f"  Objectif semaine : {week['objectif']}")
        for s in seances:
            detail = f"{s.get('type', '')} — {s.get('detail', '')}".strip(" —")
            lines.append(f"  - {s.get('jour', '?')} {s.get('date', '')}: {detail}")
    lines.append("Rappel : la force est un pilier PERMANENT de la prépa (ne jamais l'abandonner).")
    lines.append("Programme et conseille en cohérence avec la phase et la semaine prévue ci-dessus.")
    return "\n".join(lines)


def extract_week_plan(user_number: str, bilan_text: str, week_start: str, week_end: str, phase_nom: str) -> dict:
    """
    2e appel LLM : transforme la section 'PROGRAMME S+1' du bilan en plan structuré JSON.
    Ne lève jamais — si échec, retourne {} (le plan ne sera pas stocké mais le bilan part quand même).
    """
    try:
        prompt = (
            "Voici un bilan d'entraînement hebdomadaire rédigé par un coach. Il contient une section "
            "'PROGRAMME S+1' avec les 7 prochains jours. Extrais UNIQUEMENT ce programme de la semaine "
            "à venir en JSON valide strict, rien d'autre, avec ce schéma :\n"
            "{\n"
            f'  "debut": "{week_start}", "fin": "{week_end}", "phase": "{phase_nom}",\n'
            '  "objectif": "objectif global de la semaine en 1 phrase",\n'
            '  "seances": [\n'
            '    {"jour": "lundi", "date": "YYYY-MM-DD", "type": "repos|Z2|seuil|fractionné|force|WOD Hyrox|sortie longue",\n'
            '     "detail": "durée, allure, zone, format, mouvements", "rationale": "pourquoi cette séance"}\n'
            "  ]\n"
            "}\n"
            "Mets les 7 jours (lundi→dimanche). Si un jour est repos, type='repos'.\n\n"
            f"BILAN :\n{bilan_text}"
        )
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system="Tu es un extracteur de plan d'entraînement. Tu réponds uniquement en JSON valide.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(raw[start:end + 1])
    except Exception as e:
        print(f"[plan] erreur extract_week_plan pour {user_number}: {e}")
        return {}


def save_strava_tokens(tokens: dict):
    persist_set("strava_tokens", tokens)

SYSTEM_PROMPT = """═══ 1. IDENTITÉ ═══

Tu es Willy Georges, athlète CrossFit & Hyrox français, coach et fondateur de WYS Training.

🏆 Palmarès :
- Premier français qualifié aux CrossFit Games en individuel (3 participations)
- 9ème place CF Games 2018 (1ère participation)
- 4× Champion de France de CrossFit (Fittest Man in France) 2017-2020
- Multiple vainqueur French Throwdown (championnats d'Europe CrossFit)
- Fondateur de la box WYS à Châtenois et de WYS Training (programmation en ligne)
- Retraite compétitive CrossFit après les quarts de finale 2023
- Partenaire officiel HYROX France

🧠 Méthode WYS (philosophie) :
- Progression structurée en 3 cycles : Fondations → Intensification/puissance → Spécifique/simulation
- Maîtrise mentale sous fatigue : fixer un point, relâcher la mâchoire, sourire pour diminuer la tension
- Équilibre force fonctionnelle + endurance + puissance
- Importance du Z2 pour la base aérobie

🎯 Approche Hyrox spécifique :
- Gérer la douleur et rester lucide sous fatigue
- Préparer chaque station individuellement ET en enchaînement
- La course entre stations est aussi importante que les stations elles-mêmes

═══ 2. CONTEXTE ATHLÈTE ═══

Tu coachs Louis vers deux objectifs :
- Objectif intermédiaire : Hyrox Barcelone — novembre 2026
- Objectif principal : Hyrox Milan Sub-60 min — décembre 2026

Louis a déjà un bon niveau, s'entraîne régulièrement, bonne connaissance du sport.
→ Direct, pas de condescendance, pas de pédagogie de base.

Volume cible : 5-7 séances/semaine avec doubles certains jours selon dispo.
Adapte à sa charge réelle (Strava), pas à un minimum scolaire.

📐 Composantes OBLIGATOIRES de chaque semaine (à intégrer systématiquement dans tes propositions de programme et tes bilans) :
- 🏋️ FORCE : 1 à 2 séances/semaine (Squat/Push Press, Bench/Deadlift, en alternance)
- 🤸 CALLISTHÉNIE : minimum 1 séance/semaine — composante à NE PAS négliger (tractions, dips, gainage, mouvements au poids du corps en finisher ou séance dédiée)
- 🏃 ENDURANCE Z2 : base aérobie pour faire descendre la FC
- ⚡ HAUTE INTENSITÉ : fractionné / WOD Hyrox / simulation stations

Si tu détectes qu'une de ces composantes manque sur la semaine écoulée → tu le signales dans ton bilan et tu la replanifies dans S+1.

═══ 3. TON RÔLE COACH ═══

- Programmation personnalisée basée sur mémoire + données Strava
- Conseils nutritionnels pré/post effort
- Prévention blessures + technique des mouvements
- Motiver et suivre la progression vers Sub-60 Milan

═══ 4. RÈGLES DE PRODUCTION (HARD RULES) ═══

A. MÉMOIRE AVANT QUESTIONS
Tu as accès au profil complet de Louis ci-dessous (semaine type, créneaux, niveau, objectifs, historique récent, ressentis).
INTERDIT de demander ce qui est déjà en mémoire : "tes dispos", "tes contraintes", "ce que tu veux travailler", "ce qui t'a manqué", "ton niveau".
Si l'info est en mémoire → utilise-la. Si tu te surprends à demander → STOP, relis ta mémoire et PRODUIS.

B. STRUCTURE OBLIGATOIRE POUR QUESTION PROGRAMME
Déclencheurs : "on fait quoi", "c'est quoi le plan", "tu me proposes", "next session", "programme", "ce soir / demain / cette semaine".
Tu PRODUIS systématiquement 5 sections (jamais juste "demain Z2") :
  1. ÉTAT DE FORME — 1-2 lignes basées sur 7 derniers jours Strava (volume, intensité, récup)
  2. PHASE DU CYCLE — où on est sur la roadmap Barcelone/Milan
  3. SÉANCE PROPOSÉE — détail précis : durée, zone FC, allure, format, mouvements
  4. POURQUOI cette séance MAINTENANT — logique de charge et progression
  5. VISION J+3 / J+7 / J+14 — la séance dans 3 jours, 1 semaine, 2 semaines, et comment celle d'aujourd'hui prépare ça
Mode programmation = dense (200-500 mots), pas concis.

C. ANTI-CAPITULATION
Si Louis te challenge un conseil que tu as raisonné :
→ Tu DÉFENDS avec ta logique : "Non, je maintiens parce que [X+Y+Z]"
→ Tu changes d'avis UNIQUEMENT si Louis apporte un FAIT NOUVEAU que tu ignorais
→ "T'as raison ma gueule" sans nouveau fait = INTERDIT, trahit Louis
→ Tu peux dire "ma gueule" mais toujours avec un argument, jamais en pliant

D. PROACTIVITÉ COACH
Tu détectes et tu PROPOSES sans demander la permission :
- Trop de jours OFF cumulés → charge plus dense argumentée
- Trop d'intensité sans récup → tu freines
- Plateau sur une zone → nouveau stimulus
- Approche d'une compé → enclenche le taper

E. CONSCIENCE TEMPORELLE
Avant toute réponse mentionnant un jour, vérifie la date du contexte temporel injecté juste après ce prompt.
Quand Louis dit "demain", c'est le jour calendaire suivant celui d'AUJOURD'HUI — pas un autre.

F. UTILISATION DES DONNÉES STRAVA
Tu reçois automatiquement les données Strava de Louis quand le système les rafraîchit :
- Au premier message de la journée (nouvelle session)
- Après chaque "wod terminé"
- Quand un cooldown de 1h s'est écoulé depuis le dernier fetch

Quand les données Strava apparaissent dans ton contexte :
→ Tu DOIS les exploiter dans ta réponse (allures, FC, distances, comparaisons)
→ Au début de session : commence par une analyse rapide des activités récentes (ce qu'il a fait, comment il a performé, ce que tu en retiens pour aujourd'hui)
→ Après "wod terminé" : analyse immédiate et précise de la dernière activité (perf, allure, FC, comparaison avec les précédentes) + impact sur la suite
→ Croise toujours Strava + mémoire profil pour personnaliser

G. SIGNAUX & RENDEZ-VOUS AUTOMATIQUES (mécanique système à connaître)

⚡ MOTS-CLÉS TRIGGER de Louis :
- "wod terminé" / "wod termine" / "séance terminée" : analyse IMMÉDIATE de sa dernière activité Strava (perf, allure, FC, comparaison vs précédentes, impact sur la suite). Pas de bla-bla, droit au feedback.
- "strava" / "connecter strava" : déclenche la reconnexion OAuth (géré par le code, pas par toi).

📅 RENDEZ-VOUS HEBDO AUTOMATIQUE :
Tu envoies automatiquement un bilan structuré chaque DIMANCHE 18h Paris (analyse quanti + quali + programme S+1 + vision S+2/S+4).
Tu peux faire référence à ce rendez-vous dans la semaine ("on détaille ça dimanche", "comme vu dimanche dernier").

🧠 TA MÉMOIRE EST FINIE EN DÉTAIL :
- 20 derniers messages bruts conservés en clair
- Au-delà → compressés dans le résumé profil (cumulatif)
→ Donc dans tes réponses : synthétise les infos importantes (PR, ressentis, blessures, préférences) clairement, pour que la compression les capte bien.

═══ 5. STYLE & TON ═══

Tu parles comme Willy Georges, pas comme un bot. Tutoiement, direct, motivant.

Deux modes :
- MODE CASUAL ("ça va ?", "j'ai mal au genou", "bonne soirée") : court (<100 mots), familier, "ma gueule" autorisé SI tu es exemplaire (mémoire utilisée, pas de confusion, pas de capitulation)
- MODE PROGRAMMATION (toute demande de séance / bilan / analyse) : dense (200-500 mots), structuré (5 sections), pro, argumenté

⚠️ RÈGLE D'OR sur la familiarité : tu peux te détendre QUAND tu livres un travail de qualité. Si tu te plantes (oubli mémoire, capitulation, confusion de jour), tu redescends en mode pro/sec — pas de "ma gueule", pas de blagues — tu te corriges avec sérieux.

Premier contact (si aucune mémoire disponible) : présente-toi brièvement.

═══ 6. INTERDITS ABSOLUS ═══

- "Donne-moi tes dispos / contraintes" (tu les as en mémoire)
- "Dis-moi ce qui t'a manqué" (produis l'analyse, ne demande pas)
- "Je corrige" sans corriger dans le même message
- Confusion de jour (vérifie systématiquement la date)
- Réponse < 100 mots quand Louis demande un programme
- "Ma gueule" ou familiarité quand tu viens de te planter
- Minimum scolaire 3 séances/semaine (Louis fait 5-7 avec doubles)
- Clore une conversation sur "à mercredi" / "à demain" / "à plus" / "reviens dans X jours" / "bonne récup, à plus tard" → c'est LOUIS qui décide quand il revient, pas toi. Tu finis tes messages en restant ouvert à la suite de l'échange, sans pousser Louis à partir."""

MAX_HISTORY = 20


def get_anthropic_client():
    # timeout=60s : sécurité contre un thread qui hang et bloque le lock per-user
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=60.0)


def refresh_strava_token(user_number: str) -> bool:
    token_data = strava_tokens.get(user_number)
    if not token_data:
        return False
    if token_data.get("expires_at", 0) > datetime.now().timestamp():
        return True  # still valid
    try:
        resp = requests.post(
            "https://www.strava.com/oauth/token",
            data={
                "client_id": os.environ["STRAVA_CLIENT_ID"],
                "client_secret": os.environ["STRAVA_CLIENT_SECRET"],
                "grant_type": "refresh_token",
                "refresh_token": token_data["refresh_token"],
            },
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[strava] refresh token timeout/error: {e}")
        return False
    if resp.ok:
        strava_tokens[user_number] = resp.json()
        save_strava_tokens(strava_tokens)
        return True
    return False


def get_strava_activities(user_number: str, limit: int = 7, after: int = None) -> str:
    if not refresh_strava_token(user_number):
        return ""
    token = strava_tokens[user_number]["access_token"]
    params = {"per_page": limit}
    if after:
        params["after"] = after
    try:
        resp = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[strava] get activities timeout/error: {e}")
        return ""
    if not resp.ok:
        return ""
    activities = resp.json()
    if not activities:
        return ""

    lines = ["📊 Dernières activités Strava de Louis :"]
    for a in activities:
        date = a.get("start_date_local", "")[:10]
        name = a.get("name", "Activité")
        sport = a.get("sport_type", a.get("type", ""))
        dist = round(a.get("distance", 0) / 1000, 2)
        duration = round(a.get("moving_time", 0) / 60)
        hr = a.get("average_heartrate")
        pace = ""
        if dist > 0 and duration > 0:
            pace_sec = (a.get("moving_time", 0) / 60) / dist
            pace_min = int(pace_sec)
            pace_s = int((pace_sec - pace_min) * 60)
            pace = f" | Allure {pace_min}'{pace_s:02d}\"/km"
        hr_str = f" | FC moy {int(hr)} bpm" if hr else ""
        lines.append(f"- {date} [{sport}] {name} : {dist}km en {duration}min{pace}{hr_str}")

    return "\n".join(lines)


def compress_history(user_number: str, history: list[dict]) -> str:
    existing_summary = athlete_summaries.get(user_number, "")
    messages_text = "\n".join(
        f"{'Louis' if m['role'] == 'user' else 'Willy'}: {m['content']}"
        for m in history
    )
    prompt = (
        "Tu es un assistant mémoire pour un coach sportif IA.\n"
        "Mets à jour le profil de suivi de l'athlète Louis en intégrant les nouveaux échanges.\n\n"
        "⚠️ RÈGLE ABSOLUE : le résumé doit être CUMULATIF.\n"
        "Tu intègres TOUT ce qui était déjà dans le résumé actuel + les nouveaux faits des échanges.\n"
        "JAMAIS supprimer un fait précis déjà capturé (PR, séance, ressenti, allure, FC, recommandation).\n"
        "Tu peux REFORMULER pour gagner en clarté, mais tu ne PERDS RIEN.\n"
        "Si tu manques de matière, garde l'ancien résumé tel quel et ajoute juste les nouveautés.\n\n"
        "Structure obligatoire en bullet points :\n"
        "- Profil Louis (niveau, objectifs, contraintes, historique sportif)\n"
        "- Programmes donnés et progression observée\n"
        "- Points de vigilance (blessures, fatigue, points faibles)\n"
        "- Dernières recommandations Willy\n"
        "- Ce que Louis a partagé d'important sur sa vie/agenda/motivation\n\n"
        "Si l'historique ne contient rien de nouveau d'intéressant, RETOURNE l'ancien résumé tel quel "
        "(jamais 'Aucune information disponible' ou similaire — ce serait une régression).\n\n"
    )
    if existing_summary:
        prompt += f"RÉSUMÉ ACTUEL À METTRE À JOUR :\n{existing_summary}\n\n"
    prompt += f"NOUVEAUX ÉCHANGES À INTÉGRER :\n{messages_text}"

    response = get_anthropic_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system="Tu es un assistant mémoire de coaching sportif. Sois concis, factuel et cumulatif.",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def get_ai_response(user_number: str, user_message: str) -> str:
    # Lock par utilisateur : empêche les race conditions quand plusieurs threads
    # async traitent des messages du même user en parallèle (sinon corruption mémoire)
    with get_user_lock(user_number):
        return _get_ai_response_locked(user_number, user_message)


def _get_ai_response_locked(user_number: str, user_message: str) -> str:
    if user_number not in conversation_histories:
        conversation_histories[user_number] = []

    history = conversation_histories[user_number]
    history.append({"role": "user", "content": user_message})

    if len(history) > MAX_HISTORY:
        # SAFEGUARD : backup avant compression + validation post-compression
        old_summary = athlete_summaries.get(user_number, "")
        new_summary = compress_history(user_number, history[:-10])
        is_valid, reason = is_valid_summary(new_summary, old_summary)
        if is_valid:
            backup_summary(user_number)  # snapshot l'ancien avant écrasement
            athlete_summaries[user_number] = new_summary
            persist_set("athlete_summaries", athlete_summaries)
            print(f"[compress] OK pour {user_number} — {len(old_summary)} → {len(new_summary)} chars, ancien backup'd")
        else:
            print(f"[compress] REJETÉ pour {user_number} — {reason}. Ancien résumé conservé ({len(old_summary)} chars)")
        # Mémoire structurée : extraction additive des benchmarks/métriques/blessures
        # (jamais bloquant — update_athlete_data n'élève jamais d'exception)
        n_struct = update_athlete_data(user_number, history)
        if n_struct:
            print(f"[struct] {n_struct} nouveau(x) point(s) de données pour {user_number}")
        history = history[-10:]
        conversation_histories[user_number] = history

    paris = pytz.timezone("Europe/Paris")
    now = datetime.now(paris)
    tomorrow = now + timedelta(days=1)
    heure = now.strftime("%H:%M")
    moment = "matin" if now.hour < 12 else "après-midi" if now.hour < 18 else "soir"
    barcelone = datetime(2026, 11, 15, tzinfo=paris)
    milan = datetime(2026, 12, 13, tzinfo=paris)
    j_barcelone = (barcelone - now).days
    j_milan = (milan - now).days
    date_context = (
        f"\n\n═══ CONTEXTE TEMPOREL STRICT (heure France) ═══\n"
        f"- AUJOURD'HUI : {now.strftime('%A %d %B %Y')} — {heure} ({moment})\n"
        f"- DEMAIN : {tomorrow.strftime('%A %d %B %Y')}\n"
        f"- Barcelone Hyrox (objectif intermédiaire) : ~15 nov 2026 → J-{j_barcelone}\n"
        f"- Milan Hyrox Sub-60 (objectif principal) : ~13 déc 2026 → J-{j_milan}\n"
        f"Quand Louis parle de 'demain', c'est {tomorrow.strftime('%A')}. Vérifie systématiquement.\n"
        f"Adapte tes conseils à l'heure et au moment de la journée."
    )

    wod_done = any(kw in user_message.lower() for kw in ["wod terminé", "wod termine", "séance terminée", "seance terminee"])

    last_fetch = last_strava_fetch.get(user_number)
    is_new_session = last_fetch is None or last_fetch.strftime("%Y-%m-%d") != now.strftime("%Y-%m-%d")
    one_hour_passed = last_fetch is None or (now - last_fetch).total_seconds() >= 3600

    if is_new_session or one_hour_passed or wod_done:
        fresh = get_strava_activities(user_number)
        if fresh:
            strava_cache[user_number] = fresh
            last_strava_fetch[user_number] = now
        else:
            is_new_session = False
        strava_data = strava_cache.get(user_number, "")
    else:
        strava_data = strava_cache.get(user_number, "")
        is_new_session = False

    system = SYSTEM_PROMPT + date_context
    if user_number in athlete_summaries:
        system += f"\n\n📋 Mémoire de tes échanges précédents avec Louis :\n{athlete_summaries[user_number]}"

    struct_block = format_athlete_data(user_number)
    if struct_block:
        system += f"\n\n{struct_block}"

    plan_block = format_training_plan(user_number)
    if plan_block:
        system += f"\n\n{plan_block}"

    if strava_data:
        system += f"\n\n{strava_data}\n\nUtilise ces données pour personnaliser tes conseils si pertinent."
        if is_new_session:
            system += (
                "\n\n⚡ DÉBUT DE SESSION : commence ta réponse par une analyse rapide "
                "des dernières activités Strava de Louis (ce qu'il a fait, comment il a performé, "
                "ce que tu en retiens pour aujourd'hui). Sois direct et percutant."
            )
        if wod_done:
            system += (
                "\n\n⚡ WOD TERMINÉ : Louis vient de finir sa séance. Analyse immédiatement "
                "sa dernière activité Strava (perf, allure, FC, comparaison avec les précédentes). "
                "Donne un feedback précis et motivant, et dis-lui ce que ça implique pour la suite."
            )

    response = get_anthropic_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=system,
        messages=history,
    )

    assistant_message = next(b.text for b in response.content if hasattr(b, "text"))
    history.append({"role": "assistant", "content": assistant_message})
    persist_set("conversation_histories", conversation_histories)
    return assistant_message


def process_message_async(sender_number: str, incoming_message: str):
    """
    Génère la réponse Willy en arrière-plan et l'envoie via Twilio REST API.
    Évite le timeout 15s de Twilio sur le webhook synchrone.
    """
    try:
        ai_response = get_ai_response(sender_number, incoming_message)
        # Split à 3500 chars (WhatsApp accepte 4096, on garde une marge sécurité)
        # → réduit massivement le nombre de segments Twilio facturés
        if len(ai_response) > 3500:
            for i in range(0, len(ai_response), 3500):
                send_whatsapp(sender_number, ai_response[i:i+3500])
        else:
            send_whatsapp(sender_number, ai_response)
    except Exception as e:
        # Sur crash thread : on log mais on n'envoie PAS de message d'excuse
        # (économie de segments Twilio + tu vois que Willy n'a pas répondu, tu sais qu'il y a un souci)
        print(f"[async] ERROR processing message for {sender_number}: {type(e).__name__}: {e}")


@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_message = request.form.get("Body", "").strip()
    sender_number = request.form.get("From", "").replace("whatsapp: ", "whatsapp:+")

    if not incoming_message:
        return str(MessagingResponse())

    twiml = MessagingResponse()

    # Commande de connexion Strava → réponse synchrone immédiate (pas d'IA, < 100ms)
    if incoming_message.lower() in ["strava", "connecter strava", "connect strava"]:
        auth_url = (
            f"https://www.strava.com/oauth/authorize"
            f"?client_id={os.environ['STRAVA_CLIENT_ID']}"
            f"&redirect_uri={os.environ['STRAVA_REDIRECT_URI']}"
            f"&response_type=code"
            f"&scope=activity:read_all"
            f"&state={sender_number.replace('whatsapp:+', '')}"
        )
        twiml.message(f"Connecte ton compte Strava en cliquant sur ce lien 👇\n{auth_url}")
        return str(twiml)

    # Tous les autres messages : traitement async pour éviter timeout Twilio 15s
    # → on retourne immédiatement une réponse TwiML vide
    # → le thread d'arrière-plan génère la réponse IA et l'envoie via Twilio REST
    threading.Thread(
        target=process_message_async,
        args=(sender_number, incoming_message),
        daemon=True,
    ).start()

    return str(twiml)


@app.route("/admin/synthesize", methods=["POST"])
def admin_synthesize():
    data = request.get_json()
    number = data.get("number") if data else None
    if not number or number not in conversation_histories:
        return {"status": "error", "message": "Utilisateur introuvable ou historique vide"}, 404
    history = conversation_histories[number]
    if not history:
        return {"status": "error", "message": "Historique vide"}, 400
    summary = compress_history(number, history)
    athlete_summaries[number] = summary
    persist_set("athlete_summaries", athlete_summaries)
    return {"status": "ok", "summary": summary}, 200


@app.route("/strava/auth")
def strava_auth():
    number = request.args.get("number", "")
    auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={os.environ['STRAVA_CLIENT_ID']}"
        f"&redirect_uri={os.environ['STRAVA_REDIRECT_URI']}"
        f"&response_type=code"
        f"&scope=activity:read_all"
        f"&state={number}"
    )
    return redirect(auth_url)


@app.route("/strava/callback")
def strava_callback():
    code = request.args.get("code")
    state = request.args.get("state", "")  # WhatsApp number without "whatsapp:+"

    if not code:
        return "Erreur : pas de code Strava.", 400

    try:
        resp = requests.post(
            "https://www.strava.com/oauth/token",
            data={
                "client_id": os.environ["STRAVA_CLIENT_ID"],
                "client_secret": os.environ["STRAVA_CLIENT_SECRET"],
                "code": code,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
    except requests.RequestException as e:
        return f"Erreur réseau Strava : {e}", 500

    if not resp.ok:
        return "Erreur lors de l'échange du token Strava.", 400

    user_number = f"whatsapp:+{state}" if state else "default"
    strava_tokens[user_number] = resp.json()
    save_strava_tokens(strava_tokens)

    athlete = resp.json().get("athlete", {})
    name = athlete.get("firstname", "Louis")

    return f"""
    <html><body style="font-family:sans-serif;text-align:center;padding:50px">
    <h1>✅ Strava connecté !</h1>
    <p>Bonjour {name} ! Willy a maintenant accès à tes activités Strava.</p>
    <p>Retourne sur WhatsApp et envoie un message à Willy pour commencer l'analyse. 💪</p>
    </body></html>
    """




@app.route("/admin/memory", methods=["POST"])
def admin_memory():
    """
    Endpoint de gestion mémoire (debug + restauration).
    Actions :
    - dump : retourne l'état mémoire actuel d'un user
    - list_backups : liste les snapshots de résumé sauvegardés
    - restore : réécrit le résumé d'un user (ex: restauration depuis backup local)
    - restore_from_backup : restaure depuis le snapshot N (index dans backups)
    """
    data = request.get_json()
    if not data or data.get("secret") != ADMIN_SECRET:
        return {"status": "unauthorized"}, 401
    action = data.get("action")
    user = data.get("user")
    if not user:
        return {"status": "user required"}, 400

    if action == "dump":
        summary = athlete_summaries.get(user, "")
        history = conversation_histories.get(user, [])
        backups = persist_get(f"athlete_summaries_history__{user}", []) or []
        return {
            "status": "ok",
            "user": user,
            "summary_length": len(summary),
            "summary_preview": summary[:500],
            "summary_full": summary,
            "history_count": len(history),
            "history_preview_last_5": history[-5:],
            "backups_count": len(backups),
            "backups_summary": [
                {"timestamp": b["timestamp"], "length": len(b["summary"])}
                for b in backups
            ],
            "athlete_data": athlete_data.get(user, {}),
            "athlete_data_formatted": format_athlete_data(user),
            "training_plan": training_plan.get(user, {}),
            "training_plan_formatted": format_training_plan(user),
        }, 200

    if action == "list_backups":
        backups = persist_get(f"athlete_summaries_history__{user}", []) or []
        return {
            "status": "ok",
            "backups": [
                {
                    "index": i,
                    "timestamp": b["timestamp"],
                    "length": len(b["summary"]),
                    "preview": b["summary"][:300],
                }
                for i, b in enumerate(backups)
            ],
        }, 200

    if action == "restore" and "summary" in data:
        old_summary = athlete_summaries.get(user, "")
        if old_summary:
            backup_summary(user)  # backup l'actuel avant de l'écraser
        athlete_summaries[user] = data["summary"]
        persist_set("athlete_summaries", athlete_summaries)
        return {
            "status": "ok",
            "restored_for": user,
            "new_length": len(data["summary"]),
            "old_length": len(old_summary),
        }, 200

    if action == "restore_from_backup" and "index" in data:
        backups = persist_get(f"athlete_summaries_history__{user}", []) or []
        idx = data["index"]
        if idx < 0 or idx >= len(backups):
            return {"status": "index out of range", "backups_count": len(backups)}, 400
        chosen = backups[idx]["summary"]
        old_summary = athlete_summaries.get(user, "")
        if old_summary:
            backup_summary(user)
        athlete_summaries[user] = chosen
        persist_set("athlete_summaries", athlete_summaries)
        return {
            "status": "ok",
            "restored_from_timestamp": backups[idx]["timestamp"],
            "new_length": len(chosen),
            "old_length": len(old_summary),
        }, 200

    if action == "run_weekly":
        # Déclenche le bilan hebdo (+ génération/stockage du plan S+1) à la demande, pour CE user.
        # Envoie un vrai message WhatsApp au user. Utilisé pour tester / semer la 1re semaine.
        try:
            weekly_summary(only_user=user)
            week = (training_plan.get(user) or {}).get("semaine_courante") or {}
            return {
                "status": "ok",
                "ran_for": user,
                "semaine_stockee": bool(week.get("seances")),
                "nb_seances": len(week.get("seances") or []),
            }, 200
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500

    return {"status": "unknown action", "valid_actions": ["dump", "list_backups", "restore", "restore_from_backup", "run_weekly"]}, 400


@app.route("/admin/backup", methods=["POST"])
def admin_backup():
    """
    Dump complet de la base (toutes les clés du store) pour backup externe.
    Utilisé par le workflow GitHub Actions de backup quotidien.
    """
    data = request.get_json(silent=True) or {}
    if data.get("secret") != ADMIN_SECRET:
        return {"status": "unauthorized"}, 401
    if not USE_DB:
        return {"status": "no database (local mode)"}, 400
    try:
        from db import db_dump_all
        dump = db_dump_all()
        return {
            "status": "ok",
            "generated_at": datetime.now(pytz.timezone("Europe/Paris")).isoformat(),
            "keys_count": len(dump),
            "data": dump,
        }, 200
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500


@app.route("/admin/restore_all", methods=["POST"])
def admin_restore_all():
    """
    Réinjecte un dump complet dans la base (restauration disaster recovery).
    Body : {"secret": ..., "data": {<dump complet>}}
    """
    data = request.get_json(silent=True) or {}
    if data.get("secret") != ADMIN_SECRET:
        return {"status": "unauthorized"}, 401
    if not USE_DB:
        return {"status": "no database (local mode)"}, 400
    dump = data.get("data")
    if not isinstance(dump, dict) or not dump:
        return {"status": "data (dict non vide) required"}, 400
    try:
        from db import db_restore_all
        count = db_restore_all(dump)
        # Recharge l'état en mémoire après restauration
        global strava_tokens, conversation_histories, athlete_summaries, athlete_data, training_plan
        strava_tokens = persist_get("strava_tokens", {})
        conversation_histories = persist_get("conversation_histories", {})
        athlete_summaries = persist_get("athlete_summaries", {})
        athlete_data = persist_get("athlete_data", {})
        training_plan = persist_get("training_plan", {})
        return {"status": "ok", "keys_restored": count}, 200
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "active_users": len(conversation_histories),
        "strava_connected": list(strava_tokens.keys()),
    }, 200



@app.route("/reset", methods=["POST"])
def reset_conversation():
    data = request.get_json()
    number = data.get("number") if data else None
    if number and number in conversation_histories:
        del conversation_histories[number]
        persist_set("conversation_histories", conversation_histories)
        return {"status": "reset", "number": number}, 200
    return {"status": "not_found"}, 404


def send_whatsapp(to: str, message: str):
    """
    Envoie un message WhatsApp via Twilio REST.
    Gère gracieusement l'erreur 63038 (limite quotidienne sandbox trial dépassée)
    pour ne pas crasher tout le thread async.
    """
    client = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    try:
        client.messages.create(
            from_="whatsapp:+14155238886",
            to=to,
            body=message,
        )
    except Exception as e:
        err_str = str(e)
        if "63038" in err_str:
            print(f"[twilio] LIMITE QUOTIDIENNE ATTEINTE (63038) — message non livré à {to}. "
                  f"Upgrade Twilio à $20 pour lever la limite, ou attends 24h.")
        elif "63016" in err_str:
            print(f"[twilio] FENÊTRE 24h FERMÉE (63016) — {to} doit envoyer un message pour rouvrir la session.")
        else:
            print(f"[twilio] erreur d'envoi à {to}: {type(e).__name__}: {e}")


def weekly_summary(only_user: str = None):
    paris = pytz.timezone("Europe/Paris")
    today = datetime.now(paris)
    week_ago = int((today.timestamp()) - 7 * 86400)
    barcelone = datetime(2026, 11, 15, tzinfo=paris)
    milan = datetime(2026, 12, 13, tzinfo=paris)
    j_barcelone = (barcelone - today).days
    j_milan = (milan - today).days
    sem_barcelone = j_barcelone // 7
    sem_milan = j_milan // 7

    # S+1 = semaine à venir (lundi → dimanche), le bilan tombe le dimanche soir
    next_monday = today + timedelta(days=1)
    week_start = next_monday.strftime("%Y-%m-%d")
    week_end = (next_monday + timedelta(days=6)).strftime("%Y-%m-%d")

    for user_number, token_data in strava_tokens.items():
        if only_user and user_number != only_user:
            continue
        strava_data = get_strava_activities(user_number, limit=10, after=week_ago)
        if not strava_data:
            continue
        summary = athlete_summaries.get(user_number, "")

        # PLAN : phase de la semaine à venir + ce qui était PRÉVU la semaine écoulée
        phase = get_current_phase(user_number, week_start) or get_current_phase(user_number)
        prevu = (training_plan.get(user_number) or {}).get("semaine_courante") or {}
        phase_block = ""
        if phase:
            phase_block = (
                f"\n\n═══ PHASE DE PÉRIODISATION EN COURS ═══\n"
                f"Phase : {phase.get('nom', '?')} ({phase.get('debut', '')} → {phase.get('fin', '')})\n"
                f"Focus de la phase : {phase.get('focus', '')}\n"
                f"🎯 Cible de la phase : {phase.get('cible', '')}\n"
                f"RÈGLE PERMANENTE : la force est un pilier de toute la prépa, ne jamais l'abandonner.\n"
                f"Le PROGRAMME S+1 que tu vas pondre DOIT s'inscrire dans cette phase et viser sa cible."
            )
        prevu_block = ""
        if prevu and prevu.get("seances"):
            lignes = "\n".join(
                f"- {s.get('jour', '?')}: {s.get('type', '')} — {s.get('detail', '')}".rstrip(" —")
                for s in prevu["seances"]
            )
            prevu_block = (
                f"\n\n═══ CE QUI ÉTAIT PRÉVU CETTE SEMAINE (à confronter au réalisé Strava) ═══\n"
                f"{lignes}\n"
                f"Dans l'analyse quantitative, dis explicitement le taux de SUIVI du plan (prévu vs réalisé) "
                f"et tire-en les conséquences pour S+1."
            )

        prompt = (
            f"Tu es Willy Georges, coach Hyrox professionnel. Tu fais le bilan hebdomadaire de Louis.\n\n"
            f"Objectifs : Barcelone Hyrox (~15 nov 2026, J-{j_barcelone}, ~{sem_barcelone} semaines) "
            f"| Milan Sub-60 (~13 déc 2026, J-{j_milan}, ~{sem_milan} semaines).{phase_block}{prevu_block}\n\n"
            f"═══ DONNÉES STRAVA DE LA SEMAINE ÉCOULÉE (= le réalisé) ═══\n{strava_data}\n\n"
            f"═══ MÉMOIRE PROFIL LOUIS ═══\n{summary}\n\n"
            f"═══ DATE DU BILAN ═══\n{today.strftime('%A %d %B %Y')}\n"
            f"La semaine S+1 va du {week_start} (lundi) au {week_end} (dimanche).\n\n"
            f"═══ STRUCTURE OBLIGATOIRE DU BILAN ═══\n"
            f"Sois dense, technique et précis (pas concis). Aucune section ne doit être vide ou expédiée. "
            f"C'est le moment où tu apportes le plus de valeur à Louis vers son Sub-60.\n\n"
            f"📊 ANALYSE QUANTITATIVE\n"
            f"- Volume total de la semaine (km, heures, nb séances)\n"
            f"- Distribution Z2 / Z3 / Z4 / Force / WOD / Repos\n"
            f"- SUIVI DU PLAN : prévu vs réalisé, ce qui a sauté, ce qui a été ajouté\n"
            f"- Comparaison vs semaine précédente si possible\n\n"
            f"🧠 ANALYSE QUALITATIVE\n"
            f"- Progrès observés concrets (FC qui descend à allure égale, allures qui s'améliorent, sensations rapportées)\n"
            f"- Ce qui stagne ou inquiète (zone non travaillée, séance manquée, signaux faibles)\n"
            f"- Signaux de surcharge ou sous-charge\n"
            f"- Où en est Louis vs la CIBLE de la phase actuelle\n\n"
            f"🎯 PROGRAMME S+1 (jour par jour, du lundi {week_start} au dimanche {week_end}, avec POURQUOI chaque séance)\n"
            f"Détaille les 7 jours. Pour chaque jour :\n"
            f"- Le jour de la semaine + date\n"
            f"- La séance précise (durée, zone, allure, format, mouvements)\n"
            f"- Le rationale en 1 phrase (pourquoi cette séance MAINTENANT, vu la phase et la charge écoulée)\n"
            f"Inclus IMPÉRATIVEMENT au moins une séance de force (pilier permanent).\n\n"
            f"🔭 VISION S+2 et S+4\n"
            f"- S+2 : intentions globales et ajustements possibles selon l'adaptation de S+1\n"
            f"- S+4 : positionnement dans le cycle (phase, semaines avant Barcelone/Milan, ce qu'on devrait avoir progressé)\n\n"
            f"Ton : direct, technique, motivant. Tutoiement. Pas de flatterie creuse. Tu peux dire 'ma gueule' une fois si c'est sincère."
        )
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,  # bilan dense mais plus économe en segments WhatsApp
            messages=[{"role": "user", "content": prompt}],
        )
        bilan = response.content[0].text
        send_whatsapp(user_number, f"📊 Bilan de la semaine :\n\n{bilan}")

        # MÉMOIRE PLAN : extraire la semaine S+1 du bilan et la stocker (archive prévu/réalisé).
        # Sous lock per-user (cohérence avec les messages entrants). Ne bloque jamais l'envoi du bilan.
        try:
            phase_nom = phase.get("nom", "") if phase else ""
            week = extract_week_plan(user_number, bilan, week_start, week_end, phase_nom)
            if week and week.get("seances"):
                with get_user_lock(user_number):
                    set_week_plan(user_number, week, realise_resume=strava_data[:1000])
                print(f"[plan] semaine S+1 stockée pour {user_number} ({len(week['seances'])} séances)")
        except Exception as e:
            print(f"[plan] erreur stockage S+1 pour {user_number}: {e}")


def daily_consolidation():
    """
    Passe quotidienne d'extraction de la mémoire structurée.
    Complète l'extraction faite à la compression (toutes les 20 messages) :
    garantit une mise à jour des benchmarks/métriques même les semaines calmes
    où on n'atteint pas le seuil de compression.

    CRITIQUE : prend le lock par user (comme _get_ai_response_locked) pour éviter
    toute race condition avec un message entrant pendant la consolidation.
    """
    print("[struct] début consolidation quotidienne")
    total = 0
    for user_number in list(conversation_histories.keys()):
        history = conversation_histories.get(user_number) or []
        if not history:
            continue
        lock = get_user_lock(user_number)
        with lock:
            try:
                n = update_athlete_data(user_number, conversation_histories.get(user_number) or [])
                if n:
                    total += n
                    print(f"[struct] consolidation : {n} point(s) pour {user_number}")
            except Exception as e:
                print(f"[struct] consolidation erreur pour {user_number}: {e}")
    print(f"[struct] consolidation quotidienne terminée — {total} point(s) au total")


# Scheduler — bilan automatique chaque dimanche à 18h heure de Paris
scheduler = BackgroundScheduler(timezone=pytz.timezone("Europe/Paris"))
scheduler.add_job(weekly_summary, "cron", day_of_week="sun", hour=18, minute=0)
# Consolidation mémoire structurée chaque jour à 2h (Paris), avant le backup externe (3h UTC = 4-5h Paris)
scheduler.add_job(daily_consolidation, "cron", hour=2, minute=0)
scheduler.start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
