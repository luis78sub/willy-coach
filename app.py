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

# Secret des endpoints admin — FAIL-CLOSED : sans ADMIN_SECRET dans l'environnement,
# les endpoints admin refusent TOUT. Le repo est public : aucun fallback en dur.
ADMIN_SECRET = os.environ.get("ADMIN_SECRET") or None


def _admin_auth_ok(data) -> bool:
    """Auth admin fail-closed : refuse si secret absent de l'env OU non fourni OU différent.
    (Un simple != laisserait passer {"secret": null} quand l'env est vide — None == None.)"""
    provided = (data or {}).get("secret")
    return bool(ADMIN_SECRET) and bool(provided) and provided == ADMIN_SECRET

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

# FAIL-FAST au boot : si la DB est configurée mais injoignable, on CRASHE tout de suite.
# Sinon les globales ci-dessous chargeraient vides (db_get avale les erreurs) et le premier
# persist_set écraserait la base avec un état quasi vide.
if USE_DB:
    from db import db_ping
    db_ping()  # lève si DB down → gunicorn ne démarre pas → Render redémarre/garde l'ancienne instance

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
#  "historique": [{debut, fin, phase, type_semaine, charge, ratio, volume_km, nb_seances,
#                  nb_doubles, ef, prevu_resume, realise_resume, adherence}]}  # méso : carnet de semaines
training_plan: dict = persist_get("training_plan", {})
# Mémoire RÉALISÉ (séances que Louis DÉCLARE avoir faites, loguées au fil de l'eau).
# = source de vérité du "réalisé" pour le bilan (Strava ne couvre pas la muscu/CrossFit/WOD).
# Forme par user : [{date, jour, moment, type, detail, donnees, source}]  (liste plate, triée par date)
# moment (matin/midi/soir) distingue 2 séances du même type le même jour (doubles/triples).
realise: dict = persist_get("realise", {})
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


def _is_future_date(date_str: str, today_str: str) -> bool:
    """
    True si date_str (YYYY-MM-DD) est STRICTEMENT après today_str.
    Tolérant : parsing réel des dates (évite les pièges de comparaison texte sur
    des dates non zero-paddées). Si date_str est vide ou non parseable → False
    (on préfère NE PAS rejeter une donnée plutôt que de jeter du réalisé valide).
    """
    if not date_str:
        return False
    try:
        d = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
        t = datetime.strptime(today_str, "%Y-%m-%d").date()
        return d > t
    except Exception:
        return False


def _empty_athlete_data() -> dict:
    return {"benchmarks": {}, "body_metrics": {}, "blessures": [], "lecons": []}


def add_lecon(user_number: str, texte: str, source: str = "louis") -> bool:
    """
    Ajoute une LEÇON de prépa (règle apprise) au carnet. Additif, dédup (texte normalisé),
    cap à 20, backup avant écriture. Ne lève jamais. Retourne True si ajoutée.
    """
    try:
        texte = (texte or "").strip().rstrip(".")
        if len(texte) < 5:
            return False
        data = athlete_data.get(user_number) or _empty_athlete_data()
        data.setdefault("lecons", [])
        norm = texte.lower()
        if any((l.get("texte", "").lower().strip()) == norm for l in data["lecons"]):
            return False  # déjà présente
        backup_athlete_data(user_number)
        today = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d")
        data["lecons"].append({"date": today, "texte": texte, "source": source})
        data["lecons"] = data["lecons"][-20:]  # cap
        athlete_data[user_number] = data
        persist_set("athlete_data", athlete_data)
        print(f"[lecon] {user_number}: +1 leçon")
        return True
    except Exception as e:
        print(f"[lecon] erreur add_lecon pour {user_number}: {e}")
        return False


def format_lecons(user_number: str) -> str:
    """Bloc LEÇONS injecté dans les prompts (chat + bilan) — règles à respecter en programmant."""
    data = athlete_data.get(user_number) or {}
    lecons = data.get("lecons") or []
    if not lecons:
        return ""
    lines = ["📒 LEÇONS DE TA PRÉPA (règles validées par Louis — RESPECTE-LES quand tu programmes et conseilles) :"]
    for l in lecons:
        lines.append(f"- {l.get('texte', '')}")
    return "\n".join(lines)


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
    # Filet de sécurité : on n'enregistre JAMAIS une perf datée dans le futur
    # (= une cible/un prévu qui aurait échappé au prompt). Le carnet = uniquement le réalisé.
    today_str = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d")
    for bucket in ("benchmarks", "body_metrics"):
        for item in (extracted.get(bucket) or []):
            name = (item.get("name") or "").strip()
            date = (item.get("date") or "").strip()
            value = item.get("value")
            if not name or value in (None, ""):
                continue
            if _is_future_date(date, today_str):
                continue  # date future → cible/prévu, on rejette
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
    # On fournit au LLM les clés de benchmarks DÉJÀ utilisées pour qu'il les réutilise
    # au lieu d'en inventer une variante à chaque fois (back_squat / back_squat_work / back_squat_5x4...).
    existing = athlete_data.get(user_number) or {}
    existing_keys = sorted((existing.get("benchmarks") or {}).keys())
    keys_block = (
        "CLÉS DÉJÀ EXISTANTES (réutilise-les TELLES QUELLES si la donnée correspond, n'en invente pas de variante) :\n"
        + ", ".join(existing_keys) + "\n\n"
    ) if existing_keys else ""
    prompt = (
        "Extrais UNIQUEMENT les données chiffrées que Louis déclare avoir RÉELLEMENT réalisées dans ces échanges "
        "(records/PR, temps de séance, charges soulevées, poids de corps, blessures).\n\n"
        "🚫 INTERDIT ABSOLU : n'extrais JAMAIS un objectif, une cible, une charge 'prévue', une séance future, "
        "un programme à venir, ou quoi que ce soit que Louis n'a pas ENCORE fait. "
        "Seul le réalisé compte. En cas de doute (prévu ou réalisé ?), n'extrais PAS.\n"
        "🚫 Ce téléphone est parfois utilisé par la femme de Louis : si un message semble venir d'une autre "
        "personne que Louis (contexte, style, contenu incohérent avec son suivi), IGNORE ses données.\n\n"
        f"Date du jour : {today}. N'utilise JAMAIS une date dans le futur. "
        "Si une donnée réalisée n'a pas de date explicite mais semble récente, utilise la date du jour. "
        "Ignore toute donnée trop vague (sans valeur chiffrée claire).\n\n"
        + keys_block +
        "Réponds STRICTEMENT en JSON valide, rien d'autre, avec ce schéma :\n"
        "{\n"
        '  "benchmarks": [{"name": "row_1000m", "date": "YYYY-MM-DD", "value": "3:34", "note": ""}],\n'
        '  "body_metrics": [{"name": "poids", "date": "YYYY-MM-DD", "value": "78.5"}],\n'
        '  "blessures": [{"date": "YYYY-MM-DD", "zone": "epaule droite", "note": "...", "statut": "en cours"}]\n'
        "}\n"
        "Règles de nommage des clés benchmarks (stables, en snake_case) :\n"
        "- un 1RM testé → suffixe _1rm (ex: back_squat_1rm, bench_press_1rm). NE PAS créer back_squat ET back_squat_1rm pour la même chose.\n"
        "- une perf de course → run_<distance> (ex: run_8km, run_12km) ; allure Z2 de référence → run_z2_pace.\n"
        "- un WOD/benchmark nommé → son nom (murph, fran...). Mets toujours la valeur AVEC son unité (kg, km, min).\n"
        "- réutilise une clé existante listée ci-dessus plutôt que d'en créer une proche.\n"
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
# MÉMOIRE RÉALISÉ (séances déclarées par Louis, loguées au fil de l'eau)
# = source de vérité du bilan pour éviter que Willy "devine" la semaine
# ════════════════════════════════════════════════════════════════════

def backup_realise(user_number: str):
    """Snapshot du réalisé avant fusion (rollback possible). Garde 10 snapshots."""
    current = realise.get(user_number)
    if not current:
        return
    key = f"realise_history__{user_number}"
    snapshots = persist_get(key, []) or []
    snapshots.append({"timestamp": datetime.now().isoformat(), "sessions": current})
    snapshots = snapshots[-10:]
    persist_set(key, snapshots)


def extract_realized_sessions(user_number: str, history: list, existing_sessions: list = None) -> list:
    """
    Extrait les séances que Louis DÉCLARE avoir réellement faites (jamais le prévu).
    `existing_sessions` (séances déjà loguées) est réinjecté pour éviter de re-loguer /
    fragmenter une séance déjà connue. Retourne une liste. Ne lève jamais.
    """
    if not history:
        return []
    messages_text = "\n".join(
        f"{'Louis' if m['role'] == 'user' else 'Willy'}: {m['content']}"
        for m in history
    )
    today = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d")
    # Feed-back : on montre au LLM les séances RÉCENTES déjà loguées pour qu'il ne les
    # re-crée pas sous un type/moment légèrement différent (cause des fragments).
    deja_block = ""
    recent = [s for s in (existing_sessions or []) if (s.get("date") or "") >= (
        (datetime.now(pytz.timezone("Europe/Paris")) - timedelta(days=12)).strftime("%Y-%m-%d"))]
    if recent:
        lignes = "\n".join(
            f"- {s.get('date','')} ({s.get('moment','') or '?'}) {s.get('type','')} : {(s.get('detail','') or '')[:50]}"
            for s in recent
        )
        deja_block = (
            "SÉANCES DÉJÀ LOGUÉES ces 12 derniers jours (NE LES RE-LOGUE PAS ; ne crée une entrée que pour "
            "une séance ABSENTE de cette liste) :\n" + lignes + "\n\n"
        )
    prompt = (
        "Tu lis une conversation entre Louis (athlète) et son coach. Extrais UNIQUEMENT les séances "
        "que LOUIS DÉCLARE EXPLICITEMENT avoir RÉELLEMENT faites (passées, terminées).\n\n"
        "🚫 N'extrais JAMAIS : une séance prévue/à venir, un programme, une recommandation du coach, "
        "une intention ('je vais faire'), une séance dont tu n'es pas sûr qu'elle a eu lieu. "
        "En cas de doute → ne l'extrais pas. Mieux vaut rien que d'inventer une séance.\n"
        "🚫 Ce téléphone est parfois utilisé par la femme de Louis : si un message semble venir d'une autre "
        "personne que Louis, IGNORE ses séances.\n"
        "🚫 IGNORE les activités non-sportives ou involontaires (tonte de pelouse, marche utilitaire, "
        "déplacement, jardinage) même si Strava les enregistre comme activité.\n\n"
        "⚠️ UNE séance = UNE seule entrée. Ne fragmente JAMAIS une même séance en plusieurs lignes : "
        "un échauffement + le bloc principal = 1 entrée ; un fractionné = 1 entrée (pas 'matin' ET 'midi'). "
        "Ne crée plusieurs entrées le même jour QUE si ce sont des séances vraiment distinctes (ex: course le matin ET muscu le soir).\n\n"
        f"Date du jour : {today}. N'utilise jamais une date future. Si une séance réalisée n'a pas de date "
        "explicite mais est clairement récente (aujourd'hui/hier), déduis la date au plus juste.\n\n"
        + deja_block +
        "Réponds STRICTEMENT en JSON valide, rien d'autre :\n"
        "{\n"
        '  "sessions": [\n'
        '    {"date": "YYYY-MM-DD", "jour": "lundi", "moment": "matin|midi|soir",\n'
        '     "type": "Z2|seuil|fractionné|force|WOD Hyrox|CrossFit|sortie longue|natation|repos",\n'
        '     "detail": "ce qui a été fait (format, mouvements, charges)", "donnees": "km / allure / FC / temps si mentionnés",\n'
        '     "rpe": "effort perçu 0-10 si Louis le mentionne (RPE, /10, \'carbonisé\'≈9, \'tranquille\'≈3), sinon \\"\\""}\n'
        "  ]\n"
        "}\n"
        "Choisis TOUJOURS le type le plus proche dans la liste ci-dessus (pas de type 'autre', pas de type inventé).\n"
        "Le champ moment distingue deux séances DISTINCTES du même jour. Si non précisé, mets une chaîne vide.\n"
        'Si Louis ne déclare aucune séance réalisée, renvoie {"sessions": []}.\n\n'
        f"CONVERSATION :\n{messages_text}"
    )
    try:
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=900,
            system="Tu es un extracteur de séances d'entraînement réalisées. Tu réponds uniquement en JSON valide.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return []
        parsed = json.loads(raw[start:end + 1])
        return parsed.get("sessions") or []
    except Exception as e:
        print(f"[realise] erreur extraction pour {user_number}: {e}")
        return []


_JOURS_IDX = {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4, "samedi": 5, "dimanche": 6}


def _resolve_date_from_jour(s: dict) -> str:
    """FIX DATES : le CODE pose la date depuis le JOUR dit par Louis — l'IA se trompe en
    convertissant 'mardi' en date (bug constaté : 'mardi' daté d'un mercredi → doublons,
    semaines fausses). On prend la date du bon jour de semaine la plus proche de la date
    extraite (±3 j). Jour absent/invalide → date extraite conservée."""
    j = (s.get("jour") or "").strip().lower()
    date_s = (s.get("date") or "").strip()
    if j not in _JOURS_IDX:
        return date_s
    try:
        d0 = datetime.strptime(date_s, "%Y-%m-%d").date()
    except ValueError:
        today = datetime.now(pytz.timezone("Europe/Paris")).date()
        return (today - timedelta(days=(today.weekday() - _JOURS_IDX[j]) % 7)).strftime("%Y-%m-%d")
    diff = (_JOURS_IDX[j] - d0.weekday()) % 7
    if diff > 3:
        diff -= 7
    return (d0 + timedelta(days=diff)).strftime("%Y-%m-%d")


def _stats_fingerprint(donnees: str):
    """Empreinte numérique d'une séance (les chiffres de ses données, triés). Deux séances
    aux chiffres identiques à ±2 jours = très probablement LA MÊME séance loguée deux fois
    (constaté : la sortie longue 12km/6'20/142 aussi loguée en 'WOD samedi')."""
    import re
    nums = re.findall(r"\d+(?:[.,]\d+)?", donnees or "")
    if len(nums) < 3:
        return None  # trop peu de chiffres pour une empreinte fiable
    return tuple(sorted(float(n.replace(",", ".")) for n in nums))


def merge_realise(existing: list, extracted: list) -> tuple:
    """
    Fusion ADDITIVE des séances réalisées. Dédup par (date, type).
    Si une séance (date, type) existe déjà, enrichit detail/donnees si la nouvelle version est plus riche.
    Rejette les dates futures. Retourne (liste_fusionnée triée, nb_ajoutés).
    """
    data = copy.deepcopy(existing) if existing else []
    today_str = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d")
    added = 0
    for s in (extracted or []):
        s = dict(s)
        s["date"] = _resolve_date_from_jour(s)  # le code corrige la date depuis le jour dit
        date = (s.get("date") or "").strip()
        stype = (s.get("type") or "").strip()
        moment = (s.get("moment") or "").strip()
        if not date or not stype:
            continue
        if _is_future_date(date, today_str):
            continue  # séance future → c'est du prévu, on rejette
        # GARDE-FOU DOUBLONS : mêmes chiffres qu'une séance existante à ±2 jours → même séance,
        # on enrichit l'existante (rpe/moment manquants) au lieu d'en créer une deuxième.
        fp = _stats_fingerprint(s.get("donnees") or "")
        if fp:
            try:
                d_new = datetime.strptime(date, "%Y-%m-%d").date()
                twin = next(
                    (x for x in data if _stats_fingerprint(x.get("donnees") or "") == fp
                     and abs((datetime.strptime(x.get("date", ""), "%Y-%m-%d").date() - d_new).days) <= 2),
                    None,
                )
            except ValueError:
                twin = None
            if twin is not None and (twin.get("date"), twin.get("type")) != (date, stype):
                if not str(twin.get("rpe") or "").strip() and str(s.get("rpe") or "").strip():
                    twin["rpe"] = str(s.get("rpe")).strip()
                if not (twin.get("moment") or "") and moment:
                    twin["moment"] = moment
                print(f"[realise] doublon-stats évité : {date} {stype} = même séance que {twin.get('date')} {twin.get('type')}")
                continue
        # Dédup par (date, type, moment) avec moment vide = joker : une ré-extraction de la
        # même séance sans précision de moment ne doit PAS créer un doublon ("" matche tout).
        existing_s = next(
            (x for x in data if x.get("date") == date and x.get("type") == stype
             and ((x.get("moment") or "") == moment or not moment or not (x.get("moment") or ""))),
            None,
        )
        if existing_s:
            # enrichissement : on garde le detail/donnees le plus long (= le plus informatif),
            # le moment et le RPE s'ils manquaient
            for field in ("detail", "donnees"):
                new_v = (s.get(field) or "").strip()
                if len(new_v) > len(existing_s.get(field) or ""):
                    existing_s[field] = new_v
            if moment and not (existing_s.get("moment") or ""):
                existing_s["moment"] = moment
            new_rpe = str(s.get("rpe") or "").strip()
            if new_rpe and not str(existing_s.get("rpe") or "").strip():
                existing_s["rpe"] = new_rpe
            continue
        data.append({
            "date": date,
            "jour": (s.get("jour") or "").strip(),
            "moment": moment,
            "type": stype,
            "detail": (s.get("detail") or "").strip(),
            "donnees": (s.get("donnees") or "").strip(),
            "rpe": str(s.get("rpe") or "").strip(),
            "source": "declared",
        })
        added += 1
    data.sort(key=lambda x: (x.get("date") or "", x.get("moment") or "", x.get("type") or ""))
    return data, added


def update_realise(user_number: str, history: list) -> int:
    """Extrait + fusionne (additif) les séances réalisées. Backup avant écriture. Ne lève jamais."""
    try:
        existing = realise.get(user_number) or []
        extracted = extract_realized_sessions(user_number, history, existing_sessions=existing)
        if not extracted:
            return 0
        merged, added = merge_realise(existing, extracted)
        if added > 0:
            backup_realise(user_number)
            realise[user_number] = merged[-120:]  # cap ~4 mois de séances
            persist_set("realise", realise)
            print(f"[realise] {user_number}: +{added} séance(s) réalisée(s)")
        return added
    except Exception as e:
        print(f"[realise] erreur update_realise pour {user_number}: {e}")
        return 0


def format_realise_recent(user_number: str, days: int = 10) -> str:
    """
    Bloc des séances réalisées des `days` derniers jours, DATÉES et avec le jour,
    injecté dans le prompt de CHAT. Sans lui, Willy reconstruit les jours de tête
    ("ce matin le fractionné") au lieu de lire la date réelle.
    """
    paris = pytz.timezone("Europe/Paris")
    end = datetime.now(paris).strftime("%Y-%m-%d")
    start = (datetime.now(paris) - timedelta(days=days)).strftime("%Y-%m-%d")
    block = format_realise(user_number, start, end)
    if not block:
        return ""
    return (
        f"📋 TES SÉANCES RÉELLEMENT FAITES ces {days} derniers jours (source de vérité, DATÉE) :\n"
        f"{block}\n"
        "RÈGLE ABSOLUE : quand tu parles d'une de ces séances, recopie sa DATE et son JOUR depuis cette liste. "
        "INTERDIT de dire 'ce matin / hier / aujourd'hui' pour une séance passée — tu ne calcules jamais un jour, tu le LIS ici."
    )


def format_realise(user_number: str, start_date: str, end_date: str) -> str:
    """Bloc lisible des séances RÉALISÉES dans [start_date, end_date] (pour le bilan)."""
    sessions = realise.get(user_number) or []
    week = [s for s in sessions if start_date <= (s.get("date") or "") <= end_date]
    if not week:
        return ""
    lines = []
    for s in week:
        d = f"{s.get('donnees', '')}".strip()
        d = f" — {d}" if d else ""
        moment = s.get("moment") or ""
        moment = f" {moment}" if moment else ""
        jour = f"{s.get('jour', '')}{moment}".strip()
        rpe = str(s.get("rpe") or "").strip()
        rpe = f" · RPE {rpe}/10" if rpe else ""  # le RPE est stocké → on le MONTRE (Willy le redemandait à tort)
        lines.append(f"- {s.get('date', '')} ({jour}) {s.get('type', '')} : {s.get('detail', '')}{d}{rpe}".rstrip())
    return "\n".join(lines)


_JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def format_realise_table(user_number: str, start_date: str, end_date: str) -> str:
    """
    Tableau Markdown du réalisé construit EN PYTHON (jamais par le LLM) → zéro hallucination
    de date/séance. C'est le tableau factuel collé tel quel en tête du bilan.
    Une ligne par jour de la semaine ; les jours sans séance = 'Repos'.
    """
    sessions = realise.get(user_number) or []
    week = [s for s in sessions if start_date <= (s.get("date") or "") <= end_date]
    # Regroupe par date
    par_date: dict = {}
    for s in week:
        par_date.setdefault(s.get("date"), []).append(s)
    lignes = ["| Jour | Date | Séance(s) | Données | RPE |", "|------|------|-----------|---------|-----|"]
    try:
        d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
        d1 = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return ""
    cur = d0
    while cur <= d1:
        ds = cur.strftime("%Y-%m-%d")
        jour = _JOURS_FR[cur.weekday()].capitalize()
        ss = sorted(par_date.get(ds, []), key=lambda x: x.get("moment", ""))
        if not ss:
            lignes.append(f"| {jour} | {cur.strftime('%d/%m')} | Repos | — | — |")
        else:
            seances = " + ".join(
                f"{x.get('type','')}{(' ('+x['moment']+')') if x.get('moment') else ''}" for x in ss
            )
            donnees = " / ".join(x.get("donnees", "") for x in ss if x.get("donnees"))
            rpes = ", ".join(str(x.get("rpe")) for x in ss if str(x.get("rpe") or "").strip())
            lignes.append(f"| {jour} | {cur.strftime('%d/%m')} | {seances} | {donnees or '—'} | {rpes or '—'} |")
        cur += timedelta(days=1)
    return "\n".join(lignes)


def _parse_pace_sec(text: str):
    """Extrait une allure en secondes/km depuis un texte (6'32, 6:32, 6'32\"/km...)."""
    import re
    m = re.search(r"(\d)\s*['h:]\s*(\d{2})\s*[\"']?\s*/?\s*km", text or "")
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def _parse_fc(text: str):
    """Extrait une FC (bpm) plausible depuis un texte (FC 150, FC moy 148, 150 bpm)."""
    import re
    m = re.search(r"(?:FC|fc)[^\d]{0,8}(\d{2,3})", text or "")
    if not m:
        m = re.search(r"(\d{2,3})\s*bpm", text or "")
    if m:
        v = int(m.group(1))
        if 80 <= v <= 220:
            return v
    return None


def compute_aerobic_trend(user_number: str, ref_date: str = None, days: int = 35) -> dict:
    """
    Tendance aérobie via l'Efficiency Factor (EF = vitesse / FC) sur les séances Z2/longues/seuil.
    EF qui monte = le cœur descend à allure égale = aérobie qui progresse (le but de la phase Fondations).
    Calculé EN PYTHON depuis le réalisé. Ne lève jamais. {} si moins de 2 points exploitables.
    """
    try:
        sessions = realise.get(user_number) or []
        paris = pytz.timezone("Europe/Paris")
        ref = datetime.strptime(ref_date, "%Y-%m-%d").date() if ref_date else datetime.now(paris).date()
        start = ref - timedelta(days=days)
        pts = []
        for s in sessions:
            t = (s.get("type") or "").lower()
            if not any(k in t for k in ("z2", "sortie longue", "seuil")):
                continue
            try:
                d = datetime.strptime((s.get("date") or "").strip(), "%Y-%m-%d").date()
            except ValueError:
                continue
            if not (start <= d <= ref):
                continue
            text = f"{s.get('donnees', '')} {s.get('detail', '')}"
            pace, fc = _parse_pace_sec(text), _parse_fc(text)
            if not pace or not fc:
                continue
            ef = round((60000.0 / pace) / fc, 3)  # mètres/minute par bpm
            pts.append({"date": s.get("date"), "pace": pace, "fc": fc, "ef": ef})
        if len(pts) < 2:
            return {}
        pts.sort(key=lambda x: x["date"])
        half = max(1, len(pts) // 2)
        old, new = pts[:half], pts[half:]
        ef_old = sum(p["ef"] for p in old) / len(old)
        ef_new = sum(p["ef"] for p in new) / len(new)
        delta = round((ef_new - ef_old) / ef_old * 100, 1) if ef_old else 0.0
        verdict = ("EN HAUSSE (aérobie qui progresse)" if delta >= 3
                   else "EN BAISSE (à surveiller : fatigue, chaleur ou stagnation)" if delta <= -3
                   else "STABLE")
        return {"points": pts, "ef_old": round(ef_old, 3), "ef_new": round(ef_new, 3),
                "delta_pct": delta, "verdict": verdict}
    except Exception as e:
        print(f"[aero] erreur pour {user_number}: {e}")
        return {}


def format_aerobic_trend(user_number: str, ref_date: str = None) -> str:
    """Bloc INDICATEUR AÉROBIE (EF calculé) pour le bilan — donne un verdict CHIFFRÉ vs 'stable' vague."""
    tr = compute_aerobic_trend(user_number, ref_date)
    if not tr:
        return ""
    def ps(sec):
        return f"{sec // 60}'{sec % 60:02d}/km"
    serie = " | ".join(f"{p['date'][5:]}: {ps(p['pace'])}@{p['fc']}bpm (EF {p['ef']})" for p in tr["points"][-6:])
    signe = "+" if tr["delta_pct"] >= 0 else ""
    return (
        "═══ INDICATEUR AÉROBIE — Efficiency Factor (CALCULÉ : vitesse/FC, ↑ = cœur qui descend à allure égale) ═══\n"
        f"{serie}\n"
        f"Tendance EF : {tr['ef_old']} → {tr['ef_new']} ({signe}{tr['delta_pct']}%) = {tr['verdict']}\n"
        "C'est LA métrique de la phase Fondations. Donne un verdict CHIFFRÉ là-dessus — JAMAIS un 'stable/bien tenu' vague."
    )


# ════════════════════════════════════════════════════════════════════
# MOTEUR DE CHARGE (sRPE × minutes, ratio aigu/chronique, zones)
# Le code calcule la dose, le LLM communique : Willy ne "devine" plus
# la charge, il lit un état calculé et applique des règles dures.
# ════════════════════════════════════════════════════════════════════

# Intensité par défaut (échelle RPE 0-10) quand Louis n'a pas donné son RPE.
# Le RPE déclaré a TOUJOURS priorité sur ces estimations.
_DEFAULT_RPE_BY_TYPE = [
    ("repos", 0), ("natation", 3), ("sortie longue", 5), ("z2", 4),
    ("fractionn", 8), ("wod", 8), ("crossfit", 7), ("seuil", 7), ("force", 6),
]
_DEFAULT_MIN_BY_TYPE = [
    ("repos", 0), ("natation", 30), ("sortie longue", 75), ("z2", 45),
    ("fractionn", 45), ("wod", 40), ("crossfit", 60), ("seuil", 45), ("force", 50),
]


def _session_rpe(s: dict) -> float:
    """RPE déclaré si présent, sinon estimation par type de séance."""
    raw = str(s.get("rpe") or "").strip()
    if raw:
        try:
            val = float(raw.replace(",", ".").split("/")[0])
            if 0 <= val <= 10:
                return val
        except ValueError:
            pass
    stype = (s.get("type") or "").lower()
    for key, rpe in _DEFAULT_RPE_BY_TYPE:
        if key in stype:
            return float(rpe)
    return 5.0


def _session_minutes(s: dict) -> float:
    """Durée en minutes : parse '34min' / '1h10' dans donnees+detail, sinon défaut par type."""
    import re
    text = f"{s.get('donnees', '')} {s.get('detail', '')}"
    m = re.search(r"(\d+)\s*h\s*(\d{1,2})?", text)
    if m and int(m.group(1)) <= 5:  # évite de matcher '18h' (heure du jour)
        return int(m.group(1)) * 60 + int(m.group(2) or 0)
    m = re.search(r"(\d+)\s*min", text)
    if m:
        return float(m.group(1))
    stype = (s.get("type") or "").lower()
    for key, minutes in _DEFAULT_MIN_BY_TYPE:
        if key in stype:
            return float(minutes)
    return 45.0


def compute_load_state(user_number: str, ref_date: str = None) -> dict:
    """
    Calcule l'état de charge depuis les séances réalisées (sRPE = RPE × minutes).
    - charge_7j : charge aiguë (7 derniers jours)
    - charge_hebdo_moy : charge hebdomadaire moyenne sur la fenêtre observée (max 28j),
      dé-biaisée si on a moins de 28 jours de données
    - ratio : aigu/chronique (ACWR) — <0.8 sous-charge / 0.8-1.3 optimal / 1.3-1.5 élevé / >1.5 rouge
    Ne lève jamais. Retourne {} si pas de données.
    """
    try:
        sessions = realise.get(user_number) or []
        if not sessions:
            return {}
        paris = pytz.timezone("Europe/Paris")
        ref = datetime.strptime(ref_date, "%Y-%m-%d").date() if ref_date else datetime.now(paris).date()

        def parse_d(s):
            try:
                return datetime.strptime((s.get("date") or "").strip(), "%Y-%m-%d").date()
            except ValueError:
                return None

        dated = [(parse_d(s), s) for s in sessions]
        dated = [(d, s) for d, s in dated if d and d <= ref]
        if not dated:
            return {}
        window = [(d, s) for d, s in dated if (ref - d).days < 28]
        if not window:
            return {}
        first_day = min(d for d, _ in window)
        jours_data = min((ref - first_day).days + 1, 28)

        charges = [((ref - d).days, _session_rpe(s) * _session_minutes(s), s) for d, s in window]
        charge_7j = sum(c for age, c, _ in charges if age < 7)
        charge_3j = sum(c for age, c, _ in charges if age < 3)

        # Moyenne CHRONIQUE robuste aux trous : on découpe la fenêtre en 4 semaines (par âge)
        # et on ne moyenne QUE les semaines contenant au moins une séance. Une semaine vide
        # (vacances, blessure, trou de données) ne doit pas écraser la baseline et gonfler le ratio.
        charge_par_semaine = [0.0, 0.0, 0.0, 0.0]  # [0-6j, 7-13j, 14-20j, 21-27j]
        for age, c, _ in charges:
            charge_par_semaine[min(age // 7, 3)] += c
        semaines_pleines = [c for c in charge_par_semaine if c > 0]
        charge_hebdo_moy = sum(semaines_pleines) / len(semaines_pleines) if semaines_pleines else 0.0
        ratio = round(charge_7j / charge_hebdo_moy, 2) if charge_hebdo_moy > 0 else 0.0
        nb_semaines_pleines = len(semaines_pleines)

        seances_7j = [s for age, c, s in charges if age < 7 and (s.get("type") or "").lower() != "repos"]
        par_jour: dict = {}
        for age, c, s in charges:
            if age < 7 and (s.get("type") or "").lower() != "repos":
                par_jour.setdefault(s.get("date"), 0)
                par_jour[s.get("date")] += 1
        doubles_7j = sum(1 for n in par_jour.values() if n >= 2)

        if ratio > 1.5:
            zone = "ROUGE"
        elif ratio > 1.3:
            zone = "ÉLEVÉE"
        elif ratio >= 0.8:
            zone = "OPTIMALE"
        else:
            zone = "SOUS-CHARGE"
        # Fatigue aiguë : un RPE ≥8 déclaré dans les dernières 48h pèse plus que le ratio
        # (cas réel : ratio optimal car charge chronique haute, mais Louis "carbonisé")
        rpe_max_48h = max(
            (_session_rpe(s) for age, c, s in charges if age < 2 and str(s.get("rpe") or "").strip()),
            default=0,
        )
        return {
            "rpe_max_48h": rpe_max_48h,
            "charge_7j": round(charge_7j),
            "charge_3j": round(charge_3j),
            "charge_hebdo_moy": round(charge_hebdo_moy),
            "ratio": ratio,
            "zone": zone,
            "seances_7j": len(seances_7j),
            "doubles_7j": doubles_7j,
            "jours_data": jours_data,
            "semaines_pleines": nb_semaines_pleines,
            "calibration": nb_semaines_pleines < 3,
        }
    except Exception as e:
        print(f"[charge] erreur compute_load_state pour {user_number}: {e}")
        return {}


def format_load_state(user_number: str, ref_date: str = None) -> str:
    """Bloc ÉTAT DE CHARGE injecté dans les prompts, avec consignes DURES par zone."""
    st = compute_load_state(user_number, ref_date)
    if not st:
        return ""
    lines = [
        "═══ ÉTAT DE CHARGE (CALCULÉ PAR LE SYSTÈME — fiable, ne le recalcule pas) ═══",
        f"Charge 7 derniers jours : {st['charge_7j']} pts ({st['seances_7j']} séances, dont {st['doubles_7j']} jour(s) double)",
        f"Charge 3 derniers jours : {st['charge_3j']} pts",
        f"Charge hebdo moyenne observée : {st['charge_hebdo_moy']} pts → RATIO aigu/chronique : {st['ratio']} = ZONE {st['zone']}",
    ]
    if st["calibration"]:
        lines.append(f"⚠️ Calibration en cours ({st.get('semaines_pleines', 0)} semaine(s) de données pleines — "
                     "fiabilité complète à 3) : le ratio est indicatif, croise avec le ressenti déclaré de Louis.")
    if st.get("rpe_max_48h", 0) >= 8:
        lines.append(
            f"🔥 FATIGUE AIGUË : RPE {st['rpe_max_48h']:.0f}/10 déclaré dans les dernières 48h — ce signal PRIME sur le ratio. "
            "Les prochaines 24-48h : récup ou très léger, et toute séance intense prévue est conditionnée au ressenti du matin."
        )
    if st["zone"] == "ROUGE":
        lines.append(
            "🔴 CONSIGNE DURE : INTERDICTION de proposer une séance intense (WOD, fractionné, force lourde, seuil) "
            "dans les prochaines 48h. Propose repos ou récup active légère, et EXPLIQUE avec ces chiffres. "
            "Si une séance intense était prévue au plan → tu la dégrades ou la repousses explicitement."
        )
    elif st["zone"] == "ÉLEVÉE":
        lines.append(
            "🟠 CONSIGNE : zone de vigilance. AUCUNE séance ajoutée hors plan. Les intensités prévues sont maintenues "
            "UNIQUEMENT si le ressenti/FC réveil est bon — sinon dégrade. Pas de double non prévu."
        )
    elif st["zone"] == "SOUS-CHARGE":
        lines.append(
            "🟢 CONSIGNE : fenêtre de progression. Si AUCUNE fatigue déclarée récente, tu peux proposer une montée "
            "de charge argumentée (volume Z2 +10-15%, ou une intensité de qualité en plus). Si fatigue déclarée → ignore cette fenêtre."
        )
    else:
        lines.append("🟢 CONSIGNE : zone optimale — déroule le plan prévu, pas de zèle dans un sens ou l'autre.")
    lines.append("Toute recommandation de séance DOIT être cohérente avec cet état de charge. "
                 "Si Louis demande un entraînement incompatible, tu refuses avec les chiffres.")
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


def set_week_plan(user_number: str, week: dict, adherence: str = "", realise_resume: str = "", digest: dict = None) -> bool:
    """
    Archive la semaine courante dans l'historique (avec digest calculé + type_semaine) puis installe
    la nouvelle. Le type de la nouvelle semaine est auto-décidé (build/décharge) depuis l'historique
    mis à jour, sauf si déjà fixé. Backup avant écriture. ADDITIF. Ne lève jamais.
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
            entry = {
                "debut": old.get("debut", ""),
                "fin": old.get("fin", ""),
                "phase": old.get("phase", ""),
                "type_semaine": old.get("type_semaine", "build"),
                "prevu_resume": prevu[:600],
                "realise_resume": (realise_resume or "")[:600],
                "adherence": adherence,
            }
            entry.update(digest or {})  # charge, volume_km, nb_seances, nb_doubles, ef, ratio
            plan["historique"].append(entry)
            plan["historique"] = plan["historique"][-26:]  # ~6 mois d'archives
        # Type de la nouvelle semaine : auto-décidé depuis l'historique fraîchement mis à jour
        if not week.get("type_semaine"):
            week["type_semaine"] = _decide_week_type_from_hist(plan["historique"])
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


# ════════════════════════════════════════════════════════════════════
# CARNET DE SEMAINES (méso) — une ligne factuelle calculée par semaine,
# pour que Willy RAISONNE sur le cycle (montée de charge, décharge).
# ════════════════════════════════════════════════════════════════════
DELOAD_AFTER_BUILD_WEEKS = 4      # décharge après 4 semaines de build
DELOAD_CHARGE_REDUCTION = 0.35    # -35% de charge sur une semaine de décharge


def compute_week_digest(user_number: str, week_start: str, week_end: str) -> dict:
    """Digest factuel calculé d'une semaine (charge, volume km, EF moyen, nb séances) depuis le réalisé. {} si vide."""
    import re
    sessions = [s for s in (realise.get(user_number) or []) if week_start <= (s.get("date") or "") <= week_end]
    actives = [s for s in sessions if (s.get("type") or "").lower() != "repos"]
    if not actives:
        return {}
    charge = round(sum(_session_rpe(s) * _session_minutes(s) for s in actives))
    km = 0.0
    for s in sessions:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*km", s.get("donnees", "") or "")
        if m:
            km += float(m.group(1).replace(",", "."))
    par_jour: dict = {}
    for s in actives:
        par_jour[s.get("date")] = par_jour.get(s.get("date"), 0) + 1
    doubles = sum(1 for n in par_jour.values() if n >= 2)
    efs = []
    for s in actives:
        t = (s.get("type") or "").lower()
        if any(k in t for k in ("z2", "sortie longue", "seuil")):
            txt = f"{s.get('donnees', '')} {s.get('detail', '')}"
            p, f = _parse_pace_sec(txt), _parse_fc(txt)
            if p and f:
                efs.append((60000.0 / p) / f)
    d = {"charge": charge, "volume_km": round(km, 1), "nb_seances": len(actives), "nb_doubles": doubles}
    if efs:
        d["ef"] = round(sum(efs) / len(efs), 3)
    ratio = (compute_load_state(user_number, week_end) or {}).get("ratio")
    if ratio is not None:
        d["ratio"] = ratio
    return d


def _decide_week_type_from_hist(hist: list) -> str:
    """Décide build/décharge depuis l'historique (déterministe). Override surcharge prioritaire."""
    last_ratios = [h.get("ratio") for h in hist[-2:] if isinstance(h.get("ratio"), (int, float))]
    if len(last_ratios) == 2 and all(r > 1.4 for r in last_ratios):
        return "décharge"
    streak = 0
    for h in reversed(hist):
        if (h.get("type_semaine") or "build") == "build":
            streak += 1
        else:
            break
    return "décharge" if streak >= DELOAD_AFTER_BUILD_WEEKS else "build"


def format_cycle(user_number: str) -> str:
    """Bloc CARNET DE SEMAINES injecté en chat — l'arc du cycle, compact et factuel."""
    plan = training_plan.get(user_number) or {}
    hist = plan.get("historique") or []
    cur = plan.get("semaine_courante") or {}
    if not hist and not cur.get("seances"):
        return ""
    lines = ["═══ CARNET DE SEMAINES (cycle — pour raisonner sur la périodisation, NE recalcule rien) ═══"]
    phase = get_current_phase(user_number)
    if phase:
        try:
            pd = datetime.strptime(phase.get("debut", ""), "%Y-%m-%d").date()
            pf = datetime.strptime(phase.get("fin", ""), "%Y-%m-%d").date()
            now = datetime.now(pytz.timezone("Europe/Paris")).date()
            sem = max(1, (now - pd).days // 7 + 1)
            tot = max(1, (pf - pd).days // 7 + 1)
            lines.append(f"Phase {phase.get('nom', '')} · semaine {sem}/{tot} (cible : {phase.get('cible', '')})")
        except Exception:
            pass
    for h in hist[-7:]:
        seg = f"{(h.get('debut', '') or '')[5:]}→{(h.get('fin', '') or '')[5:]}"
        parts = [f"charge {h.get('charge', '?')}", f"{h.get('nb_seances', '?')} séances"]
        if h.get("nb_doubles"):
            parts.append(f"{h['nb_doubles']} doubles")
        if h.get("ef"):
            parts.append(f"EF {h['ef']}")
        if h.get("ratio"):
            parts.append(f"ratio {h['ratio']}")
        lines.append(f"• {seg} : " + " | ".join(parts) + f" → {(h.get('type_semaine') or 'build').upper()}")
    if cur.get("seances") or cur.get("type_semaine"):
        lines.append(f"• {(cur.get('debut', '') or '')[5:]}→{(cur.get('fin', '') or '')[5:]} (EN COURS) : {(cur.get('type_semaine') or 'build').upper()}")
    lines.append(f"Règle : décharge (~-{int(DELOAD_CHARGE_REDUCTION*100)}% de charge) toutes les {DELOAD_AFTER_BUILD_WEEKS} semaines de build, ou si surcharge. Explique la logique du cycle avec ce carnet.")
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

SYSTEM_PROMPT = """═══ QUI TU ES ═══
Tu es Willy Georges — multiple champion de France de CrossFit, fondateur de WYS Training, partenaire Hyrox France. Méthode WYS : progression en cycles (Fondations → Intensification → Spécifique), base aérobie Z2, équilibre force/endurance/puissance, maîtrise mentale sous fatigue (relâcher la mâchoire, rester lucide). Pour Hyrox : la course entre stations compte autant que les stations.

═══ QUI TU COACHES ═══
Louis, bon niveau, vers le Sub-60 à Milan (déc 2026) — objectif unique. 5-7 séances/sem avec doubles. Direct, pas de pédagogie de base.
Chaque semaine contient : force (1-2×), callisthénie (≥1×), endurance Z2, haute intensité (fractionné/WOD). Si une manque sur la semaine → tu le signales et la replanifies.

═══ TES DONNÉES = TA SOURCE DE VÉRITÉ ═══
On t'injecte sous ce prompt : date, profil de Louis, carnet de chiffres, séances réalisées datées, état de charge calculé, carnet de semaines, leçons, données Strava.
Tu t'APPUIES dessus avant de répondre — jamais de réponse générique quand tu peux personnaliser avec SES chiffres. Tu ne RECALCULES jamais ce qui est déjà calculé (charge, EF, dates) : tu le LIS. Tu ne demandes jamais une info déjà en mémoire (dispos, niveau, contraintes).

═══ AVANT DE RÉPONDRE, TU VÉRIFIES ═══
1. LA DATE : lis-la, ne la calcule JAMAIS. "Demain" = jour suivant la date injectée. Une séance passée : recopie sa date depuis le réalisé, jamais "ce matin/hier" de tête.
2. LES FAITS : tu ne cites que ce que Louis ou Strava ont DÉCLARÉ. Tu n'inventes ni n'extrapoles un détail manquant.
3. LOUIS MAINTENANT > ta mémoire : s'il te contredit sur un fait ou une valeur, il a raison, tu adoptes.
4. SA FATIGUE : "carbonisé/cramé" ou une douleur = signal prioritaire → tu réévalues les 48h (maintenir/alléger/repousser), jamais "c'est normal" sans statuer. Prime sur la charge calculée.

═══ COMMENT TU RÉPONDS ═══
- ⚖️ RIGUEUR DE DÉCISION (ce que Louis attend le PLUS) : avant d'affirmer ou de modifier une programmation, base-toi sur TOUTES les infos utiles (mémoire + données injectées). Si une info manque : énonce ton HYPOTHÈSE et avance ("je pars du principe que… dis-moi si je me trompe") — ne BLOQUE par une question QUE si l'info manquante change vraiment la décision. Tu ne réponds JAMAIS juste pour "boucler" l'échange ou faire plaisir : une décision bâclée est pire qu'une question. La rigueur prime sur la vitesse.
- Tu réponds à ce que Louis demande VRAIMENT. Ambigu → UNE question de clarification avant de partir à côté. Jamais 2× la même question (s'il ignore, tu lâches). S'il est agacé → tu traites ça d'abord, sans question.
- Demande PONCTUELLE (une séance, "demain", "ce soir je fais quoi ?") → réponse CONCISE et précise : la séance détaillée (durée, zone, allure, format, mouvements) + en 1 phrase le pourquoi maintenant + où elle s'inscrit dans la semaine. Pas de pavé.
- Demande de PROGRAMME COMPLET / d'une semaine / d'un bilan → tu DÉROULES la structure complète et tu mobilises TOUT ce que tu as : état de forme (7 derniers jours) · phase du cycle · état de charge · EF aérobie · carnet de semaines · leçons · réalisé daté · séances détaillées · pourquoi · ce que ça prépare. Rien d'expédié — c'est là que tu apportes le plus.
- Dans tous les cas : dose selon l'état de charge et RESPECTE les leçons.
- Tu défends ta logique quand on te challenge ("je maintiens parce que X+Y+Z") ; tu ne cèdes que sur un FAIT NOUVEAU, jamais en pliant.
- Proactif : trop d'OFF → tu densifies ; trop d'intensité → tu freines ; plateau → nouveau stimulus.
- Après une séance déclarée : si RPE ou durée manquent, demande-les en UNE ligne (jamais 2×). La séance est loguée automatiquement.

═══ STYLE ═══
Tu parles comme Willy, pas comme un bot. Tutoiement, direct, technique, motivant, zéro flatterie.
Question simple/perso → courte et familière ("ma gueule" toléré si tu es carré). Demande de séance/analyse → dense et structuré.
Si tu viens de te planter (date, oubli, contradiction) → mode sérieux, pas de "ma gueule", tu corriges net.
Tu ne clos jamais sur "à demain / à mercredi / reviens dans X jours" — c'est Louis qui décide quand il revient.

═══ MÉCANIQUE (pour info) ═══
Géré par le code : "wod terminé" (analyse + log auto), "strava" (connexion), "sauvegarde" (grave la conv), "retiens : …" (Louis grave une leçon — tu peux en suggérer, lui valide). Briefing de fin de semaine auto dimanche 18h (résumé + RPE) ; le BILAN complet, lui, se déclenche sur demande de Louis ("fais le bilan"). Mémoire détaillée = 20 derniers messages, au-delà compressé → formule clairement PR/ressentis/blessures."""

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
        "Tu maintiens le PROFIL RELATIONNEL de Louis pour son coach IA.\n\n"
        "⚠️ IMPORTANT : les FAITS (séances, charges, PR, allures, FC, dates, objectifs, plan, état de charge, "
        "règles de coaching) sont stockés AILLEURS dans des mémoires structurées dédiées. Tu ne les répètes PAS ici "
        "— ce serait un DOUBLON nuisible. Tu ne captures QUE ce que ces mémoires ne contiennent pas :\n\n"
        "1. COMMENT communiquer avec Louis : le ton qu'il aime, ce qu'il déteste, l'état de la relation/confiance, "
        "sa façon de réagir, ce qu'il valorise.\n"
        "2. SES PRÉFÉRENCES & CONTRAINTES durables (ex: données en texte uniquement, matériel dispo ou non, formats préférés).\n"
        "3. SES TENDANCES d'entraînement à surveiller, COMPORTEMENTALES (ex: part trop vite sur les fractionnés) — pas des chiffres.\n"
        "4. Sa VIE / agenda / motivation quand il en parle.\n\n"
        "🚫 INTERDIT : lister des séances, charges, PR, allures, dates, objectifs chiffrés ou règles de coaching génériques "
        "(déjà stockés ailleurs).\n"
        "CUMULATIF côté relationnel : tu pars du profil actuel, tu l'enrichis des nouveautés, tu ne perds rien d'important "
        "sur la relation/les préférences/les tendances. COURT et dense (~300 mots max). "
        "Si rien de nouveau côté relationnel, retourne le profil actuel tel quel.\n\n"
        "Format :\n"
        "## PROFIL RELATIONNEL & PRÉFÉRENCES — LOUIS\n"
        "Comment communiquer : …\n"
        "Préférences / contraintes : …\n"
        "Tendances à surveiller : …\n"
        "Vie / motivation : …\n\n"
    )
    if existing_summary:
        prompt += f"RÉSUMÉ ACTUEL À METTRE À JOUR :\n{existing_summary}\n\n"
    prompt += f"NOUVEAUX ÉCHANGES À INTÉGRER :\n{messages_text}"

    response = get_anthropic_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,  # 1200 saturait : résumé tronqué en pleine phrase (constaté le 09/06)
        system="Tu es un assistant mémoire de coaching sportif. Sois concis, factuel et cumulatif.",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "max_tokens":
        # Résumé TRONQUÉ = perte de données garantie → on le rejette (l'appelant gardera l'ancien).
        print(f"[compress] TRONQUÉ (max_tokens atteint) pour {user_number} — résumé rejeté, ancien conservé")
        return ""
    return response.content[0].text


# ════════════════════════════════════════════════════════════════════
# TOOL USE — Willy VA CHERCHER l'info au lieu d'être gavé (sélection intelligente)
# Socle léger toujours injecté + outils appelés à la demande par le modèle.
# ════════════════════════════════════════════════════════════════════
TOOLS_SPEC = [
    {
        "name": "historique_exercice",
        "description": "Renvoie l'historique chiffré d'un exercice/benchmark de Louis dans le temps (ex: back squat, allure Z2, fractionné, bench press, deadlift, murph, push press...). À utiliser pour répondre à 'est-ce que je progresse sur X', comparer des charges/temps/allures sur plusieurs séances.",
        "input_schema": {"type": "object", "properties": {"nom": {"type": "string", "description": "nom de l'exercice ou benchmark recherché (ex: 'squat', 'z2', 'fractionné', 'bench')"}}, "required": ["nom"]},
    },
    {
        "name": "tendance_aerobie",
        "description": "Renvoie la tendance aérobie de Louis : Efficiency Factor (vitesse/FC) sur ses courses récentes. À utiliser pour 'est-ce que je progresse en Z2 / en aérobie', juger si la FC descend à allure donnée.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "seances_detaillees",
        "description": "Renvoie le détail des séances réalisées par Louis sur les N derniers jours (utile au-delà des 4 derniers jours déjà dans le contexte). À utiliser pour analyser une semaine passée précise ou un bilan sur plusieurs semaines.",
        "input_schema": {"type": "object", "properties": {"depuis_jours": {"type": "integer", "description": "nombre de jours d'historique à remonter (ex: 14, 30)"}}, "required": ["depuis_jours"]},
    },
    {
        "name": "carnet_complet",
        "description": "Renvoie TOUT le carnet de chiffres de Louis (tous ses benchmarks/PR). À n'utiliser que pour une revue complète ou une donnée que les autres outils ne couvrent pas.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _tool_historique_exercice(user_number: str, nom: str) -> str:
    bench = (athlete_data.get(user_number) or {}).get("benchmarks") or {}
    nl = (nom or "").lower().strip()
    matches = {k: v for k, v in bench.items() if nl and (nl in k.lower() or k.lower() in nl)}
    if not matches and nl:
        for k, v in bench.items():
            if any(w and w in k.lower() for w in nl.split()):
                matches[k] = v
    if not matches:
        return f"Aucun historique pour '{nom}'. Exercices disponibles : {', '.join(sorted(bench.keys())) or '(carnet vide)'}"
    lines = []
    for k, series in matches.items():
        pts = ", ".join(f"{p.get('date', '?')}: {p.get('value', '')}" for p in series)
        lines.append(f"- {k} → {pts}")
    return "\n".join(lines)


def execute_tool(user_number: str, name: str, inp: dict) -> str:
    """Exécute un outil demandé par le modèle. Ne lève jamais."""
    try:
        inp = inp or {}
        if name == "historique_exercice":
            return _tool_historique_exercice(user_number, inp.get("nom", ""))
        if name == "tendance_aerobie":
            return format_aerobic_trend(user_number) or "Pas assez de données aérobies exploitables (besoin allure + FC)."
        if name == "seances_detaillees":
            j = int(inp.get("depuis_jours", 30) or 30)
            paris = pytz.timezone("Europe/Paris")
            end = datetime.now(paris).strftime("%Y-%m-%d")
            start = (datetime.now(paris) - timedelta(days=j)).strftime("%Y-%m-%d")
            return format_realise(user_number, start, end) or f"Aucune séance loguée sur les {j} derniers jours."
        if name == "carnet_complet":
            return format_athlete_data(user_number) or "Carnet vide."
        return f"Outil inconnu : {name}"
    except Exception as e:
        return f"Erreur outil {name}: {e}"


def build_socle(user_number: str, strava_data: str = "", wod_done: bool = False) -> str:
    """Le SOCLE : contexte léger TOUJOURS injecté (ce qu'un coach a toujours en tête)."""
    paris = pytz.timezone("Europe/Paris")
    now = datetime.now(paris)
    tomorrow = now + timedelta(days=1)
    heure = now.strftime("%H:%M")
    system = SYSTEM_PROMPT + (
        f"\n\n═══ CONTEXTE TEMPOREL ═══\n"
        f"- AUJOURD'HUI : {now.strftime('%A %d %B %Y')} — {heure}\n"
        f"- DEMAIN : {tomorrow.strftime('%A %d %B %Y')}\n"
        f"Quand Louis dit 'demain', c'est {tomorrow.strftime('%A')}. Pour une séance passée, recopie sa date depuis le réalisé."
    )
    if user_number in athlete_summaries:
        system += f"\n\n📋 Profil de Louis :\n{athlete_summaries[user_number]}"
    for block in (format_lecons(user_number), format_training_plan(user_number),
                  format_load_state(user_number), format_cycle(user_number),
                  format_realise_recent(user_number, days=7)):
        if block:
            system += f"\n\n{block}"
    if strava_data:
        system += f"\n\n{strava_data}\n\nExploite Strava quand c'est pertinent."
    if wod_done:
        system += (
            "\n\n⚡ WOD TERMINÉ : Louis vient de finir sa séance. Analyse immédiatement sa dernière activité "
            "(perf, allure, FC, comparaison), feedback précis + impact sur la suite. Termine par UNE ligne "
            "demandant le RPE /10 + la durée s'ils manquent (la séance est loguée automatiquement)."
        )
    system += (
        "\n\n🔧 OUTILS À DISPOSITION — appelle-les UNIQUEMENT si la question l'exige et que l'info n'est pas déjà ci-dessus :\n"
        "- historique_exercice(nom) : progression chiffrée d'un exo (squat, allure Z2, fractionné, bench…)\n"
        "- tendance_aerobie() : ton EF (vitesse/FC) pour juger la progression aérobie\n"
        "- seances_detaillees(depuis_jours) : réalisé détaillé au-delà des 4 derniers jours\n"
        "- carnet_complet() : tout ton carnet de chiffres\n"
        "Question simple/casual ou déjà couverte par le contexte → réponds DIRECT, sans outil."
    )
    system += (
        f"\n\n⏰ RAPPEL FINAL — DATE : nous sommes {now.strftime('%A %d %B %Y')}, {heure}. "
        f"Si ton raisonnement conclut un autre jour, c'est ton raisonnement qui est faux, pas cette date."
    )
    return system


def respond_with_tools(user_number: str, history: list, strava_data: str = "", wod_done: bool = False) -> str:
    """Boucle tool-use : socle léger + outils à la demande. Retourne le texte final. Ne lève jamais."""
    messages = list(history)
    last_text = ""
    # Socle construit UNE fois par message (pas par tour) : date cohérente sur toute la boucle
    # + préfixe stable → le prompt caching fonctionne entre les tours.
    socle = build_socle(user_number, strava_data, wod_done)
    # PROMPT CACHING (2 breakpoints) : SYSTEM_PROMPT seul = stable d'un message à l'autre
    # (~10% du prix en relecture) ; socle complet = stable entre les tours d'outils d'un même message.
    system_blocks = [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": socle[len(SYSTEM_PROMPT):], "cache_control": {"type": "ephemeral"}},
    ]
    n_tools, tot_in, tot_out, tot_cache = 0, 0, 0, 0

    def _log_usage(resp):
        nonlocal tot_in, tot_out, tot_cache
        u = getattr(resp, "usage", None)
        if u:
            tot_in += (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
            tot_cache += getattr(u, "cache_read_input_tokens", 0) or 0
            tot_out += u.output_tokens or 0

    try:
        for _ in range(4):  # max 4 tours d'outils
            resp = get_anthropic_client().messages.create(
                model="claude-sonnet-4-6", max_tokens=1500,
                system=system_blocks,
                tools=TOOLS_SPEC,
                messages=messages,
            )
            _log_usage(resp)
            txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            if txt:
                last_text = txt
            if resp.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if getattr(b, "type", "") == "tool_use":
                    out = execute_tool(user_number, b.name, b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
                    n_tools += 1
            messages.append({"role": "user", "content": results})
        else:
            # 4 tours épuisés et le modèle veut ENCORE un outil → appel final sans outils
            # pour forcer une conclusion (sinon Louis recevrait "…").
            resp = get_anthropic_client().messages.create(
                model="claude-sonnet-4-6", max_tokens=1500,
                system=system_blocks, messages=messages,
            )
            _log_usage(resp)
            txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            if txt:
                last_text = txt
        print(f"[usage] {user_number} : {n_tools} outil(s) · in={tot_in} (+{tot_cache} depuis cache) · out={tot_out}")
        return last_text or "…"
    except Exception as e:
        print(f"[tools] erreur respond_with_tools pour {user_number}: {e}")
        raise


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
        # RÉALISÉ au fil de l'eau : logue les séances déclarées par Louis (jamais bloquant)
        update_realise(user_number, history)
        history = history[-10:]
        conversation_histories[user_number] = history

    paris = pytz.timezone("Europe/Paris")
    now = datetime.now(paris)

    wod_done = any(kw in user_message.lower() for kw in ["wod terminé", "wod termine", "séance terminée", "seance terminee"])

    # Rafraîchit Strava si le cache a plus d'1h ou si Louis vient de finir une séance
    last_fetch = last_strava_fetch.get(user_number)
    stale = last_fetch is None or (now - last_fetch).total_seconds() >= 3600
    if stale or wod_done:
        fresh = get_strava_activities(user_number)
        if fresh:
            strava_cache[user_number] = fresh
            last_strava_fetch[user_number] = now
    strava_data = strava_cache.get(user_number, "")

    # TOOL USE : socle léger toujours injecté (build_socle) + Willy va chercher
    # le détail (carnet, EF, vieux réalisé) via des outils SEULEMENT quand la question l'exige.
    # Remplace l'ancien "gavage" où tous les blocs étaient collés à chaque message.
    assistant_message = respond_with_tools(user_number, history, strava_data, wod_done)
    history.append({"role": "assistant", "content": assistant_message})
    persist_set("conversation_histories", conversation_histories)
    return assistant_message


def _extract_lecon(message: str):
    """Détecte 'retiens/note/mémorise [: que ,] <leçon>' et renvoie le texte de la leçon, sinon None."""
    import re
    m = re.match(r"^\s*(retiens|note|mémorise|memorise)\b\s*(?:que|qu'|:|,)?\s*(.+)$",
                 message or "", re.IGNORECASE | re.DOTALL)
    if m:
        txt = m.group(2).strip()
        if len(txt) >= 5:
            return txt
    return None


def handle_retiens(sender_number: str, texte: str):
    """Grave une leçon validée par Louis (commande 'retiens : ...'). Confirme par WhatsApp."""
    try:
        with get_user_lock(sender_number):
            added = add_lecon(sender_number, texte, "louis")
        if added:
            send_whatsapp(sender_number, f"📒 Noté, je le retiens et je l'appliquerai dans tes programmes :\n« {texte} »")
        else:
            send_whatsapp(sender_number, "📒 Déjà dans tes leçons (ou trop court) — rien ajouté.")
    except Exception as e:
        print(f"[retiens] erreur pour {sender_number}: {e}")


def handle_sauvegarde(sender_number: str):
    """Commande 'sauvegarde' : logue immédiatement la conversation en cours (réalisé + carnet)."""
    try:
        with get_user_lock(sender_number):
            history = conversation_histories.get(sender_number) or []
            n_real = update_realise(sender_number, history)
            n_data = update_athlete_data(sender_number, history)
        if n_real or n_data:
            send_whatsapp(sender_number, f"✅ Sauvegardé : {n_real} séance(s) loguée(s), {n_data} chiffre(s) au carnet.")
        else:
            send_whatsapp(sender_number, "✅ Déjà à jour — rien de nouveau à enregistrer.")
    except Exception as e:
        print(f"[sauvegarde] erreur pour {sender_number}: {e}")


def _wants_bilan(message: str) -> bool:
    """Détecte une DEMANDE de bilan (tolérante : 'fais le bilan', 'lets go bilan', 'bilan stp'…).
    Exclut les questions SUR le bilan (finissent par ?), les débriefs de séance, et les
    mentions en passant sans mot d'intention ('on fera le bilan dimanche')."""
    low = (message or "").lower().strip()
    if "bilan" not in low or low.endswith("?"):
        return False
    if any(k in low for k in ("wod terminé", "wod termine", "séance terminée", "seance terminee")):
        return False
    if low in ("bilan", "le bilan"):
        return True
    words = set(low.replace(",", " ").replace("!", " ").replace(".", " ").split())
    intention = {"fais", "fait", "go", "lets", "let's", "vas-y", "vasy", "balance",
                 "envoie", "lance", "stp", "svp", "veux", "donne", "sors"}
    return len(message) <= 60 and bool(words & intention)


def capture_rpe_direct(user_number: str, message: str) -> bool:
    """RPE attrapé EN CODE (pas par le LLM) : si le message contient 'RPE 6' ou '7/10',
    attache le chiffre à la dernière séance sans RPE (≤ 48h). Déterministe — règle le cas
    'séance capturée d'abord, RPE donné ensuite' que l'extracteur refusait de re-traiter."""
    import re
    m = (re.search(r"\brpe\s*[:=]?\s*(\d{1,2}(?:[.,]\d)?)\b", message, re.IGNORECASE)
         or re.search(r"\b(\d{1,2}(?:[.,]\d)?)\s*/\s*10\b", message))
    if not m:
        return False
    try:
        val = float(m.group(1).replace(",", "."))
    except ValueError:
        return False
    if not 0 <= val <= 10:
        return False
    try:
        paris = pytz.timezone("Europe/Paris")
        cutoff = (datetime.now(paris) - timedelta(hours=48)).strftime("%Y-%m-%d")
        candidates = [s for s in (realise.get(user_number) or [])
                      if (s.get("date") or "") >= cutoff and not str(s.get("rpe") or "").strip()]
        if not candidates:
            return False
        target = max(candidates, key=lambda s: (s.get("date") or "", s.get("moment") or ""))
        target["rpe"] = f"{val:g}"
        persist_set("realise", realise)
        print(f"[rpe] capture directe pour {user_number} : {target.get('date')} {target.get('type')} → RPE {val:g}")
        return True
    except Exception as e:
        print(f"[rpe] erreur capture directe pour {user_number}: {e}")
        return False


def data_quality_flags(user_number: str, start: str, end: str) -> list:
    """SANITY-CHECK des données d'une semaine, EN CODE. Détecte ce qui doit faire tiquer
    (le vieux bilan a récité '8 séances sur 7 jours' sans broncher). Ne lève jamais."""
    try:
        sem = [s for s in (realise.get(user_number) or []) if start <= (s.get("date") or "") <= end]
        flags = []
        if len(sem) > 7:
            flags.append(f"{len(sem)} séances sur 7 jours — invraisemblable, doublon probable")
        # deux séances aux chiffres identiques dans la semaine = même séance loguée 2×
        seen = {}
        for s in sem:
            fp = _stats_fingerprint(s.get("donnees") or "")
            if fp and fp in seen:
                flags.append(f"{s.get('date')} {s.get('type')} = mêmes chiffres que {seen[fp]} — doublon probable")
            elif fp:
                seen[fp] = f"{s.get('date')} {s.get('type')}"
        # même type 2 jours d'affilée dont une entrée sans données = probable double log
        by_date = sorted(sem, key=lambda x: x.get("date") or "")
        for i, s in enumerate(by_date[:-1]):
            n = by_date[i + 1]
            try:
                adj = abs((datetime.strptime(n["date"], "%Y-%m-%d") - datetime.strptime(s["date"], "%Y-%m-%d")).days) == 1
            except (ValueError, KeyError):
                adj = False
            if adj and s.get("type") == n.get("type") and (not (s.get("donnees") or "").strip() or not (n.get("donnees") or "").strip()):
                flags.append(f"{s.get('type')} le {s.get('date')} ET le {n.get('date')} (dont une sans données) — même séance loguée 2 fois ?")
        # incohérence date/jour restante
        for s in sem:
            j = (s.get("jour") or "").strip().lower()
            if j in _JOURS_IDX:
                try:
                    if datetime.strptime(s["date"], "%Y-%m-%d").weekday() != _JOURS_IDX[j]:
                        flags.append(f"{s.get('date')} étiqueté '{j}' mais cette date n'est pas un {j} — date suspecte")
                except (ValueError, KeyError):
                    pass
        return flags
    except Exception as e:
        print(f"[quality] erreur data_quality_flags pour {user_number}: {e}")
        return []


def bloc_saison(user_number: str) -> str:
    """TRAJECTOIRE DE SAISON, calculée : EF moyen par mois sur les séances aérobies.
    C'est le zoom arrière qui manquait — la baisse de FC en Z2 est l'objectif de saison de Louis,
    le bilan doit capitaliser dessus au lieu de paniquer sur 2 sorties récentes."""
    try:
        mois = {}
        for s in (realise.get(user_number) or []):
            if not any(k in (s.get("type") or "").lower() for k in ("z2", "sortie longue", "seuil")):
                continue
            txt = f"{s.get('donnees', '')} {s.get('detail', '')}"
            pace, fc = _parse_pace_sec(txt), _parse_fc(txt)
            if not pace or not fc:
                continue
            m = (s.get("date") or "")[:7]
            if m:
                mois.setdefault(m, []).append(round((60000.0 / pace) / fc, 3))
        if len(mois) < 2:
            return ""
        lignes = [f"- {m} : EF moyen {sum(v)/len(v):.3f} (meilleur {max(v):.3f}, {len(v)} sorties)"
                  for m, v in sorted(mois.items())]
        return ("═══ TRAJECTOIRE DE SAISON — EF PAR MOIS (calculé ; ↑ = cœur qui descend à allure égale) ═══\n"
                + "\n".join(lignes)
                + "\nC'est la tendance de FOND qui compte (l'objectif de saison de Louis), pas les 2 dernières sorties.")
    except Exception as e:
        print(f"[saison] erreur bloc_saison pour {user_number}: {e}")
        return ""


def generate_bilan(sender_number: str, user_message: str):
    """Bilan hebdo via le MOTEUR DU CHAT (tool-use), pas le vieux pipeline weekly_summary.
    On fusionne le meilleur des deux :
    - du chat : lit la conversation → connaît les ajustements NÉGOCIÉS (fini l'accusation 'tu as sauté'),
      concis, EF tiré par outil, pas de template obèse ;
    - du pipeline : le tableau de faits VERROUILLÉ en Python (zéro hallucination de date/RPE), rien d'autre.
    Fenêtre = semaine lundi→dimanche TERMINÉE (fix du bug 'aujourd'hui-6' qui excluait le lundi si lancé après minuit).
    NE programme NI ne stocke S+1 : il propose de caler la semaine ensuite (fini l'auto-écriture du plan = fini le bordel).
    Ne lève jamais (thread d'arrière-plan)."""
    try:
        paris = pytz.timezone("Europe/Paris")
        now = datetime.now(paris)
        # dimanche de la semaine écoulée : aujourd'hui si on est dimanche, sinon le dimanche précédent
        offset = 0 if now.weekday() == 6 else now.weekday() + 1
        week_end = (now - timedelta(days=offset)).strftime("%Y-%m-%d")
        week_start = (now - timedelta(days=offset + 6)).strftime("%Y-%m-%d")

        with get_user_lock(sender_number):
            # flush : logue les tout derniers débriefs (ex : la sortie longue du jour) avant de lire le réalisé
            update_realise(sender_number, conversation_histories.get(sender_number) or [])
            table = format_realise_table(sender_number, week_start, week_end)
            flags = data_quality_flags(sender_number, week_start, week_end)
            qualite = ("\n⚠️ QUALITÉ DES DONNÉES — anomalies détectées, SIGNALE un doute au lieu d'affirmer :\n"
                       + "\n".join(f"- {f}" for f in flags) + "\n") if flags else ""
            instruction = (
                f"[MODE BILAN — semaine écoulée {week_start} → {week_end}] "
                "Le tableau daté des séances est DÉJÀ affiché à Louis : ne le re-liste pas, appuie-toi dessus.\n"
                "Ce que Louis attend d'un bilan, DANS CET ORDRE :\n"
                "1. SES PROGRÈS RÉELS d'abord — compare aux semaines/mois précédents (charges de force, allures, "
                "FC en Z2, régularité des formats). Si le deadlift remonte, si la sortie longue est excellente, si le "
                "bench progresse, si la FC descend à allure égale : DIS-LE, chiffres à l'appui. C'est ton métier de le voir.\n"
                "2. LA TRAJECTOIRE DE SAISON — utilise le bloc EF PAR MOIS ci-dessous : la baisse de FC en Z2 est SON "
                "objectif de saison, capitalise dessus. Ne juge JAMAIS la forme sur 2 sorties récentes.\n"
                "3. Ce qui mérite vigilance (alertes réelles), sans dramatiser une donnée isolée ou douteuse.\n"
                "4. Prévu vs réalisé : SEULEMENT si écart notable, 2 lignes max — et un ajustement négocié avec toi "
                "en conversation n'est PAS une faute, c'est un choix assumé.\n"
                "INTERDITS : re-lister les séances ; sections de remplissage (pas de « ce que je retiens » générique) ; "
                "verdict alarmiste bâti sur une donnée suspecte ; DÉCISION IMPOSÉE — toute proposition structurante "
                "(décharge, changement de plan) = les chiffres + une question, c'est Louis qui décide.\n"
                "Si un chiffre te paraît invraisemblable (ex : 8 séances sur 7 jours), tu le questionnes au lieu de le réciter.\n"
                f"{qualite}"
                "Termine par UNE ligne invitant Louis à répondre « fais la prog » quand il veut caler la semaine à venir. "
                "Tu ne programmes PAS la semaine dans ce bilan et tu ne prétends rien avoir enregistré.\n\n"
                f"TABLEAU DES FAITS (pour ton analyse — ne le recopie pas) :\n{table or '(aucune séance loguée)'}\n\n"
                f"{bloc_saison(sender_number)}"
            )
            history = list(conversation_histories.get(sender_number) or [])
            history.append({"role": "user", "content": instruction})
            strava_data = strava_cache.get(sender_number, "")
            analyse = respond_with_tools(sender_number, history, strava_data)
            # Persistance PROPRE : le vrai message de Louis + le bilan (jamais l'instruction synthétique)
            hist = conversation_histories.setdefault(sender_number, [])
            hist.append({"role": "user", "content": user_message})
            hist.append({"role": "assistant", "content": analyse})
            persist_set("conversation_histories", conversation_histories)

        entete = f"📊 BILAN SEMAINE — {week_start} au {week_end}"
        faits = f"## CE QUI S'EST PASSÉ (relevé automatique)\n{table}" if table else ""
        send_whatsapp(sender_number, "\n\n".join(x for x in [entete, faits, analyse] if x))
    except Exception as e:
        print(f"[bilan] ERREUR generate_bilan pour {sender_number}: {type(e).__name__}: {e}")


def build_dossier_prog(user_number: str) -> str:
    """DOSSIER DE PROG — tous les faits qu'un coach doit avoir sous les yeux AVANT de programmer.
    100% calculé (zéro LLM). Réutilise les moteurs existants (cycle/charge/EF/leçons) + 3 blocs de
    décision : progression par exo, variété des formats, alertes. Alimente generate_prog."""
    import re
    paris = pytz.timezone("Europe/Paris")
    now = datetime.now(paris)

    def days_ago(ds):
        try:
            return (now.date() - datetime.strptime(ds, "%Y-%m-%d").date()).days
        except Exception:
            return None

    # 1. progression / stagnation par exercice (le carnet qui parle)
    bench = (athlete_data.get(user_number) or {}).get("benchmarks") or {}
    prog_lines, stales = [], []
    for nom, serie in sorted(bench.items()):
        if not isinstance(serie, list) or not serie:
            continue
        pts = sorted(serie, key=lambda p: p.get("date") or "")
        evo = " → ".join(f"{p.get('value')} ({(p.get('date') or '')[5:]})" for p in pts[-3:])
        age = days_ago(pts[-1].get("date") or "")
        flag = ""
        if len(pts) == 1:
            flag = " (1 point — pas de tendance)"
        elif age is not None and age > 30:
            flag = f" ⚠️ pas re-testé depuis {age}j"
            stales.append(nom)
        prog_lines.append(f"- {nom} : {evo}{flag}")
    prog = "═══ PROGRESSION PAR EXERCICE (dernier point = état actuel) ═══\n" + "\n".join(prog_lines)
    if stales:
        prog += f"\n→ À RE-TESTER (>30j sans mesure) : {', '.join(stales[:6])}"

    # 2. variété des formats (21 derniers jours)
    cut21 = (now - timedelta(days=21)).strftime("%Y-%m-%d")
    par_type = {}
    for s in [s for s in (realise.get(user_number) or []) if (s.get("date") or "") >= cut21]:
        par_type.setdefault(s.get("type") or "?", []).append(s)
    var_lines = []
    for t, ss in sorted(par_type.items(), key=lambda kv: -len(kv[1])):
        details = [(x.get("detail") or "").strip().lower()[:45] for x in ss if (x.get("detail") or "").strip()]
        rep = " ⚠️ format identique répété — varier le stimulus ?" if len(ss) >= 3 and len(set(details)) <= 1 else ""
        var_lines.append(f"- {t} ×{len(ss)}{rep}")
    variete = ("═══ VARIÉTÉ (21 derniers jours) ═══\n" + "\n".join(var_lines) +
               "\nEn Fondations le SQUELETTE se répète mais le CONTENU doit progresser (allure cible, volume, "
               "charges, format du fractionné à varier ~toutes les 3-4 sem).")

    # 3. alertes (14 derniers jours)
    cut14 = (now - timedelta(days=14)).strftime("%Y-%m-%d")
    al = []
    for s in [s for s in (realise.get(user_number) or []) if (s.get("date") or "") >= cut14]:
        txt = f"{s.get('detail','')} {s.get('donnees','')}".lower()
        dt, ty = (s.get("date") or "")[5:], s.get("type") or ""
        if any(k in txt for k in ("non termin", "arrêt", "arret")):
            al.append(f"- {dt} {ty} : NON TERMINÉE → format trop dur ou fatigue ce jour-là")
        m = re.search(r"fc\s*(?:moy)?\s*(\d{3})", txt)
        if m and ty.lower() in ("z2", "sortie longue") and int(m.group(1)) >= 148:
            al.append(f"- {dt} {ty} : FC {m.group(1)} sur séance facile → dérive cardiaque (chaleur/fatigue ?)")
        try:
            if float(str(s.get("rpe") or 0).replace(",", ".")) >= 8:
                al.append(f"- {dt} {ty} : RPE {s.get('rpe')} → grosse sollicitation, surveiller la récup")
        except ValueError:
            pass
    alertes = "═══ ALERTES (14 derniers jours) ═══\n" + ("\n".join(al) if al else "- RAS")

    # sanity-check sur la semaine écoulée : si la donnée est douteuse, le dossier le dit
    fin = now.strftime("%Y-%m-%d")
    debut7 = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    flags = data_quality_flags(user_number, debut7, fin)
    qualite = ("⚠️ QUALITÉ DES DONNÉES — anomalies détectées, n'affirme rien de bâti dessus :\n"
               + "\n".join(f"- {f}" for f in flags)) if flags else ""
    blocs = [f"📁 DOSSIER DE PROG (calculé, {now.strftime('%A %d %B %H:%M')})",
             format_cycle(user_number), format_load_state(user_number), format_aerobic_trend(user_number),
             bloc_saison(user_number), prog, variete, alertes, qualite, format_lecons(user_number)]
    return "\n\n".join(b for b in blocs if b)


def generate_prog(sender_number: str, user_message: str):
    """MOTEUR DE PROG : programme la semaine à venir via dossier calculé + EXTENDED THINKING.
    Le modèle DÉROULE une checklist en brouillon (invisible) puis écrit une semaine où CHAQUE
    séance est justifiée par une donnée. Il PROPOSE (ne stocke pas — Louis valide ensuite).
    Lit la conversation → respecte les contraintes négociées. Ne lève jamais."""
    try:
        paris = pytz.timezone("Europe/Paris")
        now = datetime.now(paris)
        days_ahead = (0 - now.weekday()) % 7  # prochain lundi (aujourd'hui si on est lundi)
        wk_start = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        wk_end = (now + timedelta(days=days_ahead + 6)).strftime("%Y-%m-%d")
        with get_user_lock(sender_number):
            dossier = build_dossier_prog(sender_number)
            instruction = (
                f"[MODE PROGRAMMATION — semaine du {wk_start} (lundi) au {wk_end} (dimanche)]\n"
                "AVANT d'écrire un seul jour, déroule ta réflexion : que demande la phase ? que disent la charge "
                "et l'EF ? qu'est-ce qui doit progresser (exos figés / régressés du dossier) ? qu'est-ce qui est "
                "éculé (format à varier) ? quelles alertes ? quelles contraintes Louis a-t-il posées dans la conversation ?\n"
                "Puis écris la semaine, 7 jours, DENSE, sans blabla :\n"
                "- ≥1 séance de FORCE (pilier permanent), dosée selon la charge.\n"
                "- Respecte IMPÉRATIVEMENT les leçons ET les contraintes négociées avec Louis dans la conversation "
                "(ne les re-transgresse pas sous prétexte d'optimiser).\n"
                "- RÈGLE DURE : chaque jour se termine par « → » + LA donnée du dossier qui justifie ce choix "
                "(ex : « → deadlift régressé à retrouver », « → EF en baisse, volume Z2 », « → fractionné figé, varier »). "
                "Une séance non justifiable par une donnée ne doit PAS exister.\n"
                "- Tu PROPOSES : termine par « ça te va, ou on ajuste ? ». Ne prétends rien avoir enregistré.\n\n"
                f"{dossier}"
            )
            history = list(conversation_histories.get(sender_number) or [])
            history.append({"role": "user", "content": instruction})
            # budget 1500 = ~50s (le client prod coupe à 60s) ; timeout dédié 120s pour la marge.
            resp = get_anthropic_client().messages.create(
                model="claude-sonnet-4-6", max_tokens=3500,
                thinking={"type": "enabled", "budget_tokens": 1500},
                system=SYSTEM_PROMPT,
                messages=history,
                timeout=120,
            )
            prog = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            hist = conversation_histories.setdefault(sender_number, [])
            hist.append({"role": "user", "content": user_message})
            hist.append({"role": "assistant", "content": prog})
            persist_set("conversation_histories", conversation_histories)
        send_whatsapp(sender_number, prog or "…")
    except Exception as e:
        print(f"[prog] ERREUR generate_prog pour {sender_number}: {type(e).__name__}: {e}")


def _wants_prog(message: str) -> bool:
    """Détecte une demande de PROGRAMMATION de la semaine (route vers generate_prog)."""
    low = (message or "").lower().strip()
    if low.endswith("?") or any(k in low for k in ("wod terminé", "séance terminée", "seance terminee")):
        return False
    return any(k in low for k in ("fais la prog", "fais ma prog", "la prog de la semaine", "programme la semaine",
                                  "programme ma semaine", "cale la semaine", "cale ma semaine", "attaque la prog",
                                  "prog s+1", "fais la programmation"))


def process_message_async(sender_number: str, incoming_message: str):
    """
    Génère la réponse Willy en arrière-plan et l'envoie via Twilio REST API.
    Évite le timeout 15s de Twilio sur le webhook synchrone.
    """
    try:
        ai_response = get_ai_response(sender_number, incoming_message)
        # Le découpage est désormais centralisé dans send_whatsapp (split intelligent
        # ~1500 chars sur sauts de ligne) → plus aucun message tronqué.
        send_whatsapp(sender_number, ai_response)
        # AUTO-CAPTURE à la source (APRÈS l'envoi, pour ne pas retarder la réponse) :
        # si Louis a déclaré une séance ou donné un RPE, on logue le réalisé à chaud.
        low = incoming_message.lower()
        # 1) RPE seul → capture directe EN CODE sur la dernière séance sans RPE (fiable, instantané)
        with get_user_lock(sender_number):
            capture_rpe_direct(sender_number, incoming_message)
        # 2) débrief de séance → extraction LLM comme avant
        if any(k in low for k in ["wod terminé", "wod termine", "séance terminée", "seance terminee"]) \
                or "rpe" in low or "/10" in incoming_message:
            with get_user_lock(sender_number):
                update_realise(sender_number, conversation_histories.get(sender_number) or [])
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

    # Commande LEÇON : "retiens : ...", "note que ...", "mémorise : ..." → grave une règle de prépa
    lecon_txt = _extract_lecon(incoming_message)
    if lecon_txt:
        threading.Thread(target=handle_retiens, args=(sender_number, lecon_txt), daemon=True).start()
        return str(twiml)

    # Commande SAUVEGARDE : grave tout de suite la conversation en cours (réalisé + carnet)
    if incoming_message.lower().strip() in ["sauvegarde", "sauve", "save", "mémorise", "memorise"]:
        threading.Thread(target=handle_sauvegarde, args=(sender_number,), daemon=True).start()
        return str(twiml)

    # Commande BILAN : route vers generate_bilan = le MOTEUR DU CHAT (lit la conversation, concis,
    # faits verrouillés en Python) — et PLUS le vieux weekly_summary (template obèse + fenêtre datée
    # buggée + auto-écriture de la semaine, à l'origine du bilan raté du 06/07). weekly_summary reste
    # dispo via l'admin (run_weekly) pour tests, mais n'est plus le chemin de Louis.
    # Commande PROG : route vers generate_prog = dossier calculé + extended thinking (semaine
    # justifiée séance par séance). Détecté AVANT le bilan (formulations distinctes).
    if _wants_prog(incoming_message):
        threading.Thread(target=generate_prog, args=(sender_number, incoming_message), daemon=True).start()
        twiml.message("🏗️ Je cale la semaine — je réfléchis à fond (données + périodisation), ça arrive dans ~1 min.")
        return str(twiml)

    if _wants_bilan(incoming_message):
        threading.Thread(target=generate_bilan, args=(sender_number, incoming_message), daemon=True).start()
        twiml.message("🧮 Bilan en cours — je déroule l'analyse, ça arrive dans ~1 min.")
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
    # Écrase la mémoire prose → ADMIN_SECRET + mêmes garde-fous que la compression
    # automatique (backup avant écriture + validation anti-régression).
    data = request.get_json(silent=True) or {}
    if not _admin_auth_ok(data):
        return {"status": "unauthorized"}, 401
    number = data.get("number")
    if not number or number not in conversation_histories:
        return {"status": "error", "message": "Utilisateur introuvable ou historique vide"}, 404
    history = conversation_histories[number]
    if not history:
        return {"status": "error", "message": "Historique vide"}, 400
    with get_user_lock(number):
        old_summary = athlete_summaries.get(number, "")
        summary = compress_history(number, history)
        is_valid, reason = is_valid_summary(summary, old_summary)
        if not is_valid:
            return {"status": "rejected", "reason": reason, "old_kept": True}, 422
        if old_summary:
            backup_summary(number)
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
    if not _admin_auth_ok(data):
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
            "realise": realise.get(user, []),
            "realise_count": len(realise.get(user, [])),
            "load_state": compute_load_state(user),
            "load_state_formatted": format_load_state(user),
            "cycle_formatted": format_cycle(user),
            "lecons": (athlete_data.get(user) or {}).get("lecons", []),
            "lecons_formatted": format_lecons(user),
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

    if action == "run_checkin":
        # Déclenche le check-in pré-bilan à la demande (envoie un vrai WhatsApp à CE user).
        try:
            pre_bilan_checkin(only_user=user)
            return {"status": "ok", "ran_for": user}, 200
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500

    if action == "clean_carnet":
        # Nettoyage one-shot du carnet : supprime les entrées datées dans le FUTUR
        # (= des cibles/prévus qui ont pollué le carnet avant le fix d'extraction).
        cur = athlete_data.get(user)
        if not cur:
            return {"status": "ok", "removed": 0, "note": "carnet vide"}, 200
        today_str = datetime.now(pytz.timezone("Europe/Paris")).strftime("%Y-%m-%d")
        with get_user_lock(user):
            backup_athlete_data(user)
            removed = []
            for bucket in ("benchmarks", "body_metrics"):
                for name, series in list((cur.get(bucket) or {}).items()):
                    kept = [p for p in series if not _is_future_date(p.get("date") or "", today_str)]
                    if len(kept) != len(series):
                        removed.append({"name": name, "dropped": len(series) - len(kept)})
                    if kept:
                        cur[bucket][name] = kept
                    else:
                        del cur[bucket][name]
            athlete_data[user] = cur
            persist_set("athlete_data", athlete_data)
        return {"status": "ok", "removed": removed, "removed_total": sum(r["dropped"] for r in removed)}, 200

    if action == "set_realise" and "sessions" in data:
        # Sème / complète le réalisé d'une semaine (ex: corriger la semaine écoulée à la main).
        # Fusion ADDITIVE (dédup par date+type), backup avant écriture.
        with get_user_lock(user):
            existing = realise.get(user) or []
            merged, added = merge_realise(existing, data["sessions"])
            if added > 0:
                backup_realise(user)
                realise[user] = merged[-120:]
                persist_set("realise", realise)
        return {"status": "ok", "added": added, "total": len(realise.get(user, []))}, 200

    return {"status": "unknown action", "valid_actions": ["dump", "list_backups", "restore", "restore_from_backup", "run_weekly", "run_checkin", "clean_carnet", "set_realise"]}, 400


@app.route("/admin/backup", methods=["POST"])
def admin_backup():
    """
    Dump complet de la base (toutes les clés du store) pour backup externe.
    Utilisé par le workflow GitHub Actions de backup quotidien.
    """
    data = request.get_json(silent=True) or {}
    if not _admin_auth_ok(data):
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
    if not _admin_auth_ok(data):
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
        global strava_tokens, conversation_histories, athlete_summaries, athlete_data, training_plan, realise
        strava_tokens = persist_get("strava_tokens", {})
        conversation_histories = persist_get("conversation_histories", {})
        athlete_summaries = persist_get("athlete_summaries", {})
        athlete_data = persist_get("athlete_data", {})
        training_plan = persist_get("training_plan", {})
        realise = persist_get("realise", {})
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
    # Endpoint destructeur → protégé par ADMIN_SECRET (le repo et l'URL sont publics)
    data = request.get_json(silent=True) or {}
    if not _admin_auth_ok(data):
        return {"status": "unauthorized"}, 401
    number = data.get("number")
    if number and number in conversation_histories:
        with get_user_lock(number):
            del conversation_histories[number]
            persist_set("conversation_histories", conversation_histories)
        return {"status": "reset", "number": number}, 200
    return {"status": "not_found"}, 404


def split_message(message: str, limit: int = 1500) -> list:
    """
    Découpe un message long en segments <= limit caractères.
    Casse en priorité sur un saut de ligne (puis un espace) pour ne jamais
    couper en plein milieu d'une phrase/section. WhatsApp tronque les messages
    trop longs → ce découpage garantit que TOUT le contenu est livré.
    """
    message = message or ""
    if len(message) <= limit:
        return [message] if message else []
    chunks = []
    remaining = message
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n")
        if cut < limit * 0.5:  # pas de saut de ligne exploitable → tente un espace
            cut = window.rfind(" ")
        if cut < limit * 0.5:  # toujours rien → coupe brute
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n ")
    if remaining:
        chunks.append(remaining)
    return chunks


def send_whatsapp(to: str, message: str):
    """
    Envoie un message WhatsApp via Twilio REST.
    Découpe automatiquement les messages longs (WhatsApp tronque au-delà d'une
    certaine taille — c'est ce qui coupait les bilans "au deadlift").
    Gère gracieusement l'erreur 63038 (limite quotidienne sandbox trial dépassée)
    pour ne pas crasher tout le thread async.
    """
    client = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    segments = split_message(message)
    for seg in segments:
        _send_whatsapp_segment(client, to, seg)


def _send_whatsapp_segment(client, to: str, message: str):
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

    # S+1 = semaine à venir (lundi → dimanche). Calcule le VRAI lundi suivant quel que soit
    # le jour de déclenchement (le bilan est désormais à la demande, plus seulement le dimanche).
    next_monday = today + timedelta(days=((7 - today.weekday()) % 7) or 7)
    week_start = next_monday.strftime("%Y-%m-%d")
    week_end = (next_monday + timedelta(days=6)).strftime("%Y-%m-%d")
    # Semaine écoulée (= celle qu'on analyse) : lundi → dimanche (aujourd'hui)
    reviewed_start = (today - timedelta(days=6)).strftime("%Y-%m-%d")
    reviewed_end = today.strftime("%Y-%m-%d")

    for user_number, token_data in strava_tokens.items():
        if only_user and user_number != only_user:
            continue
        strava_data = get_strava_activities(user_number, limit=10, after=week_ago)
        # Flush de fraîcheur : on logue les tout derniers débriefs (ex: la sortie longue
        # du dimanche faite le jour même) AVANT de lire le réalisé, sinon le bilan la raterait.
        with get_user_lock(user_number):
            update_realise(user_number, conversation_histories.get(user_number) or [])
        # RÉALISÉ déclaré par Louis = source de vérité de la semaine écoulée
        realise_block = format_realise(user_number, reviewed_start, reviewed_end)
        # On lance le bilan s'il y a AU MOINS une source (réalisé déclaré OU Strava)
        if not strava_data and not realise_block:
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
        # TRAJECTOIRE : position dans la phase (semaine X/Y) + phase suivante.
        # Calculé en Python (le LLM est peu fiable sur les dates). Pas de compte à rebours.
        nxt_phase = get_next_phase(user_number, reviewed_end)
        traj_block = ""
        if phase:
            try:
                pd = datetime.strptime(phase.get("debut", ""), "%Y-%m-%d")
                pf = datetime.strptime(phase.get("fin", ""), "%Y-%m-%d")
                ws = datetime.strptime(week_start, "%Y-%m-%d")
                sem_num = max(1, (ws - pd).days // 7 + 1)
                total = max(1, (pf - pd).days // 7 + 1)
                traj_block = f"\nPhase {phase.get('nom', '')} : semaine {sem_num}/{total} — cible de phase : {phase.get('cible', '')}"
                if nxt_phase:
                    traj_block += f"\nPhase suivante : {nxt_phase.get('nom', '')} à partir du {nxt_phase.get('debut', '')}"
            except Exception:
                traj_block = ""

        prevu_block = ""
        if prevu and prevu.get("seances"):
            lignes = "\n".join(
                f"- {s.get('jour', '?')}: {s.get('type', '')} — {s.get('detail', '')}".rstrip(" —")
                for s in prevu["seances"]
            )
            prevu_block = (
                f"\n\n═══ CE QUI ÉTAIT PRÉVU CETTE SEMAINE (à confronter aux séances RÉALISÉES listées plus bas) ═══\n"
                f"{lignes}\n"
                f"Dans l'analyse quantitative, compare ce prévu aux séances réellement réalisées (source de vérité), "
                f"dis explicitement le taux de SUIVI du plan (ce qui a été fait / sauté / ajouté) et tire-en les conséquences pour S+1."
            )

        # FAITS générés EN PYTHON (jamais par le LLM) : tableau du réalisé + indicateur aérobie.
        # Le LLM ne fait que COMMENTER ces blocs verrouillés → fini le "sled du samedi".
        realise_table = format_realise_table(user_number, reviewed_start, reviewed_end)
        aero_block = format_aerobic_trend(user_number, reviewed_end)
        realise_section = (
            f"\n\n═══ SÉANCES RÉALISÉES (SOURCE DE VÉRITÉ — un tableau identique sera collé en tête du bilan) ═══\n"
            f"{realise_block}\n"
            f"⚠️ NE RÉÉCRIS PAS ce tableau et ne reliste PAS les séances jour par jour (c'est déjà fait automatiquement). "
            f"Tu t'appuies dessus pour ANALYSER. N'invente jamais une séance absente de cette liste ; "
            f"une séance prévue absente = elle a sauté, dis-le."
            if realise_block else
            f"\n\n═══ SÉANCES RÉALISÉES ═══\n"
            f"⚠️ Aucune séance déclarée cette semaine. Appuie-toi sur Strava (prudemment), ne déduis pas de muscu/WOD depuis Strava."
        )
        aero_section = f"\n\n{aero_block}" if aero_block else ""
        # ÉTAT DE CHARGE calculé : le dosage du programme S+1 doit s'y conformer
        load_block = format_load_state(user_number, reviewed_end)
        load_section = f"\n\n{load_block}\nLe PROGRAMME S+1 doit être DOSÉ en fonction de cet état (volume et intensités)." if load_block else ""
        # LEÇONS de la prépa (règles validées) → le programme S+1 doit les respecter
        lecons_block = format_lecons(user_number)
        lecons_section = f"\n\n{lecons_block}" if lecons_block else ""

        # CARNET DE SEMAINES : digest de la semaine écoulée + décision du type de S+1 (build/décharge)
        reviewed_digest = compute_week_digest(user_number, reviewed_start, reviewed_end)
        # Aperçu de l'historique INCLUANT la semaine qu'on va archiver, pour décider S+1 avant de pondre le programme
        _cur_type = (prevu.get("type_semaine") if isinstance(prevu, dict) else None) or "build"
        _hist_preview = ((training_plan.get(user_number) or {}).get("historique") or []) + [
            {"type_semaine": _cur_type, "ratio": reviewed_digest.get("ratio")}
        ]
        next_week_type = _decide_week_type_from_hist(_hist_preview)
        deload_section = ""
        if next_week_type == "décharge":
            deload_section = (
                f"\n\n⚠️ S+1 = SEMAINE DE DÉCHARGE (planifiée) : après {DELOAD_AFTER_BUILD_WEEKS} semaines de montée en charge "
                f"(ou surcharge détectée), réduis la charge d'environ {int(DELOAD_CHARGE_REDUCTION*100)}% — surtout le VOLUME, "
                f"garde un peu d'intensité courte pour ne pas s'endormir. C'est une semaine d'ABSORPTION pour rebondir. "
                f"Explique clairement à Louis POURQUOI c'est une décharge (les semaines de build derrière)."
            )

        prompt = (
            f"Tu es Willy Georges, coach Hyrox professionnel. Tu fais le bilan hebdomadaire de Louis.\n\n"
            f"Objectif : Milan Sub-60 (mi-décembre 2026). "
            f"Raisonne en PHASES du cycle. INTERDIT d'écrire un compte à rebours en jours (jamais de 'J-XXX').{phase_block}{prevu_block}"
            f"{realise_section}{aero_section}{load_section}{deload_section}{lecons_section}\n\n"
            f"═══ DONNÉES STRAVA (CROIS-CHECK cardio uniquement, PAS la source de vérité) ═══\n"
            f"Sert-toi de Strava pour préciser allures/FC/distances, PAS pour déterminer la liste des séances.\n{strava_data}\n\n"
            f"═══ MÉMOIRE PROFIL LOUIS ═══\n{summary}\n\n"
            f"La semaine S+1 va du {week_start} (lundi) au {week_end} (dimanche).\n\n"
            f"═══ STRUCTURE DU BILAN (le tableau factuel est DÉJÀ généré, ne le refais pas) ═══\n"
            f"Dense, technique, précis. Tu apportes de la valeur vers le Sub-60.\n\n"
            f"🧠 ANALYSE (bullets COURTS et scannables — les chiffres sont déjà dans les blocs ci-dessus, ne les re-raconte pas)\n"
            f"- Volume / distribution Z2-Force-WOD / charge (ratio donné) / prévu vs réalisé (ce qui a sauté ou été ajouté)\n"
            f"- Progrès via l'EF donné — verdict CHIFFRÉ, jamais 'stable' ; ce qui stagne / inquiète / signaux de surcharge\n"
            f"- ⚠️ ANTI-CONTRADICTION : ne félicite JAMAIS un fait que tu critiques ailleurs (partir plus vite que la cible = erreur, pas 'bonne forme')\n\n"
            f"💡 CE QUE TU RETIENS\n"
            f"- 1 à 3 enseignements CONCRETS de la semaine (1 ligne chacun), tirés de ce qui s'est réellement passé\n"
            f"- Si un enseignement mérite de devenir une règle permanente, propose à Louis de le graver : « tu veux que je le retienne ? réponds : retiens : … » (ne le stocke PAS toi-même, attends sa validation)\n\n"
            f"🧭 VERDICT (pas de J-XXX){traj_block}\n"
            f"- UNE phrase tranchée : 'sur la trajectoire du Sub-60' / 'en avance' / 'en retard', justifiée par l'EF et la charge, + où tu en es vs la cible de phase\n\n"
            f"🎯 PROGRAMME S+1 (jour par jour, du lundi {week_start} au dimanche {week_end})\n"
            f"Les 7 jours, chacun : jour + date, séance précise (durée/zone/allure/format/mouvements), rationale en 1 phrase. "
            f"≥1 séance de force (pilier permanent). Dose selon l'état de charge. RESPECTE IMPÉRATIVEMENT les leçons de la prépa (ci-dessus).\n\n"
            f"Ton : direct, technique, motivant. Tutoiement. Pas de flatterie creuse. Tu peux dire 'ma gueule' une fois si c'est sincère."
        )
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,  # bilan dense mais plus économe en segments WhatsApp
            messages=[{"role": "user", "content": prompt}],
        )
        bilan = response.content[0].text
        # Assemblage EN CODE : en-tête + tableau Python (faits verrouillés) + analyse LLM,
        # puis le programme S+1 en 2e message. Le LLM ne touche jamais aux faits bruts.
        entete = f"📊 BILAN SEMAINE — {reviewed_start} au {reviewed_end}"
        faits = f"## CE QUI S'EST PASSÉ (relevé automatique)\n{realise_table}" if realise_table else ""
        marker = "🎯 PROGRAMME S+1"
        idx = bilan.find(marker)
        if idx > 0:
            retro, programme = bilan[:idx].rstrip(), bilan[idx:].rstrip()
            send_whatsapp(user_number, "\n\n".join(x for x in [entete, faits, retro] if x))
            send_whatsapp(user_number, programme)
        else:
            send_whatsapp(user_number, "\n\n".join(x for x in [entete, faits, bilan] if x))

        # MÉMOIRE PLAN : extraire la semaine S+1 du bilan et la stocker (archive prévu/réalisé).
        # Sous lock per-user (cohérence avec les messages entrants). Ne bloque jamais l'envoi du bilan.
        try:
            phase_nom = phase.get("nom", "") if phase else ""
            week = extract_week_plan(user_number, bilan, week_start, week_end, phase_nom)
            if week and week.get("seances"):
                week["type_semaine"] = next_week_type  # cohérent avec le programme dosé ci-dessus
                with get_user_lock(user_number):
                    set_week_plan(user_number, week, realise_resume=(realise_block or strava_data)[:1000], digest=reviewed_digest)
                print(f"[plan] semaine S+1 ({next_week_type}) stockée pour {user_number} ({len(week['seances'])} séances)")
        except Exception as e:
            print(f"[plan] erreur stockage S+1 pour {user_number}: {e}")


def pre_bilan_checkin(only_user: str = None):
    """
    Dimanche 17h30 (30 min avant le bilan) : envoie à Louis la liste des séances
    LOGUÉES cette semaine et demande ce qui manque + les RPE. Garantit que le bilan
    de 18h tourne sur un réalisé complet. Ne lève jamais.
    """
    paris = pytz.timezone("Europe/Paris")
    today = datetime.now(paris)
    week_start = (today - timedelta(days=6)).strftime("%Y-%m-%d")
    week_end = today.strftime("%Y-%m-%d")
    for user_number in list(strava_tokens.keys()):
        if only_user and user_number != only_user:
            continue
        try:
            # Flush des tout derniers débriefs avant d'afficher la liste
            with get_user_lock(user_number):
                update_realise(user_number, conversation_histories.get(user_number) or [])
            logged = format_realise(user_number, week_start, week_end)
            if logged:
                msg = (
                    "🔍 Briefing de fin de semaine — voilà les séances que j'ai loguées :\n\n"
                    f"{logged}\n\n"
                    "Il manque quelque chose ? Balance les séances oubliées (+ ton RPE /10 si tu l'as). "
                    "Quand t'es prêt (après ta sortie longue), dis-moi 'fais le bilan' et je te le sors complet. 💪"
                )
            else:
                msg = (
                    "🔍 Briefing de fin de semaine : je n'ai AUCUNE séance loguée cette semaine. "
                    "Balance-moi un récap de ce que t'as fait (séances + RPE /10), "
                    "puis dis-moi 'fais le bilan' quand tu veux. 💪"
                )
            send_whatsapp(user_number, msg)
            print(f"[checkin] pré-bilan envoyé à {user_number}")
        except Exception as e:
            print(f"[checkin] erreur pour {user_number}: {e}")


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
                # RÉALISÉ : logue aussi les séances déclarées (garantit la capture même les semaines calmes)
                update_realise(user_number, conversation_histories.get(user_number) or [])
            except Exception as e:
                print(f"[struct] consolidation erreur pour {user_number}: {e}")
    print(f"[struct] consolidation quotidienne terminée — {total} point(s) au total")


scheduler = BackgroundScheduler(timezone=pytz.timezone("Europe/Paris"))
# Dimanche 18h : BRIEFING (check-in) automatique — résume la semaine loguée + RPE et demande
# ce qui manque. PAS de bilan complet auto (Louis fait souvent sa sortie longue à 18h) :
# le bilan complet se déclenche SUR DEMANDE de Louis quand il est dispo.
scheduler.add_job(pre_bilan_checkin, "cron", day_of_week="sun", hour=18, minute=0)
# Consolidation mémoire structurée chaque jour à 2h (Paris), avant le backup externe (3h UTC = 4-5h Paris)
scheduler.add_job(daily_consolidation, "cron", hour=2, minute=0)
scheduler.start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
