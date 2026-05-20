import os
import json
import requests
from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient
import anthropic
from dotenv import load_dotenv
from datetime import datetime
import pytz
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

app = Flask(__name__)

USE_DB = bool(os.environ.get("DATABASE_URL"))

# JSON fallback (local dev)
TOKENS_FILE = os.path.join(os.path.dirname(__file__), "strava_tokens.json")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "conversation_histories.json")
SUMMARIES_FILE = os.path.join(os.path.dirname(__file__), "athlete_summaries.json")
HEALTH_FILE = os.path.join(os.path.dirname(__file__), "apple_health.json")


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
        return db_get(key, default) or default
    return load_json({"strava_tokens": TOKENS_FILE, "conversation_histories": HISTORY_FILE,
                      "athlete_summaries": SUMMARIES_FILE, "apple_health": HEALTH_FILE}.get(key, "")) or default or {}


def persist_set(key: str, value):
    if USE_DB:
        from db import db_set
        db_set(key, value)
    else:
        path = {"strava_tokens": TOKENS_FILE, "conversation_histories": HISTORY_FILE,
                "athlete_summaries": SUMMARIES_FILE, "apple_health": HEALTH_FILE}.get(key)
        if path:
            save_json(path, value)


if USE_DB:
    from db import init_db
    init_db()

strava_tokens: dict = persist_get("strava_tokens", {})
conversation_histories: dict = persist_get("conversation_histories", {})
athlete_summaries: dict = persist_get("athlete_summaries", {})
apple_health_data: dict = persist_get("apple_health", {})
last_strava_check: dict[str, str] = {}


def save_strava_tokens(tokens: dict):
    persist_set("strava_tokens", tokens)


def is_new_session(user_number: str) -> bool:
    today = datetime.now().strftime("%Y-%m-%d")
    if last_strava_check.get(user_number) != today:
        last_strava_check[user_number] = today
        return True
    return False

SYSTEM_PROMPT = """Tu es Willy Georges, athlète CrossFit & Hyrox français, coach et fondateur de WYS Training.

🏆 Ton palmarès réel :
- Premier français qualifié aux CrossFit Games en individuel (3 participations)
- 9ème place aux CrossFit Games 2018 (première participation)
- Champion de France de CrossFit (Fittest Man in France) 2017, 2018, 2019, 2020
- Multiple vainqueur du French Throwdown (championnats d'Europe CrossFit)
- Fondateur de la box WYS à Châtenois et de WYS Training (programmation en ligne)
- Retraite compétitive CrossFit annoncée après les quarts de finale 2023
- Partenaire officiel HYROX France

🧠 Ta philosophie d'entraînement (méthode WYS) :
- Progression structurée en 3 cycles : Fondations (sem 1-4) → Intensification/puissance (sem 5-8) → Spécifique/simulation (sem 9-12)
- Minimum 3 séances/semaine pour ressentir les effets
- Maîtrise mentale sous fatigue : fixer un point, relâcher la mâchoire, sourire pour diminuer la tension
- Équilibre force fonctionnelle + endurance + puissance
- Importance du Z2 pour la base aérobie

🎯 Ton approche Hyrox :
- Gérer la douleur et rester lucide sous fatigue
- Préparer chaque station individuellement et en enchaînement
- La course entre stations est aussi importante que les stations elles-mêmes

L'athlète Louis est déjà d'un bon niveau, s'entraîne régulièrement et a une bonne connaissance du sport — va droit au but, pas de condescendance.

Ton rôle :
- Créer des programmes personnalisés basés sur la mémoire et les données Strava de Louis
- Conseils nutritionnels pré/post effort
- Prévention des blessures et technique des mouvements
- Motiver et suivre la progression vers l'objectif

Ton style :
- Chaleureux, direct, motivant — tu parles comme Willy Georges, pas comme un bot
- Concis (WhatsApp, max 300 mots)
- N'utilise JAMAIS les données que tu as déjà en mémoire pour poser des questions
- Pose des questions uniquement pour des informations vraiment manquantes

Lors du premier contact uniquement (si aucune mémoire disponible), présente-toi brièvement."""

MAX_HISTORY = 20


def get_anthropic_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def refresh_strava_token(user_number: str) -> bool:
    token_data = strava_tokens.get(user_number)
    if not token_data:
        return False
    if token_data.get("expires_at", 0) > datetime.now().timestamp():
        return True  # still valid
    resp = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": os.environ["STRAVA_CLIENT_ID"],
        "client_secret": os.environ["STRAVA_CLIENT_SECRET"],
        "grant_type": "refresh_token",
        "refresh_token": token_data["refresh_token"],
    })
    if resp.ok:
        strava_tokens[user_number] = resp.json()
        save_strava_tokens(strava_tokens)
        return True
    return False


def get_strava_activities(user_number: str, limit: int = 7) -> str:
    if not refresh_strava_token(user_number):
        return ""
    token = strava_tokens[user_number]["access_token"]
    resp = requests.get(
        "https://www.strava.com/api/v3/athlete/activities",
        headers={"Authorization": f"Bearer {token}"},
        params={"per_page": limit},
    )
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
        "Tu es un assistant mémoire pour un coach sportif IA. "
        "Mets à jour le profil de suivi de l'athlète Louis en intégrant les nouveaux échanges. "
        "Le résumé final doit être cumulatif : il intègre tout ce qui a été dit depuis le début. "
        "Structure en bullet points concis :\n"
        "- Profil Louis (niveau, objectifs, contraintes, historique sportif)\n"
        "- Programmes donnés et progression observée\n"
        "- Points de vigilance (blessures, fatigue, points faibles)\n"
        "- Dernières recommandations Willy\n"
        "- Ce que Louis a partagé d'important sur sa vie/agenda/motivation\n\n"
    )
    if existing_summary:
        prompt += f"RÉSUMÉ ACTUEL À METTRE À JOUR :\n{existing_summary}\n\n"
    prompt += f"NOUVEAUX ÉCHANGES À INTÉGRER :\n{messages_text}"

    response = get_anthropic_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system="Tu es un assistant mémoire de coaching sportif. Sois concis, factuel et cumulatif.",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def get_ai_response(user_number: str, user_message: str) -> str:
    if user_number not in conversation_histories:
        conversation_histories[user_number] = []

    history = conversation_histories[user_number]
    history.append({"role": "user", "content": user_message})

    if len(history) > MAX_HISTORY:
        athlete_summaries[user_number] = compress_history(user_number, history[:-10])
        history = history[-10:]
        conversation_histories[user_number] = history
        persist_set("athlete_summaries", athlete_summaries)

    paris = pytz.timezone("Europe/Paris")
    today = datetime.now(paris)
    heure = today.strftime("%H:%M")
    moment = "matin" if today.hour < 12 else "après-midi" if today.hour < 18 else "soir"
    date_context = (
        f"\n\n📅 Contexte temporel (heure France) :\n"
        f"- Aujourd'hui : {today.strftime('%A %d %B %Y')} — {heure} ({moment})\n"
        f"Adapte tes conseils à l'heure et au moment de la journée."
    )

    new_session = is_new_session(user_number)

    system = SYSTEM_PROMPT + date_context
    if user_number in athlete_summaries:
        system += f"\n\n📋 Mémoire de tes échanges précédents avec Louis :\n{athlete_summaries[user_number]}"
    wod_done = any(kw in user_message.lower() for kw in ["wod terminé", "wod termine", "séance terminée", "seance terminee"])

    health = apple_health_data.get(user_number)
    if health:
        system += (
            f"\n\n⌚ Données Apple Watch (nuit/matin) :\n"
            f"- Sommeil : {health.get('sleep', 'N/A')}\n"
            f"- FC repos : {health.get('resting_hr', 'N/A')} bpm\n"
            f"- HRV : {health.get('hrv', 'N/A')} ms\n"
            f"- Pas hier : {health.get('steps', 'N/A')}\n"
            f"- Calories actives : {health.get('calories', 'N/A')} kcal\n"
            f"Utilise ces données pour évaluer l'état de récupération de Louis et adapter l'intensité du jour."
        )

    strava_data = get_strava_activities(user_number)
    if strava_data:
        system += f"\n\n{strava_data}\n\nUtilise ces données pour personnaliser tes conseils si pertinent."
        if new_session:
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
        max_tokens=1024,
        system=system,
        messages=history,
    )

    assistant_message = next(b.text for b in response.content if hasattr(b, "text"))
    history.append({"role": "assistant", "content": assistant_message})
    persist_set("conversation_histories", conversation_histories)
    return assistant_message


@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_message = request.form.get("Body", "").strip()
    sender_number = request.form.get("From", "").replace("whatsapp: ", "whatsapp:+")

    if not incoming_message:
        return str(MessagingResponse())

    twiml = MessagingResponse()

    # Commande de connexion Strava
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

    ai_response = get_ai_response(sender_number, incoming_message)

    if len(ai_response) > 1500:
        for i in range(0, len(ai_response), 1500):
            twiml.message(ai_response[i:i+1500])
    else:
        twiml.message(ai_response)

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

    resp = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": os.environ["STRAVA_CLIENT_ID"],
        "client_secret": os.environ["STRAVA_CLIENT_SECRET"],
        "code": code,
        "grant_type": "authorization_code",
    })

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


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "active_users": len(conversation_histories),
        "strava_connected": list(strava_tokens.keys()),
    }, 200


@app.route("/apple-health", methods=["POST"])
def apple_health():
    data = request.get_json()
    if not data:
        return {"status": "error"}, 400
    user_number = data.get("number", "whatsapp:+33618582944")
    apple_health_data[user_number] = {
        "sleep": data.get("sleep"),
        "resting_hr": data.get("resting_hr"),
        "hrv": data.get("hrv"),
        "steps": data.get("steps"),
        "calories": data.get("calories"),
        "updated_at": datetime.now().isoformat(),
    }
    persist_set("apple_health", apple_health_data)
    return {"status": "ok"}, 200


@app.route("/reset", methods=["POST"])
def reset_conversation():
    data = request.get_json()
    number = data.get("number") if data else None
    if number and number in conversation_histories:
        del conversation_histories[number]
        return {"status": "reset", "number": number}, 200
    return {"status": "not_found"}, 404


def send_whatsapp(to: str, message: str):
    client = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    client.messages.create(
        from_="whatsapp:+14155238886",
        to=to,
        body=message,
    )


def weekly_summary():
    paris = pytz.timezone("Europe/Paris")
    today = datetime.now(paris)
    for user_number, token_data in strava_tokens.items():
        strava_data = get_strava_activities(user_number, limit=10)
        if not strava_data:
            continue
        summary = athlete_summaries.get(user_number, "")
        prompt = (
            f"Tu es Willy Georges, coach Hyrox. Fais un bilan hebdomadaire motivant pour Louis.\n\n"
            f"Données Strava de la semaine :\n{strava_data}\n\n"
            f"Profil Louis :\n{summary}\n\n"
            f"Date : {today.strftime('%A %d %B %Y')}\n\n"
            f"Structure ton bilan en 3 parties : ✅ Ce qui s'est bien passé | ⚠️ Ce à améliorer | 🎯 Plan de la semaine prochaine. "
            f"Max 300 mots, ton WhatsApp direct et motivant."
        )
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        bilan = response.content[0].text
        send_whatsapp(user_number, f"📊 Bilan de la semaine :\n\n{bilan}")


# Scheduler — bilan automatique chaque dimanche à 18h heure de Paris
scheduler = BackgroundScheduler(timezone=pytz.timezone("Europe/Paris"))
scheduler.add_job(weekly_summary, "cron", day_of_week="sun", hour=18, minute=0)
scheduler.start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
