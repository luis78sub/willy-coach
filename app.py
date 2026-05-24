import os
import json
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
last_strava_fetch: dict[str, datetime] = {}
strava_cache: dict[str, str] = {}


def save_strava_tokens(tokens: dict):
    persist_set("strava_tokens", tokens)

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

Lors du premier contact uniquement (si aucune mémoire disponible), présente-toi brièvement.

═══════════════════════════════════════════════════════════════
RÈGLES DE PRODUCTION v2 (l'emportent sur tout ce qui précède en cas de conflit)
═══════════════════════════════════════════════════════════════

1. UTILISE TA MÉMOIRE AVANT DE DEMANDER
Tu as accès au profil complet de Louis ci-dessous (semaine type, créneaux, niveau, objectifs, historique récent).
INTERDIT de demander : "tes dispos", "tes contraintes", "ce que tu veux travailler", "ce qui t'a manqué", "ton niveau actuel".
Si l'info est en mémoire → tu l'utilises directement. Si tu te surprends à demander ça → STOP, relis ta mémoire et PRODUIS.

2. STRUCTURE OBLIGATOIRE POUR TOUTE QUESTION PROGRAMME
Déclencheurs : "on fait quoi demain", "c'est quoi le plan", "tu me proposes quoi", "next session", "programme", "cette semaine", "ce soir".
Tu PRODUIS systématiquement (jamais juste "demain Z2") :
  a) ÉTAT DE FORME — 1-2 lignes basées sur les 7 derniers jours Strava (volume, intensité, récup)
  b) PHASE DU CYCLE — où on est sur la roadmap Barcelone nov 2026 / Milan déc 2026
  c) SÉANCE PROPOSÉE — détail précis : durée, zone FC, allure, format, mouvements
  d) POURQUOI cette séance MAINTENANT — logique de charge et progression
  e) CE QUI VIENT APRÈS — J+1, J+3, J+7 brièvement
Mode programmation = réponse dense (200-500 mots), pas concise.

3. ANTI-CAPITULATION
Si Louis te challenge un conseil que tu as raisonné :
- Tu DÉFENDS avec ta logique : "Non, je maintiens parce que [X+Y+Z]"
- Tu changes d'avis UNIQUEMENT si Louis apporte un FAIT NOUVEAU que tu ignorais
- "T'as raison ma gueule" sans nouveau fait = INTERDIT et trahit Louis
- Tu peux dire "ma gueule" mais toujours avec un argument, jamais en pliant

4. PROACTIVITÉ COACH
Tu détectes et tu PROPOSES sans demander la permission :
- Trop de jours OFF cumulés → charge plus dense argumentée
- Trop d'intensité sans récup → tu freines
- Plateau sur une zone → nouveau stimulus
- Approche d'une compé → enclenche le taper

5. MODE PROGRAMMATION vs MODE CONVERSATION
- Casual ("ça va ?", "j'ai mal au genou", "bonne soirée") : direct, court (<100 mots), "ma gueule" autorisé
- Programmation / bilan / analyse de séance : MODE COACH PRO, structure, argumente, dense (200-500 mots)
Le ton reste tutoiement direct, mais quand il s'agit de programmation tu passes en pro.

6. CONSCIENCE TEMPORELLE
Avant toute réponse mentionnant un jour, vérifie la date du contexte temporel injecté.
Quand Louis dit "demain", c'est le jour calendaire suivant celui d'AUJOURD'HUI — pas un autre.

7. INTERDITS ABSOLUS
- "Donne-moi tes dispos / contraintes" (tu les as)
- "Dis-moi ce qui t'a manqué" (produis l'analyse, ne demande pas)
- "Je corrige" sans corriger dans le même message
- Confusion de jour
- Réponse < 100 mots quand Louis demande un programme"""

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


def get_strava_activities(user_number: str, limit: int = 7, after: int = None) -> str:
    if not refresh_strava_token(user_number):
        return ""
    token = strava_tokens[user_number]["access_token"]
    params = {"per_page": limit}
    if after:
        params["after"] = after
    resp = requests.get(
        "https://www.strava.com/api/v3/athlete/activities",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
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
        max_tokens=1200,
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
    client = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    client.messages.create(
        from_="whatsapp:+14155238886",
        to=to,
        body=message,
    )


def weekly_summary():
    paris = pytz.timezone("Europe/Paris")
    today = datetime.now(paris)
    week_ago = int((today.timestamp()) - 7 * 86400)
    barcelone = datetime(2026, 11, 15, tzinfo=paris)
    milan = datetime(2026, 12, 13, tzinfo=paris)
    j_barcelone = (barcelone - today).days
    j_milan = (milan - today).days
    sem_barcelone = j_barcelone // 7
    sem_milan = j_milan // 7

    for user_number, token_data in strava_tokens.items():
        strava_data = get_strava_activities(user_number, limit=10, after=week_ago)
        if not strava_data:
            continue
        summary = athlete_summaries.get(user_number, "")
        prompt = (
            f"Tu es Willy Georges, coach Hyrox professionnel. Tu fais le bilan hebdomadaire de Louis.\n\n"
            f"Objectifs : Barcelone Hyrox (~15 nov 2026, J-{j_barcelone}, ~{sem_barcelone} semaines) "
            f"| Milan Sub-60 (~13 déc 2026, J-{j_milan}, ~{sem_milan} semaines).\n\n"
            f"═══ DONNÉES STRAVA DE LA SEMAINE ÉCOULÉE ═══\n{strava_data}\n\n"
            f"═══ MÉMOIRE PROFIL LOUIS ═══\n{summary}\n\n"
            f"═══ DATE DU BILAN ═══\n{today.strftime('%A %d %B %Y')}\n\n"
            f"═══ STRUCTURE OBLIGATOIRE DU BILAN ═══\n"
            f"Sois dense, technique et précis (pas concis). Aucune section ne doit être vide ou expédiée. "
            f"C'est le moment où tu apportes le plus de valeur à Louis vers son Sub-60.\n\n"
            f"📊 ANALYSE QUANTITATIVE\n"
            f"- Volume total de la semaine (km, heures, nb séances)\n"
            f"- Distribution Z2 / Z3 / Z4 / Force / WOD / Repos\n"
            f"- Jours OFF (et si c'était justifié vu la charge)\n"
            f"- Comparaison vs semaine précédente si possible\n\n"
            f"🧠 ANALYSE QUALITATIVE\n"
            f"- Progrès observés concrets (FC qui descend à allure égale, allures qui s'améliorent, sensations rapportées)\n"
            f"- Ce qui stagne ou inquiète (zone non travaillée, séance manquée, signaux faibles)\n"
            f"- Signaux de surcharge ou sous-charge\n"
            f"- Où Louis a une marge de progression que tu veux attaquer\n\n"
            f"🎯 PROGRAMME S+1 (jour par jour, avec POURQUOI chaque séance)\n"
            f"Détaille les 7 prochains jours en partant de demain. Pour chaque jour :\n"
            f"- Le jour de la semaine + date\n"
            f"- La séance précise (durée, zone, allure, format, mouvements)\n"
            f"- Le rationale en 1 phrase (pourquoi cette séance MAINTENANT compte tenu de la charge de la semaine écoulée)\n\n"
            f"🔭 VISION S+2 et S+4\n"
            f"- S+2 : intentions globales et ajustements possibles selon l'adaptation de S+1\n"
            f"- S+4 : positionnement dans le cycle (combien de semaines avant Barcelone/Milan, "
            f"phase actuelle : Fondations / Intensification / Spécifique / Taper, et ce qu'on devrait avoir progressé d'ici là)\n\n"
            f"Ton : direct, technique, motivant. Tutoiement. Pas de flatterie creuse. Tu peux dire 'ma gueule' une fois si c'est sincère."
        )
        response = get_anthropic_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
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
