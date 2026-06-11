import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor

def get_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"], sslmode="require")

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL
                )
            """)
        conn.commit()

def db_ping():
    """
    Vérifie que la DB répond. LÈVE en cas d'échec — appelé au boot :
    mieux vaut crasher franchement (Render redémarre le service) que démarrer
    avec un état vide et écraser la base au premier persist_set.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")

def db_get(key: str, default=None):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM store WHERE key = %s", (key,))
                row = cur.fetchone()
                return row[0] if row else default
    except Exception:
        return default

def db_set(key: str, value):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO store (key, value) VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """, (key, json.dumps(value)))
            conn.commit()
    except Exception as e:
        # Échec d'écriture = divergence RAM/DB silencieuse → log BRUYANT pour les logs Render.
        # (Pas de json.dumps ici : si la valeur est non-sérialisable, le log crasherait aussi.)
        print(f"❌❌❌ [db] ÉCHEC D'ÉCRITURE clé '{key}' : {type(e).__name__}: {e} "
              f"— l'état mémoire et la base DIVERGENT jusqu'à la prochaine écriture réussie")

def db_dump_all() -> dict:
    """Retourne TOUTE la base (toutes les clés du store) sous forme de dict."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT key, value FROM store")
            rows = cur.fetchall()
            return {row[0]: row[1] for row in rows}

def db_restore_all(data: dict) -> int:
    """Réinjecte un dump complet dans le store (upsert clé par clé). Retourne le nb de clés écrites."""
    count = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for key, value in data.items():
                cur.execute("""
                    INSERT INTO store (key, value) VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """, (key, json.dumps(value)))
                count += 1
        conn.commit()
    return count
