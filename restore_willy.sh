#!/usr/bin/env bash
# Restauration de la base Willy depuis un backup chiffré (disaster recovery).
#
# Usage :
#   1. Télécharge l'artifact "willy-db-backup-XXXX" depuis l'onglet Actions de GitHub
#      (il contient backup.json.enc), puis dézippe-le dans le dossier courant.
#   2. Lance :
#        BACKUP_PASSPHRASE="<ta-passphrase>" ADMIN_SECRET="<ton-admin-secret>" \
#          ./restore_willy.sh backup.json.enc
#
# Le script déchiffre le backup puis le réinjecte dans la base via /admin/restore_all.

set -euo pipefail

ENC_FILE="${1:-backup.json.enc}"
URL="${WILLY_URL:-https://willy-coach.onrender.com}"

if [ -z "${BACKUP_PASSPHRASE:-}" ]; then
  echo "Erreur : définis BACKUP_PASSPHRASE (la passphrase du secret GitHub)."
  exit 1
fi
if [ -z "${ADMIN_SECRET:-}" ]; then
  echo "Erreur : définis ADMIN_SECRET (le secret admin défini dans Render)."
  exit 1
fi
if [ ! -f "$ENC_FILE" ]; then
  echo "Erreur : fichier '$ENC_FILE' introuvable."
  exit 1
fi

echo "Déchiffrement de $ENC_FILE ..."
openssl enc -d -aes-256-cbc -pbkdf2 \
  -in "$ENC_FILE" \
  -out backup_decrypted.json \
  -pass pass:"$BACKUP_PASSPHRASE"

echo "Extraction de la partie 'data' du dump ..."
python3 -c "import json,sys; d=json.load(open('backup_decrypted.json')); json.dump({'secret':'$ADMIN_SECRET','data':d.get('data',d)}, open('restore_payload.json','w'))"

echo "Réinjection dans la base via $URL/admin/restore_all ..."
curl -s -X POST "$URL/admin/restore_all" \
  -H "Content-Type: application/json" \
  --data-binary @restore_payload.json

echo ""
echo "Terminé. Pense à supprimer backup_decrypted.json et restore_payload.json (contiennent tes données en clair)."
