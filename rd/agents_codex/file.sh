#!/bin/sh
# Traite rd/sujets_a_traiter.md ligne par ligne (« - slug | sujet ») ; s'arrête au premier échec (quota).
cd "$(dirname "$0")/../.."
LOCK=rd/agents_codex/.file.lock
mkdir "$LOCK" 2>/dev/null || { echo "file déjà en cours ($LOCK)"; exit 3; }
trap 'rmdir "$LOCK"' EXIT
while L=$(grep -m1 '^- [a-z0-9-]* |' rd/sujets_a_traiter.md); [ -n "$L" ]; do
  SLUG=$(echo "$L" | sed 's/^- \([^ ]*\) |.*/\1/'); SUJET=$(echo "$L" | sed 's/^- [^|]*| //')
  sh rd/agents_codex/sujet.sh "$SLUG" "$SUJET" || exit 1
  sed -i "/^- $SLUG |/d" rd/sujets_a_traiter.md
done
echo "file vide"
