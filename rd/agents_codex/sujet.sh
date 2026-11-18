#!/bin/sh
# Usage : sh rd/agents_codex/sujet.sh <slug> "<sujet>"   -> rd/recherches/<slug>/rapport.md, analyse.md
set -e
. "$(dirname "$0")/modele.sh"
cd "$(dirname "$0")/../.."
A=rd/agents_codex; SLUG=$1; SUJET=$2; D=rd/recherches/$SLUG
mkdir -p $D
P=$(cat $A/commun.md; sed "s|{SUJET}|$SUJET|" $A/recherche_sujet.md)
echo "$(date '+%H:%M') recherche $SLUG"
test -s $D/rapport.md || { printf '%s' "$P" | codex --search exec -m "$CODEX_MODEL" -s read-only --skip-git-repo-check -o $D/rapport.md - > $D/recherche.log 2>&1 || true; }
test -s $D/rapport.md || { echo "échec recherche, voir $D/recherche.log"; exit 1; }
echo "$(date '+%H:%M') analyse $SLUG"
cat $D/rapport.md | codex --search exec -m "$CODEX_MODEL" -s read-only --skip-git-repo-check -o $D/analyse.md "$(cat $A/commun.md; sed "s|{SUJET}|$SUJET|" $A/analyse_sujet.md)" > $D/analyse.log 2>&1 || true
test -s $D/analyse.md || { echo "échec analyse, voir $D/analyse.log"; exit 1; }
V=$(grep -oE "(À INTÉGRER|À TESTER|À SURVEILLER|ÉCARTÉ).*" $D/analyse.md | tail -1)
echo "- $(date '+%F') | $SLUG | $SUJET | ${V:-verdict non trouvé} | $D/" >> rd/recherches/INDEX.md
echo "$(date '+%H:%M') fin $SLUG : $V"
