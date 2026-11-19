#!/bin/sh
# Usage : sh rd/agents_codex/veille.sh [depuis AAAA-MM-JJ]   -> rd/veille_actu/<date>.md
set -e
cd "$(dirname "$0")/../.."
A=rd/agents_codex; T=$(date '+%F'); O=rd/veille_actu/$T.md
DEPUIS=${1:-$(ls rd/veille_actu/20*.md 2>/dev/null | tail -1 | sed 's|.*/||; s|\.md||')}
DEPUIS=${DEPUIS:-$(date -d '-30 days' '+%F')}
P=$(cat $A/commun.md; sed "s|{DEPUIS}|$DEPUIS|" $A/veille_actu.md)
printf '%s' "$P" | bash _codex/garde_codex.sh --search exec -s read-only --skip-git-repo-check -o $O - > rd/veille_actu/$T.log 2>&1 || true
test -s $O || { echo "échec veille, voir rd/veille_actu/$T.log"; exit 1; }
echo "- $T | depuis $DEPUIS | $O" >> rd/veille_actu/INDEX.md
echo "fin veille $T"
