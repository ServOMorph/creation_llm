#!/bin/sh
# Usage : sh rd/agents_codex/lancer.sh <sujet>   (sujet : lecun | frugal)
set -e
cd "$(dirname "$0")/../.."
A=rd/agents_codex
S=$1
case $S in lecun) R=rd/veille_lecun_jepa.md; X=rd/analyse_lecun_jepa.md;; frugal) R=rd/veille_frugalite.md; X=rd/analyse_frugalite.md;; *) exit 2;; esac
echo "$(date '+%H:%M') recherche $S"
cat $A/commun.md $A/recherche_$S.md | bash _codex/garde_codex.sh --search exec -s read-only --skip-git-repo-check -o $R - > rd/agents_codex/$S.recherche.log 2>&1
test -s $R || { echo "rapport de recherche vide"; exit 1; }
echo "$(date '+%H:%M') analyse $S"
cat $R | bash _codex/garde_codex.sh --search exec -s read-only --skip-git-repo-check -o $X "$(cat $A/commun.md $A/analyse_$S.md)" > rd/agents_codex/$S.analyse.log 2>&1
echo "$(date '+%H:%M') fin $S : $R $X"
