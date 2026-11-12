import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from make_transfo import (  # noqa: E402
    TEMPLATES,
    bench_sources,
    build_tasks,
    check,
    clean,
    corrupt,
    good_sentence,
    kinds_for,
    normalize,
    render,
    split_sentences,
)


def test_clean_and_split():
    assert clean("Paris est grand[1]. ") == "Paris est grand."
    assert clean("Il dit » bonjour.") == "Il dit bonjour."
    assert split_sentences("Le chat dort. Le chien court ! Fin") == ["Le chat dort.", "Le chien court !", "Fin"]


def test_good_sentence():
    assert good_sentence("Le petit chat dort sur le canapé.")
    assert not good_sentence("Trop court.")
    assert not good_sentence("Il est né en 1950 à Lyon dans le Rhône.")
    assert not good_sentence("Le mot (rare) est utilisé ici souvent.")
    assert not good_sentence("le chat dort sur le canapé du salon.")
    assert not good_sentence("Biographie Il est né le , dans la ville de Lyon.")
    assert not good_sentence("Tiens, je crois que c'est à toi, dit Lulu.")
    assert not good_sentence("Vie privée Il est le fils du ministre des Finances.")
    assert not good_sentence("Articulation calcanéo-cuboïdienne L'articulation est très mobile.")
    assert not good_sentence("Ouvrages sur Bervi Osip V.")
    assert good_sentence("Le chat de Paul mange dans la cuisine.")
    assert not good_sentence("Le le chat dort sur le canapé du salon.")
    assert "pluriel" not in kinds_for("Leurs limites sont définies par arrêtés ministériels.", "wiki_fr")
    assert "pluriel" in kinds_for("L'enfant mange une soupe aux champignons.", "lecarnet")
    assert "pluriel" not in kinds_for("Ce soir-là, Lulu mange une soupe aux champignons.", "lecarnet")


def test_kinds_applicability():
    assert "present" in kinds_for("Le garçon mangeait une pomme dans le jardin.", "lecarnet")
    assert "futur" not in kinds_for("Le garçon mangeait une pomme dans le jardin.", "lecarnet")
    assert "futur" in kinds_for("Le garçon mange une pomme dans le jardin.", "lecarnet")
    assert "pluriel" in kinds_for("Le garçon mange une pomme dans le jardin.", "lecarnet")
    assert "pluriel" not in kinds_for("Les garçons mangent une pomme dans le jardin.", "lecarnet")
    assert "pluriel" not in kinds_for("Pierre mange une pomme dans le jardin.", "lecarnet")
    assert "negation" not in kinds_for("Le garçon ne mange jamais de pomme au jardin.", "lecarnet")
    assert "politesse" in kinds_for("Donne-moi ce livre tout de suite !", "lecarnet")
    assert "simplifier" not in kinds_for("Le garçon mange une pomme dans le jardin.", "lecarnet")


def test_corrupt_changes_sentence_and_is_reproducible():
    s = "Les enfants sont allés à la plage et ils ont joué."
    a, b = corrupt(s, random.Random(1)), corrupt(s, random.Random(1))
    assert a == b and a != s
    assert corrupt("Xyz.", random.Random(1)) is None


def test_check_filters():
    task = {"type": "question", "entree": "Le chat dort sur le lit."}
    assert check(task, "Le chat dort-il sur le lit ?") == []
    assert "recopie" in check(task, "Le chat dort sur le lit.")
    long_q = {"type": "question", "entree": "Léo aimait beaucoup fabriquer des objets avec les choses qu'il trouvait dans la forêt."}
    assert check(long_q, "Léo aimait-il beaucoup fabriquer des objets avec les choses qu'il trouvait dans la forêt ?") == []
    assert "recopie" in check({"type": "paraphrase", "entree": long_q["entree"]}, long_q["entree"][:-1] + " !")
    assert "pas_de_?" in check(task, "Le chat dort sur le lit, vraiment.")
    assert check({"type": "negation", "entree": "Il mange."}, "Il mange encore.") == ["pas_negation"]
    assert "pas_futur" in check({"type": "futur", "entree": "Il mange une pomme."}, "Il mangeait une pomme.")
    assert check({"type": "futur", "entree": "Il mange une pomme."}, "Il mangera une pomme.") == []
    assert "plusieurs_phrases" in check({"type": "resume", "entree": "A b c d e f g h. I j k l m n."}, "A b. C d.")
    assert check(task, None) == ["vide"]


def test_build_tasks_unique_sentence_and_correction_target():
    sents = [{"source": "s", "texte": f"Le chat numéro {i} est sur le lit.", "types": ["question", "correction", "negation"]} for i in range(20)]
    tasks = build_tasks(sents, {"question": 5, "correction": 5, "negation": 5})
    ids = [t["id"].split("-")[1] for t in tasks]
    assert len(ids) == len(set(ids)) == 15
    for t in tasks:
        if t["type"] == "correction":
            assert t["sortie"] in {s["texte"] for s in sents} and t["entree"] != t["sortie"]


def test_render_contains_source_and_answer():
    rng = random.Random(0)
    task = {"type": "pluriel", "entree": "Le chat dort.", "sortie": "Les chats dorment."}
    for _ in range(20):
        user, assistant = render(task, rng)
        assert "Le chat dort." in user["content"] and user["role"] == "user"
        assert "Les chats dorment." in assistant["content"] and assistant["role"] == "assistant"
    assert set(TEMPLATES) >= {"politesse", "correction", "resume"}


def test_bench_sources_loaded():
    srcs = bench_sources()
    assert len(srcs) == 25 and normalize("Ouvre la porte.") in srcs


def test_post_filters():
    from make_transfo import glued, output_ok, source_ok

    vocab = {"dès": 50, "comme": 50, "c": 50}
    assert glued("Contexte historique Dès son accession au trône, il règne.", vocab)
    assert not glued("Il part à Paris avec Marie.", vocab)
    t = lambda kind, src: {"type": kind, "entree": src, "sortie": src}
    assert not source_ok(t("futur", "Son campus de est situé au sud de Waco."), vocab)
    assert not source_ok(t("futur", "Il naît à environ d'altitude dans les Alpes."), vocab)
    assert not source_ok(t("futur", "À sa sortie, est bien accueilli par la presse."), vocab)
    assert not source_ok(t("pluriel", "L'ensemble de son gouvernement est assermenté."), vocab)
    assert source_ok(t("pluriel", "Le chat dort sur le lit."), vocab)
    assert not source_ok(t("correction", "Le château a été construit au début du dans le style baroque."), vocab)
    assert not source_ok(t("correction", "Notes et références Liens externes Jean Diodati, dans le Dictionnaire."), vocab)
    assert not output_ok({"type": "politesse"}, "Accompagne-moi, je te montrerai.")
    assert not output_ok({"type": "politesse"}, "Léo, pourrais-tu faire attention.")
    assert output_ok({"type": "politesse"}, "Pourrais-tu fermer la porte, s'il te plaît ?")
    assert not output_ok({"type": "question"}, "Elle a-t-elle expliqué la règle ?")
    assert output_ok({"type": "question"}, "A-t-elle expliqué la règle ?")
