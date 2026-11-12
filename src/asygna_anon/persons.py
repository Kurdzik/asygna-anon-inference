"""One number per person (API.md A5): a numbering key per name part with the Polish case endings stripped by
a rule, so "Jan Kowalski" and "Jana Kowalskiego" share a label. A key only compares forms within one request;
it is never shown. A surname alone shares the label of the one full name of the request ending in it, and
stays its own with two such names; a first name alone stays its own, as it is too often another person's.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

WORD = re.compile(r"[^\W\d_]+")
# A surname declined as an adjective keeps its gender: a Kowalska is not the Kowalski of the same request.
# Three letters before the ending, as Jacek's Jacka is no such surname
ADJECTIVE_MAN = re.compile(r"(\w{3,}(?:s|c|dz)k)(?:iego|iemu|im|i)")
ADJECTIVE_WOMAN = re.compile(r"(\w{3,}(?:s|c|dz)k)(?:iej|a|ą)")
# Endings that change the stem's last consonant: Ryszardzie, Małgorzacie, Kindze, Agnieszce, Piotrze
ALTERED = {"dzie": "d", "cie": "t", "dze": "g", "ce": "k", "rze": "r"}
ALTERED_ENDING = re.compile(r"(dzie|cie|dze|ce|rze)$")
# Longest first by the search: Orzeszkowej, Jerzego, Janowi, Markiem, Annie, Anna, Annę. An i before a vowel
# stays in the stem (Zofia, Zofii, Zofię), or Wiktoria would be the Wiktor of the same request
ENDING = re.compile(r"(?:(?<=[wn])ej|iego|iemu|ego|emu|ych|ymi|ym|owi|iem|em|ie|iu|a|ą|ę|e|i|o|u|y)$")
SHORTEST_STEM = 2


def stem(word: str) -> str:
    """A casefolded name part without its case ending: Kowalskiego -> kowalski, Kowalskiej -> kowalska,
    Jana -> jan, Pawła -> pawl, Marka -> mark, Kacpra -> kacpr, Piotrze -> piotr."""
    if man := ADJECTIVE_MAN.fullmatch(word):
        return f"{man.group(1)}i"
    if woman := ADJECTIVE_WOMAN.fullmatch(word):
        return f"{woman.group(1)}a"
    stripped = ALTERED_ENDING.sub(lambda found: ALTERED[found.group(1)], word)
    if stripped == word:
        stripped = ENDING.sub("", word)
    if len(stripped) < SHORTEST_STEM:
        stripped = word
    stripped = re.sub(r"ł$", "l", stripped)  # Paweł, Michale
    # The e the other forms drop: Marek, Paweł, Kacper. Not Aleksander, whose Aleksandra is as often a woman
    return re.sub(r"(?<!and)e([klr])$", r"\1", stripped)


def merged(names: Iterable[str]) -> dict[str, str]:
    """Each casefolded name of one request and its numbering key."""
    parts = {name: [stem(word) for word in WORD.findall(name)] or [name] for name in names}
    keys = {name: tuple(sorted(set(stems))) for name, stems in parts.items()}
    holders: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for name, stems in parts.items():
        if len(keys[name]) > 1:
            holders[stems[-1]].add(keys[name])

    def whole(key: tuple[str, ...]) -> tuple[str, ...]:
        found = holders[key[0]] if len(key) == 1 else set()
        return next(iter(found)) if len(found) == 1 else key

    return {name: " ".join(whole(key)) for name, key in keys.items()}
