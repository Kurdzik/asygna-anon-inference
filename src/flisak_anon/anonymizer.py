"""The anonymisation pipeline: the regex stage, the name stage and the placeholders' numbering.

Per request: regex matches (checksum-validated) become indexed placeholders, NER runs on that materialized
text (`pipeline("ner", aggregation_strategy="simple")`, `batch_size=8`), and entities in ENTITY_MAPPING
scoring over 0.4 are snapped around placeholders and applied with the rightmost span winning an overlap.
`stages` also maps every applied span back to the original text; server.py calls it with `api=True` (API.md
api.4). The 64-token overlap is `windowed`'s, not the pipeline's (tokenizers 0.23 returns only the first
overflow window), and EMAIL_REGEX bounds the local part to 64 characters and each domain label to 63, since
an unbounded pattern is quadratic on "a.a.a...". ENGLISH adds English phones, the US SSN, the UK NI number
and card numbers (API.md api.3).
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

REQUIRED_MODEL_FILES = (
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
ENTITY_MAPPING = {
    "nam_liv_person": "PERSON",
    "nam_liv_character": "PERSON",
    "nam_liv_god": "PERSON",
    "nam_adj_person": "PERSON",
    "nam_num_phone": "PHONE",
    "PER": "PERSON",  # flisak-anon-en's one label (O, B-PER, I-PER)
}
NER_SCORE_THRESHOLD = 0.4
NER_KWARGS = {"batch_size": 8}
STRIDE = 64  # tokens two windows share, as the service's stride=64

EMAIL_REGEX = re.compile(
    r"\b[a-zA-Z0-9](?:[a-zA-Z0-9._%+-]{0,62}[a-zA-Z0-9])?@"
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"
    r"\.[a-zA-Z]{2,63}\b",
    re.IGNORECASE,
)
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b")
POLISH_BANK_ACCOUNT_REGEX = re.compile(
    r"\b(?:PL\s?)?(\d{2})\s?(\d{4})\s?(\d{4})\s?(\d{4})\s?(\d{4})\s?(\d{4})\s?(\d{4})\b",
    re.IGNORECASE,
)
IBAN_REGEX = re.compile(r"\b([A-Z]{2})(\d{2})([A-Z0-9]{11,30})\b", re.IGNORECASE)
PESEL_REGEX = re.compile(r"\b\d{11}\b")
ID_CARD_REGEX = re.compile(r"\b[A-Z]{3}\s?\d{6}\b", re.IGNORECASE)
PHONE_REGEX = re.compile(
    r"(?<![\d+])(?:(?:\+|00)?48[\s-]?)?"
    r"(?:\(\d{2}\)[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}"
    r"|\d{3}[\s-]?\d{3}[\s-]?\d{3}"
    r"|\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2})"
    r"(?!\d)"
)
# ANON_LANGUAGE=en adds the English rules below to the Polish ones (API.md api.3); pl, the default, runs the
# Polish ones alone, as the service does
ENGLISH = os.environ.get("ANON_LANGUAGE", "pl") == "en"
ENGLISH_PHONE_REGEX = re.compile(
    r"(?<![\w+/-])(?:"
    # E.164 but +48, the Polish rule's: +44 20 7946 0958, +44 (0)20 7946 0958, +1 (212) 555-0198, +12125550198
    r"\+(?!48)\d+(?:[ .-]?\(\d{1,4}\)[ .-]?\d+|[ .-]\d+)*"
    # UK and Irish national, never 00 (a call abroad): (020) 7946 0958, 07700 900123, 087 123 4567
    r"|\(0[1-9]\d{0,3}\) ?\d{3,6}(?:[ -]\d{3,4})?"
    r"|0[1-9]\d{0,3}[ -]\d{3,6}(?:[ -]\d{3,4})?"
    r"|07\d{9}"  # a UK mobile written in one piece: 07700900123
    # US and Canada: (212) 555-0198, 212-555-0198, 212.555.0198, 650 723 1976, 1-800-555-0199
    r"|(?:1[ .-]?)?(?:\([2-9]\d{2}\) ?|[2-9]\d{2}[ .-])[2-9]\d{2}[ .-]\d{4}"
    r")(?!\d)"
)
SSN_REGEX = re.compile(r"(?<![\d/-])(\d{3})([ -])(\d{2})\2(\d{4})(?![\d-])")
NI_NUMBER_REGEX = re.compile(r"\b[A-Z]{2} ?\d{2} ?\d{2} ?\d{2} ?[A-D]\b", re.IGNORECASE)
CARD_REGEX = re.compile(  # in groups: one piece of 13 to 19 digits is as often a reference or a barcode
    r"(?<![\d+/-])(?<!\d[ -])(?:"  # never glued to a number, + or slash, nor inside a longer run (an IBAN's)
    r"\d{4}([ -])\d{4}\1\d{4}\1\d{1,4}(?:\1\d{1,3})?"  # 4111 1111 1111 1111, 4-4-4-4-3
    r"|\d{4}([ -])\d{6}\2\d{4,5})"  # 3782 822463 10005 (American Express, Diners)
    r"(?![\d-])"
)
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22, "BR": 29,
    "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29,
    "ES": 24, "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28,
    "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28,
    "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MR": 27,
    "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24,
    "RS": 22, "SA": 24, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "TN": 24, "TR": 26, "UA": 29, "VA": 22,
    "VG": 24, "XK": 20,
}  # fmt: skip

Record = dict[str, Any]
Ner = Callable[..., list[list[Record]]]  # transformers' pipeline("ner") called on a list of texts


class WindowError(RuntimeError):
    """A text the windows would not show the model whole; the server answers 500, never a half-masked text."""


def _windows(text: str, tokenizer: Any) -> list[tuple[int, int]]:
    """[start, end) character spans of `text`, each at most the model's tokens with its special ones, the next
    starting STRIDE tokens before the previous ends, cut between words where one allows it. WindowError when
    a character the model could read lies outside them, or a window would be truncated."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    offsets, words, n = enc["offset_mapping"], enc.word_ids(), len(enc["input_ids"])
    limit = tokenizer.model_max_length
    size = limit - tokenizer.num_special_tokens_to_add()

    def boundary(i: int) -> bool:
        return i in (0, n) or words[i] != words[i - 1]

    def fits(a: int, b: int) -> bool:  # the window as the pipeline will tokenize it, special tokens included
        return len(tokenizer(text[offsets[a][0] : offsets[b - 1][1]], verbose=False)["input_ids"]) <= limit

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        end = next((e for e in range(end, start + size // 2, -1) if boundary(e)), end)
        while not fits(start, end):
            end -= 1  # a cut inside a word can tokenize longer on its own
            if end == start:
                raise WindowError("a window holds no token")
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end == n:
            break
        nxt = max(start + 1, end - STRIDE)
        start = next((s for s in range(nxt, end) if boundary(s)), nxt)
    covered = bytearray(len(text))
    for s, e in spans:
        covered[s:e] = b"\1" * (e - s)
    # BertNormalizer drops whitespace, U+FFFD and C* characters (control, format, private, unassigned);
    # anything else outside the windows is text the model never read, whatever the tokenizer reported
    unread = (c for i, c in enumerate(text) if not covered[i])
    if any(not c.isspace() and c != "\ufffd" and unicodedata.category(c)[0] != "C" for c in unread):
        raise WindowError("text outside every window")
    if spans:  # the edges hold nothing to read: a one-window text goes to the model exactly as sent
        spans[0] = (0, spans[0][1])
        spans[-1] = (spans[-1][0], len(text))
    return spans


def _merge_overlapping(entities: list[Record]) -> list[Record]:
    """transformers' TokenClassificationPipeline.aggregate_overlapping_entities (5.12.0), which its stride ran
    on the chunks' entities: of two overlapping ones the longer wins, then the higher score."""
    if not entities:
        return entities
    entities = sorted(entities, key=lambda x: x["start"])
    merged, previous = [], entities[0]
    for entity in entities:
        if previous["start"] <= entity["start"] < previous["end"]:
            current_length = entity["end"] - entity["start"]
            previous_length = previous["end"] - previous["start"]
            if current_length > previous_length or (
                current_length == previous_length and entity["score"] > previous["score"]
            ):
                previous = entity
        else:
            merged.append(previous)
            previous = entity
    merged.append(previous)
    return merged


def windowed(ner: Ner, tokenizer: Any) -> Ner:
    """`ner` over windows of each text (`_windows`), all of them in one call, each entity shifted back to its
    text and the overlaps merged as the pipeline's stride merged them. The pipeline's own stride is not used:
    tokenizers 0.23 returns only the first overflow window, so past ~512 tokens names went unread."""

    def run(texts: list[str], **kwargs: Any) -> list[list[Record]]:
        spans = [_windows(t, tokenizer) for t in texts]
        pieces = [t[s:e] for t, sp in zip(texts, spans, strict=True) for s, e in sp]
        found = ner(pieces, **kwargs) if pieces else []
        if len(found) != len(pieces):
            raise WindowError("the model answered for fewer windows than it was given")
        per_window = iter(found)
        out = []
        for sp in spans:
            ents = [
                {**e, "start": e["start"] + s, "end": e["end"] + s} for s, _ in sp for e in next(per_window)
            ]
            out.append(_merge_overlapping(ents) if len(sp) > 1 else ents)
        return out

    return run


def validate_email(email: str) -> bool:
    if len(email) < 6 or len(email) > 320 or email.count("@") != 1:
        return False
    local, domain = email.split("@")
    if len(local) == 0 or len(local) > 64 or local.startswith(".") or local.endswith(".") or ".." in local:
        return False
    if len(domain) == 0 or len(domain) > 255:
        return False
    if any(domain.startswith(x) or domain.endswith(x) for x in [".", "-"]):
        return False
    return not (".." in domain or "." not in domain)


def validate_polish_bank_account(account_number: str) -> bool:
    digits = re.sub(r"[^0-9]", "", account_number.upper().replace("PL", ""))
    if len(digits) != 26:
        return False
    return int(digits[2:] + "2521" + digits[:2]) % 97 == 1


def validate_iban(iban: str) -> bool:
    iban_clean = iban.replace(" ", "").upper()
    if len(iban_clean) < 15 or len(iban_clean) > 34:
        return False
    if not (iban_clean[:2].isalpha() and iban_clean[2:4].isdigit()):
        return False
    if iban_clean[:2] in IBAN_LENGTHS and len(iban_clean) != IBAN_LENGTHS[iban_clean[:2]]:
        return False
    rearranged = iban_clean[4:] + iban_clean[:4]
    numeric = "".join(str(ord(c) - ord("A") + 10) if c.isalpha() else c for c in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def validate_pesel(pesel: str) -> bool:
    if len(pesel) != 11 or not pesel.isdigit():
        return False
    checksum = sum(int(d) * w for d, w in zip(pesel, (1, 3, 7, 9, 1, 3, 7, 9, 1, 3), strict=False)) % 10
    return (10 - checksum) % 10 == int(pesel[10])


def validate_id_card(id_card: str) -> bool:
    cleaned = id_card.replace(" ", "").upper()
    if len(cleaned) != 9 or not (cleaned[:3].isalpha() and cleaned[3:].isdigit()):
        return False
    values = [ord(c) - ord("A") + 10 if c.isalpha() else int(c) for c in cleaned]
    return sum(v * w for v, w in zip(values, (7, 3, 1, 9, 7, 3, 1, 7, 3), strict=True)) % 10 == 0


def validate_english_phone(phone: str) -> bool:
    """E.164 has at least 8 digits with the country code, a run past its 15 masked whole rather than left;
    a UK, Irish or US national number has 9 to 11."""
    digits = re.sub(r"\D", "", phone.replace("(0)", ""))
    return len(digits) >= 8 if phone.startswith("+") else 9 <= len(digits) <= 11


def validate_ssn(ssn: str) -> bool:
    """The SSA never issues area 000, 666 or 900-999, group 00 or serial 0000."""
    area, group, serial = re.findall(r"\d+", ssn)
    return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"


def validate_ni_number(number: str) -> bool:
    """HMRC's prefix rules: no D, F, I, Q, U or V, no O second, and none of the prefixes never issued."""
    prefix = number[:2].upper()
    if prefix[0] in "DFIQUV" or prefix[1] in "DFIOQUV":
        return False
    return prefix not in ("BG", "GB", "KN", "NK", "NT", "TN", "ZZ")


def validate_card(card: str) -> bool:
    """Luhn: every second digit from the right doubled (less 9 past 9), the sum a multiple of 10."""
    digits = [int(d) for d in reversed(re.sub(r"\D", "", card))]
    return sum(d if i % 2 == 0 else d * 2 - 9 * (d > 4) for i, d in enumerate(digits)) % 10 == 0


def _normalize_phone(value: str) -> str:
    return re.sub(r"[^\d*?]", "", value)


def _entity_key(entity_type: str, raw: str) -> str:
    if entity_type == "PERSON":
        return " ".join(raw.casefold().split())
    if entity_type == "EMAIL":
        return raw.lower()
    if entity_type == "PHONE":
        digits = _normalize_phone(raw.replace("(0)", ""))  # +44 (0)20 is +44 20
        if digits.startswith("0048") and len(digits) >= 13:
            digits = digits[4:]
        elif digits.startswith("48") and len(digits) >= 11:
            digits = digits[2:]
        return digits
    if entity_type == "BANK_ACCOUNT":
        value = raw.replace(" ", "").upper()
        return value[2:] if value.startswith("PL") else value
    if entity_type in ("PESEL", "CARD_NUMBER"):  # a PESEL is also an SSN or NI number in English
        return re.sub(r"[\W_]", "", raw.upper())
    if entity_type == "ID_CARD":
        return raw.replace(" ", "").upper()
    return raw


class EntityRegistry:
    """Per-request (type, key) -> 1-based index, assigned on first sight."""

    def __init__(self) -> None:
        self._indices: dict[tuple[str, str], int] = {}
        self._counts: dict[str, int] = {}

    def assign(self, entity_type: str, key: str) -> int:
        idx = self._indices.get((entity_type, key))
        if idx is None:
            idx = self._counts.get(entity_type, 0) + 1
            self._counts[entity_type] = idx
            self._indices[(entity_type, key)] = idx
        return idx


def _rightmost(replacements: list[Record]) -> list[Record]:
    """The replacements in text order, less each one overlapping a later one (the rightmost span wins)."""
    filtered, last_start = [], float("inf")
    for r in sorted(replacements, key=lambda x: x["start"], reverse=True):
        if r["end"] > last_start:
            continue
        filtered.append(r)
        last_start = r["start"]
    return sorted(filtered, key=lambda x: x["start"])


def _materialize(text: str, replacements: list[Record]) -> tuple[str, list[Record]]:
    """Apply replacements, the rightmost span winning any overlap; placed tokens carry new_start/new_end."""
    filtered = _rightmost(replacements)
    out: list[str] = []
    placed: list[Record] = []
    cursor = pos = 0
    for r in filtered:
        segment = text[cursor : r["start"]]
        out.append(segment)
        pos += len(segment)
        out.append(r["label"])
        placed.append({**r, "new_start": pos, "new_end": pos + len(r["label"])})
        pos += len(r["label"])
        cursor = r["end"]
    out.append(text[cursor:])
    return "".join(out), placed


def collect_regex_matches(text: str, api: bool = False) -> list[Record]:
    """Regex-matched PII as {start, end, type, key} (no whitelists); `api=True`
    skips a PHONE or ID_CARD match starting inside an earlier one, which would win it: jan600100200@wp.pl.
    ENGLISH adds cards and English phones before the Polish phone rule, and SSN and NI numbers as PESEL after
    PESEL's, each only where no earlier match lies."""
    if not text or len(text.strip()) < 3:
        return []
    records: list[Record] = []
    covered = bytearray(len(text))  # 1 inside a match: the service's `any(start <= pos < end)` in O(1)

    def add(m: re.Match[str], kind: str) -> None:
        records.append(
            {"start": m.start(), "end": m.end(), "type": kind, "key": _entity_key(kind, m.group())}
        )
        covered[m.start() : m.end()] = b"\1" * (m.end() - m.start())

    def english(regex: re.Pattern[str], kind: str, valid: Callable[[str], bool]) -> None:
        for m in regex.finditer(text):
            if not any(covered[m.start() : m.end()]) and valid(m.group()):
                add(m, kind)

    for m in EMAIL_REGEX.finditer(text):
        if validate_email(m.group()):
            add(m, "EMAIL")
    for m in IP_REGEX.finditer(text):
        add(m, "IP_ADDRESS")
    for m in POLISH_BANK_ACCOUNT_REGEX.finditer(text):
        if validate_polish_bank_account(m.group()):
            add(m, "BANK_ACCOUNT")
    for m in IBAN_REGEX.finditer(text):
        if not covered[m.start()] and validate_iban(m.group()):
            add(m, "BANK_ACCOUNT")
    if ENGLISH:
        english(CARD_REGEX, "CARD_NUMBER", validate_card)
        english(ENGLISH_PHONE_REGEX, "PHONE", validate_english_phone)
    for m in PHONE_REGEX.finditer(text):
        if not ((api or ENGLISH) and covered[m.start()]):
            add(m, "PHONE")
    for m in PESEL_REGEX.finditer(text):
        if covered[m.start()]:
            continue
        if validate_pesel(m.group()):
            add(m, "PESEL")
    if ENGLISH:
        english(SSN_REGEX, "PESEL", validate_ssn)
        english(NI_NUMBER_REGEX, "PESEL", validate_ni_number)
    for m in ID_CARD_REGEX.finditer(text):
        if not (api and covered[m.start()]) and validate_id_card(m.group()):
            add(m, "ID_CARD")
    return records


def collect_ner_matches(text: str, ner_entities: list[Record]) -> list[Record]:
    """NER output filtered to ENTITY_MAPPING above the score threshold (no protected spans)."""
    records = []
    for entity in ner_entities:
        entity_type = ENTITY_MAPPING.get(entity["entity_group"])
        if not entity_type or entity["score"] <= NER_SCORE_THRESHOLD:
            continue
        start, end = entity["start"], entity["end"]
        records.append(
            {
                "start": start,
                "end": end,
                "type": entity_type,
                "key": _entity_key(entity_type, text[start:end]),
            }
        )
    return records


def _snap_span(record: Record, token_spans: list[tuple[int, int]]) -> Record | None:
    """Widen an NER span to whole placeholders it cuts; None when only placeholders are left."""
    start, end = record["start"], record["end"]
    overlapping = [(ts, te) for ts, te in token_spans if start < te and end > ts]
    if not overlapping:
        return record
    new_start = min([start] + [ts for ts, _ in overlapping])
    new_end = max([end] + [te for _, te in overlapping])
    covered = sum(max(0, min(new_end, te) - max(new_start, ts)) for ts, te in token_spans)
    if covered >= new_end - new_start:
        return None
    return {**record, "start": new_start, "end": new_end}


@dataclass
class Masked:
    """What the service replaced in one text, as [start, end) spans of the original text."""

    persons: list[tuple[int, int]]  # applied NER PERSON spans
    spans: list[tuple[int, int]]  # everything applied: regex and NER


def _original(pos: int, placed: list[Record], end: bool) -> int:
    """A position of the materialized text in the original one (a position inside a placeholder maps to
    the matched text's start, or its end for an end position)."""
    shift = 0
    for p in placed:
        if pos >= p["new_end"]:
            shift += (p["new_end"] - p["new_start"]) - (p["end"] - p["start"])
        elif pos > p["new_start"]:
            return p["end"] if end else p["start"]
    return pos - shift


def regex_stage(text: str, api: bool = False) -> tuple[str, list[Record]]:
    """Phases 1-2 of one request: the text NER sees (regex matches as indexed placeholders) and the placed
    replacements (start/end in `text`, new_start/new_end in the result)."""
    registry = EntityRegistry()
    records = sorted(collect_regex_matches(text, api), key=lambda r: r["start"])
    for r in records:
        r["label"] = f"<{r['type']}_{registry.assign(r['type'], r['key'])}>"
    return _materialize(text or "", records)


Stages = list[tuple[list[Record], list[Record]]]


def stages(texts: Sequence[str], ner: Ner | None, api: bool = False) -> Stages:
    """Run the service's four phases, each text as its own request, NER over all of them in one call (none
    when `ner` is None). Per text: the placed regex replacements and the applied NER records (type, start/end
    in the original text); with `api=True` every snapped NER record, overlapping ones too (server.py trims
    them first)."""
    materialized, placed_per_text = [], []
    for text in texts:  # phases 1-2
        mat, placed = regex_stage(text, api)
        materialized.append(mat)
        placed_per_text.append(placed)
    needing = [i for i, t in enumerate(materialized) if t and len(t.strip()) >= 3]
    entities: list[list[Record]] = [[] for _ in texts]
    if ner is not None and needing:  # phase 3
        for i, found in zip(needing, ner([materialized[i] for i in needing], **NER_KWARGS), strict=True):
            entities[i] = found
    result = []
    for mat, placed, ents in zip(materialized, placed_per_text, entities, strict=True):  # phase 4
        tokens = [(p["new_start"], p["new_end"]) for p in placed]
        snapped = []
        for r in sorted(collect_ner_matches(mat, ents), key=lambda r: r["start"]):
            if (adj := _snap_span(r, tokens)) is not None:
                snapped.append({**adj, "label": f"<{adj['type']}>"})
        applied = snapped if api else _materialize(mat, snapped)[1]
        back = [
            {**a, "start": _original(a["start"], placed, False), "end": _original(a["end"], placed, True)}
            for a in applied
        ]
        result.append((placed, back))
    return result


def mask(texts: Sequence[str], ner: Ner) -> list[Masked]:
    """The service's phases over `texts` as `Masked` spans."""
    return [
        Masked(
            persons=[(a["start"], a["end"]) for a in applied if a["type"] == "PERSON"],
            spans=[(p["start"], p["end"]) for p in placed] + [(a["start"], a["end"]) for a in applied],
        )
        for placed, applied in stages(texts, ner)
    ]
