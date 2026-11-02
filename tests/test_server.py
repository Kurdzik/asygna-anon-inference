import hashlib
import json
import logging
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flisak_anon import anonymizer, server
from flisak_anon.anonymizer import REQUIRED_MODEL_FILES

KEY = "k" * 32
AUTH = {"Authorization": f"Bearer {KEY}"}
MODEL = "flisak-anon-0.1.0"


def fake_ner(people: list, calls: list | None = None):
    """A pipeline("ner") stand-in: each of `people` found in the text the model reads is a person, or of the
    group it is paired with."""
    pairs = [p if isinstance(p, tuple) else (p, "nam_liv_person") for p in people]

    def ner(texts: list[str], **kwargs) -> list[list[dict]]:
        if calls is not None:
            calls.append(texts)
        return [
            [
                {"entity_group": group, "score": 0.99, "start": t.index(p), "end": t.index(p) + len(p)}
                for p, group in pairs
                if p in t
            ]
            for t in texts
        ]

    return ner


def client(ner=None) -> TestClient:
    return TestClient(server.create_app(MODEL, ner or fake_ner(["Jan Kowalski"]), KEY))


def post(c: TestClient, texts: list[str], features: list[str], **extra):
    return c.post(
        "/v1/anonymize", headers=AUTH, json={"model": MODEL, "texts": texts, "features": features, **extra}
    )


def test_the_example_of_api_md_numbers_across_texts_and_returns_spans() -> None:
    texts = ["Jan Kowalski", "jan.kowalski@example.com", "Reklamacja biletu",
             "Proszę o zwrot. Jan Kowalski, tel. 600 100 200"]  # fmt: skip
    r = post(
        client(),
        texts,
        ["PERSON", "EMAIL", "PHONE", "PESEL"],
        keep={"emails": ["zgloszenia@urzad.example.pl"]},
    )
    assert r.status_code == 200
    assert r.json() == {
        "model": MODEL,
        "features": ["PERSON", "EMAIL", "PHONE", "PESEL"],
        "texts": [
            "<PERSON_1>",
            "<EMAIL_1>",
            "Reklamacja biletu",
            "Proszę o zwrot. <PERSON_1>, tel. <PHONE_1>",
        ],
        "entities": [
            {"text": 0, "start": 0, "end": 12, "type": "PERSON", "label": "<PERSON_1>", "source": "model"},
            {"text": 1, "start": 0, "end": 24, "type": "EMAIL", "label": "<EMAIL_1>", "source": "regex"},
            {"text": 3, "start": 16, "end": 28, "type": "PERSON", "label": "<PERSON_1>", "source": "model"},
            {"text": 3, "start": 35, "end": 46, "type": "PHONE", "label": "<PHONE_1>", "source": "regex"},
        ],
    }


def test_only_requested_features_are_replaced_and_numbers_have_no_gaps() -> None:
    text = "Jan Kowalski, jan@firma.pl, tel. 600 100 200, PESEL 44051401359, druga: anna@firma.pl"
    calls: list = []
    r = post(client(fake_ner(["Jan Kowalski"], calls)), [text], ["EMAIL", "PESEL"]).json()
    assert r["texts"] == ["Jan Kowalski, <EMAIL_1>, tel. 600 100 200, PESEL <PESEL_1>, druga: <EMAIL_2>"]
    assert {e["type"] for e in r["entities"]} == {"EMAIL", "PESEL"}
    assert calls == []  # A4: regex features only never run the model
    r = post(client(fake_ner(["Jan Kowalski"], calls)), [text], ["PERSON"]).json()
    assert r["texts"] == [
        "<PERSON_1>, jan@firma.pl, tel. 600 100 200, PESEL 44051401359, druga: anna@firma.pl"
    ]
    assert calls == [["Jan Kowalski, <EMAIL_1>, tel. <PHONE_1>, PESEL <PESEL_1>, druga: <EMAIL_2>"]]  # A3


@pytest.mark.parametrize(
    ("feature", "valid", "invalid"),
    [
        ("EMAIL", "jan.kowalski@firma.pl", "jan.kowalski@firma"),
        ("PHONE", "+48 600-100-200", "600 100 20"),
        ("PESEL", "44051401359", "44051401358"),
        ("PESEL", "48010100000", "48010100001"),  # born 1948: 48 and nine digits, a phone number too
        ("BANK_ACCOUNT", "PL61 1090 1014 0000 0712 1981 2874", "PL61 1090 1014 0000 0712 1981 2875"),
        ("BANK_ACCOUNT", "DE89370400440532013000", "DE89370400440532013001"),
        ("ID_CARD", "ABA 300000", "ABA 300001"),
        ("IP_ADDRESS", "192.168.1.10", "192.168.1"),
    ],
)
def test_golden_vectors_per_feature(feature: str, valid: str, invalid: str) -> None:
    r = post(client(), [f"nr {valid} koniec", f"nr {invalid} koniec"], [feature]).json()
    assert r["texts"] == [f"nr <{feature}_1> koniec", f"nr {invalid} koniec"]


def test_a_1948_pesel_is_replaced_whichever_of_pesel_and_phone_is_requested() -> None:
    texts = ["PESEL 48010100000, tel. +48010100000"]  # the same digits with a plus are a phone
    for features, out in (
        (["PHONE"], "PESEL <PHONE_1>, tel. <PHONE_1>"),  # one number, as a phone
        (["PESEL", "PHONE"], "PESEL <PESEL_1>, tel. <PHONE_1>"),
    ):
        assert post(client(), texts, features).json()["texts"] == [out]
    r = post(client(), texts, ["PESEL", "PHONE"], keep={"phones": ["48010100000"]}).json()
    assert r["texts"] == texts and r["entities"] == []  # a kept phone stays one


def test_the_wkm_rule_is_not_ported() -> None:
    r = post(client(), ["karta pacjenta, PESEL 44051401359"], ["PESEL"]).json()
    assert r["texts"] == ["karta pacjenta, PESEL <PESEL_1>"]


def test_a_number_inside_an_email_address_stays_part_of_it() -> None:
    texts = ["kowalski600100200@wp.pl", "Pisz na jan.kowalski.600-100-200@firma.pl", "jan.aba300000@firma.pl"]
    for features in (
        ["EMAIL"],
        list(server.FEATURES),
    ):  # the phone or ID card number no longer wins the address
        assert post(client(), texts, features).json()["texts"] == [
            "<EMAIL_1>",
            "Pisz na <EMAIL_2>",
            "<EMAIL_3>",
        ]
    r = post(client(), texts[:1], ["EMAIL", "PHONE"], keep={"emails": ["Kowalski600100200@wp.pl"]}).json()
    assert r["texts"] == ["kowalski600100200@wp.pl"] and r["entities"] == []


def test_the_regexes_stay_linear_on_hostile_text() -> None:
    for text in ("a." * 100_000, "a@" + "a." * 99_999):  # 200,000 characters, the limit
        started = time.monotonic()
        server.anonymize([text], list(server.FEATURES), server.Keep(), None)
        assert time.monotonic() - started < 2  # 0.1 s; the service's unbounded e-mail pattern took 129 s


def test_kept_values_stay_readable_and_take_no_number() -> None:
    texts = ["Od: Biuro@Urzad.pl, do jan@firma.pl", "tel. +48 600 100 200 lub 601 234 567"]
    keep = {"emails": ["biuro@urzad.pl"], "phones": ["600100200"]}
    r = post(client(), texts, ["EMAIL", "PHONE"], keep=keep).json()
    assert r["texts"] == ["Od: Biuro@Urzad.pl, do <EMAIL_1>", "tel. +48 600 100 200 lub <PHONE_1>"]


def test_a_name_running_into_a_regex_match_is_trimmed_never_dropped() -> None:
    ner = fake_ner(["Jan Kowalski <PHONE_1>"])  # the model's span swallows the placeholder
    text = "Jan Kowalski 600 100 200"
    r = post(client(ner), [text], ["PERSON", "PHONE"]).json()
    assert r["texts"] == ["<PERSON_1> <PHONE_1>"]
    assert [(e["type"], text[e["start"] : e["end"]]) for e in r["entities"]] == [
        ("PERSON", "Jan Kowalski"),
        ("PHONE", "600 100 200"),
    ]
    assert post(client(ner), [text], ["PERSON"]).json()["texts"] == ["<PERSON_1> 600 100 200"]
    ner = fake_ner(["Anna <EMAIL_1> Nowak"])  # one span around a match: two pieces, one label
    r = post(client(ner), ["Anna x@firma.pl Nowak"], ["PERSON"]).json()
    assert r["texts"] == ["<PERSON_1> x@firma.pl <PERSON_1>"]
    ner = fake_ner(["Anna Nowak <EM"])  # ends inside the placeholder: snapped over all of it, then trimmed
    text = "Anna Nowak anna@firma.pl"
    r = post(client(ner), [text], ["PERSON", "EMAIL"]).json()
    assert r["texts"] == ["<PERSON_1> <EMAIL_1>"]
    assert [(e["type"], e["source"]) for e in r["entities"]] == [("PERSON", "model"), ("EMAIL", "regex")]
    assert post(client(ner), [text], ["PERSON"]).json()["texts"] == ["<PERSON_1> anna@firma.pl"]


def test_overlapping_model_spans_are_resolved_after_the_trim() -> None:
    ner = fake_ner([("Jan Kowalski <PH", "nam_liv_person"), ("PHONE_1> wew. 12", "nam_num_phone")])
    text = "Jan Kowalski 600 100 200 wew. 12"  # both spans snap over <PHONE_1>, so they overlap until trimmed
    assert post(client(ner), [text], ["PERSON"]).json()["texts"] == ["<PERSON_1> 600 100 200 wew. 12"]
    r = post(client(ner), [text], ["PERSON", "PHONE"]).json()
    assert r["texts"] == ["<PERSON_1> <PHONE_1> <PHONE_2>"]
    assert [e["source"] for e in r["entities"]] == ["model", "regex", "model"]


def test_spans_are_code_points_of_the_text_as_sent() -> None:
    text = "lukasz@zolc.pl — zażółć gęślą jaźń 😀, Łukasz Żółć, tel. 600 100 200, Żaneta Ćwik"
    r = post(client(fake_ner(["Łukasz Żółć", "Żaneta Ćwik"])), [text], ["PERSON", "EMAIL", "PHONE"]).json()
    values = [text[e["start"] : e["end"]] for e in r["entities"]]
    assert values == ["lukasz@zolc.pl", "Łukasz Żółć", "600 100 200", "Żaneta Ćwik"]
    rebuilt = text
    for e in reversed(r["entities"]):
        rebuilt = rebuilt[: e["start"]] + e["label"] + rebuilt[e["end"] :]
    assert r["texts"] == [rebuilt]


def test_auth() -> None:
    c = client()
    assert c.get("/health").json() == {"status": "ok"}
    body = {"model": MODEL, "texts": ["x"], "features": ["EMAIL"]}
    for headers in (
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": KEY},
        {"Authorization": "Bearer "},
    ):
        for r in (c.get("/v1/models", headers=headers), c.post("/v1/anonymize", headers=headers, json=body)):
            assert r.status_code == 401
            assert r.json()["error"]["code"] == "invalid_api_key"
    assert c.get("/v1/models", headers=AUTH).json() == {
        "object": "list",
        "data": [{"id": MODEL, "object": "model"}],
    }
    assert c.post("/v1/anonymize", headers=AUTH, json=body).status_code == 200


@pytest.mark.parametrize(
    ("body", "status", "code", "param"),
    [
        (b"{not json", 400, "invalid_request", None),
        (b'{"texts": ["a"], "features": ["EMAIL"]}', 400, "invalid_request", "model"),
        ({"texts": ["a"], "features": ["EMAIL"], "jan.kowalski@firma.pl": 1}, 400, "invalid_request", None),
        ({"texts": "a", "features": ["EMAIL"]}, 400, "invalid_request", "texts"),
        ({"texts": [1], "features": ["EMAIL"]}, 400, "invalid_request", "texts"),
        ({"texts": [], "features": ["EMAIL"]}, 400, "invalid_request", "texts"),
        ({"texts": ["a"], "features": []}, 400, "invalid_request", "features"),
        ({"texts": ["a"], "features": ["EMAIL", "EMAIL"]}, 400, "invalid_request", "features"),
        ({"texts": ["a"], "features": ["ADDRESS"]}, 422, "unsupported_feature", "ADDRESS"),
        ({"texts": ["a"], "features": ["Jan Kowalski"]}, 422, "unsupported_feature", None),
        ({"texts": ["a"], "features": ["EMAIL"], "model": "gpt-4o"}, 404, "model_not_found", "model"),
        ({"texts": ["a"] * 65, "features": ["EMAIL"]}, 413, "too_large", None),
        ({"texts": ["a" * 100_000] * 2 + ["a"], "features": ["EMAIL"]}, 413, "too_large", None),
        (
            {"texts": ["a"], "features": ["EMAIL"], "keep": {"emails": ["x@a.pl"] * 65}},
            413,
            "too_large",
            None,
        ),
        (
            {"texts": ["a"], "features": ["EMAIL"], "keep": {"emails": ["@firma.pl"]}},
            400,
            "invalid_request",
            "keep",
        ),
        (
            {"texts": ["a"], "features": ["EMAIL"], "keep": {"emails": ["*@firma.pl"]}},
            400,
            "invalid_request",
            "keep",
        ),
        (
            {"texts": ["a"], "features": ["EMAIL"], "keep": {"phones": ["600 1"]}},
            400,
            "invalid_request",
            "keep",
        ),
        ({"texts": ["a"], "features": ["EMAIL"], "keep": {"names": []}}, 400, "invalid_request", "keep"),
    ],
)
def test_validation(body, status: int, code: str, param: str | None) -> None:
    if isinstance(body, dict):
        body = json.dumps({"model": MODEL, **body}).encode()
    r = client().post("/v1/anonymize", headers=AUTH, content=body)
    assert (r.status_code, r.json()["error"]["code"], r.json()["error"]["param"]) == (status, code, param)
    assert r.json()["error"]["message"] == server.ERRORS[code][2]


def test_the_body_limit_holds_before_parsing() -> None:
    r = client().post("/v1/anonymize", headers=AUTH, content=b"x" * (server.MAX_BODY + 1))
    assert (r.status_code, r.json()["error"]["code"]) == (413, "too_large")
    r = client().post("/v1/anonymize", headers=AUTH, content=iter([b"x" * server.MAX_BODY, b"x"]))  # chunked
    assert (r.status_code, r.json()["error"]["code"]) == (413, "too_large")
    assert client().get("/v2/nothing").json()["error"]["code"] == "not_found"


def test_a_full_queue_answers_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "QUEUE", 0)
    inner: list = []

    def ner(texts: list[str], **kwargs) -> list[list[dict]]:  # more requests while the model runs
        inner.append(post(c, ["Jan Kowalski"], ["PERSON"]))
        inner.append(post(c, ["jan@firma.pl"], ["EMAIL"]))
        return [[] for _ in texts]

    c = client(ner)
    assert post(c, ["Jan Kowalski"], ["PERSON"]).status_code == 200
    assert inner[0].status_code == 429 and inner[0].headers["Retry-After"] == "1"
    assert inner[0].json()["error"]["code"] == "busy"
    assert inner[1].json()["texts"] == ["<EMAIL_1>"]  # A4: regex features only never wait for the model


def test_nothing_the_caller_sent_reaches_the_logs_or_an_error(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    secret = ["Grzegorz Brzęczyszczykiewicz", "g.brzeczyszczykiewicz@chrzaszczyrzewoszyce.pl", "601 234 567"]
    text = ", ".join(secret)

    def failing(texts: list[str], **kwargs) -> list[list[dict]]:
        raise RuntimeError(f"cannot read {texts[0]}")

    ok = post(client(fake_ner([secret[0]])), [text], ["PERSON", "EMAIL", "PHONE"])
    assert ok.status_code == 200 and not any(s in json.dumps(ok.json()["texts"]) for s in secret)
    errors = [
        post(client(failing), [text], ["PERSON"]),
        post(client(), [text], [secret[0]]),
        post(client(), [text], ["EMAIL"], keep={"emails": [secret[1] + "!"]}),
        post(client(), [text], ["EMAIL"], extra=secret[0]),
        client().post("/v1/anonymize", headers=AUTH, content=text.encode()),
        client().post(
            "/v1/anonymize", headers={"Authorization": f"Bearer {secret[1]}"}, content=text.encode()
        ),
    ]
    assert [r.status_code for r in errors] == [500, 422, 400, 400, 400, 401]
    assert errors[0].json()["error"]["code"] == "server_error"
    assert "RuntimeError" in caplog.text and "status=200" in caplog.text
    for s in [*secret, "Brzęczyszczykiewicz", "601234567", KEY]:
        assert s not in caplog.text
        assert not any(s in r.text for r in errors)


def manifest_dir(tmp_path: Path) -> Path:
    for f in REQUIRED_MODEL_FILES + ("LICENSE", "ATTRIBUTION.md"):
        (tmp_path / f).write_text(f"{f} bytes")
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in tmp_path.iterdir()}
    (tmp_path / "manifest.json").write_text(
        json.dumps({"name": "flisak-anon", "version": "0.1.0", "files": files})
    )
    return tmp_path


def test_the_model_directory_must_match_its_manifest(tmp_path: Path) -> None:
    d = manifest_dir(tmp_path)
    assert server.verify(d) == MODEL
    (d / "tokenizer.json").write_text("{}")  # transformers would read it before vocab + merges
    with pytest.raises(ValueError, match="exactly the files"):
        server.verify(d)
    (d / "tokenizer.json").unlink()
    weights = (d / "pytorch_model.bin").read_bytes()
    (d / "pytorch_model.bin").write_bytes(bytes([weights[0] ^ 1]) + weights[1:])  # one changed bit
    with pytest.raises(ValueError, match="pytorch_model.bin does not match"):
        server.verify(d)


def word_tokenizer():
    """A real fast tokenizer, one token per word or punctuation mark as HerBERT's normalizer and pre-tokenizer
    split them, 512 tokens with <s> and </s>, built in memory."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2}, unk_token="<unk>"))
    tok.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=False, lowercase=False
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, model_max_length=512, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )


def reading_pipeline(tok, names: list[str], calls: list):
    """pipeline("ner") as it reads a text: only what fits its 512 tokens, each name found there a person."""

    def ner(texts: list[str], **kwargs) -> list[list[dict]]:
        calls.append(texts)
        out = []
        for t in texts:
            seen = max((e for _, e in tok(t, truncation=True, return_offsets_mapping=True)["offset_mapping"]))
            found = [m for n in names for m in re.finditer(re.escape(n), t) if m.end() <= seen]
            out.append([{"entity_group": "nam_liv_person", "score": 0.99, "start": m.start(), "end": m.end()}
                        for m in found])  # fmt: skip
        return out

    return ner


LONG = " ".join(
    f"Akapit {i}: dziękuję za wiadomość, Kasiu, odpiszę jutro, pozdrawiam, Marek Nowak." for i in range(100)
)


def test_a_long_text_is_read_to_its_end() -> None:
    tok, calls = word_tokenizer(), []
    assert len(tok(LONG)["input_ids"]) > 1500
    ner = reading_pipeline(tok, ["Kasiu", "Marek Nowak"], calls)
    r = post(client(anonymizer.windowed(ner, tok)), [LONG, "Kasiu"], ["PERSON"]).json()
    assert "Kasiu" not in r["texts"][0] and "Marek" not in r["texts"][0] and r["texts"][1] == "<PERSON_1>"
    labels = [e["label"] for e in r["entities"]]
    assert (labels.count("<PERSON_1>"), labels.count("<PERSON_2>")) == (
        101,
        100,
    )  # each name once, none twice
    assert len(calls) == 1 and len(calls[0]) > 3  # every window of both texts in one call
    assert all(len(tok(w)["input_ids"]) <= 512 for w in calls[0])
    leaky = post(client(ner), [LONG], ["PERSON"]).json()["texts"][0]  # the pipeline alone stops at 512 tokens
    assert leaky.count("Kasiu") > 60


def test_a_short_text_goes_to_the_model_as_before() -> None:
    tok, text = word_tokenizer(), "  Proszę o zwrot. Jan Kowalski, tel. 600 100 200, Anna Nowak\n"
    before, after = [], []
    r1 = post(
        client(reading_pipeline(tok, ["Jan Kowalski", "Anna Nowak"], before)), [text], ["PERSON", "PHONE"]
    )
    ner = anonymizer.windowed(reading_pipeline(tok, ["Jan Kowalski", "Anna Nowak"], after), tok)
    assert post(client(ner), [text], ["PERSON", "PHONE"]).json() == r1.json()
    assert before == after == [["  Proszę o zwrot. Jan Kowalski, tel. <PHONE_1>, Anna Nowak\n"]]


def test_a_replacement_character_at_an_edge_is_not_refused() -> None:
    tok, calls = word_tokenizer(), []  # the normalizer drops U+FFFD, a mis-decoded byte, as it drops controls
    ner = anonymizer.windowed(reading_pipeline(tok, ["Kasiu", "Marek Nowak"], calls), tok)
    r = post(client(ner), ["\ufffd" + LONG + "\ufffd", "Kasiu\ufffd", "\ufffd\ufffd\ufffd"], ["PERSON"])
    assert r.status_code == 200
    assert "Kasiu" not in r.text and "Marek" not in r.text


class Truncating:
    """A tokenizer that, like tokenizers 0.23's overflow, returns only the first 512 tokens of a text."""

    def __init__(self, tok) -> None:
        self.tok, self.model_max_length = tok, tok.model_max_length

    def num_special_tokens_to_add(self) -> int:
        return self.tok.num_special_tokens_to_add()

    def __call__(self, text: str, **kwargs):
        return self.tok(
            text, **kwargs, truncation=True, max_length=510 if "add_special_tokens" in kwargs else 512
        )


def test_a_window_that_would_be_lost_is_refused(caplog: pytest.LogCaptureFixture) -> None:
    tok, calls = word_tokenizer(), []
    ner = reading_pipeline(tok, ["Kasiu", "Marek Nowak"], calls)

    def dropping(texts: list[str], **kwargs) -> list[list[dict]]:
        return ner(texts[:1], **kwargs)

    for c in (client(anonymizer.windowed(ner, Truncating(tok))), client(anonymizer.windowed(dropping, tok))):
        r = post(c, [LONG], ["PERSON"])
        assert (r.status_code, r.json()["error"]["code"]) == (500, "server_error")
        assert "Kasiu" not in r.text
    assert caplog.text.count("error=WindowError") == 2


# The English rules (ANON_LANGUAGE=en): each number they take, what they must leave alone, and the Polish
# instance without them.
ENGLISH_TAKES = [
    ("PHONE", "+44 20 7946 0958", "+44 20"),
    ("PHONE", "+44 (0)20 7946 0958", "+44 (0)20"),
    ("PHONE", "+12125550198", "+1212555"),
    ("PHONE", "+8613800138002", "+8613800"),
    ("PHONE", "+44 20 7946 0958 0161 496 0000", "+44 20 7"),  # longer than E.164: masked whole, never left
    ("PHONE", "+1 (212) 555-0198", "+1 (212) 555"),
    ("PHONE", "+353 1 234 5678", "+353 1 234"),  # E.164: 8 to 15 digits
    ("PHONE", "020 7946 0958", "020 7946"),
    ("PHONE", "(020) 7946 0958", "(020) 7946"),
    ("PHONE", "07700 900123", "07700 900"),
    ("PHONE", "07700900123", "0770090012"),
    ("PHONE", "01 234 5678", "01 234 567"),
    ("PHONE", "087 123 4567", "087 123"),
    ("PHONE", "(212) 555-0198", "(212) 155-0198"),  # no exchange starts with 0 or 1
    ("PHONE", "212-555-0198", "112-555-0198"),
    ("PHONE", "1-800-555-0199", "1-800-055-0199"),
    ("PHONE", "650 723 1976", "650 123 1976"),  # spaced, as Enron's mail writes them
    ("PESEL", "123-45-6789", "666-45-6789"),  # the US SSN; area 666 is never issued
    ("PESEL", "123 45 6789", "123-45-0000"),
    ("PESEL", "219-09-9999", "900-12-3456"),
    ("PESEL", "AB 12 34 56 C", "AB 12 34 56 E"),  # the UK NI number; its suffix is A-D
    ("PESEL", "AB123456C", "QQ123456C"),
    ("PESEL", "ab123456d", "GB123456A"),
    ("CARD_NUMBER", "4111 1111 1111 1111", "4111 1111 1111 1112"),  # Luhn
    ("CARD_NUMBER", "5555-5555-5555-4444", "5555-5555-5555 4444"),  # one separator throughout
    ("CARD_NUMBER", "4222 2222 2222 2", "4111 1111 1111"),  # 13 and 12 digits
    ("CARD_NUMBER", "3782 822463 10005", "3782 822463 10006"),
    ("CARD_NUMBER", "4111-1111-1111-1111", "ZAM/4111-1111-1111-1111"),
    ("CARD_NUMBER", "4111 1111 1111 1111 110", "4111 1111 1111 1111 111"),
]
ENGLISH_LEAVES = [
    "Invoice INV-0123456789", "ref 0123456789", "PO 4500012345", "2125550198", "2024-11-01", "01/11/2024",
    "10:30 - 11:45", "1,234,567.89", "£1 234 567", "+12.50", "sort code 20-00-00", "SW1A 1AA", "12 345 678",
    "Order 1234567", "0 800 123", "order 123-4567-8901234", "qty 100 200 3000", "PO 4111111111111111",
    "EAN 5901234123457", "ref/123-45-6789",
    "IBAN DE62 3704 0044 5320 1300 08",  # its digits in fours pass Luhn, a card's test
]  # fmt: skip


@pytest.fixture
def english(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anonymizer, "ENGLISH", True)


def all_features() -> list[str]:
    return [*server.FEATURES, *server.ENGLISH_FEATURES]


@pytest.mark.parametrize(("feature", "valid", "invalid"), ENGLISH_TAKES)
def test_english_golden_vectors(english: None, feature: str, valid: str, invalid: str) -> None:
    r = post(client(), [f"no. {valid} end", f"no. {invalid} end"], [feature]).json()
    assert r["texts"] == [f"no. <{feature}_1> end", f"no. {invalid} end"]


def test_the_english_rules_leave_amounts_dates_and_references(english: None) -> None:
    features = [f for f in all_features() if f != "PERSON"]
    texts = [f"x {text} y" for text in ENGLISH_LEAVES]
    assert post(client(), texts, features).json()["texts"] == texts


def test_the_polish_instance_has_no_english_rules() -> None:
    texts = [f"no. {valid} end" for _, valid, _ in ENGLISH_TAKES]
    r = post(client(), texts, list(server.FEATURES)).json()
    changed = [t for t, sent in zip(r["texts"], texts, strict=True) if t != sent]
    assert changed == ["no. <PHONE_1> end"]  # 01 234 5678 reads as a Polish 2-3-2-2 number too
    assert post(client(), ["a"], ["CARD_NUMBER"]).json()["error"]["code"] == "unsupported_feature"
    assert post(client(), ["a"], ["EMAIL"], keep={"phones": ["020 7946 0958"]}).status_code == 400


def test_polish_numbers_read_as_before_with_the_english_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = ["tel. +48 600 100 200, 600 100 200, (22) 123 45 67, 0048 601 234 567, +48 (22) 123 45 67",
             "tel. 0048601234007, 0048221234502, +48 600 100 200 300 sztuk",  # Luhn-valid, no cards
             "PESEL 44051401359",
             "dowód ABA 300000, konto PL61 1090 1014 0000 0712 1981 2874, IP 192.168.1.10"]  # fmt: skip
    polish = post(client(), texts, list(server.FEATURES)).json()
    assert polish["texts"] == [
        "tel. <PHONE_1>, <PHONE_1>, <PHONE_2>, <PHONE_3>, <PHONE_2>",
        "tel. <PHONE_4>, <PHONE_5>, <PHONE_1> 300 sztuk",
        "PESEL <PESEL_1>",
        "dowód <ID_CARD_1>, konto <BANK_ACCOUNT_1>, IP <IP_ADDRESS_1>",
    ]
    monkeypatch.setattr(anonymizer, "ENGLISH", True)
    assert post(client(), texts, list(server.FEATURES)).json() == polish


def test_one_number_written_two_ways_takes_one_label_and_stays_kept(english: None) -> None:
    texts = ["+44 (0)20 7946 0958 or +44 20 7946 0958, mobile 07700 900123"]
    assert post(client(), texts, ["PHONE"]).json()["texts"] == ["<PHONE_1> or <PHONE_1>, mobile <PHONE_2>"]
    r = post(client(), texts, ["PHONE"], keep={"phones": ["+44 20 7946 0958"]}).json()
    assert r["texts"] == ["+44 (0)20 7946 0958 or +44 20 7946 0958, mobile <PHONE_1>"]


def test_a_2007_pesel_the_uk_mobile_rule_takes_is_still_a_pesel(english: None) -> None:
    pesel = "07222800001"
    assert anonymizer.validate_pesel(pesel)
    assert post(client(), [f"PESEL {pesel}"], ["PESEL", "PHONE"]).json()["texts"] == ["PESEL <PESEL_1>"]
    assert post(client(), [f"PESEL {pesel}"], ["PHONE"]).json()["texts"] == ["PESEL <PHONE_1>"]


def test_the_english_rules_stay_linear_on_hostile_text(english: None) -> None:
    for text in ("1 " * 100_000, "+1 " * 66_666, "0-" * 100_000, "4111 " * 40_000, "AB12" * 50_000):
        started = time.monotonic()
        server.anonymize([text], all_features(), server.Keep(), None)
        assert time.monotonic() - started < 2


def test_two_pesels_in_other_digits_take_two_labels() -> None:
    fullwidth = str.maketrans("0123456789", "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19")
    text = f"PESEL {'44051401359'.translate(fullwidth)} i {'02070803628'.translate(fullwidth)}"
    assert post(client(), [text], ["PESEL"]).json()["texts"] == ["PESEL <PESEL_1> i <PESEL_2>"]
