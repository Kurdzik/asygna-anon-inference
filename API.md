# Anonymisation API

The HTTP contract and the rules of the service in `src/flisak_anon/`. It runs on premises, stateless, one model a
process, and it is meant to be called by servers. A caller names what it wants masked in each request.

Contract version 1.1: `ANON_LANGUAGE=en` adds the English rules of api.3 to the regex stage and offers `CARD_NUMBER`;
with the default `pl` the service offers the seven other features.

## api.1 Rules

| # | Rule | Why |
|---|---|---|
| A2 | **The caller names the features per request** from a closed list (api.3). The service finds every feature it knows, but replaces only the requested ones. Anything not requested stays readable as written, including a name inside an e-mail address when EMAIL is not requested. | Each install decides what leaves readable. |
| A3 | **Model + regex, in a fixed order.** The regex stage runs first. Every regex match becomes a placeholder in the text the model reads, whether requested or not, and then the model runs. | The models were trained on text that had been through the regex stage, so they must read the same kind of text. |
| A4 | **The model runs only when a model feature is requested** (PERSON, or PHONE, which the Polish model also tags). A request for regex features only never runs the model. | Cost: regexes take microseconds. |
| A5 | **Placeholders are `<TYPE_n>`, numbered per request across all its texts**, in reading order (text, then offset). The same value gets the same number. A name is compared by its casefolded form, so "Jan Kowalski", "Jana Kowalskiego" and "Kowalski" get three numbers; a phone number is compared by its digits. Only replaced entities take a number, so the numbers have no gaps. | A reader can tell one person from two across a sender line and a signature. Numbers are deterministic. |
| A6 | **The answer gives spans as well as texts**: each replaced entity's text index, `[start, end)` in Unicode code points of the text as sent, type, label and source (`regex` or `model`). It never repeats the value it found. | The caller can check the answer, and an evaluation can score it without parsing placeholders. |
| A7 | **OpenAI-compatible conventions**: base URL ending in `/v1`, `Authorization: Bearer`, `GET /v1/models`, and errors in OpenAI's `{"error": {...}}` shape. | Existing OpenAI clients and model lists work as they are. |
| A8 | **The service stores and logs no text.** Logs hold a request id, character and entity counts per type, latency and status, and never text, values or keys. An error never echoes input. Nothing is cached. | The input is exactly the personal data being protected. |
| A9 | **The model is a directory, checked at start.** The service checks every file's sha256 against manifest.json, then warms up, and refuses to start on a mismatch or on a file manifest.json does not list (transformers would read a stray `tokenizer.json` or `model.safetensors` first). | The weights served are exactly the weights measured. |

## api.2 The HTTP contract

| Method and path | Auth | Answer |
|---|---|---|
| `GET /health` | none | 200 `{"status": "ok"}`: the port opens only once the model passed A9 and warmed up, so nothing answers before |
| `GET /v1/models` | Bearer | `{"object": "list", "data": [{"id": "flisak-anon-0.1.0", "object": "model"}]}`, the id taken from manifest.json |
| `POST /v1/anonymize` | Bearer | below |

Request (`Content-Type: application/json`, at most 2 MiB):

```json
{
  "model": "flisak-anon-0.1.0",
  "texts": ["Jan Kowalski", "jan.kowalski@example.com", "Reklamacja biletu",
            "Proszę o zwrot. Jan Kowalski, tel. 600 100 200"],
  "features": ["PERSON", "EMAIL", "PHONE", "PESEL"],
  "keep": {"emails": ["zgloszenia@urzad.example.pl"], "phones": []}
}
```

- `model` must be the id the service serves, else 404 `model_not_found`.
- `texts` holds 1 to 64 strings, empty ones allowed, at most 200,000 characters together, else 413 `too_large`.
  A text has no length limit of its own and is read to its end: the service cuts one longer than the model's 512
  tokens into overlapping windows itself (api.4 step 3), and answers 500 `server_error`, never a partly masked text,
  if a window would be lost.
- `features` holds at least one code from api.3. Any unknown code gives 422 `unsupported_feature`, and `param` names
  the first one when it looks like a code (`[A-Z][A-Z0-9_]{0,31}`), else null; a repeated code gives 400
  `invalid_request`.
- `keep` is optional. It lists values that stay readable: e-mail addresses compared whole and without regard to case,
  and phone numbers compared by their digits, with a leading 48 or 0048 dropped. It takes
  no domains or wildcards: a colleague's address in the organisation's domain is personal data. Each entry must be
  one whole address (the e-mail regex and validator of api.3) or one whole number the phone regex matches, else
  400 `invalid_request` (`param` "keep"); each list holds at most 64 entries, else 413 `too_large`. A kept phone
  number stays readable whether the regex or the model found it.
- An unknown field is refused (400 `invalid_request`).

Answer 200:

```json
{
  "model": "flisak-anon-0.1.0",
  "features": ["PERSON", "EMAIL", "PHONE", "PESEL"],
  "texts": ["<PERSON_1>", "<EMAIL_1>", "Reklamacja biletu", "Proszę o zwrot. <PERSON_1>, tel. <PHONE_1>"],
  "entities": [
    {"text": 0, "start": 0, "end": 12, "type": "PERSON", "label": "<PERSON_1>", "source": "model"},
    {"text": 1, "start": 0, "end": 24, "type": "EMAIL", "label": "<EMAIL_1>", "source": "regex"},
    {"text": 3, "start": 16, "end": 28, "type": "PERSON", "label": "<PERSON_1>", "source": "model"},
    {"text": 3, "start": 35, "end": 46, "type": "PHONE", "label": "<PHONE_1>", "source": "regex"}
  ]
}
```

Entities are sorted by text, then start, and never overlap within a text. `texts[i]` equals the text as sent with
each of its spans replaced by the span's label. Every `type` is one of the requested features.

Errors use OpenAI's shape, `{"error": {"message": "<fixed text>", "type": "...", "code": "...", "param": null}}`.
The message is a fixed sentence per code and never quotes the input.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | malformed JSON, a missing or unknown field, a wrong type, a repeated feature |
| 401 | `invalid_api_key` | missing or wrong key (compared in constant time) |
| 404 | `model_not_found` | `model` is not the served one |
| 404 | `not_found` | an unknown path (405 for a wrong method, `code` `invalid_request`) |
| 413 | `too_large` | over a limit above |
| 422 | `unsupported_feature` | a feature code this service does not offer (`param`: the code) |
| 429 | `busy` | the concurrency queue is full (`Retry-After: 1`) |
| 500 | `server_error` | anything else, a text the windows would not cover whole included (api.4 step 3) |

## api.3 Features

| Code | Checklist (pl / en) | Found by | What is replaced |
|---|---|---|---|
| `PERSON` | Imiona i nazwiska / People's names | model: flisak-anon's `nam_liv_person`, `nam_liv_character`, `nam_liv_god`, `nam_adj_person` above score 0.4 | names in every case form, diminutives, signatures, lowercase and uppercase. Companies, streets, institutions and brands named after people stay. |
| `EMAIL` | Adresy e-mail / E-mail addresses | regex + validation | the whole address |
| `PHONE` | Numery telefonów / Phone numbers | regex (Polish formats, `+48`/`0048`; English below) + model (`nam_num_phone`) | the number |
| `PESEL` | Numery PESEL / PESEL numbers | regex, checksum | 11 digits with a valid check digit; English below |
| `BANK_ACCOUNT` | Numery rachunków bankowych / Bank account numbers | regex, checksum | NRB (26 digits, `PL` optional) and IBAN (mod 97, the country's length) |
| `ID_CARD` | Numery dowodów osobistych / ID card numbers | regex, checksum | three letters and six digits with a valid check digit |
| `IP_ADDRESS` | Adresy IP / IP addresses | regex | IPv4 and full IPv6 |
| `CARD_NUMBER` | Numery kart płatniczych / Payment card numbers | regex, Luhn; only with `ANON_LANGUAGE=en` (422 `unsupported_feature` otherwise) | 13 to 19 digits in fours (4-4-4-4, 4-4-4-4-3) or as 4-6-5 and 4-6-4, spaces or dashes, one separator throughout; never glued to a number, `+` or `/`, nor inside a longer run of groups (an IBAN's) |

**English rules** (`ANON_LANGUAGE=en` on the server, `pl` by default; any other value refuses to start). They run
beside the Polish ones, which stay as they are, and each takes a number only where no earlier match lies:
- `PHONE`: E.164 but `+48` (`+` and 8 digits or more in groups, `(0)` and an area code in parentheses allowed:
  +44 (0)20 7946 0958, +1 (212) 555-0198, +12125550198; a run past E.164's 15 digits is masked whole, never left);
  UK and Irish national numbers (0 but never 00, then 9 to 11 digits in groups, or a UK mobile in one piece: 020
  7946 0958, (01) 234 5678, 07700 900123, 07700900123); US and Canadian ones with a separator (no area code or
  exchange starting 0 or 1: (212) 555-0198, 212.555.0198, 650 723 1976, 1-800-555-0199). They run before the Polish
  rule, so a foreign number such as +420 601 234 567 is one match, and the Polish rule then skips a number starting
  inside an earlier match as the API does; `+48` and `0048` numbers stay the Polish rule's and read as on the
  Polish instance. A number's key drops `(0)`, so +44 (0)20 7946 0958 and +44 20 7946 0958 share a label; the
  national form keeps a label of its own.
- `PESEL` also takes the US Social Security number (3-2-4 with dashes or spaces, never area 000, 666 or 9xx, group
  00 or serial 0000) and the UK National Insurance number (AB 12 34 56 C, HMRC's prefix rules, suffix A-D): both
  are a person's national number, the place PESEL holds in Polish mail.
- `CARD_NUMBER` above. It is a code of its own: a card number is no bank account.
What they leave: amounts, dates, times, sort codes, postcodes, a ten-digit number in one piece (a US number written
so, or an invoice or order number), a card written in one piece (as often a reference or a barcode), a number glued to a
dash or slash (INV-0123456789). What they take too, failing closed: a `+` before an amount in groups (+12 345 678),
a date or count written right after an E.164 number (+44 20 7946 0958 20 Oct), and a 3-3-4 or 3-2-4 list of
numbers read as a US number or an SSN (250 500 1000).

Not offered yet. Each needs a model or regex change and is measured before it is offered:
- `ADDRESS`, a private person's street, number, postal code and town. This needs training: the model was taught streets as
  text to keep (hard negatives), and FastPDN's head has road and city labels that nothing maps yet. It also needs a
  postal-code regex.
- `PASSPORT` (check digit).
- `NIP` (checksum). A sole trader's NIP identifies a person, a company's does not.
- `CARD_NUMBER` for Polish mail (built for English, Luhn; measured on Polish text before it runs there).
- `PLATE` (vehicle registration).
- `BIRTH_DATE`.


## api.4 Inside the service

**Pipeline per request**:
1. Find regex matches of every type in each text, validated. Mark the `keep` values as kept.
   A phone or ID card number starting inside an earlier match is skipped, as a PESEL is:
   the number wins an address such as `kowalski600100200@wp.pl`, which then stays readable when EMAIL alone is
   requested, and loses its name and domain to `<PHONE_1>` otherwise.
   A PESEL of someone born in 1948 is also 48 and a phone number, and the phone rule finds it first: it is a
   PESEL when PESEL is requested and a phone otherwise, so either feature replaces it. The cost: a phone written
   as 48 and nine digits with no `+` or space that happens to pass the PESEL checksum (1 in 10) reads as a PESEL
   when PESEL is requested, unless it is a kept phone. The same holds for a PESEL of someone born in 2007, which the
   English UK mobile rule takes (07 and nine digits).
2. Build the text the model reads, with every regex match as a placeholder (`regex_stage`).
3. If PERSON or PHONE is requested, run NER over all texts in one call:
   `pipeline("ner", aggregation_strategy="simple")`, `batch_size=8`, score above 0.4. A text longer than the
   model's 512 tokens (special ones included) is cut by the service, not by the pipeline's `stride=64`
   (`anonymizer.windowed`): tokenized once whole, it becomes windows of at most 512 tokens, each starting 64 tokens
   before the previous one ends and cut between words where a word boundary lies in the window's second half; every
   window of every text goes to the model in the one call, its entities are shifted back to the text, and
   overlapping ones are merged as the pipeline's stride merged them (the longer wins, then the higher score). A text
   of one window goes to the model exactly as sent. Fail closed: a character the tokenizer would read (not
   whitespace, U+FFFD or a control or format character) left outside every window, a window that would tokenize past
   512, or fewer answers than windows raise `WindowError`, which the request answers with 500 `server_error`. The
   startup log names the `tokenizers` version. Why: tokenizers 0.23.2 (in v0.29.0) returns only the first overflow
   window, so the pipeline's stride left everything past ~512 tokens unread.
4. Snap the NER spans around placeholders and map them back to the original text (`_snap_span`, `_original`). An
   NER span that overlaps a regex match is trimmed to its characters outside the match, less the whitespace at
   their ends (split in two around a match inside it), and never dropped; a piece with no letter or digit left
   goes, and the pieces of one span share one label. Then drop unrequested and kept pieces, and where two pieces
   still overlap, the rightmost one wins (`_rightmost`). The regex match keeps its own
   characters: replaced if requested and not kept, readable otherwise. A snapped span never swallows a placeholder: "Jan Kowalski 600 100 200" is `<PERSON_1> <PHONE_1>`, or `<PERSON_1> 600 100 200`
   when PHONE is not requested.
5. Number and apply only the requested matches that are not kept (A5). Return the texts and the spans.

**Layout.** `server.py`: routes, limits, errors, auth, the manifest check (`verify`), the load and warm-up (`load`), and
step 4's trim and overlap rule and step 5 (`anonymize`). `anonymizer.py`: steps 1-3 and step 4's snap and mapping. `create_app`
takes the NER function, so `tests/test_server.py` injects a fake one.

**Runtime.** CUDA when the container sees a GPU, else CPU. On CPU flisak-anon reads about 1,000 characters in 168
ms of wall time on 6 threads , so a 5,000-character complaint takes under a second and the
200,000-character limit about 34 s. Text of one token per character costs three times that: 200,000 dots or emoji
are 449 windows and took about 100 s in the image on 6 CPUs (2026-10-14). One model run at a time; up to 8 more
requests wait for it, and a further one is answered 429 `busy`. A request for regex features only never waits for the
model and takes no place in its queue.
The regexes run in linear time: the e-mail pattern takes at most 64 characters before the `@` and 63 per domain
label, as the validator and DNS allow (also in `mask`). An unbounded pattern is quadratic: 200,000
characters of "a.a.a…" held Python's lock, and so every request, for 129 s.

**Security.** One API key, from the environment (`ANON_API_KEY`, at least 16 characters, else the server does not
start), compared with `hmac.compare_digest`. No CORS: callers are servers. No docs UI and no OpenAPI document. The
body limit is enforced while the body is read, before parsing, and the key is checked before the body is read. The
service listens on plain HTTP on a private network, or sits behind the host's TLS proxy. uvicorn's access log is off; the one log line per request
holds a request id, the status and error code, the counts of texts and characters, the features, the entities per
type and the latency, and a failure logs only the exception's type, since its message can quote the text. The Hub
is never contacted (`HF_HUB_OFFLINE=1`).
