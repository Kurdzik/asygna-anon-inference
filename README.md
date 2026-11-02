# flisak-anon-inference

The anonymisation API. It replaces personal data in text with numbered placeholders, on your own server, so the text
can go on to an AI model or a reader who should not see the data.

```
"Jan Kowalski, tel. 600 100 200" -> "<PERSON_1>, tel. <PHONE_1>"
```

It has two stages, and only the second is the model you download:

1. **Regex rules** in this code find e-mail addresses, IP addresses, bank accounts, phone numbers, ID numbers and,
   in English, payment cards. Numbers with a checksum are validated.
2. **A name model** finds people's names in the text the rules left.

The numbering is also in this code, so a model on its own gives you neither the rules nor the placeholders.

| Model | Language | `ANON_LANGUAGE` |
|---|---|---|
| [Kurdzik/flisak-anon](https://huggingface.co/Kurdzik/flisak-anon) | Polish | `pl` (default) |
| [Kurdzik/flisak-anon-en](https://huggingface.co/Kurdzik/flisak-anon-en) | English | `en` |

Run one process per model. `ANON_LANGUAGE=en` adds English phone, SSN, National Insurance and card rules to the Polish
rules.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12. This installs a CPU build of torch.

```
uv sync
uv run hf download Kurdzik/flisak-anon-en --local-dir models/flisak-anon-en --exclude .gitattributes
rm -rf models/flisak-anon-en/.cache

export ANON_API_KEY=$(openssl rand -hex 24)
ANON_MODEL_DIR=models/flisak-anon-en ANON_LANGUAGE=en uv run python -m flisak_anon.server
```

Remove `.cache` and leave out `.gitattributes` as above: the server checks the directory against the model's
`manifest.json` and refuses to start on any file the manifest does not list.

The Polish model's current version on the Hub (0.1.0) has no `manifest.json` yet, so the server refuses it as downloaded.

The server listens on port 8000. In another shell, with the same `ANON_API_KEY`:

```
curl -s localhost:8000/v1/anonymize -H "Authorization: Bearer $ANON_API_KEY" -H "Content-Type: application/json" -d '{
  "model": "flisak-anon-en-0.1.0",
  "texts": ["John Smith called Mary Jones on 020 7946 0958. Later John Smith wrote back."],
  "features": ["PERSON", "PHONE"]
}'
```

`GET /v1/models` gives the model id to send. [API.md](API.md) has the whole contract: features, `keep`, errors, limits.

For a GPU, run the same module in an image with torch and CUDA, with `src/` on `PYTHONPATH` and the model directory
mounted read-only.

## What the numbers mean

Numbering runs per request, across all its texts. The same written form gets the same number, so a reader can tell
two people apart and a caller can restore an answer from its own map:

```
John Smith called Mary Jones. Later John Smith wrote to Mary. Mr Smith will call back.
<PERSON_1> called <PERSON_2>. Later <PERSON_1> wrote to <PERSON_3>. Mr <PERSON_4> will call back.
```

It counts written forms, not people: "Mary Jones" and "Mary" get two numbers, as do "Jan Kowalski" and "Jana Kowalskiego".
Nothing leaks through this, but a reader may see one person as several. Numbers restart with each request.

## Limits

A model misses some names. Check its card for the measured leak rates before relying on it where a missed name matters.
Home addresses are not detected. Card numbers written in one piece and foreign IBANs printed in groups stay readable.
The service stores nothing and logs only counts and codes, never text.

## Tests

```
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The tests inject a fake model, so they need no download.

## Licence

The code is proprietary. The models have their own licence, CC BY 4.0, on their cards.
