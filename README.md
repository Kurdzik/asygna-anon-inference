# asygna-anon-inference

The anonymisation API. It replaces personal data in text with numbered placeholders, on your own server, so the text
can go on to an AI model or a reader who should not see the data.

```
"Jan Kowalski, tel. 600 100 200" -> "<PERSON_1>, tel. <PHONE_1>"
```

It has two stages, and only the second is the model you download:

1. **Regex rules** in this code find e-mail addresses, IP addresses, bank accounts, phone numbers, ID numbers and
   payment cards. Numbers with a checksum are validated.
2. **A name model** finds people's names in the text the rules left.

The numbering is also in this code, so a model on its own gives you neither the rules nor the placeholders.

| Model | Language | `ANON_LANGUAGE` |
|---|---|---|
| [Kurdzik/asygna-anon](https://huggingface.co/Kurdzik/asygna-anon) | Polish | `pl` (default) |
| [Kurdzik/asygna-anon-en](https://huggingface.co/Kurdzik/asygna-anon-en) | English | `en` |

Run one process per model. `ANON_LANGUAGE=en` adds English phone, SSN and National Insurance rules to the Polish rules.

## Run it

With Docker. This builds a CPU image and downloads both models once, into a volume (about 1 GB each):

```
cp .env.example .env    # then set ANON_API_KEY in it
docker compose up -d --build
```

The Polish model is on `localhost:8000` and the English one on `localhost:8001` once `docker compose ps` shows them
healthy. Try the English one with the key from `.env`:

```
curl -s localhost:8001/v1/anonymize -H "Authorization: Bearer $(grep ^ANON_API_KEY= .env | cut -d= -f2)" -H "Content-Type: application/json" -d '{
  "model": "flisak-anon-en-0.1.0",
  "texts": ["John Smith called Mary Jones on 020 7946 0958. Later John Smith wrote back."],
  "features": ["PERSON", "PHONE"]
}'
```

`GET /v1/models` gives the model id to send. [API.md](API.md) has the whole contract: features, `keep`, errors, limits.

Without Docker, with [uv](https://docs.astral.sh/uv/) and Python 3.12:

```
uv sync
uv run hf download Kurdzik/asygna-anon-en --local-dir models/asygna-anon-en --exclude .gitattributes
rm -rf models/asygna-anon-en/.cache
ANON_API_KEY=$(openssl rand -hex 24) ANON_MODEL_DIR=models/asygna-anon-en ANON_LANGUAGE=en uv run python -m asygna_anon.server
```

Remove `.cache` and leave out `.gitattributes` as above: the server checks the directory against the model's
`manifest.json` and refuses to start on any file the manifest does not list.

For a GPU, run the same module in an image with torch and CUDA, with `src/` on `PYTHONPATH` and the model directory
mounted read-only.

## What the numbers mean

Numbering runs per request, across all its texts. The same value gets the same number, so a reader can tell two people
apart and a caller can restore an answer from its own map.

On the Polish instance one person keeps one number through their case forms, by a rule that strips the endings, and a
surname alone takes the number of the one full name of the request ending in it:

```
Jan Kowalski napisał do Anny Nowak. Odpowiedź dla Jana Kowalskiego: Kowalski oddzwoni. Pisze Jan.
<PERSON_1> napisał do <PERSON_2>. Odpowiedź dla <PERSON_1>: <PERSON_1> oddzwoni. Pisze <PERSON_3>.
```

A first name alone keeps its own number, as it is too often another person's, and two people the rule cannot tell apart
(Jan and Jana share a stem) share a number. Nothing leaks through this, but a reader may see one person as several, or
two as one.

The English instance counts written forms, not people: "Mary Jones", "Mary" and "Smith" each get a number of their own.
Numbers restart with each request.

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

The code is under the [Apache License 2.0](LICENSE). The models have their own licence, CC BY 4.0, on their cards.
