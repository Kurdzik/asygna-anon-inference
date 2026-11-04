"""The anonymisation API (API.md api.2-api.4): POST /v1/anonymize, GET /v1/models, GET /health.

`python3 -m flisak_anon.server` serves the checkpoint directory in ANON_MODEL_DIR on port 8000, Bearer key
ANON_API_KEY, once every file matched manifest.json (A9) and one warm-up ran, on CUDA when there is one; with
ANON_LANGUAGE=en (pl by default) the regex stage runs its English rules too. It needs only fastapi, uvicorn,
pydantic, torch and transformers, all in vllm/vllm-openai:v0.29.0, and logs counts, codes and timings, never a
text, value or key (A8).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from anyio import from_thread
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException

from flisak_anon import anonymizer, persons
from flisak_anon.anonymizer import EntityRegistry, Ner, Record

FEATURES = (
    "PERSON",
    "EMAIL",
    "PHONE",
    "PESEL",
    "BANK_ACCOUNT",
    "ID_CARD",
    "IP_ADDRESS",
    "CARD_NUMBER",
)  # api.3
MODEL_FEATURES = {"PERSON", "PHONE"}  # A4: only these run the model
MAX_BODY = 2 * 1024 * 1024
MAX_TEXTS = 64
MAX_CHARS = 200_000  # all texts together
MAX_KEEP = 64  # per list
QUEUE = 8  # requests waiting for the model beside the one it runs; more get 429 busy
PORT = 8000
WARMUP = "Jan Kowalski mieszka w Warszawie i pisze na biuro@example.com."
ERRORS = {  # code: status, OpenAI type, fixed message (api.2: never the input)
    "invalid_request": (400, "invalid_request_error", "The request is not valid."),
    "invalid_api_key": (401, "authentication_error", "Missing or invalid API key."),
    "model_not_found": (404, "invalid_request_error", "That model is not served here."),
    "not_found": (404, "invalid_request_error", "No such endpoint."),
    "too_large": (413, "invalid_request_error", "The request is over a limit."),
    "unsupported_feature": (422, "invalid_request_error", "A feature is not offered."),
    "busy": (429, "rate_limit_error", "The server is busy, try again shortly."),
    "server_error": (500, "server_error", "The server failed."),
    "client_left": (499, "invalid_request_error", "The client left."),  # only the log reads it
}
log = logging.getLogger("flisak_anon.server")


class ApiError(Exception):
    def __init__(self, code: str, param: str | None = None) -> None:
        self.code, self.param = code, param


class Keep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)


class AnonymizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    texts: list[str]
    features: list[str]
    keep: Keep = Field(default_factory=Keep)


def error(code: str, param: str | None = None, status: int | None = None) -> JSONResponse:
    default, kind, message = ERRORS[code]
    return JSONResponse(
        {"error": {"message": message, "type": kind, "code": code, "param": param}},
        status or default,
        headers={"Retry-After": "1"} if code == "busy" else None,
    )


def parse(body: bytes, model_id: str) -> AnonymizeRequest:
    """The request, checked against api.2's limits; ApiError otherwise. A pydantic error carries the input,
    so only its location is read."""
    try:
        req = AnonymizeRequest.model_validate_json(body)
    except ValidationError as e:
        loc = e.errors()[0]["loc"]
        field = str(loc[0]) if loc and loc[0] in AnonymizeRequest.model_fields else None
        raise ApiError("invalid_request", field) from None
    if req.model != model_id:
        raise ApiError("model_not_found", "model")
    if not req.texts or not req.features:
        raise ApiError("invalid_request", "texts" if not req.texts else "features")
    if (
        len(req.texts) > MAX_TEXTS
        or sum(map(len, req.texts)) > MAX_CHARS
        or max(len(req.keep.emails), len(req.keep.phones)) > MAX_KEEP
    ):
        raise ApiError("too_large")
    for f in req.features:
        if f not in FEATURES:  # the code back only when it looks like one, never free text
            raise ApiError("unsupported_feature", f if re.fullmatch(r"[A-Z][A-Z0-9_]{0,31}", f) else None)
    if len(set(req.features)) != len(req.features):
        raise ApiError("invalid_request", "features")
    emails_ok = all(
        anonymizer.EMAIL_REGEX.fullmatch(e) and anonymizer.validate_email(e) for e in req.keep.emails
    )
    if not emails_ok or not all(map(_phone, req.keep.phones)):
        raise ApiError("invalid_request", "keep")  # exact addresses and numbers only: no domains or wildcards
    return req


def _phone(value: str) -> bool:
    """One whole number a phone rule takes, the English ones too when they run."""
    if anonymizer.PHONE_REGEX.fullmatch(value):
        return True
    english = anonymizer.ENGLISH and anonymizer.ENGLISH_PHONE_REGEX.fullmatch(value)
    return bool(english) and anonymizer.validate_english_phone(value)


def _trim(text: str, start: int, end: int, matches: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """An NER span less the regex matches inside it, each piece without its edge whitespace; a piece with no
    letter or digit goes (api.4 step 4)."""
    cuts = sorted((s, e) for s, e in matches if s < end and e > start)
    if not cuts:
        return [(start, end)]
    bounds, pos = [], start
    for s, e in cuts:
        bounds.append((pos, s))
        pos = max(pos, e)
    bounds.append((pos, end))
    pieces = []
    for a, b in bounds:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if any(c.isalnum() for c in text[a:b]):
            pieces.append((a, b))
    return pieces


def anonymize(
    texts: Sequence[str], features: Sequence[str], keep: Keep, ner: Ner | None
) -> tuple[list[str], list[Record]]:
    """api.4 steps 1-5: the texts with every requested, not kept match replaced by `<TYPE_n>`, numbered
    across the request in reading order, and the spans in code points of the texts as sent."""
    kept = {("EMAIL", anonymizer._entity_key("EMAIL", e)) for e in keep.emails}
    kept |= {("PHONE", anonymizer._entity_key("PHONE", p)) for p in keep.phones}

    def wanted(s: Record) -> bool:
        return s["type"] in features and (s["type"], s["key"]) not in kept

    def pesel(text: str, p: Record) -> Record:
        """A PESEL of someone born in 1948 is also 48 and a phone number, which the phone rule finds first: it
        is a PESEL when PESEL is requested and the number is not a kept phone, else a phone (api.4 step 1); so
        is one born in 2007 that the English rules take for a UK mobile (07 and nine digits)."""
        value = text[p["start"] : p["end"]]
        if p["type"] != "PHONE" or "PESEL" not in features or ("PHONE", p["key"]) in kept:
            return p
        return {**p, "type": "PESEL", "key": value} if anonymizer.validate_pesel(value) else p

    found = []
    stages = anonymizer.stages(texts, ner, api=True)
    for i, (text, (placed, snapped)) in enumerate(zip(texts, stages, strict=True)):
        placed = [pesel(text, p) for p in placed]
        matches = [(p["start"], p["end"]) for p in placed]
        model = []
        for a in snapped:
            pieces = _trim(text, a["start"], a["end"], matches)
            key = anonymizer._entity_key(a["type"], " ".join(text[s:e] for s, e in pieces))
            model += [
                {"start": s, "end": e, "type": a["type"], "key": key, "source": "model"} for s, e in pieces
            ]
        # step 4: the overlap rule last, so an unrequested, kept or trimmed-away span never wins over a name
        model = anonymizer._rightmost([m for m in model if wanted(m)])
        spans = [{**p, "source": "regex"} for p in placed if wanted(p)] + model
        found += [(i, s) for s in sorted(spans, key=lambda s: s["start"])]
    registry = EntityRegistry()
    # Polish case endings mean nothing to an English instance's names, and it was never measured with them
    names = (s["key"] for _, s in found if s["type"] == "PERSON")
    people = {} if anonymizer.ENGLISH else persons.merged(names)
    entities = [
        {"text": i, "start": s["start"], "end": s["end"], "type": s["type"],
         "label": f"<{s['type']}_{registry.assign(s['type'], people.get(s['key'], s['key']))}>",
         "source": s["source"]}
        for i, s in found
    ]  # fmt: skip
    out = [
        anonymizer._materialize(t, [e for e in entities if e["text"] == i])[0] for i, t in enumerate(texts)
    ]
    return out, entities


def create_app(model_id: str, ner: Ner, api_key: str) -> FastAPI:
    """The API over a loaded NER pipeline (tests inject a fake one)."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    slots = threading.BoundedSemaphore(QUEUE + 1)
    # ponytail: one model run at a time; a pool of pipelines (or GPU batching across requests) for throughput
    model_lock = threading.Lock()

    def locked_ner(request: Request, texts: list[str], **kwargs: Any) -> list[list[Record]]:
        with model_lock:
            # A client that gave up while this waited reads nothing: its run would only hold up the next one.
            if from_thread.run(request.is_disconnected):
                raise ApiError("client_left")
            return ner(texts, **kwargs)

    def authorized(request: Request) -> bool:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.encode(), api_key.encode())

    @app.exception_handler(HTTPException)
    async def unknown_route(_: Request, exc: HTTPException) -> JSONResponse:
        return error("not_found" if exc.status_code == 404 else "invalid_request", status=exc.status_code)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}  # the port opens only after the manifest check and the warm-up

    @app.get("/v1/models")
    def models(request: Request) -> Any:
        if not authorized(request):
            return error("invalid_api_key")
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/anonymize")
    async def anonymize_route(request: Request) -> JSONResponse:
        started, rid = time.monotonic(), uuid.uuid4().hex[:12]
        try:
            if not authorized(request):
                raise ApiError("invalid_api_key")
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > MAX_BODY:
                raise ApiError("too_large")
            body = bytearray()
            async for chunk in request.stream():
                body += chunk
                if len(body) > MAX_BODY:
                    raise ApiError("too_large")
            req = parse(bytes(body), model_id)
            model = bool(MODEL_FEATURES & set(req.features))  # A4: regex-only requests skip the model's queue
            if model and not slots.acquire(blocking=False):
                raise ApiError("busy")
            try:
                use = partial(locked_ner, request) if model else None
                texts, entities = await run_in_threadpool(anonymize, req.texts, req.features, req.keep, use)
            finally:
                if model:
                    slots.release()
        except ApiError as e:
            log.info("anonymize id=%s status=%d code=%s ms=%d", rid, ERRORS[e.code][0], e.code, _ms(started))
            return error(e.code, e.param)
        except Exception as e:  # only the type: a message can quote the text
            log.error("anonymize id=%s status=500 error=%s ms=%d", rid, type(e).__name__, _ms(started))
            return error("server_error")
        counts = dict(Counter(e["type"] for e in entities))
        log.info(
            "anonymize id=%s status=200 texts=%d chars=%d features=%s entities=%s ms=%d",
            rid, len(texts), sum(map(len, req.texts)), ",".join(req.features), counts, _ms(started),
        )  # fmt: skip
        return JSONResponse(
            {"model": model_id, "features": req.features, "texts": texts, "entities": entities}
        )

    return app


def _ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def verify(model_dir: Path) -> str:
    """The served model's id, once the directory holds exactly manifest.json's files, each at its sha256 (A9);
    ValueError otherwise. An extra file is refused too: transformers would prefer a model.safetensors or a
    tokenizer.json to the files the manifest names."""
    manifest = json.loads((model_dir / "manifest.json").read_text("utf-8"))
    files: dict[str, str] = manifest["files"]
    present = {p.name for p in model_dir.iterdir()} - {"manifest.json"}
    if present != set(files) or not set(anonymizer.REQUIRED_MODEL_FILES) <= present:
        raise ValueError(f"{model_dir} does not hold exactly the files manifest.json lists")
    for name, digest in sorted(files.items()):
        h = hashlib.sha256()
        with (model_dir / name).open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise ValueError(f"{name} does not match its sha256 in manifest.json")
    return f"{manifest['name']}-{manifest['version']}"


def load(model_dir: Path) -> Ner:
    """The service's NER pipeline over the directory (anonymizer.py's settings, windowed), warmed up once."""
    import tokenizers
    import torch
    from transformers import pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("ner", model=str(model_dir), aggregation_strategy="simple", device=device)
    ner = anonymizer.windowed(pipe, pipe.tokenizer)
    ner([WARMUP], **anonymizer.NER_KWARGS)
    log.info("model loaded device=%s tokenizers=%s", device, tokenizers.__version__)
    return ner


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # a local directory only, never the Hub
    api_key, model_dir = os.environ.get("ANON_API_KEY", ""), os.environ.get("ANON_MODEL_DIR", "")
    if len(api_key) < 16 or not model_dir:
        sys.exit("flisak_anon.server: set ANON_MODEL_DIR and ANON_API_KEY (at least 16 characters)")
    if os.environ.get("ANON_LANGUAGE", "pl") not in ("pl", "en"):
        sys.exit("flisak_anon.server: ANON_LANGUAGE is pl or en")
    try:
        model_id = verify(Path(model_dir))
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"flisak_anon.server: refusing to start: {e}")
    log.info("manifest verified model=%s english=%s", model_id, anonymizer.ENGLISH)
    import uvicorn

    app = create_app(model_id, load(Path(model_dir)), api_key)
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=False, server_header=False)


if __name__ == "__main__":
    main()
