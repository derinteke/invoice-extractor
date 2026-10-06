"""LLM extractors: Gemini (free API tier) and an optional local Ollama vision model."""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from pydantic import ValidationError

from .loaders import Payload, payload_images
from .schema import Document, gemini_json_schema

PROMPT = """You extract structured data from a scanned invoice or receipt.

Rules:
- Copy names, numbers and identifiers exactly as printed. Never guess; use null when a field is absent.
- Amounts are plain numbers: convert printed separators, e.g. "1.234,56" -> 1234.56, "60.000" on an
  IDR/TRY receipt -> 60000, "$ 889,20" -> 889.2. Discounts are negative.
- vendor_name / customer_name: the company or person name only, without the address.
- date: the issue date as YYYY-MM-DD. US-style invoices print MM/DD/YYYY.
- date_printed: the issue date copied character for character; don't reorder day and month.
- line_items: one entry per purchased item, in printed order, including add-ons that have their
  own price. Mark choices printed under an item without their own charge (size, "- less ice",
  spice level, flavours of a bundle) with is_option=true; a separately sold item is never an option.
  Skip sub-total, tax, discount, payment and change lines. `total` is the line amount before
  tax and before a line discount; put a printed line discount in `discount` (negative).
- subtotal is before tax. service_charge and tax are separate: never add them together.
- total is the amount due as printed, not the cash tendered or a rounded cash payment.
"""


MAX_RETRY_WAIT_S = 120


class ExtractionError(RuntimeError):
    pass


@dataclass
class ExtractionResult:
    document: Document
    model: str
    latency_s: float
    raw_text: str
    usage: dict = field(default_factory=dict)


class Extractor(Protocol):
    name: str

    def extract(self, payload: Payload) -> ExtractionResult: ...


def _parse(raw: str) -> Document:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return Document.model_validate_json(text)
    except ValidationError as exc:
        raise ExtractionError(f"Model output does not match the schema: {exc}") from exc


def _retry_delay(exc) -> float | None:
    """Read the server's suggested wait ('retryDelay': '17s') from a 429 response."""
    try:
        for detail in exc.details["error"]["details"]:
            if "retryDelay" in detail:
                return float(detail["retryDelay"].rstrip("s")) + 1
    except (KeyError, TypeError, AttributeError, ValueError):
        pass
    return None


class GeminiExtractor:
    """Gemini with JSON-schema constrained output. Respects free-tier rate limits."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 rpm: float | None = None, max_retries: int = 5, client=None):
        from google import genai

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.name = f"gemini/{self.model}"
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if client is None and not api_key:
            raise ExtractionError("GEMINI_API_KEY is not set. Get a free key at "
                                  "https://aistudio.google.com/apikey and put it in .env.")
        timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", 120))
        self.client = client or genai.Client(
            api_key=api_key, http_options=genai.types.HttpOptions(timeout=int(timeout_s * 1000))
        )
        self.min_interval = 60.0 / float(rpm or os.getenv("GEMINI_RPM", 8))
        self.max_retries = max_retries
        self._last_call = 0.0

    def _throttle(self):
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def extract(self, payload: Payload) -> ExtractionResult:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_json_schema=gemini_json_schema(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = [types.Part.from_bytes(data=payload.data, mime_type=payload.mime_type), PROMPT]
        for attempt in range(self.max_retries + 1):
            self._throttle()
            start = time.monotonic()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except errors.APIError as exc:
                retriable = exc.code == 429 or (exc.code or 0) >= 500
                if not retriable or attempt == self.max_retries:
                    raise ExtractionError(f"Gemini API error {exc.code}: {exc.message}") from exc
                delay = _retry_delay(exc) or min(60, 5 * 2 ** attempt)
                if delay > MAX_RETRY_WAIT_S:  # daily quota, not a per-minute limit
                    raise ExtractionError(
                        f"Gemini API error 429: quota exhausted for {self.model}; "
                        f"resets in {delay / 3600:.1f} h."
                    ) from exc
                print(f"[gemini] {exc.code}, retrying in {delay:.0f} s", file=sys.stderr)
                time.sleep(delay)
                continue
            except httpx.TimeoutException as exc:
                if attempt == self.max_retries:
                    raise ExtractionError("Gemini request timed out.") from exc
                print("[gemini] timeout, retrying", file=sys.stderr)
                continue
            latency = time.monotonic() - start
            raw = resp.text or ""
            if not raw:
                raise ExtractionError("Gemini returned an empty response.")
            meta = resp.usage_metadata
            usage = {
                "input_tokens": getattr(meta, "prompt_token_count", None),
                "output_tokens": getattr(meta, "candidates_token_count", None),
            } if meta else {}
            return ExtractionResult(_parse(raw), self.name, latency, raw, usage)
        raise ExtractionError("unreachable")


class OllamaExtractor:
    """Local vision model through Ollama's /api/chat with a JSON-schema `format`."""

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 600):
        self.model = model or os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
        self.name = f"ollama/{self.model}"
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def extract(self, payload: Payload) -> ExtractionResult:
        body = {
            "model": self.model,
            "stream": False,
            "format": gemini_json_schema(),
            "options": {"temperature": 0, "num_predict": 2048},
            "messages": [{
                "role": "user",
                "content": PROMPT,
                "images": [base64.b64encode(img).decode() for img in payload_images(payload)],
            }],
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read())
        except OSError as exc:
            raise ExtractionError(f"Ollama request failed: {exc}") from exc
        raw = out.get("message", {}).get("content", "")
        usage = {"input_tokens": out.get("prompt_eval_count"), "output_tokens": out.get("eval_count")}
        return ExtractionResult(_parse(raw), self.name, time.monotonic() - start, raw, usage)


def get_extractor(provider: str | None = None, model: str | None = None) -> Extractor:
    provider = (provider or os.getenv("EXTRACTOR", "gemini")).lower()
    if provider == "gemini":
        return GeminiExtractor(model=model)
    if provider == "ollama":
        return OllamaExtractor(model=model)
    raise ValueError(f"Unknown provider '{provider}' (use gemini or ollama)")
