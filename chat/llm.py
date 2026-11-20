"""LLM interaction layer: Gemini (google-genai SDK, Interactions API) in normal mode,
or a local Gemma 4 via Ollama in most-secure mode.

The LLM does two things only:
  1. `extract` – read the applicant's message and pull out field values,
     note anything ambiguous, and detect a yes/no confirmation.
  2. `reply`   – write the next conversational message (clarifications,
     next questions, answering side questions).
It never sees or makes the credit decision, and never sees PII in clear text:
callers wrap it in `chat.pii.PIISafeLLM`, which redacts prompts and restores outputs.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Literal, Protocol

from google import genai
from langsmith import traceable
from pydantic import BaseModel, Field, ValidationError, create_model

from . import config
from .fields import FIELDS
from .observability import record_llm_usage

# One optional string per registry field; validation/parsing happens in fields.py.
ExtractedFields = create_model(
    "ExtractedFields",
    **{name: (str | None, Field(default=None, description=spec.description)) for name, spec in FIELDS.items()},
)


class Extraction(BaseModel):
    fields: ExtractedFields = Field(description="Values stated in the LATEST applicant message only; null if not mentioned.")
    confirmation: Literal["yes", "no", "none"] = Field(
        description="If the assistant just asked the applicant to confirm their details: 'yes' if they confirmed, "
                    "'no' if they want changes, otherwise 'none'.")
    ambiguities: list[str] = Field(
        description="Short notes on anything the applicant said that is unclear and needs a clarifying question "
                    "(e.g. unclear whether income is monthly or yearly). Empty if none.")


class LLM(Protocol):
    def extract(self, transcript: str, collected: dict, wanted: list[str]) -> Extraction: ...
    def reply(self, transcript: str, instructions: str) -> str: ...


EXTRACT_SYSTEM = """You extract loan-application data from a chat between ABC Credit's two-wheeler loan assistant and an applicant.
Only extract values the applicant actually stated in their latest message (they may also be correcting an earlier answer).
Map free-text answers to the listed codes where a description gives codes (e.g. "I run a shop" -> SEP).
Keep numbers as the applicant gave them (e.g. "25k", "1.2 lakh"). Never guess or invent values; use null.
If something is ambiguous, leave the field null and describe the ambiguity.
Personal details (name, email, phone, IDs) have been replaced by placeholders such as [NAME_1], [EMAIL_1],
[PHONE_1]. For the Name, Email and Phone fields return the placeholder exactly as written; never invent a value."""

REPLY_SYSTEM = """You are the friendly assistant of ABC Credit, an NBFC, helping a customer apply for a two-wheeler loan by chat.
Style: warm, brief (1-3 short sentences), plain language, no markdown headings. Ask at most two things per message.
You only gather information and explain the process. You must NEVER say or hint whether the loan will be approved
or declined, and never estimate eligibility – a separate credit model makes that decision.
Name, email and mobile number are collected on the first form; never ask for PAN, Aadhaar or bank details.
Personal details appear as placeholders such as [NAME_1], [EMAIL_1] or [PHONE_1]. You may use a placeholder
exactly as written (e.g. "Thanks, [NAME_1]!") but never guess, spell out or alter what it stands for."""


class GeminiLLM:
    def __init__(self, model: str = config.LLM_MODEL, client: genai.Client | None = None):
        self.model = model
        self.client = client or genai.Client()   # reads GEMINI_API_KEY (or GOOGLE_API_KEY)

    @traceable(run_type="llm", name="gemini",
               metadata={"ls_provider": "google_genai", "ls_model_name": config.LLM_MODEL},
               process_inputs=lambda i: {"system": i["system"], "prompt": i["prompt"],
                                         "structured": bool(i.get("response_format"))})
    def _call(self, system: str, prompt: str, response_format: dict | None = None) -> str | None:
        kw: dict = dict(
            model=self.model, system_instruction=system, input=prompt,
            generation_config={"thinking_level": config.LLM_THINKING_LEVEL,
                               "max_output_tokens": config.LLM_MAX_TOKENS},
            store=False,   # don't keep applicant data on Google's side
        )
        if response_format:
            kw["response_format"] = response_format
        interaction = self.client.interactions.create(**kw)
        record_llm_usage(getattr(interaction, "usage", None))
        if interaction.status != "completed":
            return None
        return (interaction.output_text or "").strip() or None

    def extract(self, transcript: str, collected: dict, wanted: list[str]) -> Extraction:
        prompt = (f"Fields we still need or may be corrected: {', '.join(wanted)}\n"
                  f"Already collected: {collected}\n\nConversation:\n{transcript}")
        text = self._call(EXTRACT_SYSTEM, prompt, {"type": "text", "mime_type": "application/json",
                                                   "schema": Extraction.model_json_schema()})
        try:
            return Extraction.model_validate_json(text) if text else _empty_extraction()
        except ValidationError:
            return _empty_extraction()

    def reply(self, transcript: str, instructions: str) -> str:
        text = self._call(REPLY_SYSTEM, f"Conversation so far:\n{transcript}\n\n"
                                        f"Write the assistant's next message. {instructions}")
        return text or "Sorry, I couldn't process that. Could you rephrase?"


def parse_extraction(text: str | None) -> Extraction:
    """Lenient parse of a model's JSON extraction: strips code fences and trailing junk, and if the whole
    object is broken (e.g. truncated), still salvages the flat "fields" block. Unknown keys are dropped."""
    if not text:
        return _empty_extraction()
    s = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data: dict = {}
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            data = {}
    if not isinstance(data.get("fields"), dict):
        m = re.search(r'"fields"\s*:\s*(\{[^{}]*\})', s, re.S)     # fields is flat: no nested braces
        try:
            data = {"fields": json.loads(m.group(1))} if m else {}
        except json.JSONDecodeError:
            data = {}
    raw = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    fields = {k: (None if raw.get(k) in (None, "", "null") else str(raw.get(k))) for k in FIELDS}
    conf = data.get("confirmation") if data.get("confirmation") in ("yes", "no", "none") else "none"
    amb = [str(a) for a in data.get("ambiguities") or [] if a] if isinstance(data.get("ambiguities"), list) else []
    return Extraction(fields=ExtractedFields(**fields), confirmation=conf, ambiguities=amb)


def _empty_extraction() -> Extraction:
    return Extraction(fields=ExtractedFields(**{k: None for k in FIELDS}), confirmation="none", ambiguities=[])


class OllamaLLM:
    """Local model (Gemma 4) served by Ollama – used in most-secure mode, so prompts never leave this machine.
    Plain HTTP via the standard library; deliberately not @traceable."""

    def __init__(self, model: str = config.SECURE_LLM_MODEL, host: str = config.OLLAMA_HOST,
                 timeout: float = config.SECURE_LLM_TIMEOUT_S):
        self.model, self.host, self.timeout = model, host, timeout

    def _call(self, system: str, prompt: str, max_tokens: int = config.LLM_MAX_TOKENS,
              temperature: float = 0.6) -> str | None:
        # No Ollama `format` (JSON-schema) mode: with Gemma it often emits the right JSON and then pads with
        # whitespace until the token cap, so the object never closes. JSON is requested in the prompt instead.
        body = {"model": self.model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "options": {"num_predict": max_tokens, "temperature": temperature}}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Local model unavailable at {self.host} ({self.model}): {e}") from e
        return ((data.get("message") or {}).get("content") or "").strip() or None

    def extract(self, transcript: str, collected: dict, wanted: list[str]) -> Extraction:
        guide = "\n".join(f"- {k}: {FIELDS[k].description}" for k in wanted if k in FIELDS)
        prompt = (f"Fields to look for:\n{guide}\n\nAlready collected: {collected}\n\nConversation:\n{transcript}\n\n"
                  "Reply with ONLY one JSON object (no markdown, nothing after it):\n"
                  '{"fields": {"<field name>": "<value>", ...only fields stated in the latest applicant message},\n'
                  ' "confirmation": "yes" | "no" | "none",\n "ambiguities": ["short note", ...]}')
        return parse_extraction(self._call(EXTRACT_SYSTEM, prompt, max_tokens=800, temperature=0.1))

    def reply(self, transcript: str, instructions: str) -> str:
        text = self._call(REPLY_SYSTEM, f"Conversation so far:\n{transcript}\n\n"
                                        f"Write the assistant's next message. {instructions}")
        return text or "Sorry, I couldn't process that. Could you rephrase?"

    # ---- multimodal input (most-secure mode only; audio and images never leave this machine) ----------
    def _media_call(self, prompt: str, media: bytes, max_tokens: int) -> str:
        """Gemma 4 takes audio (16 kHz mono WAV) and images through Ollama's `images` field.
        No `format` schema here: with media attached it can make Gemma loop until the token cap."""
        body = {"model": self.model, "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
                "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(media).decode()]}]}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return ((json.loads(r.read()).get("message") or {}).get("content") or "").strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Local model unavailable at {self.host} ({self.model}): {e}") from e

    def transcribe(self, wav: bytes) -> str:
        text = self._media_call(TRANSCRIBE_PROMPT, wav, 400)
        return "" if "NO_SPEECH" in text else text

    def read_id(self, image: bytes) -> dict:
        """-> {"is_id_document": bool, "full_name": str, "date_of_birth": "YYYY-MM-DD", "age": str, "pincode": str,
        "email": str, "phone": str}.
        The ID number is never requested, and anything else the model returns is dropped."""
        text = self._media_call(READ_ID_PROMPT, image, 200)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
        out = {k: str(data.get(k) or "").strip() for k in ("full_name", "date_of_birth", "age", "pincode", "email", "phone")}
        out["age"] = out["age"] if re.fullmatch(r"\d{1,3}", out["age"]) else ""
        # Keep phone/email only if they really are one: the model sometimes puts the 12-digit Aadhaar number in phone.
        phone = re.sub(r"^(?:\+?91|0)(?=\d{10}$)", "", re.sub(r"[\s()-]", "", out["phone"]))
        out["phone"] = phone if re.fullmatch(r"[6-9]\d{9}", phone) else ""
        out["email"] = out["email"] if re.fullmatch(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}", out["email"]) else ""
        # Any card with a person's details is accepted. The model's yes/no flag is flaky (it flips on cards
        # marked SAMPLE/SPECIMEN even while reading every field), so a name plus one other detail also counts.
        flag = str(data.get("is_id_document")).strip().lower() == "true"
        out["is_id_document"] = flag or bool(out["full_name"] and any(out[k] for k in ("date_of_birth", "age", "pincode", "email", "phone")))
        return out


TRANSCRIBE_PROMPT = ("Transcribe this audio exactly as spoken. The speaker may use English, Hindi or a mix "
                     "(write Hindi words in Latin script). Output only the transcript, nothing else. "
                     "If there is no clear human speech (silence, a tone, music or noise), output exactly: NO_SPEECH")

READ_ID_PROMPT = """Look at this image. If it is any card or document showing a person's details (government ID, employee or student badge, membership or profile card, etc.), read that person's details. Reply with ONLY this JSON object and nothing else:
{"is_id_document": true or false, "full_name": "", "date_of_birth": "YYYY-MM-DD", "age": "", "pincode": "", "email": "", "phone": ""}
is_id_document is true if the image shows a person's name with other personal details, false otherwise (e.g. scenery, objects, blank pages).
Use "" for anything not clearly visible. age is a number only if an age is printed. pincode is the 6-digit postal code in the address. phone is the person's mobile number as printed.
Never output any ID number, card number or barcode/QR contents."""
