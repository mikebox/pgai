"""LLM interaction layer (Gemini via the google-genai SDK, Interactions API).

The LLM does two things only:
  1. `extract` – read the applicant's message and pull out field values,
     note anything ambiguous, and detect a yes/no confirmation.
  2. `reply`   – write the next conversational message (clarifications,
     next questions, answering side questions).
It never sees or makes the credit decision, and never sees PII in clear text:
callers wrap it in `chat.pii.PIISafeLLM`, which redacts prompts and restores outputs.
"""
from __future__ import annotations

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


def _empty_extraction() -> Extraction:
    return Extraction(fields=ExtractedFields(**{k: None for k in FIELDS}), confirmation="none", ambiguities=[])
