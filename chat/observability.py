"""LangSmith tracing (optional – off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set).

What gets traced, per applicant turn (one trace, grouped into a LangSmith thread by session_id):
  * the LangGraph run and every node (extract / ask / confirm / predict / closed) with state in/out
  * each Gemini call (prompt, response, token usage, latency)
  * the ML model call (features, probability, decision, latency)

All trace payloads pass through the same PII masker as the audit log before
leaving the machine (name, phone, email, PAN, Aadhaar, long digit runs, pincodes).
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from . import audit, config  # noqa: F401  (config loads chat/.env first)

PROJECT = os.getenv("LANGSMITH_PROJECT", "abc-credit-loan-chatbot")


def enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def _mask(value: str, path) -> str:
    key = str(path[-1]) if path else ""
    if (m := audit.mask_value(key, value)) is not None:
        return m
    return audit.mask_text(value)


_client = None
if enabled():
    from langsmith import Client
    from langsmith.anonymizer import create_anonymizer

    _client = Client(anonymizer=create_anonymizer(_mask))


def run_config(session_id: str) -> dict:
    """Extra LangGraph config for one turn: tracer callback + thread metadata."""
    if not enabled():
        return {}
    from langchain_core.tracers.langchain import LangChainTracer

    return {
        "callbacks": [LangChainTracer(project_name=PROJECT, client=_client)],
        "run_name": "loan_chat_turn",
        "tags": ["loan-chatbot", f"llm:{config.LLM_MODEL}"],
        "metadata": {"session_id": session_id, "llm_model": config.LLM_MODEL},
    }


@contextmanager
def trace_context(session_id: str):
    """Makes @traceable children (Gemini, ML model) use the masked client and same thread."""
    if not enabled():
        yield
        return
    from langsmith import tracing_context

    with tracing_context(enabled=True, client=_client, project_name=PROJECT,
                         metadata={"session_id": session_id}):
        yield


def record_llm_usage(usage) -> None:
    """Attach Gemini token counts to the current LangSmith LLM run."""
    if not enabled() or usage is None:
        return
    from langsmith import get_current_run_tree

    rt = get_current_run_tree()
    if rt is None:
        return
    inp = getattr(usage, "total_input_tokens", None) or 0
    out = getattr(usage, "total_output_tokens", None) or 0
    thought = getattr(usage, "total_thought_tokens", None) or 0
    rt.set(usage_metadata={"input_tokens": inp, "output_tokens": out + thought,
                           "total_tokens": inp + out + thought,
                           "output_token_details": {"reasoning": thought}})
