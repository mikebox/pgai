"""LangGraph flow for the loan-application chatbot.

One graph run per applicant input. Input is either a submitted form (answers
for one question group), a free-text chat message, or the "submit" action on
the review card.

    START ─┬─> closed                       (application already decided)
           └─> extract ─┬─> ask             (next group's form / fix answers / clarify via Gemini)
                        ├─> confirm         (all collected -> review card)
                        └─> predict         (submitted -> ML model -> decision)

Questions are asked in themed groups (fields.GROUPS), one form per group.
Form answers are validated directly in code (no LLM call); Gemini is used to
read free-text messages, map "Other" answers to codes, and write clarifying
questions. Every Gemini call goes through `pii.PIISafeLLM`: name, email, phone
and other identifiers are swapped for placeholders before the prompt leaves
the process and restored in the output. State is persisted per session by the checkpointer.
"""
from __future__ import annotations

import operator
import threading
import uuid
from datetime import date, datetime
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import audit, config, observability
from .pii import PIISafeLLM
from .fields import (FIELDS, build_model_input, cross_validate, display, field_ui,
                     fields_to_collect, groups_for)
from .llm import LLM, GeminiLLM, OllamaLLM
from .ml_model import LoanModel, load_model, timed_predict

Stage = Literal["collecting", "confirming", "done"]
InputKind = Literal["form", "text", "submit"]


class ChatState(TypedDict, total=False):
    session_id: str
    # user:      {"role": "user", "content": str, "form"?: {field: raw}, "action"?: "submit"}
    # assistant: {"role": "assistant", "content": str, "form"?: <form spec for the UI>}
    messages: Annotated[list[dict], operator.add]
    application: dict          # validated answers, keyed by FIELDS name
    errors: dict               # field -> problem with the latest answer
    ambiguities: list[str]
    stage: Stage
    input_kind: InputKind
    confirmed: bool
    model_output: dict | None


def _transcript(messages: list[dict], last: int = 10) -> str:
    who = {"user": "Applicant", "assistant": "Assistant"}
    return "\n".join(f"{who[m['role']]}: {m['content']}" for m in messages[-last:])


def _group_form(groups, app: dict, errors: dict) -> dict | None:
    """Form spec for the first question group that still has missing or invalid answers."""
    for i, (gid, title, names) in enumerate(groups):
        if any(n not in app or n in errors for n in names):
            return {"type": "group", "id": gid, "title": title, "step": i + 1, "steps": len(groups),
                    "fields": [field_ui(n, app.get(n), errors.get(n)) for n in names]}
    return None


def build_graph(llm: LLM, model: LoanModel, checkpointer=None):
    collect = fields_to_collect(model.required_fields)
    groups = groups_for(collect)

    def group_form(app: dict, errors: dict) -> dict | None:
        return _group_form(groups, app, errors)

    def review_form(app: dict) -> dict:
        return {"type": "review", "id": "review", "title": "Review & submit", "step": len(groups), "steps": len(groups),
                "fields": [field_ui(n, app.get(n)) | {"display": display(n, app.get(n))} for n in collect]}

    # ---- helpers -------------------------------------------------------------------
    def apply_values(raw: dict, app: dict, errors: dict) -> bool:
        changed = False
        for name, value in raw.items():
            if name not in collect or value in (None, ""):
                continue
            clean, err = FIELDS[name].validate(value)
            if err:
                errors[name] = err
                app.pop(name, None)
            else:
                changed |= app.get(name) != clean
                app[name] = clean
                errors.pop(name, None)
        return changed

    # ---- nodes ---------------------------------------------------------------------
    def extract(state: ChatState) -> dict:
        app = dict(state.get("application", {}))
        errors = dict(state.get("errors", {}))
        stage = state.get("stage", "collecting")
        sid = state["session_id"]
        msg = state["messages"][-1]

        if msg.get("action") == "submit":
            audit.log_event(sid, "user_submit")
            ok = stage == "confirming" and not errors and all(f in app for f in collect)
            return {"input_kind": "submit", "confirmed": ok, "ambiguities": []}

        ambiguities: list[str] = []
        confirmation = "none"
        if msg.get("form") is not None:
            kind: InputKind = "form"
            raw = {k: v for k, v in msg["form"].items() if k in collect}
            audit.log_event(sid, "user_form", values=audit.mask_fields(raw))
            # "Other" free-text answers to choice questions -> let Gemini map them to a code.
            other = {k: v for k, v in raw.items()
                     if FIELDS[k].kind == "choice" and v and FIELDS[k].validate(v)[1]}
            if other:
                text = "; ".join(f"{FIELDS[k].question} {v}" for k, v in other.items())
                ex = PIISafeLLM(llm, app).extract(f"Applicant: {text}", {}, list(other))
                for k in other:
                    mapped = getattr(ex.fields, k, None)
                    raw[k] = mapped if mapped else raw[k]
            changed = apply_values(raw, app, errors)
        else:
            kind = "text"
            audit.log_event(sid, "user_message", text=audit.mask_text(msg["content"], app))
            ex = PIISafeLLM(llm, app).extract(_transcript(state["messages"]), audit.mask_fields(app), collect)
            changed = apply_values(ex.fields.model_dump(), app, errors)
            ambiguities = list(ex.ambiguities)
            confirmation = ex.confirmation

        for name, err in cross_validate(app).items():
            errors[name] = err
            app.pop(name, None)

        confirmed = False
        if stage == "confirming":
            if kind == "text" and confirmation == "yes" and not changed and not errors:
                confirmed = True
            elif changed or confirmation == "no":
                stage = "collecting"
                if kind == "text" and not changed:
                    ambiguities.append("The applicant wants to change something; ask which detail "
                                       "(they can also press Edit on the review card).")

        audit.log_event(sid, "extraction", input_kind=kind, application=audit.mask_fields(app),
                        errors=errors, ambiguities=ambiguities, confirmation=confirmation)
        return {"application": app, "errors": errors, "ambiguities": ambiguities,
                "stage": stage, "confirmed": confirmed, "input_kind": kind}

    def ask(state: ChatState) -> dict:
        app, errors = state.get("application", {}), state.get("errors", {})
        ambiguities = state.get("ambiguities", [])
        form = group_form(app, errors) or review_form(app)

        if state.get("input_kind") != "text" and not ambiguities:
            # Form path: no LLM needed – errors are shown inline on the form.
            if errors:
                text = "Almost there – please check the highlighted answer" + ("s" if len(errors) > 1 else "") + "."
            elif state["messages"][-1].get("source") == "id":
                text = ("I've filled in what I could read from your ID card. Please check it"
                        + (" and fill in the rest." if form["type"] == "group" else " and submit."))
            else:
                text = f"Thanks! Next up: {form['title'].lower()}."
            return {"messages": [{"role": "assistant", "content": text, "form": form}]}

        parts = ["If the applicant asked a question, answer it briefly first."]
        if errors:
            parts.append("Some answers need correcting – explain kindly: "
                         + "; ".join(f"{FIELDS[k].label}: {v}" for k, v in errors.items()))
        if ambiguities:
            parts.append("Ask a clarifying question about: " + "; ".join(ambiguities))
        if form["type"] == "group":
            parts.append(f"A form titled '{form['title']}' is shown right below your message asking for: "
                         + ", ".join(f["label"] for f in form["fields"])
                         + ". Invite them to fill it in (or just type the answers); don't list every question again.")
        else:
            parts.append("All details are collected and a review card is shown below; ask them to check it and "
                         "press Submit, or tell you what to change.")
        text = PIISafeLLM(llm, app).reply(_transcript(state["messages"]), " ".join(parts))
        return {"messages": [{"role": "assistant", "content": text, "form": form}]}

    def confirm(state: ChatState) -> dict:
        text = "That's everything! Please review your details and submit."
        return {"messages": [{"role": "assistant", "content": text, "form": review_form(state["application"])}],
                "stage": "confirming"}

    def predict(state: ChatState) -> dict:
        features = build_model_input(state["application"], model.required_fields)
        out = timed_predict(model, features)
        audit.log_event(state["session_id"], "prediction", features=audit.mask_fields(out.features),
                        probability_decline=out.probability_decline, threshold=out.threshold,
                        decision=out.decision, model=out.model_name, model_version=out.model_version,
                        latency_ms=out.latency_ms, is_placeholder=out.is_placeholder)
        if out.decision == "APPROVED":
            text = ("🎉 Good news – your loan application is **approved in principle**.\n\n"
                    "Next steps: visit your ABC dealership with your KYC documents (PAN, Aadhaar, address proof) "
                    "and income proof to complete verification and sign the loan agreement.")
        else:
            text = ("Thank you for applying. We're unable to approve this application at the moment.\n\n"
                    "Next steps: you may reapply after 3 months, consider a higher down payment or a lower-priced "
                    "vehicle, or speak to a dealership executive about other options.")
        if out.is_placeholder:
            text += "\n\n_⚠️ Placeholder model in use – this is not a real credit decision._"
        return {"messages": [{"role": "assistant", "content": text}], "stage": "done", "model_output": out.to_dict()}

    def closed(state: ChatState) -> dict:
        audit.log_event(state["session_id"], "user_message_after_decision",
                        text=audit.mask_text(state["messages"][-1]["content"], state.get("application")))
        return {"messages": [{"role": "assistant", "content":
                "Your application has already been submitted and decided. Start a new application to apply again."}]}

    # ---- routing ---------------------------------------------------------------------
    def route_start(state: ChatState) -> str:
        return "closed" if state.get("stage") == "done" else "extract"

    def route_after_extract(state: ChatState) -> str:
        app = state.get("application", {})
        if state.get("errors") or state.get("ambiguities") or any(f not in app for f in collect):
            return "ask"
        if state.get("confirmed"):
            return "predict"
        if state.get("stage") == "confirming" and state.get("input_kind") == "text":
            return "ask"            # user typed something else while reviewing -> answer, show card again
        return "confirm"

    g = StateGraph(ChatState)
    for name, fn in [("extract", extract), ("ask", ask), ("confirm", confirm), ("predict", predict), ("closed", closed)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, route_start, ["extract", "closed"])
    g.add_conditional_edges("extract", route_after_extract, ["ask", "confirm", "predict"])
    for n in ("ask", "confirm", "predict", "closed"):
        g.add_edge(n, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


Mode = Literal["standard", "secure"]
MODES: tuple[Mode, ...] = ("standard", "secure")


def _age_from_dob(dob: str) -> str:
    """Age in whole years, computed in code (not by the model). '' if the date can't be read."""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            d = datetime.strptime(dob.strip(), fmt).date()
            break
        except ValueError:
            continue
    else:
        return ""
    t = date.today()
    return str(t.year - d.year - ((t.month, t.day) < (d.month, d.day)))


class LoanChatbot:
    """Small façade used by the UIs: one instance serves many sessions.

    Each session runs in a fixed mode chosen when it starts:
      - "standard": Gemini (cloud) for the conversation.
      - "secure":   local Gemma 4 via Ollama; nothing is sent to a cloud LLM and LangSmith tracing is off.
    Both graphs share one checkpointer; each LLM is created only when a session first needs it."""

    def __init__(self, llm: LLM | None = None, model: LoanModel | None = None, secure_llm: LLM | None = None):
        self.model = model or load_model()
        self._llms: dict[str, LLM | None] = {"standard": llm, "secure": secure_llm or llm}
        self._graphs: dict[str, object] = {}
        self._modes: dict[str, Mode] = {}
        self._checkpointer = InMemorySaver()
        self._lock = threading.Lock()
        self.collect = fields_to_collect(self.model.required_fields)
        self.n_questions = len(self.collect)
        self.n_steps = len(groups_for(self.collect))

    def _graph(self, mode: Mode):
        with self._lock:
            if mode not in self._graphs:
                llm = self._llms.get(mode) or (OllamaLLM() if mode == "secure" else GeminiLLM())
                self._llms[mode] = llm
                self._graphs[mode] = build_graph(llm, self.model, self._checkpointer)
            return self._graphs[mode]

    def mode(self, session_id: str) -> Mode:
        return self._modes.get(session_id, "standard")

    def llm_name(self, mode: Mode) -> str:
        llm = self._llms.get(mode)
        return getattr(llm, "model", None) or (config.SECURE_LLM_MODEL if mode == "secure" else config.LLM_MODEL)

    def new_session(self, mode: Mode = "standard") -> tuple[str, dict]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        sid = uuid.uuid4().hex
        graph = self._graph(mode)
        self._modes[sid] = mode
        greeting = {"role": "assistant", "form": _group_form(groups_for(self.collect), {}, {}), "content":
                    f"Let's get started – {self.n_questions} quick questions in {self.n_steps} short steps. "
                    "First, your contact details. " + (
                        "You can also type or speak your answers, upload a photo of your ID card to fill in your "
                        "name, age, pincode, email and phone, or ask me anything below." if mode == "secure" else
                        "You can also type your answers or ask me anything below.")}
        graph.update_state(self._cfg(sid), {
            "session_id": sid, "messages": [greeting], "application": {}, "errors": {}, "ambiguities": [],
            "stage": "collecting", "input_kind": "text", "confirmed": False, "model_output": None,
        })
        audit.log_event(sid, "session_start", model=self.model.name, model_version=self.model.version,
                        mode=mode, llm=self.llm_name(mode))
        return sid, greeting

    def send(self, session_id: str, text: str | None = None, form: dict | None = None,
             action: str | None = None, source: str | None = None, shown: str | None = None) -> dict:
        """Send one input: free text, a submitted form {field: value}, or action='submit'. Returns the bot message."""
        if action == "submit":
            msg = {"role": "user", "content": "Submit application", "action": "submit"}
        elif form is not None:
            shown = shown or "; ".join(f"{FIELDS[k].label.capitalize()}: {v}" for k, v in form.items() if k in FIELDS and v)
            msg = {"role": "user", "content": shown or "(empty form)", "form": form, **({"source": source} if source else {})}
        else:
            msg = {"role": "user", "content": (text or "").strip()}
        mode = self.mode(session_id)
        graph = self._graph(mode)
        if mode == "secure":   # keep everything on this machine: no LangSmith tracing either
            with observability.tracing_disabled():
                state = graph.invoke({"messages": [msg]}, self._cfg(session_id))
        else:
            cfg = {**self._cfg(session_id), **observability.run_config(session_id)}
            with observability.trace_context(session_id):
                state = graph.invoke({"messages": [msg]}, cfg)
        return state["messages"][-1]

    # ---- voice and ID-card input: most-secure mode only (local Gemma, nothing leaves this machine) ----
    def _secure_llm(self, session_id: str):
        if not self.state(session_id):
            raise KeyError("unknown session")
        if self.mode(session_id) != "secure":
            raise PermissionError("voice and ID-card input are only available in most secure mode")
        if self.state(session_id).get("stage") == "done":
            raise PermissionError("this application has already been submitted")
        self._graph("secure")
        return self._llms["secure"]

    def transcribe(self, session_id: str, wav: bytes) -> str:
        """Speech -> text. The text goes back to the applicant to check and send; it isn't sent automatically."""
        text = self._secure_llm(session_id).transcribe(wav)
        audit.log_event(session_id, "voice_input", audio_kb=round(len(wav) / 1024), chars=len(text))
        return text

    def read_id(self, session_id: str, image: bytes) -> tuple[dict, list[str]]:
        """Read name / age / pincode / email / phone from an ID-card photo and apply them like a submitted form.
        The image is processed in memory and never stored or logged. Returns (bot message, fields filled)."""
        r = self._secure_llm(session_id).read_id(image)
        values = {"Name": r["full_name"], "Age": _age_from_dob(r["date_of_birth"]) or r.get("age", ""),
                  "Pincode": r["pincode"], "Email": r.get("email", ""), "Phone": r.get("phone", "")}
        values = {k: v for k, v in values.items() if v and k in self.collect} if r["is_id_document"] else {}
        audit.log_event(session_id, "id_upload", is_id_document=r["is_id_document"], fields_read=sorted(values))
        if not values:
            why = ("That doesn't look like an ID card" if not r["is_id_document"]
                   else "I couldn't read the details on that ID card")
            msg = {"role": "assistant", "content": why + ". You can try a clearer photo, or just type your answers."}
            self._graph("secure").update_state(self._cfg(session_id), {"messages": [
                {"role": "user", "content": "(uploaded an image)", "source": "id"}, msg]}, as_node="ask")
            return msg, []
        shown = "Uploaded ID card – " + ", ".join(FIELDS[k].label for k in values)
        return self.send(session_id, form=values, source="id", shown=shown), sorted(values)

    def progress(self, session_id: str) -> dict:
        s = self.state(session_id)
        app, stage = s.get("application", {}), s.get("stage")
        groups = groups_for(self.collect)
        done = sum(all(f in app for f in g[2]) for g in groups)
        return {"collected": sum(f in app for f in self.collect), "total": self.n_questions,
                "steps": len(groups), "steps_done": done, "stage": stage}

    def state(self, session_id: str) -> ChatState:
        graph = self._graphs.get(self.mode(session_id)) or next(iter(self._graphs.values()), None)
        return graph.get_state(self._cfg(session_id)).values if graph is not None else {}

    @staticmethod
    def _cfg(sid: str) -> dict:
        return {"configurable": {"thread_id": sid}}
