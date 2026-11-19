"""End-to-end graph tests with a scripted fake LLM (no API calls).
Run from the project root:  python -m chat.tests.test_flow   (or pytest chat/tests)"""
import pathlib
import tempfile

from chat import config
from chat.fields import FIELDS
from chat.graph import LoanChatbot
from chat.llm import Extraction, ExtractedFields
from chat.ml_model import ModelOutput, decide


class FakeLLM:
    """Returns pre-scripted extractions, one per extract() call."""
    def __init__(self, script=()):
        self.script = list(script)
        self.calls = 0
        self.prompts = []          # everything that would have been sent to the LLM

    def extract(self, transcript, collected, wanted):
        self.calls += 1
        self.prompts.append(f"{transcript}\n{collected}")
        fields, conf, amb = self.script.pop(0)
        return Extraction(fields=ExtractedFields(**{k: fields.get(k) for k in FIELDS}),
                          confirmation=conf, ambiguities=amb)

    def reply(self, transcript, instructions):
        self.calls += 1
        self.prompts.append(f"{transcript}\n{instructions}")
        return f"[ask] {instructions}"


class FakeModel:
    name, version = "fake", "1"
    required_fields = ["Pincode", "Age", "Employment_Type", "Net_salary", "PAST_LOANS_ACTIVE",
                       "Model_Description", "Loan_Amount", "LTV"]

    def predict(self, app):
        p = 0.8 if app["LTV"] > 90 else 0.2
        return ModelOutput(decide(p, 0.5), p, 0.5, self.name, self.version, features=dict(app))


CONTACT = {"Name": "Ravi Kumar", "Email": "ravi.k@example.com", "Phone": "+91 98765 43210"}
PII = ("Ravi", "Kumar", "ravi.k@example.com", "98765", "43210")


def _bot(tmp, script=()):
    config.AUDIT_LOG = pathlib.Path(tmp) / "audit.jsonl"
    llm = FakeLLM(script)
    return LoanChatbot(llm=llm, model=FakeModel()), llm


def test_form_flow(tmp_path):
    bot, llm = _bot(tmp_path, [({"Employment_Type": "AGR"}, "none", [])])   # only for the "Other…" answer
    sid, greeting = bot.new_session()
    assert greeting["form"]["title"] == "Your contact details" and greeting["form"]["step"] == 1
    assert [f["name"] for f in greeting["form"]["fields"]] == ["Name", "Email", "Phone"]

    # step 1: contact details, validated in code
    m = bot.send(sid, form={**CONTACT, "Email": "not-an-email"})
    assert next(f for f in m["form"]["fields"] if f["name"] == "Email")["error"]
    m = bot.send(sid, form=CONTACT)
    assert bot.state(sid)["application"]["Phone"] == "9876543210"
    assert m["form"]["title"] == "About you" and m["form"]["step"] == 2
    assert [f["name"] for f in m["form"]["fields"]] == ["Age", "Pincode", "Employment_Type"]

    # group 1, with an "Other" free-text employment answer mapped by the LLM
    m = bot.send(sid, form={"Age": "30", "Pincode": "560002", "Employment_Type": "I farm paddy"})
    assert bot.state(sid)["application"]["Employment_Type"] == "AGR"
    assert m["form"]["title"] == "Your finances" and m["form"]["step"] == 3

    # group 2, invalid income -> same form back with inline error, no LLM call
    calls = llm.calls
    m = bot.send(sid, form={"Net_salary": "65", "PAST_LOANS_ACTIVE": "NO_PAST_LOANS"})
    assert llm.calls == calls
    assert m["form"]["title"] == "Your finances"
    assert next(f for f in m["form"]["fields"] if f["name"] == "Net_salary")["error"]
    m = bot.send(sid, form={"Net_salary": "35k", "PAST_LOANS_ACTIVE": "NO_PAST_LOANS"})
    assert m["form"]["title"] == "Your vehicle"

    # group 3, loan > price -> error; fix it -> review card
    m = bot.send(sid, form={"Model_Description": "JUPITER 125 BSVI - DISC", "Vehicle_Price": "1 lakh", "Loan_Amount": "120000"})
    assert next(f for f in m["form"]["fields"] if f["name"] == "Loan_Amount")["error"]
    m = bot.send(sid, form={"Model_Description": "JUPITER 125 BSVI - DISC", "Vehicle_Price": "1 lakh", "Loan_Amount": "80000"})
    assert m["form"]["type"] == "review" and bot.state(sid)["stage"] == "confirming"
    assert {f["name"]: f["display"] for f in m["form"]["fields"]}["Net_salary"] == "₹35,000"
    assert {f["name"]: f["display"] for f in m["form"]["fields"]}["Vehicle_Price"] == "₹1,00,000"

    # edit from the review card -> review again
    m = bot.send(sid, form={"Loan_Amount": "75000"})
    assert m["form"]["type"] == "review" and bot.state(sid)["application"]["Loan_Amount"] == 75000

    m = bot.send(sid, action="submit")
    s = bot.state(sid)
    assert s["stage"] == "done" and s["model_output"]["decision"] == "APPROVED"
    assert s["model_output"]["features"]["LTV"] == 75.0
    assert not set(CONTACT) & set(s["model_output"]["features"])     # contact details never reach the model
    assert bot.progress(sid)["steps_done"] == 4
    assert llm.calls == 1   # whole form path used the LLM only for the "Other" answer

    assert not any(p in prompt for prompt in llm.prompts for p in PII)   # no PII reached the LLM

    log = config.AUDIT_LOG.read_text()
    assert "560002" not in log and "560***" in log and '"event": "prediction"' in log
    assert not any(p in log for p in PII) and "******3210" in log


def test_free_text_flow(tmp_path):
    bot, llm = _bot(tmp_path, [
        ({"Name": "[NAME_1]", "Email": "[EMAIL_1]", "Phone": "[PHONE_1]"}, "none", []),
        ({"Age": "30", "Pincode": "560002", "Employment_Type": "SAL"}, "none", []),
        ({"Net_salary": "35k", "PAST_LOANS_ACTIVE": "NO_PAST_LOANS"}, "none", ["is 35k monthly?"]),
        ({"Model_Description": "Jupiter", "Vehicle_Price": "100000", "Loan_Amount": "80000"}, "none", []),
        ({}, "yes", []),
    ])
    sid, _ = bot.new_session()
    m = bot.send(sid, "Hi, my name is Ravi Kumar, mail me at ravi.k@example.com or call +91 98765 43210")
    assert {k: bot.state(sid)["application"][k] for k in CONTACT} == {
        "Name": "Ravi Kumar", "Email": "ravi.k@example.com", "Phone": "9876543210"}
    assert "[NAME_1]" in llm.prompts[0] and "[EMAIL_1]" in llm.prompts[0] and "[PHONE_1]" in llm.prompts[0]
    assert m["form"]["title"] == "About you"
    m = bot.send(sid, "30, 560002, salaried. Ravi again, btw")
    assert m["form"]["title"] == "Your finances"
    m = bot.send(sid, "35k, no loans")
    assert "is 35k monthly?" in m["content"] and m["form"]["title"] == "Your vehicle"
    m = bot.send(sid, "Jupiter, 1 lakh, loan 80k")
    assert m["form"]["type"] == "review"
    bot.send(sid, "yes")
    assert bot.state(sid)["stage"] == "done"
    assert not any(p in prompt for prompt in llm.prompts for p in PII)
    assert "already been submitted" in bot.send(sid, "hello?")["content"]
    log = config.AUDIT_LOG.read_text()
    assert not any(p in log for p in PII) and "[NAME_1] again" in log


def test_trained_model_flow(tmp_path):
    """Full form flow against the real trained model (skipped if models/ has no artifact)."""
    if not config.TRAINED_MODEL_PATH.exists():
        print("skip: no trained model")
        return
    from chat.ml_model import load_model
    config.AUDIT_LOG = pathlib.Path(tmp_path) / "audit.jsonl"
    llm = FakeLLM()
    bot = LoanChatbot(llm=llm, model=load_model("trained"))
    assert bot.n_questions == 10 and bot.n_steps == 3
    results = {}
    for price in ("100000", "84000"):           # LTV 80% vs 95%
        sid, greeting = bot.new_session()
        assert [f["name"] for f in greeting["form"]["fields"]] == ["Name", "Email", "Phone"]
        m = bot.send(sid, form=CONTACT)
        assert [f["name"] for f in m["form"]["fields"]] == ["Age", "Pincode", "Employment_Type", "PAST_LOANS_ACTIVE"]
        m = bot.send(sid, form={"Age": "30", "Pincode": "560002", "Employment_Type": "SAL",
                                "PAST_LOANS_ACTIVE": "NO_PAST_LOANS"})
        assert m["form"]["title"] == "Your vehicle"
        m = bot.send(sid, form={"Model_Description": "JUPITER 125 BSVI - DISC", "Vehicle_Price": price,
                                "Loan_Amount": "80000"})
        assert m["form"]["type"] == "review"
        bot.send(sid, action="submit")
        out = bot.state(sid)["model_output"]
        assert not out["is_placeholder"] and out["threshold"] == 0.09
        results[price] = out["decision"]
    assert results == {"100000": "APPROVED", "84000": "DECLINED"}
    assert llm.calls == 0                        # form path needs no LLM at all
    log = config.AUDIT_LOG.read_text()
    assert "560002" not in log and '"model": "loan-decision-hgb"' in log


def test_pii_redactor():
    from chat.pii import PIISafeLLM, Redactor
    r = Redactor({"Name": "Ravi Kumar", "Email": "ravi@x.co.in", "Phone": "9876543210"})
    out = r.redact("Name: Ravi Kumar; Email: ravi@x.co.in; call me on 98765-43210. PAN ABCDE1234F, "
                   "Aadhaar 1234 5678 9012, a/c 123456789012345. Age 28, pincode 560002, loan 1,10,000")
    assert out == ("Name: [NAME_1]; Email: [EMAIL_1]; call me on [PHONE_1]. PAN [PAN_1], "
                   "Aadhaar [AADHAAR_1], a/c [ACCOUNT_1]. Age 28, pincode 560002, loan 1,10,000")
    assert r.redact("ravi here") == "[NAME_1] here"
    assert r.redact("this is my first loan") == "this is my first loan"
    assert r.restore("Thanks, [NAME_1]!") == "Thanks, Ravi Kumar!"

    # A contact value that isn't a placeholder can't have come from the applicant -> dropped.
    llm = FakeLLM([({"Name": "Some Guess", "Email": "[EMAIL_1]"}, "none", [])])
    ex = PIISafeLLM(llm, {"Email": "ravi@x.co.in"}).extract("Applicant: ravi@x.co.in", {}, ["Name", "Email"])
    assert ex.fields.Name is None and ex.fields.Email == "ravi@x.co.in"
    assert "ravi@x.co.in" not in llm.prompts[0]


if __name__ == "__main__":
    test_form_flow(pathlib.Path(tempfile.mkdtemp()))
    test_free_text_flow(pathlib.Path(tempfile.mkdtemp()))
    test_trained_model_flow(pathlib.Path(tempfile.mkdtemp()))
    test_pii_redactor()
    print("ok")
