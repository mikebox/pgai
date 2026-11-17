"""Terminal chat.   Run from the project root:   python -m chat.cli
Shows each question group; answer in free text (Gemini reads it) or type 'submit' on the review step."""
from .graph import LoanChatbot


def _print(msg: dict) -> None:
    print(f"\nBot: {msg['content']}")
    form = msg.get("form")
    if not form:
        return
    print(f"\n  [{form['title']}]  step {form['step']} of {form['steps']}")
    for f in form["fields"]:
        if form["type"] == "review":
            print(f"   {f['label'].capitalize():16} {f['display']}")
            continue
        err = f"   ← {f['error']}" if f.get("error") else ""
        print(f"   • {f['question']}{err}")
        for i, o in enumerate(f.get("options") or [], 1):
            print(f"       {i}. {o['label']}" + (f" – {o['hint']}" if o["hint"] else ""))
    if form["type"] == "review":
        print("\n  Type 'submit' to send, or tell me what to change.")
    print()


def main() -> None:
    bot = LoanChatbot()
    sid, greeting = bot.new_session()
    _print(greeting)
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        reviewing = bot.state(sid).get("stage") == "confirming"
        msg = bot.send(sid, action="submit") if reviewing and text.lower() == "submit" else bot.send(sid, text)
        _print(msg)
        if bot.state(sid).get("stage") == "done":
            break


if __name__ == "__main__":
    main()
