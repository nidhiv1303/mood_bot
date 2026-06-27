from dotenv import load_dotenv
load_dotenv()

import sys
import yaml
from pathlib import Path
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ── Load mood config ─────────────────────────────────────────────────────────
MOODS_FILE = Path(__file__).parent / "moods.yaml"

def load_moods():
    if MOODS_FILE.exists():
        with open(MOODS_FILE) as f:
            return yaml.safe_load(f)["moods"]
    # fallback defaults
    return {
        "angry": {"label": "Angry", "emoji": "😡", "prompt": "You are an extremely angry AI. You are irritable, impatient, and short-tempered. Express frustration openly."},
        "funny": {"label": "Funny", "emoji": "😂", "prompt": "You are a hilarious AI. Use jokes, puns, and witty observations. Keep things light and entertaining."},
        "sad":   {"label": "Sad",   "emoji": "😢", "prompt": "You are a melancholic AI. You speak with a heavy heart, expressing gloom and sorrow in every response."},
    }

MOODS = load_moods()

# ── Model ─────────────────────────────────────────────────────────────────────
model = ChatMistralAI(model="mistral-small-2506", streaming=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_system_prompt(selected_keys: list[str], intensity: int) -> str:
    """Build a combined system prompt from one or more moods + intensity."""
    intensity_desc = {
        range(1, 4):  "slightly",
        range(4, 7):  "moderately",
        range(7, 9):  "very",
        range(9, 11): "EXTREMELY",
    }
    level = next((v for k, v in intensity_desc.items() if intensity in k), "moderately")

    if len(selected_keys) == 1:
        mood = MOODS[selected_keys[0]]
        base = mood["prompt"]
        return f"{base} Your emotional intensity is {level} high (level {intensity}/10). Scale every response accordingly."
    else:
        labels = [MOODS[k]["label"] for k in selected_keys]
        prompts = [MOODS[k]["prompt"] for k in selected_keys]
        blend = " AND ".join(f"[{l}] {p}" for l, p in zip(labels, prompts))
        return (
            f"You are a mood-blended AI combining: {blend}. "
            f"Weave both moods together naturally in every reply. "
            f"Your emotional intensity is {level} (level {intensity}/10)."
        )

def display_menu():
    print("\n" + "═" * 50)
    print("  🎭  AI MOOD CHATBOT")
    print("═" * 50)
    print("\nAvailable moods:\n")
    for i, (key, cfg) in enumerate(MOODS.items(), 1):
        print(f"  [{i}] {cfg['emoji']}  {cfg['label']}")
    print("\n  [q] Quit")
    print()

def pick_moods() -> list[str]:
    keys = list(MOODS.keys())
    while True:
        display_menu()
        raw = input("Enter mood number(s) — separate with comma to blend (e.g. 1,3): ").strip()
        if raw.lower() == "q":
            sys.exit(0)
        choices = [c.strip() for c in raw.split(",")]
        try:
            selected = [keys[int(c) - 1] for c in choices if c]
            if selected:
                return list(dict.fromkeys(selected))  # dedupe, preserve order
        except (ValueError, IndexError):
            pass
        print("  ⚠  Invalid choice, try again.")

def pick_intensity() -> int:
    while True:
        raw = input("Mood intensity 1–10 [default 7]: ").strip()
        if not raw:
            return 7
        if raw.isdigit() and 1 <= int(raw) <= 10:
            return int(raw)
        print("  ⚠  Enter a number between 1 and 10.")

def stream_response(messages) -> str:
    """Stream tokens to stdout, return full text."""
    print("\nBot: ", end="", flush=True)
    full = ""
    for chunk in model.stream(messages):
        token = chunk.content
        print(token, end="", flush=True)
        full += token
    print("\n")
    return full

def commands_help():
    print("\n  Commands:")
    print("    /mood    — change mood mid-conversation")
    print("    /clear   — clear chat history")
    print("    /save    — save conversation to file")
    print("    /history — show last 5 messages")
    print("    /exit    — quit\n")

def save_chat(history: list[dict], mood_label: str):
    fname = f"chat_{mood_label.lower().replace(' ','_')}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"AI Mood Chat — {mood_label}\n{'─'*40}\n\n")
        for msg in history:
            role = "You" if msg["role"] == "user" else "Bot"
            f.write(f"{role}: {msg['content']}\n\n")
    print(f"  ✅ Saved to {fname}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🎭 Welcome to AI Mood Chatbot!")
    print("Type /help for commands.\n")

    selected = pick_moods()
    intensity = pick_intensity()
    system_prompt = build_system_prompt(selected, intensity)
    mood_label = " + ".join(MOODS[k]["label"] for k in selected)
    emojis = " ".join(MOODS[k]["emoji"] for k in selected)

    messages: list = [SystemMessage(content=system_prompt)]
    chat_history: list[dict] = []

    print(f"\n{'═'*50}")
    print(f"  {emojis}  {mood_label} Mode  (intensity {intensity}/10)")
    print(f"{'═'*50}")
    print("  Type your message, or /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit"):
                print("\nGoodbye! 👋")
                break
            elif cmd == "/help":
                commands_help()
            elif cmd == "/mood":
                selected = pick_moods()
                intensity = pick_intensity()
                system_prompt = build_system_prompt(selected, intensity)
                mood_label = " + ".join(MOODS[k]["label"] for k in selected)
                emojis = " ".join(MOODS[k]["emoji"] for k in selected)
                messages = [SystemMessage(content=system_prompt)]
                chat_history.clear()
                print(f"\n  ✅ Switched to {emojis} {mood_label} (intensity {intensity}/10)\n")
            elif cmd == "/clear":
                messages = [SystemMessage(content=system_prompt)]
                chat_history.clear()
                print("  ✅ Chat history cleared.\n")
            elif cmd == "/save":
                save_chat(chat_history, mood_label)
            elif cmd == "/history":
                recent = chat_history[-10:]
                if not recent:
                    print("  (no messages yet)\n")
                else:
                    print()
                    for m in recent:
                        who = "You" if m["role"] == "user" else "Bot"
                        print(f"  {who}: {m['content'][:100]}{'…' if len(m['content'])>100 else ''}")
                    print()
            else:
                print(f"  ⚠  Unknown command: {user_input}. Type /help.\n")
            continue

        # ── Normal message ───────────────────────────────────────────────────
        messages.append(HumanMessage(content=user_input))
        chat_history.append({"role": "user", "content": user_input})

        reply = stream_response(messages)
        messages.append(AIMessage(content=reply))
        chat_history.append({"role": "bot", "content": reply})

if __name__ == "__main__":
    main()