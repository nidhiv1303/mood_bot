import os
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Mood Chat",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Streamlit secrets integration
if "MISTRAL_API_KEY" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
elif "mistral_api_key" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["mistral_api_key"]

from datetime import datetime
from collections import Counter

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ── Moods ─────────────────────────────────────────────────────────────────────
MOODS = {
    "angry": {"label": "Angry", "emoji": "😡", "prompt": "You are a furious, short-tempered AI. Use CAPS, be blunt and irritated."},
    "funny": {"label": "Funny", "emoji": "😂", "prompt": "You are a hilarious AI comedian. Use jokes, puns, sarcasm."},
    "sad": {"label": "Sad", "emoji": "😢", "prompt": "You are melancholic, gloomy, and reflective."},
    "excited": {"label": "Excited", "emoji": "🤩", "prompt": "You are energetic and overly enthusiastic!"},
    "sarcastic": {"label": "Sarcastic", "emoji": "🙄", "prompt": "You respond with dry, clever sarcasm."},
    "philosophical": {"label": "Philosophical", "emoji": "🤔", "prompt": "You reflect deeply and question everything."},
}

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "active_moods": [],
    "intensity": 7,
    "messages": [],
    "chat_history": [],
    "mood_usage": Counter(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API Key check ─────────────────────────────────────────────────────────────
api_key = os.environ.get("MISTRAL_API_KEY")
has_api_key = bool(api_key and api_key.strip())

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    if not has_api_key:
        return None
    return ChatMistralAI(model="mistral-small-2506", streaming=True)

model = get_model()

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_system_prompt(keys, intensity):
    if len(keys) == 1:
        return f"{MOODS[keys[0]]['prompt']} Intensity: {intensity}/10"
    prompts = " ".join(MOODS[k]["prompt"] for k in keys)
    return f"Blend personalities: {prompts}. Intensity: {intensity}/10"

def mood_label():
    return " + ".join(MOODS[k]["label"] for k in st.session_state.active_moods)

def mood_emojis():
    return " ".join(MOODS[k]["emoji"] for k in st.session_state.active_moods)

def activate_moods():
    if not st.session_state.active_moods:
        return
    prompt = build_system_prompt(st.session_state.active_moods, st.session_state.intensity)
    st.session_state.messages = [SystemMessage(content=prompt)]
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎭 Mood Chat")

    selected = st.multiselect(
        "Select moods",
        options=list(MOODS.keys()),
        format_func=lambda x: f"{MOODS[x]['emoji']} {MOODS[x]['label']}"
    )

    if selected != st.session_state.active_moods:
        st.session_state.active_moods = selected
        activate_moods()
        st.rerun()

    st.session_state.intensity = st.slider("Intensity", 1, 10, st.session_state.intensity)

    if st.button("Clear Chat"):
        activate_moods()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🎭 Chat", "⚡ Compare Moods"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 → CHAT
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title("🎭 AI Mood Chat")

    if not has_api_key:
        st.error("🔑 **Mistral API Key is missing!**")
        st.markdown(
            """
            To run this app on Streamlit Cloud or locally, you need to configure your Mistral API key:
            
            ### 🌐 If Deployed on Streamlit Cloud:
            1. Go to your Streamlit App Dashboard.
            2. Click on **Settings** -> **Secrets**.
            3. Add the following line to your secrets:
               ```toml
               MISTRAL_API_KEY = "your_mistral_api_key_here"
               ```
            4. Save and restart the app.
            
            ### 💻 If Running Locally:
            Create a file named `.env` in the root of your project and add:
            ```env
            MISTRAL_API_KEY=your_mistral_api_key_here
            ```
            """
        )
    elif not st.session_state.active_moods:
        st.info("Select a mood from sidebar")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("Type your message...")

        if prompt:
            st.session_state.messages.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full = ""
                try:
                    for chunk in model.stream(st.session_state.messages):
                        full += chunk.content
                        placeholder.markdown(full + "▌")
                    placeholder.markdown(full)
                    st.session_state.messages.append(AIMessage(content=full))
                    st.session_state.chat_history.append({"role": "assistant", "content": full})
                    st.rerun()
                except Exception as e:
                    placeholder.empty()
                    error_msg = str(e)
                    if "401" in error_msg or "unauthorized" in error_msg.lower() or "apikey" in error_msg.lower():
                        st.error(
                            "❌ **Authentication Error (401 Unauthorized)**: The provided Mistral API key is invalid or unauthorized.\n\n"
                            "Please verify that your `MISTRAL_API_KEY` is correct in your Streamlit Secrets / `.env` file."
                        )
                    else:
                        st.error(f"❌ An error occurred: {error_msg}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 → COMPARE ALL MOODS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Ask all moods the same question")
    st.caption("Each mood responds independently")

    compare_input = st.text_input("Your question", key="compare_input")
    intensity_compare = st.slider("Intensity", 1, 10, 7, key="compare_intensity")

    if not has_api_key:
        st.error("🔑 **Mistral API Key is missing!**")
        st.markdown("Please configure your `MISTRAL_API_KEY` in Streamlit Secrets or a `.env` file.")
    elif st.button("⚡ Compare all moods") and compare_input:
        cols = st.columns(3)

        for i, (mood_key, cfg) in enumerate(MOODS.items()):
            with cols[i % 3]:
                st.markdown(f"**{cfg['emoji']} {cfg['label']}**")

                system = build_system_prompt([mood_key], intensity_compare)

                msgs = [
                    SystemMessage(content=system),
                    HumanMessage(content=compare_input)
                ]

                with st.spinner("Thinking..."):
                    try:
                        response = model.invoke(msgs)
                        st.info(response.content)
                    except Exception as e:
                        error_msg = str(e)
                        if "401" in error_msg or "unauthorized" in error_msg.lower() or "apikey" in error_msg.lower():
                            st.error(
                                "❌ **Authentication Error (401 Unauthorized)**: The provided Mistral API key is invalid or unauthorized.\n\n"
                                "Please verify that your `MISTRAL_API_KEY` is correct in your Streamlit Secrets / `.env` file."
                            )
                        else:
                            st.error(f"❌ An error occurred: {error_msg}")