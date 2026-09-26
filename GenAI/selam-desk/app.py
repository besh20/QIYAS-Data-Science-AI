import streamlit as st
from src import agent

MAX_MESSAGES_PER_SESSION = 40  # protects the free-tier API quota from runaway usage

st.set_page_config(page_title="Selam Desk")
st.title("Selam Desk")
st.caption("Bilingual (Amharic/English) university registration agent")

# STAFF_WEBHOOK_URL is the app OWNER's own Slack/Discord channel -- set once
# by whoever deploys the app, not something visitors provide.
webhook_url = st.secrets.get("STAFF_WEBHOOK_URL", None)

with st.sidebar:
    st.subheader("Your Gemini API key")
    st.write(
        "This app runs on YOUR own free Gemini API key, not a shared one. "
        "Your key is only used in your browser session and is never stored "
        "or shown to other visitors."
    )
    st.markdown("[Get a free key at Google AI Studio](https://aistudio.google.com/apikey)")
    # value= pre-fills from a LOCAL secrets.toml purely for the developer's
    # own convenience while testing -- a deployed Streamlit Cloud instance
    # simply won't have this secret set, so it's empty (and mandatory) for
    # every real visitor, including the deployer themselves in production.
    api_key_input = st.text_input(
        "Gemini API key", type="password",
        value=st.secrets.get("GEMINI_API_KEY", ""),
    )

if not api_key_input:
    st.info("Enter your free Gemini API key in the sidebar to start chatting with Selam Desk.")
    st.stop()

# (Re)configure whenever the key changes, and start a fresh session so the
# new client is actually used (a chat object is bound to the client that
# created it).
if st.session_state.get("configured_key") != api_key_input:
    agent.configure_gemini(api_key_input, webhook_url)
    st.session_state.configured_key = api_key_input
    for key in ["chat", "messages", "message_count"]:
        st.session_state.pop(key, None)

if "chat" not in st.session_state:
    with st.spinner("Starting session..."):
        st.session_state.chat = agent.create_chat_session()
    st.session_state.messages = [{"role": "assistant", "content": agent.WELCOME_MESSAGE}]
    st.session_state.message_count = 0

# --- render history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- rate limit check ---
if st.session_state.message_count >= MAX_MESSAGES_PER_SESSION:
    st.warning(
        "This session has reached its message limit (to protect the free API quota "
        "during the demo). Click 'Reset conversation' in the sidebar to start a new one."
    )
else:
    # --- chat input ---
    user_input = st.chat_input("Type in Amharic or English...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("..."):
                reply = agent.send_message(st.session_state.chat, user_input)
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.message_count += 1

with st.sidebar:
    st.subheader("About")
    st.write(
        "This is a real tool-calling agent: the model decides for itself, turn by "
        "turn, whether to look something up, collect registration details, log a "
        "complaint, or escalate to staff. Conversation history is kept for the "
        "whole session, so follow-up questions work naturally."
    )
    st.write("Registration is only saved after the student explicitly confirms a preview.")
    st.write("Data resets on redeploy (SQLite demo limitation).")
    st.write(f"Messages this session: {st.session_state.get('message_count', 0)}/{MAX_MESSAGES_PER_SESSION}")
    if webhook_url:
        st.write("✅ Staff notifications: configured")
    else:
        st.write("⚠️ Staff notifications: not configured (see README for STAFF_WEBHOOK_URL)")
    if st.button("Reset conversation"):
        for key in ["chat", "messages", "message_count"]:
            st.session_state.pop(key, None)
        st.rerun()

    with st.expander("Debug: recent errors"):
        import os
        log_path = os.path.join("data", "app.log")
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                lines = [l for l in f if '"send_failed"' in l or '"send_error_raised"' in l]
            if lines:
                for l in lines[-5:]:
                    st.code(l.strip(), language="json")
            else:
                st.write("No API errors logged yet.")
        else:
            st.write("No log file yet.")

