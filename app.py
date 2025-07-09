# app.py

import streamlit as st
import uuid
from langchain_core.messages import AIMessage, HumanMessage

# --- Import your graph and helper functions ---
from graph.graph import app
from agents.linkedin_scraper import linkedin_scraper, format_profile_data

# --- Page Configuration ---
st.set_page_config(page_title="AI LinkedIn Coach", layout="wide")
st.title("🚀 AI LinkedIn Profile Coach")

# --- Initialize Session State ---
if "episodes" not in st.session_state:
    st.session_state.episodes = {}
    st.session_state.active_episode_name = None
    st.session_state.profile_loaded = False
    # --- CHANGED: Only store the user-provided secrets ---
    st.session_state.apify_key = ""
    st.session_state.cookie_content = ""

# Helper function to get the active thread_id
def get_active_thread_id():
    if st.session_state.active_episode_name:
        return st.session_state.episodes.get(st.session_state.active_episode_name)
    return None

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Your Credentials (Required)")
    st.info("Your keys and cookies are not stored. They are only used for this session.")
    
    # --- CHANGED: Simplified inputs for the user ---
    st.session_state.apify_key = st.text_input(
        "Your Apify API Key",
        type="password",
        value=st.session_state.apify_key,
        help="Get your free key from https://console.apify.com/account/integrations"
    )
    st.session_state.cookie_content = st.text_area(
        "Your LinkedIn Cookie JSON",
        height=150,
        placeholder="Paste your exported cookie JSON here.",
        value=st.session_state.cookie_content,
        help="This is required to analyze profiles."
    )
    
    st.divider()
    st.header("💬 Chat Episodes")
    # ... (Episode selection logic is the same) ...
    episode_names = list(st.session_state.episodes.keys())
    new_episode_option = "✨ Create New Episode..."
    episode_names.append(new_episode_option)
    selected_episode = st.selectbox(
        "Select an Episode",
        options=episode_names,
        index=episode_names.index(st.session_state.active_episode_name) if st.session_state.active_episode_name else len(episode_names) - 1,
    )
    if selected_episode == new_episode_option:
        st.session_state.active_episode_name = None
        st.session_state.profile_loaded = False
    elif selected_episode != st.session_state.active_episode_name:
        st.session_state.active_episode_name = selected_episode
        st.session_state.profile_loaded = True
        st.rerun()

    st.divider()

    # --- URL Input Section ---
    if selected_episode == new_episode_option:
        st.subheader("Create a New Episode")
        new_episode_name = st.text_input("Give this episode a name", key="new_episode_name")
        linkedin_url = st.text_input("Enter LinkedIn Profile URL", key="linkedin_url_input")

        if st.button("Start New Episode", key="start_episode_button"):
            # --- CHANGED: Simplified validation ---
            if not st.session_state.apify_key or not st.session_state.cookie_content:
                st.error("Apify API Key and LinkedIn Cookie JSON are required. Please provide them above.")
            elif not new_episode_name or not linkedin_url:
                st.warning("Please provide an episode name and a valid LinkedIn URL.")
            else:
                with st.spinner("Scraping and analyzing profile..."):
                    # Pass the user-provided credentials to the scraper
                    raw_data = linkedin_scraper(
                        linkedin_url,
                        st.session_state.apify_key,
                        st.session_state.cookie_content
                    )
                    
                    if raw_data:
                        # This logic remains the same
                        profile_text = format_profile_data(raw_data)
                        new_thread_id = str(uuid.uuid4())
                        config = {"configurable": {"thread_id": new_thread_id}}
                        initial_state = {
                            "profile_text": profile_text,
                            "messages": [AIMessage(content=f"Welcome to '{new_episode_name}'. I've analyzed the profile. How can I help?")]
                        }
                        app.update_state(config, initial_state)
                        st.session_state.episodes[new_episode_name] = new_thread_id
                        st.session_state.active_episode_name = new_episode_name
                        st.session_state.profile_loaded = True
                        st.success("New episode started!")
                        st.rerun()
                    else:
                        st.error("Failed to scrape profile. Please check the URL and your credentials.")
# --- Main Chat Interface ---
# This entire section remains the same as it correctly handles the conversation flow
# once an episode is active.

active_thread_id = get_active_thread_id()
if active_thread_id:
    config = {"configurable": {"thread_id": active_thread_id}}
    conversation = app.get_state(config)
    messages = conversation.values.get("messages", [])
else:
    messages = [AIMessage(content="Please provide your credentials and create a new episode from the sidebar to begin.")]

# Display existing messages
for msg in messages:
    if isinstance(msg, AIMessage):
        st.chat_message("AI").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("Human").write(msg.content)

# The chat input box
if prompt := st.chat_input(disabled=not st.session_state.profile_loaded):
    st.chat_message("Human").write(prompt)

    config = {"configurable": {"thread_id": active_thread_id}}
    inputs = {"messages": [HumanMessage(content=prompt)]}

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            app.invoke(inputs, config)
            final_conversation = app.get_state(config)
            ai_response = final_conversation.values["messages"][-1]
            st.write(ai_response.content)
    
    st.rerun()