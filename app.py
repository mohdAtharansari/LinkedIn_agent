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

# --- Initialize Session State for Episodes ---
if "episodes" not in st.session_state:
    # 'episodes' will be a dictionary mapping a display name to a thread_id
    st.session_state.episodes = {}
    st.session_state.active_episode_name = None
    st.session_state.profile_loaded = False

# Helper function to get the active thread_id
def get_active_thread_id():
    if st.session_state.active_episode_name:
        return st.session_state.episodes.get(st.session_state.active_episode_name)
    return None

# --- Sidebar for Episode Management and URL Input ---
with st.sidebar:
    st.header("Chat Episodes")

    # --- NEW: Episode Creation and Selection ---
    episode_names = list(st.session_state.episodes.keys())
    new_episode_option = "✨ Create New Episode..."
    episode_names.append(new_episode_option)

    # The selectbox for switching episodes
    selected_episode = st.selectbox(
        "Select an Episode",
        options=episode_names,
        index=episode_names.index(st.session_state.active_episode_name) if st.session_state.active_episode_name else len(episode_names) - 1,
        key="episode_selector"
    )

    # Logic for handling episode selection
    if selected_episode == new_episode_option:
        st.session_state.active_episode_name = None
        st.session_state.profile_loaded = False
    elif selected_episode != st.session_state.active_episode_name:
        st.session_state.active_episode_name = selected_episode
        st.session_state.profile_loaded = True # An existing episode always has a profile
        st.rerun() # Rerun to load the selected episode's chat history

    st.divider()

    # --- URL Input Section (Now conditional based on episode) ---
    if not st.session_state.active_episode_name:
        st.subheader("Create a New Episode")
        new_episode_name = st.text_input("Give this episode a name (e.g., 'For Data Scientist Role')", key="new_episode_name")
        linkedin_url = st.text_input("Enter LinkedIn Profile URL", key="linkedin_url_input")

        if st.button("Start New Episode", key="start_episode_button"):
            if new_episode_name and linkedin_url and "linkedin.com/in/" in linkedin_url:
                if new_episode_name in st.session_state.episodes:
                    st.error("An episode with this name already exists. Please choose a different name.")
                else:
                    with st.spinner("Scraping and analyzing profile... This may take a minute."):
                        try:
                            raw_data = linkedin_scraper(linkedin_url)
                            if raw_data:
                                profile_text = format_profile_data(raw_data)
                                
                                # Create a new thread for this episode
                                new_thread_id = str(uuid.uuid4())
                                config = {"configurable": {"thread_id": new_thread_id}}
                                initial_state = {
                                    "profile_text": profile_text,
                                    "messages": [AIMessage(content=f"Welcome to your new episode: '{new_episode_name}'. I've analyzed your profile. How can I help?")]
                                }
                                app.update_state(config, initial_state)
                                
                                # Save the new episode and make it active
                                st.session_state.episodes[new_episode_name] = new_thread_id
                                st.session_state.active_episode_name = new_episode_name
                                st.session_state.profile_loaded = True
                                st.success("New episode started!")
                                st.rerun()
                            else:
                                st.error("Failed to scrape profile.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
            else:
                st.warning("Please provide a name and a valid LinkedIn URL.")
    else:
        st.success(f"Active Episode: {st.session_state.active_episode_name}")


# --- Main Chat Interface ---

# Get the chat history for the active episode
active_thread_id = get_active_thread_id()
if active_thread_id:
    config = {"configurable": {"thread_id": active_thread_id}}
    # Load the conversation history from the checkpointer
    conversation = app.get_state(config)
    messages = conversation.values.get("messages", [])
else:
    messages = [AIMessage(content="Create a new episode from the sidebar to begin.")]

# Display existing messages
for msg in messages:
    if isinstance(msg, AIMessage):
        st.chat_message("AI").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("Human").write(msg.content)

# The chat input box
if prompt := st.chat_input(disabled=not st.session_state.profile_loaded):
    st.chat_message("Human").write(prompt)

    # Prepare the input for the graph
    config = {"configurable": {"thread_id": active_thread_id}}
    inputs = {"messages": [HumanMessage(content=prompt)]}

    # Invoke the graph and stream the response
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            app.invoke(inputs, config) # The invoke call updates the history in the DB
            # We need to re-fetch the state to get the latest messages
            final_conversation = app.get_state(config)
            ai_response = final_conversation.values["messages"][-1]
            st.write(ai_response.content)
    
    # Rerun the script to display the latest state of the conversation
    st.rerun()