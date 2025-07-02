# app.py

import streamlit as st
import uuid
from langchain_core.messages import AIMessage, HumanMessage


from graph.graph import app  # This is your compiled LangGraph app
from agents.linkedin_scraper import linkedin_scraper, format_profile_data # Your scraper and formatter functions

# --- Page Configuration ---
st.set_page_config(page_title="AI LinkedIn Coach", layout="wide")
st.title("🚀 AI LinkedIn Profile Coach")

# --- Initialize Session State ---
# This is Streamlit's way of having "memory" across user interactions.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = [AIMessage(content="Hello! Please provide your LinkedIn Profile URL to get started.")]
    st.session_state.profile_loaded = False # This is the key to controlling the UI flow

# --- Sidebar for URL Input ---
with st.sidebar:
    st.header("Profile Input")
    linkedin_url = st.text_input("Enter LinkedIn Profile URL", key="linkedin_url_input")

    if st.button("Analyze Profile", key="analyze_button"):
        if linkedin_url and "linkedin.com/in/" in linkedin_url:
            with st.spinner("Scraping and analyzing profile... This may take a minute."):
                try:
                    # 1. Scrape the data
                    raw_data = linkedin_scraper(linkedin_url)
                    if raw_data:
                        # 2. Format the data
                        profile_text = format_profile_data(raw_data)
                        
                        # 3. Initialize the graph state for this session
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        initial_state = {
                            "profile_text": profile_text,
                            "messages": [AIMessage(content="I've successfully analyzed your profile! How can I help you improve it?")]
                        }
                        app.update_state(config, initial_state)
                        
                        # 4. Update session state to "unlock" the chat
                        st.session_state.messages = initial_state["messages"]
                        st.session_state.profile_loaded = True
                        st.success("Profile loaded successfully!")
                        # Rerun to update the main chat UI
                        st.rerun()
                    else:
                        st.error("Failed to scrape profile. Please check the URL or your Apify setup.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid LinkedIn Profile URL.")

# --- Main Chat Interface ---

# Display existing messages
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("AI").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("Human").write(msg.content)

# The chat input box
# It is DISABLED if the profile has not been loaded yet.
if prompt := st.chat_input(disabled=not st.session_state.profile_loaded):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("Human").write(prompt)

    # Prepare the input for the graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {"messages": [HumanMessage(content=prompt)]}

    # Invoke the graph and stream the response
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            # The final state contains the full conversation history
            final_state = app.invoke(inputs, config)
            ai_response = final_state["messages"][-1]
            
            # Update session state and display the response
            st.session_state.messages.append(ai_response)
            st.write(ai_response.content)