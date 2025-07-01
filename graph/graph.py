from states.state import GraphState
from agents.analyze_jobfit import analyze_job_fit
from agents.router import route_requests
from agents.analyze_profile import analyze_profile
from agents.career_counsel import counsel_career
from agents.career_enhancer import enhance_content
from agents.general_question import general_question
from agents.end_session import end_session
from agents.linkedin_scraper import linkedin_scraper, format_profile_data
import os
import uuid
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
workflow = StateGraph(GraphState)

workflow.add_node("router", route_requests)
workflow.add_node("analyze_profile", analyze_profile)
workflow.add_node("analyze_job_fit", analyze_job_fit)
workflow.add_node("enhance_content", enhance_content)
workflow.add_node("counsel_career", counsel_career)
workflow.add_node("general_question", general_question)
workflow.add_node("end_session", end_session)

# Set the entry point of the graph
workflow.set_entry_point("router")


workflow.add_conditional_edges(
    "router",  
    lambda state: state["route"],  # The function that reads the route from the state
    {
        
        "analyze_profile": "analyze_profile",
        "analyze_job_fit": "analyze_job_fit",
        "enhance_content": "enhance_content",
        "counsel_career": "counsel_career",
        "general_question": "general_question",
        "end_session": "end_session",
    }
)


workflow.add_edge("analyze_profile", END)
workflow.add_edge("analyze_job_fit", END)
workflow.add_edge("enhance_content", END)
workflow.add_edge("counsel_career", END)
workflow.add_edge("general_question", END)
workflow.add_edge("end_session", END)


conn = sqlite3.connect("threads.sqlite", check_same_thread=False)
memory = SqliteSaver(conn=conn)


app = workflow.compile(checkpointer=memory)
if __name__ == "__main__":
    print("--- AI Career Coach ---")
    
    # --- Step 1: Get URL from User (Pre-Graph Logic) ---
    profile_url = input("Please enter a LinkedIn Profile URL to begin: ")
    if not profile_url or "linkedin.com/in/" not in profile_url:
        print("Invalid LinkedIn Profile URL. Exiting.")
        exit()

    # --- Step 2: Scrape and Format Data (Pre-Graph Logic) ---
    print("\nAnalyzing profile... This is a one-time setup for our session and may take a minute.")
    
    # Call your external functions
    raw_profile_data = linkedin_scraper(profile_url)
    
    if not raw_profile_data:
        print("Could not retrieve profile data. Exiting.")
        exit()
        
    profile_text = format_profile_data(raw_profile_data)
    
    if not profile_text:
        print("Could not format profile data. Exiting.")
        exit()

    print("\nProfile analysis complete! You can now start chatting with the AI coach.")
    print("Type 'exit' to quit.")

    # --- Step 3: Initialize the Graph Session with the Scraped Data ---
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # The initial state now includes the dynamically scraped profile_text
    initial_state = {
        "profile_text": profile_text,
        "messages": [AIMessage(content="Hello! I'm your AI Career Coach. I've successfully analyzed your profile. How can I help you today?")]
    }
    # Persist this initial state
    app.update_state(config, initial_state)

    print(f"\nAI: {initial_state['messages'][-1].content}")

    # --- Step 4: Enter the Conversational Loop ---
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Prepare the input for the graph
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # Invoke the graph. It will use the profile_text already in its memory.
        final_state = app.invoke(inputs, config=config)

        # Print the AI's last response
        ai_response = final_state["messages"][-1].content
        print(f"\nAI: {ai_response}")