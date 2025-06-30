from states.state import GraphState
from agents.analyze_jobfit import analyze_job_fit
from agents.router import route_requests
from agents.analyze_profile import analyze_profile
from agents.career_counsel import counsel_career
from agents.career_enhancer import enhance_content
from agents.general_question import general_question
from agents.end_session import end_session
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

# Define the conditional edges from the router
# The router's output string directly maps to the name of the next node.
workflow.add_conditional_edges(
    "router",  # The starting node
    lambda state: state["route"],  # The function that reads the route from the state
    {
        # The mapping from the route string to the next node
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

# Compile the graph into a runnable app, passing the checkpointer instance.
app = workflow.compile(checkpointer=memory)
if __name__ == "__main__":
    print("Starting the AI Career Coach chat.")
    print("Type 'exit' to quit.")

    # Load the profile text from the file for our test session
    try:
        with open("agents/formatted_profile.txt", "r") as f:
            profile_text = f.read()
    except FileNotFoundError:
        print("FATAL ERROR: 'formatted_profile.txt' not found. Please create it.")
        exit()

    # Each conversation needs a unique ID. This is how we track memory.
    # We can use a random UUID for each new chat session.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # The initial state for a new conversation
    initial_state = {
        "profile_text": profile_text,
        "messages": [AIMessage(content="Hello! I'm your AI Career Coach. I have your profile loaded. How can I help you today?")]
    }
    # We need to manually update the state for the first message
    app.update_state(config, initial_state)

    print(f"\nAI: {initial_state['messages'][-1].content}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Prepare the input for the graph
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # Invoke the graph. LangGraph handles the state updates.
        final_state = app.invoke(inputs, config=config)

        # Print the AI's last response
        ai_response = final_state["messages"][-1].content
        print(f"\nAI: {ai_response}")