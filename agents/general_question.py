from states.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("google_api_key"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2,
    api_key= os.getenv("groq_api_key")
    # other params...
)


def general_question(state: GraphState) -> dict:
    """
    Handles general conversation or questions that don't match any specific tool.

    Args:
        state: The current state of the graph, containing the conversation history.

    Returns:
        A dictionary with the AI's chat response.
    """
    print("---AGENT: Handling General Question---")
    
    # We simply pass the entire message history to the LLM.
    # This allows it to have a natural, context-aware conversation.
    # We can add a light instruction to keep it in character.
    
    # Prepending a system message to guide the response
    prompt_messages = [
        AIMessage(content="You are a helpful and friendly AI career coach. Answer the user's general question."),
    ] + state['messages']

    response = llm.invoke(prompt_messages)
    
    return {"messages": [response]}

# In your test file (e.g., agent_test.py)

