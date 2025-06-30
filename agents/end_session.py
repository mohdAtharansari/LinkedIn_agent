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

def end_session(state: GraphState) -> dict:
    """
    Provides a polite closing message to end the conversation.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary with the final goodbye message.
    """
    print("---AGENT: Ending Session---")
    
    goodbye_message = AIMessage(content="You're very welcome! It was a pleasure assisting you. Best of luck with your career journey, and feel free to reach out anytime you need help.")
    
    return {"messages": [goodbye_message]}