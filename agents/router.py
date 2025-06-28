from states.state import GraphState
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("google_api_key"))

def route_requests(state: GraphState) -> str:
    """
    This is the simplified central router. It reads the user's last message,
    classifies the intent, and determines which node to call next.
    """
    print("---ROUTER: Classifying user intent---")
    
    last_message = state['messages'][-1]

    # The LLM-based classification prompt remains the same, as it's very effective.
    classification_prompt = f"""
    You are an expert request router for an AI career coach.
    Your task is to classify the user's latest message into one of the following categories.
    Respond with ONLY the category name and nothing else.

    **Categories:**
    - `analyze_profile`: The user wants a general review, critique, or analysis of their LinkedIn profile.
        (Examples: "review my profile", "what do you think of my LinkedIn?", "give me feedback")
    - `analyze_job_fit`: The user wants to compare their profile against a specific job role or asks about their suitability for a job.
        (Examples: "compare me to a data scientist", "am I a good fit for a project manager role?", "how do I stack up for this job?")
    - `enhance_content`: The user wants help rewriting or improving a specific section of their profile.
        (Examples: "help me rewrite my summary", "can you make my headline better?", "improve this job description")
    - `counsel_career`: The user is asking for advice on skills to learn, how to fill gaps, or general career path guidance.
        (Examples: "what should I learn next?", "how do I get into product management?", "what skills am I missing?")
    - `end_session`: The user is saying thank you, goodbye, or indicates the conversation is over.
        (Examples: "thanks, that's all for now", "goodbye", "perfect, thank you!")
    - `general_question`: The user is asking a question that doesn't fit the other categories, or is just chatting.
        (Examples: "hi", "what can you do?", "how does this work?")

    **User's latest message:**
    ---
    "{last_message.content}"
    ---

    **Classification:**
    """

    classification_response = llm.invoke(classification_prompt)
    route = classification_response.content.strip()

    print(f"---ROUTER: Classified intent as '{route}'---")

    # The router's job is now done. It simply returns the name of the next node.
    return route