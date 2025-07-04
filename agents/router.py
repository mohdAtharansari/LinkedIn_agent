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

 
    classification_prompt =f"""
You are an expert request router for an AI career coach. Your primary goal is to accurately classify the user's latest message into one of the predefined categories.

**Your Thought Process (Internal Monologue):**
1.  Read the user's message carefully.
2.  Consider the core intent. Is the user asking for a review of their current profile (`analyze_profile`)? A comparison against a job (`analyze_job_fit`)? Help writing content (`enhance_content`)? Or advice on future actions and skills (`counsel_career`)?
3.  Compare the user's message to the examples provided for each category.
4.  Select the single best category that matches the user's primary intent.

**Final Output:**
After your internal thought process, respond with ONLY the category name and nothing else.

**Categories & Examples:**
- `analyze_profile`: The user wants a general review of their current profile.
    (Examples: "review my profile", "what do you think of my LinkedIn?", "give me feedback")
- `analyze_job_fit`: The user wants a direct comparison of their profile against a job.
    (Examples: "compare me to a data scientist", "am I a good fit for a project manager role?", "how do I stack up for this job?")
- `enhance_content`: The user wants help rewriting or improving a specific section.
    (Examples: "help me rewrite my summary", "can you make my headline better?", "improve this job description")
- `counsel_career`: The user is asking for advice on future actions, learning, or filling skill gaps.
    (Examples: "what should I learn next?", "how do I transition into AI?", "what skills am I missing for a PM role?")
- `end_session`: The user is saying thank you, goodbye, or indicates the conversation is over.
    (Examples: "thanks, that's all for now", "goodbye", "perfect, thank you!")
- `general_question`: The user is asking a general question or just chatting.
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

    
    return {"route": route}