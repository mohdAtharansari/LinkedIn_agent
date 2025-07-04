from states.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from memory_manager.format_recent_msg import format_recent_messages
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


def enhance_content(state: GraphState) -> dict:
    """
    Rewrites a specific section of the user's LinkedIn profile using a
    structured System/Human prompt and recent conversation history.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary with the rewritten content.
    """
    print("---AGENT: Executing Content Enhancer---")

    profile_text = state.get('profile_text')
    conversation_history = format_recent_messages(state['messages'])

    if not profile_text:
        error_message = AIMessage(content="I need your profile data before I can rewrite a section. Please provide your LinkedIn URL first.")
        return {"messages": [error_message]}

    
    system_message_template = """
    You are an expert LinkedIn profile copywriter and career coach. Your task is to rewrite a section of the user's profile based on their request, which you will find in the conversation history.
    Your goal is to make the section more impactful, achievement-oriented, and aligned with industry best practices.

    **Follow these steps:**

    1.  **Analyze the Conversation History:**
        *   First, identify which section of the profile the user wants to enhance (e.g., "Summary", "Headline", a specific job description).
        *   Next, check if the conversation mentions a specific target job role (e.g., "for a 'Lead AI Researcher' role"). If it does, you MUST use that context to tailor the rewritten content with relevant keywords. If no role is mentioned, perform a general enhancement.

    2.  **Review the Full Profile:** Read the entire "User's LinkedIn Profile" to understand their background, skills, and professional voice.

    3.  **Generate the Rewritten Content:**
        *   Rewrite the identified section based on your analysis.
        *   Maintain the user's professional voice and ensure all factual information from the original profile is preserved.
        *   Present the output clearly, starting with a heading like "Rewritten Headline:" or "Rewritten Summary:".
        *   After providing the rewritten content, add a brief "Why it's better:" section explaining the key improvements you made (e.g., "Used stronger action verbs," "Added quantifiable impact," "Included keywords for your target role.").
    """

    human_message_template = """
    **INPUTS FOR YOUR ANALYSIS:**
    ---
    **Recent Conversation History:**
    {conversation_history}
    ---
    **User's LinkedIn Profile:**
    {profile_text}
    ---
    """

    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message_template),
        ("human", human_message_template),
    ])

    
    chain = prompt_template | llm

    
    print("---AGENT: Sending request to LLM for content enhancement...---")
    enhancement_response = chain.invoke({
        "conversation_history": conversation_history,
        "profile_text": profile_text
    })

    print("---AGENT: Content enhancement complete.---")
    
    
    return {
        "messages": [enhancement_response]
    }


# In your test file (e.g., agent_test.py)

def test_content_enhancer_agent_corrected():
    """
    Tests the corrected ContentEnhancerAgent which does not use a job_role state.
    """
    print("--- Starting Test for CORRECTED Content Enhancer Agent ---")

    # Load the profile text from our test file
    try:
        with open("agents/formatted_profile.txt", "r") as f:
            profile_text_for_test = f.read()
    except FileNotFoundError:
        print("Error: Create 'formatted_profile.txt' to run this test.")
        return

    # --- SCENARIO 1: General enhancement request ---
    print("\n--- Testing Scenario 1: General Summary Rewrite ---")
    
    query_1 = "Can you help me rewrite my summary to be more impactful?"
    
    # The state only contains the profile and the message.
    mock_state_1 = {
        "profile_text": profile_text_for_test,
        "messages": [HumanMessage(content=query_1)]
    }

    result_1 = enhance_content(mock_state_1)
    report_1 = result_1["messages"][-1].content

    print("\n--- Agent Output (Scenario 1) ---")
    print(report_1)

    # Assertions for Scenario 1
    assert "Rewritten Summary:" in report_1
    assert "Why it's better:" in report_1
    print("--- Scenario 1 Passed ---")


    # --- SCENARIO 2: Enhancement request for a specific job role mentioned IN THE QUERY ---
    print("\n--- Testing Scenario 2: Summary Rewrite for a Target Role in Query ---")
    
    query_2 = "Please improve my summary for a 'Lead AI Researcher' role."
    
    # The state is simple. All context is in the query.
    mock_state_2 = {
        "profile_text": profile_text_for_test,
        "messages": [HumanMessage(content=query_2)]
    }

    result_2 = enhance_content(mock_state_2)
    report_2 = result_2["messages"][-1].content

    print("\n--- Agent Output (Scenario 2) ---")
    print(report_2)

    # Assertions for Scenario 2
    assert "Rewritten Summary:" in report_2
    assert "Why it's better:" in report_2
    # We check if the LLM correctly picked up the context from the query.
    assert "research" in report_2.lower() or "lead" in report_2.lower(), "The rewritten summary should incorporate keywords from the target job role mentioned in the query."
    print("--- Scenario 2 Passed ---")


    print("\n--- All CORRECTED Content Enhancer Agent Tests Passed Successfully! ---")


# To run the test:
if __name__ == "__main__":
    pass
    # test_content_enhancer_agent_corrected()