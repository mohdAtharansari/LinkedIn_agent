from states.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from memory_manager.format_recent_msg import format_recent_messages
import os
from langchain_groq import ChatGroq
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
def analyze_profile(state: GraphState) -> dict:
    """
    Performs a general analysis of the user's LinkedIn profile.

    This agent is triggered when the router classifies the user's
    intent as 'analyze_profile'.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary with the analysis to update the state.
    """
    print("---AGENT: Executing Profile Analyzer---")


    profile_text = state.get('profile_text')


    if not profile_text:
        print("---AGENT: Error - Profile text not found in state.---")

        error_message = AIMessage(content="It seems I don't have your profile data loaded yet. Please provide your LinkedIn URL first.")
        return {"messages": [error_message]}


    prompt = f"""
    You are a world-class LinkedIn profile optimization coach. Your tone is encouraging, professional, and highly constructive.
    Your task is to provide a comprehensive critique of the following LinkedIn profile.

    Analyze the profile based on these key areas, providing actionable advice for each:

    1.  **Headline:**
        - Is it impactful and keyword-rich?
        - Does it clearly state the person's value proposition beyond just their job title?
        - Suggest 1-2 alternative headlines.

    2.  **Summary (About Section):**
        - Is it written in the first person? Does it tell a compelling career story?
        - Does it focus on achievements and impact (using metrics/numbers) rather than just listing responsibilities?
        - Is it easy to read (e.g., short paragraphs, bullet points)?

    3.  **Experience Section:**
        - Is the language action-oriented (e.g., "Led," "Developed," "Achieved")?
        - Are there quantifiable results? (e.g., "Increased efficiency by 15%").
        - Point out any recent or important roles that are missing a description and emphasize the need to add one.

    4.  **Skills Section:**
        - Are the skills relevant to the user's headline and experience?
        - Is the list well-curated, or is it just a dump of keywords? Suggest organizing or prioritizing the top 5-10 skills.

    5.  **Overall First Impression:**
        - Give a final summary of the profile's 2-3 biggest strengths.
        - Identify the #1 biggest opportunity for improvement that will have the most impact.

    Provide your feedback in a clear, well-structured format using Markdown for headings and bullet points.

    **LinkedIn Profile to Analyze:**
    ---
    {profile_text}
    ---
    """


    print("---AGENT: Sending request to LLM for analysis...---")
    analysis_response = llm.invoke(prompt)


    print("---AGENT: Analysis complete.---")
    return {
        "initial_analysis": analysis_response.content,
        "messages": [analysis_response]  # The AIMessage from the LLM is added to the history
    }

def test_analyze_profile():
    print("--- Starting Profile Analyzer Test ---")


    try:
        with open("agents/formatted_profile.txt", "r") as f:
            profile_text_for_test = f.read()
    except FileNotFoundError:
        print("Error: Create a file named 'formatted_profile.txt' with the user's formatted profile data to run this test.")
        return


    mock_state = {
        "profile_text": profile_text_for_test,
        "messages": [HumanMessage(content="Please analyze my profile.")]
    }


    result = analyze_profile(mock_state)


    print("\n--- Agent Output ---")
    print(result["initial_analysis"])
    
    assert "initial_analysis" in result
    assert "messages" in result
    assert len(result["initial_analysis"]) > 100 

    print("\n--- Profile Analyzer Test Passed Successfully! ---")


# To run the test:
if __name__ == "__main__":
    pass
    # test_analyze_profile()