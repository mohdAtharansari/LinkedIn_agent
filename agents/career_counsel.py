from states.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("google_api_key"))

def counsel_career(state: GraphState) -> dict:
    """
    Identifies skill gaps for a target role mentioned in the user's query
    and provides career counseling and learning resources.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary with the career advice.
    """
    print("---AGENT: Executing Career Counselor---")

    
    profile_text = state.get('profile_text')
    last_user_message = state['messages'][-1].content

    
    if not profile_text:
        error_message = AIMessage(content="I need your profile data before I can provide career advice. Please provide your LinkedIn URL first.")
        return {"messages": [error_message]}

    
    prompt = f"""
    You are a senior career strategist and counselor for tech professionals. Your task is to provide a detailed skill gap analysis and a strategic career plan based on the user's profile and their career aspirations mentioned in their query.

    **Follow these steps precisely:**

    1.  **Identify the Goal:** Read the "User's Query" to determine their target job role or career path (e.g., "Product Manager," "move into AI," "become a team lead"). If the query is too vague, ask for clarification.

    2.  **Analyze Current Skills:** Review the "User's LinkedIn Profile" to create a clear picture of their existing skills, experiences, and qualifications.

    3.  **Perform Gap Analysis:** Compare the user's current skills (Step 2) with the typical requirements for their target role (Step 1).

    4.  **Generate a Strategic Career Plan:** Produce a report for the user with the following structure, using Markdown.

    ---

    ### Strategic Career Plan: Targeting [The Goal You Identified]

    **1. 🎯 Key Skill Gaps Identified:**
    *   Based on your profile, here are the most critical skill areas to develop to reach your goal. Be specific.
    *   Example: "Technical: While you have strong Python skills, experience with 'A/B testing frameworks' is missing."
    *   Example: "Soft Skills: To move into a leadership role, demonstrating 'stakeholder management' and 'product roadmap planning' is crucial."

    **2. 📚 Recommended Learning Path:**
    *   Provide a list of concrete learning resources to fill the identified gaps. Suggest specific types of courses, certifications, or books.
    *   **For Technical Skills:** Suggest platforms like Coursera, edX, or specific certifications (e.g., "AWS Certified Solutions Architect").
    *   **For Soft Skills:** Suggest books, workshops, or practical ways to gain experience (e.g., "Volunteer to lead a small project at your current job to build leadership experience.").

    **3. 🛠️ Actionable Next Steps:**
    *   Outline 2-3 immediate, practical steps the user can take in the next 1-3 months.
    *   Example: "1. **Start a Project:** Build a small web application that uses A/B testing and add it to your GitHub."
    *   Example: "2. **Update Your Profile:** As you learn, begin adding these new skills and project experiences to your LinkedIn to reflect your growth."

    **INPUTS FOR YOUR ANALYSIS:**
    ---
    **User's Query:** "{last_user_message}"
    ---
    **User's LinkedIn Profile:**
    {profile_text}
    ---
    """

    # Step 4: Invoke the LLM.
    print("---AGENT: Sending request to LLM for career counseling...---")
    counseling_response = llm.invoke(prompt)

    # Step 5: Return the result.
    print("---AGENT: Career counseling complete.---")
    return {
        "messages": [counseling_response]
    }

# In your test file (e.g., agent_test.py)

def test_career_counselor_agent():
    """
    Tests the CareerCounselorAgent.
    """
    print("--- Starting Test for Career Counselor Agent ---")

    # Load the profile text from our test file
    try:
        with open("agents/formatted_profile.txt", "r") as f:
            profile_text_for_test = f.read()
    except FileNotFoundError:
        print("Error: Create 'formatted_profile.txt' to run this test.")
        return

    # --- Test Scenario: User asks for advice on a specific career transition ---
    print("\n--- Testing Scenario: Career Transition Advice ---")
    
    query = "I have a background in data analysis but I want to become a 'Product Manager'. What skills am I missing and what should I do next?"
    # query = "Hi how are you?"
    # The state only contains the profile and the message.
    mock_state = {
        "profile_text": profile_text_for_test,
        "messages": [HumanMessage(content=query)]
    }

    result = counsel_career(mock_state)
    report = result["messages"][-1].content

    print("\n--- Agent Output ---")
    print(report)

    # Assertions to verify the output structure and content
    assert "Strategic Career Plan: Targeting Product Manager" in report, "The report title should contain the correct target role."
    assert "Key Skill Gaps Identified" in report, "The report must include a skill gaps section."
    assert "Recommended Learning Path" in report, "The report must include a learning path section."
    assert "Actionable Next Steps" in report, "The report must include an actionable steps section."
    # Check for content related to the target role
    assert "product" in report.lower(), "The advice should be relevant to a Product Manager role."
    assert "roadmap" in report.lower() or "user stories" in report.lower() or "stakeholder" in report.lower(), "The advice should contain PM-specific keywords."
    
    print("--- Career Counselor Agent Test Passed Successfully! ---")


# To run the test:
if __name__ == "__main__":
    pass
    # test_career_counselor_agent()