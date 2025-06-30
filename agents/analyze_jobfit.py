from states.state import GraphState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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
def analyze_job_fit(state: GraphState) -> dict:
    """
    Compares the user's profile against a generated, industry-standard
    job description for the role mentioned in the user's query.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary with the job fit report.
    """
    print("---AGENT: Executing Job Fit Analyzer (as per spec)---")

    
    profile_text = state.get('profile_text')
    # last_user_message = state['messages'][-1].content
    conversation_history = format_recent_messages(state['messages'])

    
    if not profile_text:
        error_message = AIMessage(content="I need your profile data before I can perform a job fit analysis. Please provide your LinkedIn URL first.")
        return {"messages": [error_message]}

    
    prompt = f"""
    You are an expert AI recruitment strategist. Your task is to analyze a user's LinkedIn profile against an industry-standard job description that you will generate.

    **Follow these steps precisely:**

    1.  **Identify the Job Role:** First, carefully read the "User's Query" to identify the specific job role they are interested in. If no clear job role is mentioned, state that you need one to proceed.

    2.  **Generate a Standard Job Description (Internal Step):** Based on the identified job role, mentally construct a detailed, industry-standard job description. This should include:
        *   Key Responsibilities (e.g., "Develop and deploy machine learning models," "Collaborate with product teams").
        *   Required Technical Skills (e.g., "Python, TensorFlow, SQL, AWS").
        *   Essential Soft Skills (e.g., "Problem-solving, Communication, Teamwork").

    3.  **Compare and Produce the Report:** Use the job description you just generated as a benchmark to analyze the "User's LinkedIn Profile". Produce a report with the following structure, using Markdown.

    ---

    ### Job Fit Analysis: [The Job Role You Identified]

    **Overall Match Score: [Provide a percentage from 0% to 100%]**
    *Provide a one-sentence summary explaining why you gave this score, based on the comparison to the standard job description.*

    ---

    **✅ Strengths & Alignment:**
    *Based on your generated job description, list the specific parts of the user's profile that are a strong match. Be specific.*
    *   Example: "Your experience with TensorFlow, as listed in your skills, directly aligns with a key technical requirement for this role."
    *   Example: "The project described in your Data Science internship is excellent evidence of your ability to handle end-to-end model development."

    **⚠️ Gaps & Opportunities:**
    *Based on your generated job description, identify the most significant gaps in the user's profile. Be constructive.*
    *   Example: "The standard job description for this role requires experience with cloud deployment (e.g., AWS, GCP), which is not currently mentioned in your profile."
    *   Example: "Your experience descriptions could be strengthened by adding quantifiable metrics (e.g., 'improved model accuracy by 10%') to demonstrate impact."

    **🚀 Suggested Improvements:**
    *Provide a prioritized list of actionable steps the user can take to improve their alignment.*
    *   Example: "1. **Add Project Metrics:** Edit your 'Data Analyst' experience to include specific numbers that show the results of your work."
    *   Example: "2. **Learn a Cloud Platform:** To fill the cloud technology gap, consider taking a foundational course in AWS or Azure and adding a small deployment project to your portfolio."

    **INPUTS FOR YOUR ANALYSIS:**
    ---
    **User's Query:** "{conversation_history}"
    ---
    **User's LinkedIn Profile:**
    {profile_text}
    ---
    """

  
    print("---AGENT: Sending single, spec-compliant prompt to LLM...---")
    analysis_response = llm.invoke(prompt)

   
    print("---AGENT: Analysis complete.---")
    return {
        "job_fit_report": analysis_response.content,
        "messages": [analysis_response]
    }


def test_job_fit_agent_with_spec_prompt():
    """
    Tests the JobFitAgent that uses the single, spec-compliant prompt.
    """
    print("--- Starting Test for Spec-Compliant Job Fit Agent ---")

    
    try:
        with open("agents/formatted_profile.txt", "r") as f:
            profile_text_for_test = f.read()
    except FileNotFoundError:
        print("Error: Create 'formatted_profile.txt' with formatted profile data to run this test.")
        return


    print("\n--- Testing Scenario 1: Clear Job Role ---")
    
    valid_query = "How well does my profile align with a 'Senior AI Engineer' position?"
    
    mock_state_1 = {
        "profile_text": profile_text_for_test,
        "messages": [HumanMessage(content=valid_query)]
    }

    
    result_1 = analyze_job_fit(mock_state_1)
    report_1 = result_1.get("job_fit_report", "")

    print("\n--- Agent Output (Scenario 1) ---")
    print(report_1)

  
    assert "job_fit_report" in result_1, "The result should contain a 'job_fit_report'."
    assert "Job Fit Analysis: Senior AI Engineer" in report_1, "The report title should contain the correct job role."
    assert "Overall Match Score" in report_1, "The report must include a match score."
    assert "Strengths & Alignment" in report_1, "The report must include a strengths section."
    assert "Gaps & Opportunities" in report_1, "The report must include a gaps section."
    assert "Suggested Improvements" in report_1, "The report must include an improvements section."
    print("--- Scenario 1 Passed ---")


 
    print("\n--- Testing Scenario 2: Ambiguous Job Role ---")
    
    ambiguous_query = "Do you think I can get a better job?"

    mock_state_2 = {
        "profile_text": profile_text_for_test,
        "messages": [HumanMessage(content=ambiguous_query)]
    }


    result_2 = analyze_job_fit(mock_state_2)
    report_2 = result_2.get("job_fit_report", "")

    print("\n--- Agent Output (Scenario 2) ---")
    print(report_2)


    assert "job_fit_report" in result_2, "The result should still contain a 'job_fit_report' key, even if it's an error message."
    assert ("need a specific role" in report_2.lower() or 
            "mention a clear job role" in report_2.lower() or
            "please specify a job role" in report_2.lower()), "The report should ask the user for a specific job role."
    assert "Overall Match Score" not in report_2, "The report should not contain a score for an ambiguous query."
    print("--- Scenario 2 Passed ---")


    print("\n--- All Job Fit Agent Tests Passed Successfully! ---")



if __name__ == "__main__":
    
    # test_job_fit_agent_with_spec_prompt()
    pass