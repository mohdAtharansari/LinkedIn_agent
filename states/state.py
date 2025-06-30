import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage


# --- Define the Shared Memory (State) for the Graph ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        linkedin_url: The URL of the LinkedIn profile to analyze.
        job_role: The target job role for comparison.
        profile_data: The raw JSON data scraped from Apify.
        profile_text: The cleaned, formatted text version of the profile.
        initial_analysis: The output from the general profile analysis.
        job_fit_report: The output from the job fit analysis.
        messages: The list of messages in the conversation history.
    """
    linkedin_url: str
    job_role: str
    profile_data: dict
    profile_text: str
    initial_analysis: str
    job_fit_report: str
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    route: str