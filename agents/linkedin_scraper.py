from apify_client import ApifyClient
import json
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

def linkedin_scraper(profile_url: str, apify_api_key: str, cookie_content: str) -> dict:
    """
    Scrapes a profile using the API key and cookie content provided by the user.
    """
    if not apify_api_key:
        st.error("Apify API Key was not provided.")
        return None
    if not cookie_content:
        st.error("LinkedIn cookie content was not provided.")
        return None

    try:
        cookies = json.loads(cookie_content)
    except json.JSONDecodeError:
        st.error("Invalid JSON format for the provided cookie content.")
        return None
    
    # Initialize client with the user's key
    client = ApifyClient(apify_api_key)
    run_input = {
        "cookie": cookies,
        "minDelay": 15,
        "maxDelay": 60,
        "proxy": {"useApifyProxy": True, "apifyProxyCountry": "US"},
        "urls": [profile_url], 
    }
    
    try:
        run = client.actor("curious_coder/linkedin-profile-scraper").call(run_input=run_input)
        dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        if not dataset_items:
            return None
        return dataset_items[0]
    except Exception as e:
        st.error(f"An Apify error occurred. Please check your Apify Key. Error: {e}")
        return None

def format_profile_data(item: dict) -> str:
    """
    Extracts key information from the raw Apify profile data and formats it
    into a clean, readable string for LLM processing.

    Args:
        item: The raw dictionary containing the scraped LinkedIn profile data.

    Returns:
        A formatted string containing the cleaned-up profile information.
    """
    
    
    first_name = item.get('firstName', '')
    last_name = item.get('lastName', '')
    full_name = f"{first_name} {last_name}".strip()
    headline = item.get('headline', 'N/A')

    formatted_string = f"**Profile for {full_name}**\n\n"
    formatted_string += f"**Headline:** {headline}\n\n"

    
    summary = item.get('summary', 'N/A')
    formatted_string += f"**Summary (About Section):**\n{summary}\n\n"

    
    formatted_string += "**Experience:**\n\n"
    positions = item.get('positions', [])
    if not positions:
        formatted_string += "No professional experience listed.\n\n"
    else:
        for i, pos in enumerate(positions, 1):
            title = pos.get('title', 'N/A')
            company_name = pos.get('companyName', 'N/A')
            
            
            time_period_str = "Present"
            time_period = pos.get('timePeriod', {})
            if time_period:
                start_date = time_period.get('startDate', {})
                start_month = start_date.get('month')
                start_year = start_date.get('year')
                
                end_date = time_period.get('endDate', {})
                end_month = end_date.get('month')
                end_year = end_date.get('year')

                start_str = f"{start_month}/{start_year}" if start_month and start_year else "N/A"
                end_str = f"{end_month}/{end_year}" if end_month and end_year else "Present"
                time_period_str = f"{start_str} - {end_str}"

            formatted_string += f"{i}. **{title}** at {company_name} ({time_period_str})\n"
            
            description = pos.get('description', '[No description provided]').strip()
            
            formatted_description = "\n".join([f"    * {line}" for line in description.split('\n')])
            formatted_string += f"{formatted_description}\n\n"

    
    formatted_string += "**Education:**\n\n"
    educations = item.get('educations', [])
    if not educations:
        formatted_string += "No education listed.\n\n"
    else:
        for edu in educations:
            degree = edu.get('degreeName', '')
            school = edu.get('schoolName', 'N/A')
            field = edu.get('fieldOfStudy')
            
            
            time_period = edu.get('timePeriod', {})
            start_year = time_period.get('startDate', {}).get('year', '')
            end_year = time_period.get('endDate', {}).get('year', '')
            date_range = f"({start_year} - {end_year})" if start_year and end_year else ""

            edu_str = f"*   **{degree}**"
            if field:
                edu_str += f" in {field}"
            edu_str += f" - {school} {date_range}\n"
            formatted_string += edu_str
        formatted_string += "\n"

    
    formatted_string += "**Top Skills:**\n"
    skills = item.get('skills', [])
    if not skills:
        formatted_string += "No skills listed.\n"
    else:
        for skill in skills:
            formatted_string += f"*   {skill}\n"
    formatted_string += "\n"

    
    formatted_string += "**Certifications:**\n"
    certifications = item.get('certifications', [])
    if not certifications:
        formatted_string += "No certifications listed.\n"
    else:
        for cert in certifications:
            name = cert.get('name', 'N/A')
            authority = cert.get('authority', '')
            cert_str = f"*   {name}"
            if authority:
                cert_str += f" ({authority})"
            formatted_string += f"{cert_str}\n"
    f = open(f"agents/{first_name}_formatted_profile.txt", "w")
    f.write(formatted_string)
    f.close()        
    return formatted_string.strip()