
# 🚀 AI LinkedIn Profile Coach

An interactive, AI-powered chat system designed to help users optimize their LinkedIn profiles, analyze job fit, and receive personalized career guidance. This application leverages a multi-agent architecture built with LangGraph to provide a seamless, conversational experience with persistent memory across sessions.

## ✨ Key Features

-   **Interactive Chat Interface**: A user-friendly UI built with Streamlit for a natural, conversational flow.
-   **Live Profile Scraping**: Utilizes Apify to scrape real-time LinkedIn profile data for accurate analysis.
-   **Multi-Agent System**: A sophisticated backend where different AI "agents" handle specific tasks:
    -   **Profile Analysis**: Get a comprehensive critique of your profile's strengths and weaknesses.
    -   **Job Fit Analysis**: See how your profile stacks up against an industry-standard job description for any role, complete with a match score.
    -   **Content Enhancement**: Receive AI-generated rewrites for your summary, headline, and other sections.
    -   **Career Counseling**: Identify skill gaps for your target roles and get actionable advice and learning resources.
-   **Episodic Memory**: Create and switch between multiple "episodes" or chat sessions. You can have one episode for a "Data Scientist" role and another for a "Product Manager" role, with all conversation history retained and isolated for each.

## 🛠️ Getting Started

Follow these steps to set up and run the application on your local machine.

### 1. Prerequisites

-   Python 3.9+
-   A virtual environment tool (like `venv`)

### 2. Clone the Repository

```bash
git clone <repo-url>
cd your-repo-name
```

### 3. Set Up the Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys and Environment

The application requires API keys for its AI and scraping services.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your API keys to this file in the following format:

    ```.env
    # Get from https://platform.openai.com/ or your chosen LLM provider
    OPENAI_API_KEY="sk-..."

    # Get from https://console.apify.com/account/integrations
    APIFY_API_TOKEN="apify_api_..."
    ```

### 6. Configure LinkedIn Cookies for Scraping

The Apify LinkedIn scraper requires your personal LinkedIn session cookies to function reliably and avoid being blocked.

1.  Install the [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm) Chrome extension.
2.  Log in to your LinkedIn account in your browser.
3.  Click on the Cookie-Editor extension icon in your toolbar.
4.  Click the **"Export"** button. This will copy all your LinkedIn cookies to the clipboard in JSON format.
5.  Create a new file at the path `unit_test/cookie.json`.
6.  **Paste the entire copied JSON content** into this `cookie.json` file and save it.

The `agents/linkedin_scraper.py` file is pre-configured to load cookies from this location.

### 7. Run the Application

With the setup complete, you can now launch the Streamlit application.

```bash
streamlit run app.py
```

Your default web browser will open a new tab with the AI LinkedIn Coach application running.

## 🏛️ System Architecture

The application is built around a **multi-agent system** orchestrated by **LangGraph**. This design allows for a clear separation of concerns, where each "agent" is a specialized AI tool with a single responsibility.

### The Core Component: The Router

The "brain" of the system is the **Router**. When a user sends a message, it is first processed by the Router. The Router's job is not to answer the question itself, but to understand the user's *intent* and delegate the task to the appropriate specialist agent. It classifies each request into one of the following categories:

-   `analyze_profile`
-   `analyze_job_fit`
-   `enhance_content`
-   `counsel_career`
-   `general_question`
-   `end_session`

### The Specialist Agents

Based on the Router's decision, the conversation is handed off to one of these agents:

-   **`ProfileAnalyzerAgent`**: Triggered by `analyze_profile`. This agent performs a high-level, general critique of the user's entire profile, focusing on the headline, summary, experience descriptions, and overall completeness.

-   **`JobFitAgent`**: Triggered by `analyze_job_fit`. This is a powerful agent that first internally generates an industry-standard job description for the role mentioned in the user's query. It then compares the user's profile against this standard to produce a detailed report, including a match score, strengths, gaps, and suggestions for improvement.

-   **`ContentEnhancerAgent`**: Triggered by `enhance_content`. This agent acts as a professional copywriter. It takes a specific section of the user's profile (e.g., "my summary") and rewrites it to be more impactful, using action verbs and incorporating keywords relevant to the user's goals.

-   **`CareerCounselorAgent`**: Triggered by `counsel_career`. This agent acts as a strategic advisor. It analyzes the user's profile to identify skill gaps for a target career path and provides a concrete plan, including recommended courses, certifications, and projects.

### The Memory System

The application's ability to handle multiple "episodes" and remember conversations is powered by LangGraph's `SqliteSaver` checkpointer.

-   **`thread_id`**: Each episode created in the UI is assigned a unique `thread_id`.
-   **`threads.sqlite`**: A SQLite database file is created in the project directory. This file stores the complete state of the conversation (including all messages and the profile data) for every `thread_id`.
-   **Contextual Recall**: When you switch between episodes in the UI, the application simply tells LangGraph to use a different `thread_id`. The checkpointer then automatically loads the entire history for that thread from the database, providing seamless, persistent context for every conversation.
