
from dotenv import load_dotenv
import os
from pathlib import Path

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessageChunk


# Load API Keys
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

openrouter_key = os.getenv("OPENROUTER_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")


# Page Settings
st.set_page_config(
    page_title="InfoPilot AI",
    page_icon="⚡",
    layout="wide"
)

if not openrouter_key or not serper_key:
    st.error("API keys missing. Please check your .env file.")
    st.stop()


# AI Model Setup
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1",
    streaming=True,
    temperature=0,
    max_tokens=2048
)


# Google Search Setup
search = GoogleSerperAPIWrapper(
    k=5,
    gl="in",
    hl="en",
    serper_api_key=serper_key
)


# Google Search Tool
@tool
def google_search(query: str) -> str:
    """Search Google and return useful information with source links."""

    try:
        results = search.results(query)
        organic_results = results.get("organic", [])

        if not organic_results:
            return "No relevant search results found."

        formatted_results = []

        for index, result in enumerate(organic_results[:5], start=1):

            title = result.get("title", "Title not available")
            link = result.get("link", "")
            snippet = result.get("snippet", "Description not available")
            date = result.get("date", "Date not provided")

            formatted_results.append(
                f"""
                Result {index}
                Title: {title}
                Date shown in Google: {date}
                Description: {snippet}
                Original Source: {link}
                """
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Google Search failed: {str(e)}"


tools = [google_search]


# Create Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are InfoPilot AI, a helpful and accurate AI assistant.

GENERAL RULES:
1. Answer clearly and correctly.
2. Keep explanations simple and easy to understand.
3. Use Markdown headings, bullet points and tables when useful.
4. Explain calculations step by step.
5. Never invent facts, dates, statistics or source links.

GOOGLE SEARCH RULES:
6. For latest news, current events, announcements and updated
   information, use the google_search tool.
7. Use only information supported by search results.
8. Include original source links returned by Google.
9. Never invent announcement titles, dates or product names.
10. A Google search result date is not automatically a verified
    official publication date.
11. Clearly mention when information cannot be verified.

CONVERSATION RULES:
12. Use previous messages to answer follow-up questions.
13. Never ask users to repeat information already available
    in the conversation.
"""
)


# Initialize Chat History
if "history" not in st.session_state:
    st.session_state.history = []


# Custom UI Styling
st.markdown("""
<style>
.stMainBlockContainer {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stChatMessage"] {
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 15px;
    padding: 16px;
    margin-bottom: 14px;
}

[data-testid="stChatInput"] {
    border-radius: 14px;
}

.hero {
    background: linear-gradient(120deg, #292019, #17191f);
    border: 1px solid #39343a;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
}

.hero h1 {
    margin: 0;
    font-size:  thirty;
    color: white;
}

.hero p {
    color: #aabbd0;
    margin-bottom: 0;
}
/* Make chat input smaller */
[data-testid="stChatInput"] {
    max-width: 650px;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)


# Header and New Chat Button
header_col, button_col = st.columns([5, 1])

with header_col:
    st.markdown("""
    <div class="hero">
        <h1>🚀 InfoPilot AI</h1>
        <p>Clear answers. Smarter search. Less waiting.</p>
    </div>
    """, unsafe_allow_html=True)

with button_col:
    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# Welcome Message
if not st.session_state.history:
    st.subheader("👋 What can I help you with?")
    st.caption("Ask a question, explore a topic, or search for the latest information.")


# Display Chat History
for chat_message in st.session_state.history:

    with st.chat_message(
        chat_message["role"],
        avatar="🧑" if chat_message["role"] == "user" else "⚡"
    ):
        st.markdown(chat_message["content"])


# Chat Input
query = st.chat_input("Message InfoPilot AI...")

if query:

    # Save User Message
    st.session_state.history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)


    # Generate Assistant Response
    with st.chat_message("assistant", avatar="⚡"):

        response_box = st.empty()
        message = ""

        try:

            # Show Thinking Indicator While AI Works
            with st.spinner("⏳ Thinking..."):

                response = agent.stream(
                    {
                        "messages": st.session_state.history[-4:]
                    },
                    stream_mode="messages"
                )

                # Read AI Response
                for chunk, metadata in response:

                    if not isinstance(chunk, AIMessageChunk):
                        continue

                    content = chunk.content

                    if isinstance(content, str):
                        message += content

                    elif isinstance(content, list):

                        for block in content:

                            if isinstance(block, dict) and block.get("type") == "text":
                                message += block.get("text", "")

                            elif isinstance(block, str):
                                message += block

                    if message:
                        response_box.markdown(message + " ▌")

                # Handle Empty Response
                if not message:
                    message = "I couldn't generate a text response. Please try again."

                response_box.markdown(message)


            # Save Assistant Response
            st.session_state.history.append({
                "role": "assistant",
                "content": message
            })

        except Exception as e:
            response_box.error(f"Something went wrong: {e}")


# Footer
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>"
    "InfoPilot AI · Powered by OpenRouter and Google Search"
    "</p>",
    unsafe_allow_html=True
)
