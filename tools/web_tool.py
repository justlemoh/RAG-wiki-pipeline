"""
Web Search Tool for Agentic AI
================================
Wraps the Tavily Search API as a clean Python function tool that can be
passed directly to the Gemini API for function calling.

Tavily is designed specifically for AI agents and RAG pipelines — it
returns clean, relevant text content from the web (not just links),
making it ideal for LLM consumption.

Free tier: 1,000 searches/month — https://app.tavily.com
"""

import os
import sys
import threading

from dotenv import load_dotenv

# Add project root to sys.path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Thread-local storage to capture web results during agent execution
_local_data = threading.local()


def get_last_web_results() -> list:
    """Retrieve the web results fetched during the last tool call in this thread."""
    return getattr(_local_data, 'web_results', [])


def clear_web_results():
    """Clear the thread-local web results cache."""
    _local_data.web_results = []


def search_web(query: str) -> str:
    """
    Search the internet for up-to-date information on a topic.
    Use this tool ONLY when the local Wikipedia database has no relevant results,
    or when the question is about recent events, news, or topics not covered
    by the local knowledge base.

    Args:
        query: The search query or question to look up on the internet.

    Returns:
        A concatenated string of relevant web search results with their sources.
    """
    try:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            _local_data.web_results = []
            return "Web search unavailable: TAVILY_API_KEY is not set in the environment."

        client = TavilyClient(api_key=api_key)

        # Use search_depth="basic" for speed; "advanced" gives richer content
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=False,
        )

        results = response.get("results", [])

        if not results:
            _local_data.web_results = []
            return "No web results found for this query."

        # Store structured results for UI display
        _local_data.web_results = [
            {
                "title": r.get("title", "No title"),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": float(r.get("score", 0.0)),
                "source": "web",
            }
            for r in results
        ]

        # Format results as a readable string for the LLM
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "").strip()
            formatted.append(f"[Web Source {i}] {title}\nURL: {url}\n{content}")

        return "\n\n".join(formatted)

    except ImportError:
        return "Web search unavailable: 'tavily-python' is not installed. Run: pip install tavily-python"
    except Exception as e:
        _local_data.web_results = []
        return f"Error executing web search: {str(e)}"
