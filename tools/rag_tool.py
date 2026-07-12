"""
RAG Search Tool for Agentic AI
=============================
Wraps the core retrieve.py logic as a clean Python function tool that can
be passed directly to the Gemini API.

Includes a thread-safe context to capture retrieved chunks for UI display.
"""

import os
import sys
import threading

# Add project root to sys.path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieve import retrieve

# Thread-local storage to capture retrieved chunks during agent execution
_local_data = threading.local()


def get_last_retrieved_chunks():
    """Retrieve the chunks fetched during the last tool call in this thread."""
    return getattr(_local_data, 'chunks', [])


def clear_retrieved_chunks():
    """Clear the thread-local chunks cache."""
    _local_data.chunks = []


def search_wikipedia(query: str) -> str:
    """
    Search the Wikipedia database for context regarding a specific query.
    Use this tool whenever you need factual information, historical details,
    dates, names, capital cities, or definitions.

    Args:
        query: The search term or question to query in the database.

    Returns:
        A concatenated string of retrieved matching Wikipedia document segments.
    """
    try:
        results = retrieve(query, k=3)
        # Store in thread-local storage so the API can read them
        _local_data.chunks = results

        if not results:
            return "No matching Wikipedia articles found."

        formatted_results = []
        for i, res in enumerate(results, 1):
            formatted_results.append(
                f"[Source {i}] (Score: {res['score']:.3f})\n{res['document']}"
            )
        return "\n\n".join(formatted_results)
    except Exception as e:
        _local_data.chunks = []
        return f"Error executing search: {str(e)}"
