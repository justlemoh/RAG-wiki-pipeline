"""
Wiki Agent Module
=================
Defines the Agentic AI wrapper that uses the Gemini API and can autonomously
decide when to call the search_wikipedia tool (local DB) or the search_web
tool (Tavily internet search) to answer user questions.
"""

import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Add project root to sys.path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rag_tool import search_wikipedia, get_last_retrieved_chunks, clear_retrieved_chunks
from tools.web_tool import search_web, get_last_web_results, clear_web_results

# Load environment variables
load_dotenv()

# We use gemini-2.5-flash as it is extremely capable, fast, and supports tool use natively
MODEL_NAME = 'gemini-3.1-flash-lite'


def get_gemini_client():
    """Initialize and return the GenAI client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
    return genai.Client()


def run_wiki_agent(user_message: str, history: list = None) -> tuple[str, list]:
    """
    Runs the agent for a turn, optionally taking history of prior conversation turns.
    It has access to two tools:
      1. search_wikipedia — searches the local Qdrant Wikipedia database (fast, offline)
      2. search_web       — searches the internet via Tavily (for recent/uncovered topics)

    The agent autonomously decides which tool to call based on the query.
    Returns a tuple of (agent_answer, all_sources_list).
    """
    client = get_gemini_client()
    clear_retrieved_chunks()
    clear_web_results()

    # System instruction guiding the agent's behavior with both tools
    system_instruction = (
        "You are a helpful and intelligent Agentic AI assistant with access to two search tools:\n"
        "1. `search_wikipedia`: Searches a local database of Wikipedia articles. "
        "Use this tool FIRST for any question about facts, history, science, geography, "
        "definitions, or general knowledge.\n"
        "2. `search_web`: Searches the live internet via Tavily. "
        "Use this tool ONLY IF the local Wikipedia search returns no relevant results, "
        "OR if the question is about recent events, current news, or very specific topics "
        "unlikely to be in the Wikipedia database.\n"
        "Always try `search_wikipedia` first before falling back to `search_web`. "
        "Answer strictly based on the retrieved context and keep answers concise. "
        "If neither tool returns useful information, respond with the single word 'unanswerable'."
    )

    # Process conversation history if provided
    formatted_history = []
    if history:
        for msg in history:
            role = getattr(msg, "role", msg.get("role") if isinstance(msg, dict) else None)
            text = getattr(msg, "text", msg.get("text") if isinstance(msg, dict) else None)
            if role and text:
                # Map role 'assistant' or 'model' to Gemini's expected 'model'
                gemini_role = "model" if role in ("model", "assistant") else "user"
                formatted_history.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=text)]
                    )
                )

    # Create the chat session with the parsed history
    chat = client.chats.create(
        model=MODEL_NAME,
        history=formatted_history if formatted_history else None,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[search_wikipedia, search_web],
            temperature=0.1,  # Keep it precise and factual
        )
    )

    try:
        response = chat.send_message(user_message)
        # Combine local chunks and web results into a single sources list
        all_sources = get_last_retrieved_chunks() + get_last_web_results()
        return response.text.strip(), all_sources
    except Exception as e:
        return f"Error executing agent: {str(e)}", []



def main():
    print("=" * 60)
    print("  RAG Wiki Agent — Command Line Interface")
    print("=" * 60)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nUser Query: {query}")
        print("Agent thinking...")
        answer, sources = run_wiki_agent(query)
        print(f"\nAgent Response:\n{answer}")
        if sources:
            print("\n--- Retrieved Sources ---")
            for i, src in enumerate(sources, 1):
                if src.get('source') == 'web':
                    print(f"[{i}] [WEB] {src['title']} | {src['url']}")
                else:
                    print(f"[{i}] [DB] Doc {src['doc_id']} Chunk {src['chunk_id']} (Score: {src['score']:.3f})")
        print("=" * 60)
    else:
        print("\nEntering interactive agent mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                query = input("\nAgent User > ").strip()
                if not query:
                    continue
                if query.lower() in ['exit', 'quit']:
                    break

                print("Thinking...")
                answer, sources = run_wiki_agent(query)
                print(f"\nAgent:\n{answer}")
                if sources:
                    print("\n--- Retrieved Sources ---")
                    for src in sources:
                        if src.get('source') == 'web':
                            print(f"- [WEB] {src['title']} | {src['url']}")
                        else:
                            print(f"- [DB] Doc {src['doc_id']} Chunk {src['chunk_id']} (Score: {src['score']:.3f})")
                print("-" * 60)
            except KeyboardInterrupt:
                print()
                break


if __name__ == '__main__':
    main()

