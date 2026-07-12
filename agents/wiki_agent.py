"""
Wiki Agent Module
=================
Defines the Agentic AI wrapper that uses the Gemini API and can autonomously
decide when to call the search_wikipedia tool to answer user questions.
"""

import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Add project root to sys.path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rag_tool import search_wikipedia, get_last_retrieved_chunks, clear_retrieved_chunks

# Load environment variables
load_dotenv()

# We use gemini-2.5-flash as it is extremely capable, fast, and supports tool use natively
MODEL_NAME = 'gemini-2.5-flash'


def get_gemini_client():
    """Initialize and return the GenAI client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
    return genai.Client()


def run_wiki_agent(user_message: str) -> tuple[str, list]:
    """
    Runs the agent for a single turn. It has access to the search_wikipedia tool.
    Returns a tuple of (agent_answer, retrieved_chunks_list).
    """
    client = get_gemini_client()
    clear_retrieved_chunks()

    # System instruction guiding the agent's behavior
    system_instruction = (
        "You are a helpful and intelligent Agentic AI assistant. "
        "You have access to a tool named `search_wikipedia` which allows you to query a local database. "
        "Whenever a user asks a question that requires facts, history, general knowledge, or specific details, "
        "you MUST call the `search_wikipedia` tool to gather correct context before formulating your answer. "
        "If you use the tool, answer the question based strictly on the retrieved facts and keep it concise. "
        "If the query cannot be answered by the tool, or if the tool returns no results, "
        "and you cannot find the answer, respond with the single word 'unanswerable'."
    )

    # We use chat interface because it automatically manages function calling execution loop
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[search_wikipedia],
            temperature=0.1,  # Keep it precise and factual
        )
    )

    try:
        response = chat.send_message(user_message)
        return response.text.strip(), get_last_retrieved_chunks()
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
        answer, chunks = run_wiki_agent(query)
        print(f"\nAgent Response:\n{answer}")
        if chunks:
            print("\n--- Retrieved Sources ---")
            for i, chunk in enumerate(chunks, 1):
                print(f"[{i}] Doc {chunk['doc_id']} Chunk {chunk['chunk_id']} (Score: {chunk['score']:.3f})")
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
                answer, chunks = run_wiki_agent(query)
                print(f"\nAgent:\n{answer}")
                if chunks:
                    print("\n--- Retrieved Sources ---")
                    for chunk in chunks:
                        print(f"- Doc {chunk['doc_id']} Chunk {chunk['chunk_id']} (Score: {chunk['score']:.3f})")
                print("-" * 60)
            except KeyboardInterrupt:
                print()
                break


if __name__ == '__main__':
    main()
