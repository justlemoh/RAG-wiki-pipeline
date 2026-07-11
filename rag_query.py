"""
RAG Query Module for RAG Wiki Pipeline
======================================
Integrates semantic search retrieval with the Gemini API to answer
questions based on retrieved Wikipedia context.
"""

import os
import argparse
import sys
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from retrieve import retrieve

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = 'gemini-3.1-flash-lite'


def get_gemini_client():
    """Initialize and return the GenAI client using the environment API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set it in your environment (e.g. $env:GEMINI_API_KEY='your-key' in PowerShell)", file=sys.stderr)
        print("Attempting to initialize client without key (may fail)...", file=sys.stderr)
    
    # Client automatically picks up GEMINI_API_KEY
    return genai.Client()


def generate_answer(query, context_chunks):
    """Generate an answer using the retrieved context chunks and Gemini model."""
    client = get_gemini_client()

    # Form context string
    context_str = "\n\n".join([
        f"[Doc {chunk['doc_id']} Chunk {chunk['chunk_id']}]:\n{chunk['document']}"
        for chunk in context_chunks
    ])

    prompt = f"""You are a precise question-answering assistant. Answer the question using ONLY the provided context below.
If the answer to the question cannot be answered using the context, you MUST answer with the single word "unanswerable".
Keep your answers brief and factual.

Context:
{context_str}

Question: {query}
Answer:"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return response.text.strip()
    except APIError as e:
        return f"API Error generating answer: {e}"
    except Exception as e:
        return f"Unexpected error generating answer: {e}"


def run_rag_pipeline(query, k=3):
    """Run the end-to-end RAG pipeline for a given query."""
    # 1. Retrieve
    chunks = retrieve(query, k=k)
    if not chunks:
        return "No relevant context found.", []

    # 2. Generate
    answer = generate_answer(query, chunks)
    return answer, chunks


def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG Query System")
    parser.add_argument("--query", type=str, help="The question to ask the RAG pipeline")
    parser.add_argument("-k", type=int, default=3, help="Number of context chunks to retrieve (default: 3)")
    args = parser.parse_args()

    if args.query:
        print(f"\nQuestion: {args.query}")
        print("Running RAG pipeline...")
        answer, chunks = run_rag_pipeline(args.query, k=args.k)
        
        print("\n=== Retrieved Context ===")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[{i}] Doc {chunk['doc_id']} Chunk {chunk['chunk_id']} (Score: {chunk['score']:.4f})")
            print(chunk['document'])
        
        print("\n=== Generated Answer ===")
        print(answer)
        print("=" * 30)
    else:
        # Interactive loop
        print("\nEntering interactive RAG mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                query = input("\nAsk a question > ").strip()
                if not query:
                    continue
                if query.lower() in ['exit', 'quit']:
                    break

                answer, chunks = run_rag_pipeline(query, k=args.k)
                
                print("\n--- Retrieved Sources ---")
                for chunk in chunks:
                    print(f"- Doc {chunk['doc_id']} Chunk {chunk['chunk_id']} (Score: {chunk['score']:.3f})")
                
                print("\n--- Answer ---")
                print(answer)
                print("-" * 30)
            except KeyboardInterrupt:
                print()
                break


if __name__ == '__main__':
    main()
