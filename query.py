import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to the existing ChromaDB database created by ingest.py
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pc_emporium")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
N_RESULTS = 5

SYSTEM_PROMPT = """You are a helpful assistant for PC Emporium, a PC components retailer.
Answer the user's question using only the context provided below.
If the context does not contain enough information to answer the question, say:
"I'm sorry, I don't have enough information to answer that question."
Do not make up information that is not in the context."""


def get_embedding(text):
    """Generate an embedding for a piece of text using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def retrieve(question):
    """
    Convert the question to an embedding and query ChromaDB
    to find the most semantically similar chunks.
    """
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"]
    )

    return results


def generate(question, context_chunks):
    """
    Pass the question and retrieved context chunks to the LLM
    and return its answer.
    """
    # Format the retrieved chunks into a single context string
    context = "\n\n---\n\n".join(context_chunks)

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    return response.choices[0].message.content


def ask(question):
    """
    Full RAG pipeline: retrieve relevant chunks then generate an answer.
    """
    print(f"\nQuestion: {question}")
    print("\nSearching knowledge base...")

    # Step 1: Retrieve the most relevant chunks from ChromaDB
    results = retrieve(question)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Show the reader what was retrieved (for transparency)
    print(f"\nTop {N_RESULTS} most relevant chunks found:")
    for i, (chunk, metadata, distance) in enumerate(zip(chunks, metadatas, distances)):
        print(f"\n  [{i+1}] Source: {metadata['source']} | Title: {metadata.get('title', metadata.get('question', 'N/A'))} | Distance: {round(distance, 4)}")
        print(f"       {chunk[:150]}...")

    # Step 2: Generate an answer using the retrieved chunks as context
    print("\nGenerating answer...")
    answer = generate(question, chunks)

    print(f"\nAnswer: {answer}")
    return answer


if __name__ == "__main__":
    print("PC Emporium RAG — type your question or 'quit' to exit.")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        ask(question)