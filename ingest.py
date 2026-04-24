import os
import json
import glob
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up ChromaDB with persistent storage so the data survives between runs
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pc_emporium")

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text):
    """Generate an embedding for a piece of text using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def chunk_text(text, min_chunk_length=100):
    """
    Split text into chunks by paragraph (double newline).
    Chunks shorter than min_chunk_length characters are discarded
    as they are unlikely to contain useful information on their own.
    """
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
    return chunks


def ingest_policies():
    """
    Load policies from individual markdown files in data/policies/,
    chunk the text, generate an embedding for each chunk, and store in ChromaDB.
    """
    print("\n--- Ingesting policies ---")

    policy_files = glob.glob("data/policies/*.md")

    for filepath in policy_files:
        filename = os.path.basename(filepath)
        slug = filename.replace(".md", "")

        with open(filepath, "r") as f:
            content = f.read()

        lines = content.split("\n")
        title = lines[0].replace("# ", "").strip()
        body = "\n\n".join(lines[1:]).strip()

        chunks = chunk_text(body)
        print(f"  {title}: {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            chunk_id = f"policy-{slug}-chunk-{i}"
            embedding = get_embedding(chunk)

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": "policy",
                    "slug": slug,
                    "title": title
                }]
            )

    print(f"  Done. {len(policy_files)} policies ingested.")

def ingest_faqs():
    """
    Load FAQs from faqs.json, combine the question and answer into a single
    chunk, generate an embedding, and store in ChromaDB.
    Each FAQ is treated as a single chunk as they are short by design.
    """
    print("\n--- Ingesting FAQs ---")

    with open("data/faqs.json", "r") as f:
        faqs = json.load(f)

    for faq in faqs:
        # Combine question and answer so the embedding captures both
        text = f"Q: {faq['question']}\nA: {faq['answer']}"
        embedding = get_embedding(text)
        print(f"  {faq['question']}: 1 chunk")

        collection.add(
            ids=[faq["id"]],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "source": "faq",
                "category": faq["category"],
                "question": faq["question"]
            }]
        )

    print(f"  Done. {len(faqs)} FAQs ingested.")

def ingest_blog_posts():
    """
    Load each markdown file from data/blog-posts/, chunk the body text,
    generate an embedding for each chunk, and store in ChromaDB.
    The title (taken from the first line) and slug are stored as metadata.
    """
    print("\n--- Ingesting blog posts ---")

    blog_files = glob.glob("data/blog-posts/*.md")

    for filepath in blog_files:
        filename = os.path.basename(filepath)
        slug = filename.replace(".md", "")

        with open(filepath, "r") as f:
            content = f.read()

        # The first line of each markdown file is the title (e.g. # My Title)
        lines = content.split("\n")
        title = lines[0].replace("# ", "").strip()
        body = "\n\n".join(lines[1:]).strip()

        chunks = chunk_text(body)
        print(f"  {title}: {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            chunk_id = f"blog-{slug}-chunk-{i}"
            embedding = get_embedding(chunk)

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": "blog",
                    "slug": slug,
                    "title": title
                }]
            )

    print(f"  Done. {len(blog_files)} blog posts ingested.")


if __name__ == "__main__":
    print("Starting ingestion...")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    ingest_policies()
    ingest_faqs()
    ingest_blog_posts()

    total = collection.count()
    print(f"\nIngestion complete. {total} chunks stored in ChromaDB.")