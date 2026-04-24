import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

app = FastAPI()


class Question(BaseModel):
    question: str


def get_embedding(text):
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()


@app.post("/ask")
def ask(body: Question):
    question_embedding = get_embedding(body.question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context = "\n\n---\n\n".join(chunks)

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.question}"}
        ]
    )

    answer = response.choices[0].message.content

    sources = [
        {
            "title": m.get("title", m.get("question", "Unknown")),
            "source": m["source"],
            "distance": round(d, 4),
            "excerpt": c[:200]
        }
        for c, m, d in zip(chunks, metadatas, distances)
    ]

    return {"answer": answer, "sources": sources}