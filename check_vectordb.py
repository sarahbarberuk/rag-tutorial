import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pc_emporium")

# ---- Total chunk count ----
print("=== Total chunks in ChromaDB ===")
print(f"{collection.count()} chunks stored")
print()

# ---- Chunks per source ----
print("=== Chunks by source ===")
for source in ["policy", "faq", "blog"]:
    results = collection.get(where={"source": source})
    print(f"  {source}: {len(results['ids'])} chunks")
print()

# ---- Look at a sample chunk ----
print("=== Sample chunk ===")
results = collection.get(
    ids=["policy-returns-policy-2026-chunk-0"],
    include=["embeddings", "documents"]
)

print(f"Text content (first 300 characters):")
print(f"  {results['documents'][0][:300]}...")
print()

print(f"Vector embedding (a vector representation of the text above):")
embedding = results['embeddings'][0]
print(f"  Dimensions: {len(embedding)} numbers")
print(f"  First 10 values: {[round(float(v), 4) for v in embedding[:10]]}")
