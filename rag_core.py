import os
import json
from openai import OpenAI
from pinecone import Pinecone
from urllib.parse import unquote

# Client OpenAI e Pinecone
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "confindustria-posts"

index = pc.Index(INDEX_NAME)

def decode(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unquote(s)

def build_text(item):
    def decode_local(s):
        return s.replace("%20", " ")
    parts = [
        decode_local(item.get("title", "")),
        decode_local(item.get("date", "")),
        decode_local(item.get("url", "")),
        decode_local(item.get("category", "")),
        decode_local(item.get("categoryfull", "")),
        decode_local(item.get("content", ""))
    ]
    return "\n".join([p for p in parts if p])

def chunk_text(text: str, max_chars: int = 2000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def embed_text(text: str):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# -------------------------
# RICERCA + ricomposizione
# -------------------------
def search_and_recompose(query: str, top_k: int = 5):
    query_embedding = embed_text(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    docs = {}
    for match in results["matches"]:
        metadata = match["metadata"]
        unid = metadata.get("unid")

        if unid not in docs:
            docs[unid] = {
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "date": metadata.get("date"),
                "category": metadata.get("category"),
                "chunks": {}
            }

        docs[unid]["chunks"][metadata.get("chunk_index")] = metadata.get("text")

    recomposed_docs = []
    for unid, doc in docs.items():
        ordered_chunks = [doc["chunks"][i] for i in sorted(doc["chunks"])]
        full_text = "\n".join(ordered_chunks)
        recomposed_docs.append({
            "unid": unid,
            "title": doc["title"],
            "url": doc["url"],
            "date": doc["date"],
            "category": doc["category"],
            "content": full_text
        })

    return recomposed_docs
