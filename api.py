from fastapi import FastAPI
from fastapi.responses import JSONResponse
from rag_core import search_and_recompose, client

#attivazione api

app = FastAPI()

@app.get("/rag")
def rag(query: str, top_k: int = 3):
    try:
        docs = search_and_recompose(query, top_k=top_k)

        # Costruisci contesto
        context = "\n\n".join([
            f"TITOLO: {doc['title']}\nURL: {doc['url']}\n"
            f"DATA: {doc['date']}\nCATEGORIA: {doc['category']}\n"
            f"CONTENUTO:\n{doc['content']}"
            for doc in docs
        ])

        prompt = f"""Sei un assistente che risponde basandosi sui documenti di Confindustria.
Domanda utente: {query}

Documenti rilevanti:
{context}

Rispondi in modo chiaro e sintetico.
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        return {"query": query, "answer": answer, "documents": docs}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
