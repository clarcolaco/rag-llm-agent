from uuid import uuid4
from fastapi import FastAPI
from src.models import QueryRequest, QueryResponse, FraseInput
from src.rag_pipeline import run_rag_pipeline
from src.embedding import embed_text
from src.vector_store import insert_doc

app = FastAPI(title="RAG Agent Organizacional")

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    resposta = run_rag_pipeline(request.query)
    return QueryResponse(answer=resposta)

@app.post("/frase")
def adicionar_frase(frase: FraseInput):
    insert_doc(id=int(uuid4().int >> 64), text=frase.texto, embedding=embed_text(frase.texto))
    return {"msg": "Frase adicionada com sucesso"}