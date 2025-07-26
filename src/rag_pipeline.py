from src.embedding import embed_text
from vector_store import search_similar
from src.llm_wrapper import run_llm

def run_rag_pipeline(query: str, top_k: int = 5):
    contextos = search_similar(query, embed_text, top_k=top_k)
    contexto = "\n".join(contextos)
    prompt = f"Use as frases abaixo para responder:\n\n{contexto}\n\nPergunta: {query}"
    return run_llm(prompt)