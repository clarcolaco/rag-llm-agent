from src.embedding import embed_text
from src.vector_store import insert_doc

with open("data/phrases.txt", "r") as f:
    frases = f.readlines()

for idx, frase in enumerate(frases):
    vetor = embed_text(frase.strip())
    insert_doc(id=idx, text=frase.strip(), embedding=vetor)
