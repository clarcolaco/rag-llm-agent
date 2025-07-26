from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

qdrant = QdrantClient("localhost", port=6333)
COLLECTION = "frases"

# Inicializa a coleção
qdrant.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

def insert_doc(id: int, text: str, embedding: list):
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=id, vector=embedding, payload={"text": text})]
    )

def search_similar(query: str, embed_func, top_k=5):
    vector = embed_func(query)
    hits = qdrant.search(collection_name=COLLECTION, query_vector=vector, limit=top_k)
    return [hit.payload["text"] for hit in hits]