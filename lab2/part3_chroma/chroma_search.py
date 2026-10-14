# ========== Parte 3 — Chroma: base de datos adicional ==========

# Mismo ejercicio con Chroma sobre el dataset de Netflix, para comparar contra FAISS.
# A diferencia de FAISS, Chroma guarda id + vector + documento + metadata y permite filtrar.

import chromadb

from common.data import load_netflix
from common.embeddings import embed_batch, model
from config import CHROMA_COLLECTION_NAME

BATCH_SIZE = 1000  # Chroma limita la cantidad de items por llamada a `add` (~5k)

### ========== Datos y embeddings ==========

df = load_netflix()
embeddings = embed_batch(df["text"].tolist(), normalize=True)

### ========== Colección ==========

# Cliente en memoria (como FAISS). Para persistir: chromadb.PersistentClient(path=...).
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # Misma métrica que Pinecone y FAISS
)

### ========== Carga por lotes ==========

for start in range(0, len(df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = df.iloc[start:end]

    collection.add(
        ids=batch["show_id"].tolist(),
        embeddings=embeddings[start:end].tolist(),  # Reusamos los vectores; Chroma no acepta numpy
        documents=batch["text"].tolist(),
        # Chroma no acepta None en metadata -> fillna("") para cast.
        metadatas=batch[["title", "type", "listed_in"]]
            .assign(cast=batch["cast"].fillna(""))
            .to_dict("records"),
    )

print(collection.count())  # 8807


### ========== Búsqueda semántica (+ filtro por metadata) ==========

def search_chroma(query: str, top_k: int = 5, where: dict | None = None) -> None:
    # Pasamos el vector nosotros: con `query_texts` Chroma usaría su propia embedding function.
    q = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q, n_results=top_k, where=where)

    print(f"\nQuery: {query}  | filtro: {where}")
    # Chroma devuelve distancia coseno (1 - similitud): la convertimos para comparar con FAISS.
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  {1 - dist:.3f} | {meta['title']} ({meta['type']}) | {meta['listed_in']}")


if __name__ == "__main__":
    search_chroma("a heist where a group of experts steal money")
    search_chroma("something scary to watch at night")
    search_chroma("a romantic comedy")

    # Búsqueda híbrida: el filtro `where` restringe por metadata antes de rankear.
    search_chroma("a story about friendship", where={"type": "Movie"})
    search_chroma("a story about friendship", where={"type": "TV Show"})

# ========== Observaciones ==========

# - Sin filtro, los resultados coinciden con FAISS: mismos vectores y misma métrica. La diferencia
#   entre motores no está en la calidad del ranking sino en la infraestructura:
#     Pinecone -> id + vector + metadata, en la nube, gestionado.
#     FAISS    -> solo vectores, en memoria, sin filtros.
#     Chroma   -> id + vector + documento + metadata, local, con filtros (`where`, `where_document`).
