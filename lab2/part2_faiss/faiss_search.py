# ========== Parte 2 — FAISS: búsqueda semántica local ==========

# Mismo flujo que con Pinecone pero con otro dataset de Kaggle (Netflix titles, ~8.8k filas)
# y un índice FAISS en memoria: sin cuenta, sin API key, sin red.

import faiss

from common.data import load_netflix
from common.embeddings import embed_batch, model

### ========== Cargar y preparar el dataset ==========

df = load_netflix()
print(df.shape)
print(df["text"].iloc[0])

### ========== Generar embeddings ==========

# normalize=True -> norma 1, así el producto interno (IP) equivale a similitud coseno.
embeddings = embed_batch(df["text"].tolist(), normalize=True)
print(embeddings.shape, embeddings.dtype)  # (8807, 384) float32

### ========== Crear el índice FAISS y agregar embeddings ==========

# IndexFlatIP: búsqueda exacta (compara contra todos) por producto interno.
# Para ~9k vectores es instantáneo; para millones convendría un índice aproximado (IVF / HNSW).
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)
print(faiss_index.ntotal)  # 8807

# FAISS no guarda ids ni metadata: devuelve la POSICIÓN del vector, que mapea 1:1 con la fila de `df`.
# Por eso `df` no se debe reordenar ni filtrar después de este punto.


### ========== Búsquedas semánticas ==========

def search_faiss(query: str, top_k: int = 5) -> None:
    # Va entre corchetes porque FAISS espera una matriz (1 fila).
    q = model.encode([query], normalize_embeddings=True)
    scores, positions = faiss_index.search(q, top_k)

    print(f"\nQuery: {query}")
    for score, pos in zip(scores[0], positions[0]):
        row = df.iloc[pos]
        print(f"  {score:.3f} | {row['title']} ({row['type']}) | {row['listed_in']}")


if __name__ == "__main__":
    # Concepto, sin palabras clave obvias
    search_faiss("a heist where a group of experts steal money")
    search_faiss("documentary about space and the universe")

    # Emoción / tono
    search_faiss("a feel-good movie to watch with my family")
    search_faiss("something scary to watch at night")

    # Trama específica
    search_faiss("a detective solving murders in a small town")
    search_faiss("teenagers with supernatural powers")

    # Límites del modelo
    search_faiss("una comedia romántica")
    search_faiss("a romantic film or romcom where Julia Roberts acts")

# ========== Observaciones ==========

# 1) Idioma — "una comedia romántica":
#      0.466 | Natalia Valdebenito: Gritona (Movie) | Stand-Up Comedy
#    all-MiniLM-L6-v2 se entrenó casi solo en inglés: con español matchea por parecido superficial
#    ("comedia" ~ "Comedy") y no capta "romántica". La misma consulta en inglés sí trae romcoms.
#    Alternativa: `paraphrase-multilingual-MiniLM-L12-v2` (también 384 dims) mapea ambos idiomas
#    a vectores cercanos; bastaría cambiar EMBED_MODEL y re-generar embeddings e índice.
#
# 2) Datos fuera del texto — "a romantic film or romcom where Julia Roberts acts":
#      0.607 | The New Romantic (Movie) | Comedies, Independent Movies, Romantic Movies
#      0.587 | Lady J (Movie) | Dramas, Independent Movies, International Movies
#      0.569 | Julie & Julia (Movie) | Comedies, Dramas
#    El género se respeta, pero el actor no: `cast` no está en el texto embeddeado, así que el modelo
#    no puede saberlo (solo matchea "Julia" por parecido léxico). Un dato exacto como "quién actúa"
#    se resuelve mejor con filtros por metadata (búsqueda híbrida), que FAISS no tiene -> ver parte 3.
