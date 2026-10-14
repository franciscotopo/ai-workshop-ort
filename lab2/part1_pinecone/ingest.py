# ========== Parte 1 — Pinecone: creación y carga del índice ==========

# Carga el dataset de Home Depot, genera los embeddings con all-MiniLM-L6-v2 y los inserta
# por lotes en un índice serverless de Pinecone (dimensión 384, métrica cosine).

from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

from common.data import load_home_depot
from common.embeddings import embed_batch
from config import EMBED_DIM, PINECONE_API_KEY, PINECONE_INDEX_NAME

BATCH_SIZE = 100


def get_index():
    assert PINECONE_API_KEY, "PINECONE_API_KEY is missing in .env."
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # has_index permite re-ejecutar el script sin error por "ya existe".
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,  # Debe coincidir con el tamaño del vector de MiniLM
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Región del plan gratuito
        )

    return pc.Index(PINECONE_INDEX_NAME)


def upsert_dataframe(index, df) -> None:
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[start:start + BATCH_SIZE]

        # Un solo encode por lote: la GPU/CPU procesa los 100 textos en paralelo.
        embeddings = embed_batch(batch["text"].tolist()).tolist()

        vectors = []
        for (_, row), emb in zip(batch.iterrows(), embeddings):
            vectors.append({
                "id": str(row["product_id"]),  # Pinecone exige ids string
                "values": emb,
                # La metadata vuelve con cada match: permite mostrar qué producto es.
                "metadata": {
                    "title": row["title"],
                    "brand": row["brand"],
                    "price": float(row["price"]),
                    "url": row["url"],
                },
            })

        # upsert = insert o update por id -> re-ejecutar no duplica.
        index.upsert(vectors=vectors)


if __name__ == "__main__":
    ### ========== Extracción y preparación de datos ==========
    df = load_home_depot()
    print(df.shape)
    print(df["text"].iloc[0][:300])

    ### ========== Creación y cargado del índice ==========
    index = get_index()
    upsert_dataframe(index, df)

    ### ========== Verificación ==========
    # En serverless las stats pueden tardar unos segundos en reflejar la carga.
    print(index.describe_index_stats())
    # Resultado obtenido: dimension=384, total_vector_count=2551, metric='cosine'
