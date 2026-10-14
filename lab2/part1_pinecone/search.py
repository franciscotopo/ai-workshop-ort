# ========== Parte 1 — Pinecone: búsqueda semántica ==========

# Requiere haber corrido antes `python -m part1_pinecone.ingest`.

from common.embeddings import get_embedding
from part1_pinecone.ingest import get_index

index = get_index()


def search(query: str, top_k: int = 5) -> None:
    # La consulta se embeddea con el MISMO modelo que los productos, si no los vectores no son comparables.
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    print(f"\nQuery: {query}")
    for match in results.matches:
        m = match.metadata
        print(f"  {match.score:.3f} | {m['title']} | {m['brand']} | ${m['price']}")


if __name__ == "__main__":
    search("best tool for repairing walls")
    search("comfortable shoes for work")
    search("something to keep my dog warm")

# ========== Resultados y observaciones ==========

# Query: best tool for repairing walls
#   0.436 | 4 in. Eclipse Wall Repair Patch (2-Pack) | DAP | $23.96
#   0.424 | Wallpaper Scoring Tool (6-Pack) | Zinsser | $43.56
#   0.336 | ... "Bless The Food ..." Framed Wall Art | Stupell Industries | $28.16
#
# Query: comfortable shoes for work
#   0.524 | Men's Force Slip Resistant ... Work Boots - Composite Toe | Carhartt | $159.99
#   0.512 | Men's Rugged Flex Waterproof 8'' Work Boots - Composite Toe | Carhartt | $169.99
#
# Query: something to keep my dog warm
#   0.384 | Casey Small Rectangle Indoor/Outdoor Greystone Dog Bed | Happy Hounds | $36.82
#   0.370 | Large Tapestry Bolster Bed | Carolina Pet Company | $166.83
#   0.343 | Women's ... Heated Down Vest ... | ORORO | $179.99

# - "shoes for work" trae botas de trabajo aunque "shoes" no aparezca en el título: match semántico, no léxico.
# - "repairing walls": los 2 primeros son correctos; luego aparecen cuadros y wallpaper porque el modelo
#   asocia fuertemente "wall". El dataset (2.5k productos variados) tiene pocas herramientas de reparación.
# - Scores de 0.3-0.5 son normales en MiniLM: importa el orden relativo, no el valor absoluto.
