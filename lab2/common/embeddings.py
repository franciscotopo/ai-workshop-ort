from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL

### ========== Modelo de embeddings ==========

# Sin `device` explícito, sentence-transformers elige cuda > mps > cpu.
model = SentenceTransformer(EMBED_MODEL)


def get_embedding(text: str) -> List[float]:
    """Dado un texto, retorna su vector de embeddings como lista."""
    return model.encode(text).tolist()


def embed_batch(texts: List[str], normalize: bool = False) -> np.ndarray:
    """Embeddea una lista de textos en lote (mucho más rápido que de a uno).

    `normalize=True` deja cada vector con norma 1, así producto interno == similitud coseno.
    """
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 1,
        normalize_embeddings=normalize,
    )
