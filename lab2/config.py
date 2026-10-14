import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Solo la necesita la parte 1 (Pinecone). FAISS y Chroma corren 100% local.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

PINECONE_INDEX_NAME = "home-depot"
CHROMA_COLLECTION_NAME = "netflix"

DATA_DIR = Path(__file__).parent / "data"
