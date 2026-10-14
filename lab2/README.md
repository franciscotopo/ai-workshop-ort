# 📖 Contenidos:

- Modelos de embedding: transformar texto en vectores con `all-MiniLM-L6-v2` (sentence-transformers, 384 dimensiones).
- Bases de datos vectoriales: almacenamiento y búsqueda semántica con **Pinecone**, **FAISS** y **Chroma**.
- Carga por lotes, métricas de similitud (cosine / inner product) y filtros por metadata (búsqueda híbrida).

## Al finalizar, podrás:

- Cargar un CSV de Kaggle y preparar un campo de texto combinando columnas relevantes.
- Generar embeddings y cargarlos en un índice vectorial (nube y local).
- Hacer búsquedas semánticas y entender sus límites (idioma del modelo, datos que no están en el texto).

# Lab 2 - Setup

> El laboratorio original es un notebook de Colab con GPU T4. Esta versión lo adapta a scripts locales:
> `google.colab.userdata` -> `.env`, `!pip install` -> `requirements.txt`, y el device del modelo se elige
> solo (cuda > mps > cpu). MiniLM es chico: corre bien en CPU.

## Pasos para reproducir el entorno

1. Crear entorno virtual: `python3 -m venv venv`
2. Activarlo: `source venv/bin/activate`
3. Instalar dependencias: `pip install -r requirements.txt`

## Configuración de Pinecone

1. Crear una cuenta en https://www.pinecone.io (plan Starter) y generar una API key.
2. Crear un archivo `.env` en `lab2/` con: `PINECONE_API_KEY=...` (ver `.env.example`).
3. El índice se crea solo: `home-depot`, dimensión 384, métrica cosine, serverless `aws/us-east-1`.

FAISS y Chroma no necesitan cuenta ni API key.

## Datasets

Se descargan automáticamente con `kagglehub` (quedan cacheados en `~/.cache/kagglehub`):

| Parte | Dataset | Filas | Texto embeddeado |
|---|---|---|---|
| Pinecone | [The Home Depot Products](https://www.kaggle.com/datasets/thedevastator/the-home-depot-products-dataset) | 2551 | title + brand + description |
| FAISS / Chroma | [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) | 8807 | title + type + listed_in + description |

## Estructura

```
lab2/
├── .env.example
├── config.py              # API key, modelo, dimensión, nombres de índice/colección
├── requirements.txt
├── common/
│   ├── data.py            # descarga de datasets + armado de la columna `text`
│   └── embeddings.py      # modelo MiniLM, get_embedding(), embed_batch()
├── part1_pinecone/
│   ├── ingest.py          # crear índice, upsert por lotes, describe_index_stats
│   └── search.py          # búsqueda semántica + resultados obtenidos
├── part2_faiss/
│   └── faiss_search.py    # IndexFlatIP con vectores normalizados + observaciones
└── part3_chroma/
    └── chroma_search.py   # colección cosine, carga por lotes, búsqueda con filtros `where`
```

## Cómo correr cada parte

Igual que en lab1, los scripts importan `config.py` y `common/` desde la raíz de `lab2`, así que se corren como módulo parado en `lab2/`:

```bash
python -m part1_pinecone.ingest   # una sola vez: crea y carga el índice
python -m part1_pinecone.search
python -m part2_faiss.faiss_search
python -m part3_chroma.chroma_search
```

## Conclusiones

- **Búsqueda semántica:** "comfortable shoes for work" trae botas de trabajo aunque "shoes" no esté en el título; se compara significado, no palabras.
- **Mismo modelo, mismo ranking:** con los mismos vectores y la misma métrica, FAISS y Chroma devuelven los mismos resultados. Lo que cambia entre motores es la infraestructura:

| | Pinecone | FAISS | Chroma |
|---|---|---|---|
| Dónde vive | Nube (gestionado) | Memoria local | Local (memoria o disco) |
| Qué guarda | id + vector + metadata | solo vectores | id + vector + documento + metadata |
| Filtros por metadata | Sí | No | Sí (`where`) |
| Setup | Cuenta + API key | `pip install` | `pip install` |

- **Límite de idioma:** `all-MiniLM-L6-v2` es casi solo inglés. "una comedia romántica" devuelve un stand-up en español por parecido superficial. Un modelo multilingüe (`paraphrase-multilingual-MiniLM-L12-v2`, también 384 dims) lo resolvería.
- **Lo que no está en el texto no se encuentra:** "romcom where Julia Roberts acts" respeta el género pero no el actor, porque `cast` no se embeddeó. Los datos exactos se resuelven mejor con filtros por metadata (búsqueda híbrida).
