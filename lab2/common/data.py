import glob
import os

import kagglehub
import pandas as pd

### ========== Datasets ==========

HOME_DEPOT_DATASET = "thedevastator/the-home-depot-products-dataset"
NETFLIX_DATASET = "shivamb/netflix-shows"


def _read_csv_from_kaggle(dataset: str) -> pd.DataFrame:
    # kagglehub descarga y cachea el dataset; tomamos el primer CSV que contenga.
    path = kagglehub.dataset_download(dataset)
    csv_path = glob.glob(os.path.join(path, "*.csv"))[0]
    return pd.read_csv(csv_path)


def load_home_depot() -> pd.DataFrame:
    """Home Depot (~2.5k productos). Columna `text` = título + marca + descripción."""
    df = _read_csv_from_kaggle(HOME_DEPOT_DATASET)

    # Evita ids repetidos: en Pinecone un id duplicado pisa al vector anterior.
    df = df.drop_duplicates(subset="product_id").reset_index(drop=True)

    df["text"] = (
        "Title: " + df["title"].str.strip() +
        ". Brand: " + df["brand"].str.strip() +
        ". Description: " + df["description"].str.strip()
    )
    return df


def load_netflix() -> pd.DataFrame:
    """Netflix titles (~8.8k filas). Columna `text` = título + tipo + géneros + sinopsis."""
    df = _read_csv_from_kaggle(NETFLIX_DATASET)

    # director / cast / country quedan afuera: muchos nulos y los nombres propios meten ruido.
    df["text"] = (
        "Title: " + df["title"].str.strip() +
        ". Type: " + df["type"] +
        ". Genres: " + df["listed_in"] +
        ". Description: " + df["description"].str.strip()
    )
    return df
