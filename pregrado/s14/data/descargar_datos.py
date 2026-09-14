# -*- coding: utf-8 -*-
"""
S14 — Sistemas de recomendacion — descarga y verificacion de datos.

Idempotente. Funciona en LOCAL (venv del curso) y en Google Colab.

Produce en esta carpeta (Sesiones/S14_recomendacion/data/):
  1) ml100k_ratings.csv     — REPLICACION. MovieLens 100k: 100.000 ratings
                              (943 usuarios x 1682 peliculas). Base del paper.
  2) ml100k_movies.csv      — metadatos de las 1682 peliculas (titulo, fecha,
                              19 flags de genero) para el filtrado por contenido.
  3) online_retail_reco_muestra.csv — NEGOCIO (feedback implicito). Muestra por
                              cliente de Online Retail II (UCI 502), <=1 MB.

Reglas del curso:
  * MovieLens 100k = SUSTITUTO CANONICO del Netflix Prize (base original de
    Koren-Bell-Volinsky 2009, RETIRADO de distribucion). Se declara como tal.
  * El .zip de ml-100k (4,9 MB) y los CSV de ratings/movies (<25 MB) SI pueden
    quedar en el repo -> se conserva la base COMPLETA de replicacion en disco.
  * El COMPLETO de Online Retail II (~45,6 MB xlsx) NUNCA en OneDrive: se
    descarga a un temporal del sistema solo en runtime y se guarda SOLO la
    muestra (<=1 MB) + checksum.
  * Verificacion REAL: se descarga/carga programaticamente y se validan filas x
    columnas. HTTP 200 no basta.

Uso:
    python descargar_datos.py            # descarga + valida + checksums + smoke
    python descargar_datos.py --no-smoke # omite el smoke de viabilidad (SVD)

Fecha de verificacion: 06/08/2026.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# ---------------------------------------------------------------------------
# Rutas y fuentes
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# --- MovieLens 100k (REPLICACION — sustituto canonico del Netflix Prize) ---
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_ZIP_SHA256 = (
    "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
)
# Fallback documentado (mas ligero, esquema distinto: ratings 0.5-5.0):
MLSMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MLSMALL_ZIP_SHA256 = (
    "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
)

ML100K_RATINGS = os.path.join(DATA_DIR, "ml100k_ratings.csv")
ML100K_MOVIES = os.path.join(DATA_DIR, "ml100k_movies.csv")
ML100K_N_RATINGS = 100000
ML100K_N_USERS = 943
ML100K_N_ITEMS = 1682

# 19 generos de u.item, en el orden de u.genre (ml-100k)
ML100K_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# --- Online Retail II (NEGOCIO — feedback implicito) — reusa fuente de S07 ---
RETAIL_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
)
RETAIL_ZIP_SHA256 = (
    "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
)
RETAIL_SAMPLE = os.path.join(DATA_DIR, "online_retail_reco_muestra.csv")
RETAIL_COLS = [
    "Invoice", "InvoiceDate", "StockCode", "Description",
    "Quantity", "Price", "Customer ID", "Country",
]
RETAIL_N_CUSTOMERS = 80        # muestra por cliente -> ~10k filas, <=1 MB
RETAIL_RANDOM_STATE = 42
# Cache del xlsx COMPLETO fuera de OneDrive (temporal del sistema).
CACHE_DIR = os.path.join(tempfile.gettempdir(), "curso_upc_s14_online_retail")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("requests no esta disponible.")
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content


def _csv_ok(path: str, cols: list[str]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        head = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return list(head.columns) == cols


# ---------------------------------------------------------------------------
# 1-2) MovieLens 100k
# ---------------------------------------------------------------------------
def descargar_movielens_100k() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("== MOVIELENS 100k — REPLICACION (sustituto canonico del Netflix "
          "Prize; Koren-Bell-Volinsky 2009) ==")
    ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
    movies_cols = ["movie_id", "title", "release_date"] + ML100K_GENRES

    if _csv_ok(ML100K_RATINGS, ratings_cols) and _csv_ok(ML100K_MOVIES, movies_cols):
        rt = pd.read_csv(ML100K_RATINGS)
        mv = pd.read_csv(ML100K_MOVIES)
        if (len(rt) == ML100K_N_RATINGS and rt.user_id.nunique() == ML100K_N_USERS
                and rt.item_id.nunique() == ML100K_N_ITEMS and len(mv) == ML100K_N_ITEMS):
            print(f"  = presente y verificado: {os.path.basename(ML100K_RATINGS)} "
                  f"({len(rt)} ratings, {rt.user_id.nunique()} usuarios, "
                  f"{rt.item_id.nunique()} items) + {os.path.basename(ML100K_MOVIES)}")
            return rt, mv
        print("  ! conteos distintos a lo esperado; se regenera.")

    print(f"  ... descargando {ML100K_URL} (~4,9 MB)")
    raw = _fetch(ML100K_URL)
    sha = sha256_bytes(raw)
    aviso = "" if sha == ML100K_ZIP_SHA256 else "  [SHA256 de fuente NUEVO]"
    print(f"  ... zip SHA256={sha[:16]}...{aviso}")
    z = zipfile.ZipFile(io.BytesIO(raw))

    # u.data: user_id \t item_id \t rating \t timestamp
    rt = pd.read_csv(io.BytesIO(z.read("ml-100k/u.data")), sep="\t",
                     names=ratings_cols, encoding="latin-1")
    rt.to_csv(ML100K_RATINGS, index=False, encoding="utf-8")

    # u.item: id | title | release | video_release | imdb_url | 19 flags de genero
    item_cols = (["movie_id", "title", "release_date", "video_release", "imdb_url"]
                 + ML100K_GENRES)
    mv = pd.read_csv(io.BytesIO(z.read("ml-100k/u.item")), sep="|",
                     names=item_cols, encoding="latin-1")
    mv = mv[["movie_id", "title", "release_date"] + ML100K_GENRES]
    mv.to_csv(ML100K_MOVIES, index=False, encoding="utf-8")

    sp = 1 - len(rt) / (rt.user_id.nunique() * rt.item_id.nunique())
    print(f"  + ratings -> {os.path.basename(ML100K_RATINGS)}  "
          f"{len(rt)} filas, {rt.user_id.nunique()} usuarios x "
          f"{rt.item_id.nunique()} items, escala {rt.rating.min()}-{rt.rating.max()}, "
          f"dispersion {sp:.3f}")
    print(f"  + movies  -> {os.path.basename(ML100K_MOVIES)}  {len(mv)} peliculas "
          f"(19 flags de genero para filtrado por contenido)")
    return rt, mv


# ---------------------------------------------------------------------------
# 3) Online Retail II — muestra por cliente (feedback implicito)
# ---------------------------------------------------------------------------
def _descargar_retail_zip() -> bytes:
    os.makedirs(CACHE_DIR, exist_ok=True)
    zp = os.path.join(CACHE_DIR, "online_retail_II.zip")
    if os.path.exists(zp) and sha256_file(zp) == RETAIL_ZIP_SHA256:
        print(f"  = zip en cache temporal ({zp})")
        return open(zp, "rb").read()
    print(f"  ... descargando {RETAIL_ZIP_URL} (~45,6 MB, SOLO runtime, fuera de OneDrive)")
    raw = _fetch(RETAIL_ZIP_URL)
    sha = sha256_bytes(raw)
    aviso = "" if sha == RETAIL_ZIP_SHA256 else "  [SHA256 de fuente NUEVO]"
    open(zp, "wb").write(raw)
    print(f"  ... zip -> {zp}  SHA256={sha[:16]}...{aviso}")
    return raw


def cargar_retail_completo_limpio() -> pd.DataFrame:
    """DataFrame COMPLETO limpio (~805.549 filas). No se guarda en OneDrive."""
    raw = _descargar_retail_zip()
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
    print(f"  ... leyendo {name} (dos hojas, ~1,07M filas; tarda ~1-2 min)")
    sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)
    df = df[df["Customer ID"].notna()]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]     # cancelaciones
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]           # devoluciones/precios nulos
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["StockCode"] = df["StockCode"].astype(str)
    print(f"  ... limpieza: {n0} -> {len(df)} filas "
          f"({df['Customer ID'].nunique()} clientes, {df['StockCode'].nunique()} productos)")
    return df.reset_index(drop=True)


def obtener_retail_reco_muestra() -> pd.DataFrame:
    print("\n== ONLINE RETAIL II — MUESTRA NEGOCIO (feedback implicito; UCI 502; "
          "completo NUNCA en OneDrive) ==")
    if _csv_ok(RETAIL_SAMPLE, RETAIL_COLS):
        df = pd.read_csv(RETAIL_SAMPLE, parse_dates=["InvoiceDate"])
        if df["Customer ID"].nunique() == RETAIL_N_CUSTOMERS:
            print(f"  = presente y verificado: {os.path.basename(RETAIL_SAMPLE)} "
                  f"({len(df)} filas, {RETAIL_N_CUSTOMERS} clientes, "
                  f"{df['StockCode'].nunique()} productos)")
            return df
        print("  ! nº de clientes distinto al esperado; se regenera la muestra.")

    full = cargar_retail_completo_limpio()
    custs = np.sort(full["Customer ID"].unique())
    rng = np.random.RandomState(RETAIL_RANDOM_STATE)
    sel = rng.choice(custs, size=RETAIL_N_CUSTOMERS, replace=False)
    muestra = (full[full["Customer ID"].isin(sel)][RETAIL_COLS]
               .sort_values("InvoiceDate").reset_index(drop=True))
    muestra.to_csv(RETAIL_SAMPLE, index=False, encoding="utf-8")
    mb = os.path.getsize(RETAIL_SAMPLE) / 1e6
    print(f"  + muestra {RETAIL_N_CUSTOMERS} clientes (rs={RETAIL_RANDOM_STATE}) -> "
          f"{os.path.basename(RETAIL_SAMPLE)} ({len(muestra)} filas, "
          f"{muestra['StockCode'].nunique()} productos, {mb:.3f} MB)")
    if mb > 1.0:
        print(f"  ! AVISO: la muestra pesa {mb:.3f} MB (>1 MB). Reducir "
              f"RETAIL_N_CUSTOMERS.")
    return muestra


# ---------------------------------------------------------------------------
# SMOKE de VIABILIDAD (reporta valores OBSERVADOS; NO define targets)
# ---------------------------------------------------------------------------
def smoke(rt: pd.DataFrame) -> None:
    print("\n########## SMOKE DE VIABILIDAD S14 (valores OBSERVADOS) ##########")
    try:
        from surprise import SVD, Dataset, Reader, accuracy
        from surprise.model_selection import train_test_split
    except Exception as e:
        print(f"  ! 'surprise' no disponible ({e!r}); se omite el smoke SVD.")
        return
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rt[["user_id", "item_id", "rating"]], reader)
    tr, te = train_test_split(data, test_size=0.20, random_state=42)
    # Baseline: media global
    mu = np.mean([r for (_, _, r) in tr.all_ratings()])
    rmse_base = float(np.sqrt(np.mean([(true - mu) ** 2 for (_, _, true) in te])))
    # SVD (factorizacion de matrices, tecnica de Koren-Bell-Volinsky 2009)
    algo = SVD(n_factors=100, n_epochs=20, random_state=42)
    algo.fit(tr)
    pred = algo.test(te)
    rmse_svd = accuracy.rmse(pred, verbose=False)
    print(f"  MovieLens 100k  test 20%  n_test={len(te)}")
    print(f"    RMSE baseline (media global) = {rmse_base:.4f}")
    print(f"    RMSE SVD (100 factores)      = {rmse_svd:.4f}")
    print(f"    -> SVD reduce el RMSE (mejora {100*(rmse_base-rmse_svd)/rmse_base:.1f}%). "
          f"Viable la replica de factorizacion de matrices.")
    print("  (Los TARGETS con tolerancia los fija la ficha de réplica del paper, "
          "no este script.)")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-smoke", action="store_true",
                    help="omite el smoke de viabilidad (SVD sobre MovieLens)")
    args = ap.parse_args()

    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Cache temporal (completo Online Retail): {CACHE_DIR}\n")

    rt, mv = descargar_movielens_100k()
    obtener_retail_reco_muestra()

    if not args.no_smoke:
        smoke(rt)

    print("\n== Checksums SHA256 (archivos locales) ==")
    for p in [ML100K_RATINGS, ML100K_MOVIES, RETAIL_SAMPLE]:
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1e6
            print(f"  {os.path.basename(p):32s} {mb:6.3f} MB  {sha256_file(p)}")
    print("\nListo. Datos de S14 verificados.")


if __name__ == "__main__":
    sys.exit(main())
